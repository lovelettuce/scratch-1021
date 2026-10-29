#!/usr/bin/env python3
"""Three persistent tmux workers for a prepared RQ cross-validation study."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def process_running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] not in ('Z', 'X')
    except FileNotFoundError:
        return False


def adopt_preparation(study, coordinator, inference):
    from src.experiments.rq_cross_validation.common import read_json, write_json, stage, now
    from src.experiments.rq_cross_validation.prepare import validate_embedding
    folder = study/'shared/toys/embedding_inference'
    while process_running(inference):
        print('Waiting for the existing Toys embedding inference to finish...', flush=True)
        time.sleep(15)
    receipt = folder/'inference_result.json'
    if not receipt.exists():
        raise RuntimeError(f'Shared inference did not complete; see {folder}/logs/process.log')
    manifest = read_json(study/'study_manifest.json')
    cfg = manifest['config']
    def commit_existing():
        result = validate_embedding(folder/'pickle/merged_predictions_tensor.pt',
                    read_json(study/'shared/toys/manifest.json')['catalog_items'], cfg['embedding_dim'])
        write_json(study/'shared/toys/embedding.json', result)
        return result, [folder/'pickle/merged_predictions_tensor.pt', receipt, study/'shared/toys/embedding.json']
    stage(study, 'toys_embeddings', {'study': manifest['identity']}, commit_existing)
    if process_running(coordinator):
        args = Path(f'/proc/{coordinator}/cmdline').read_bytes().split(b'\0')
        state = Path(f'/proc/{coordinator}/stat').read_text().split(') ', 1)[1][0]
        if b'src.experiments.rq_cross_validation' not in args or state != 'T':
            raise RuntimeError('Only the explicitly frozen original coordinator can be replaced')
        os.kill(coordinator, signal.SIGKILL)
    manifest.update(status='running', execution_backend='three_tmux_workers',
                    tmux_controller='scripts/run_rq_tmux.py', tmux_started_at=now())
    write_json(study/'study_manifest.json', manifest)
    write_json(study/'tmux/shared_ready.json', {'finished_at': now(), 'embedding_sha256': read_json(study/'shared/toys/embedding.json')['sha256']})


def wait_for(study, name, peers=False):
    from src.experiments.rq_cross_validation.common import read_json
    path = study/'tmux'/name
    last = 0
    while not path.exists():
        if peers:
            for status in (study/'tmux').glob('gpu*.state.json'):
                info = read_json(status)
                if info['status'] == 'failed':
                    raise RuntimeError(f'Peer worker failed: {status.name}: {info.get("error")}')
        if time.monotonic()-last > 30:
            print(f'Waiting for {name}', flush=True)
            last = time.monotonic()
        time.sleep(2)


def execute_cell(study, cfg, gpu, cell, phase):
    from src.experiments.rq_cross_validation.__main__ import isolated_env
    base = study/'experiments'/cell
    base.mkdir(parents=True, exist_ok=True)
    console = base/f'tmux_{phase}.log'
    offsets = {str(p): p.stat().st_size for p in base.rglob('logs/process.log')}
    offsets[str(console)] = console.stat().st_size if console.exists() else 0
    port = cfg['master_ports'][list(cfg['experiments']).index(cell)]
    command = [cfg['python'], '-u', '-m', 'src.experiments.rq_cross_validation.pipeline',
               '--study', str(study), '--cell', cell, '--stage', phase]
    print(f'\nSTART GPU {gpu}: {cell} / {phase}\n{shlex.join(command)}', flush=True)
    def mirror():
        for path in [console, *base.rglob('logs/process.log')]:
            if not path.exists():
                continue
            offset = offsets.get(str(path), 0)
            with path.open('rb') as source:
                source.seek(offset)
                chunk = source.read(256*1024)
                offsets[str(path)] = source.tell()
            if chunk:
                print(chunk.decode(errors='replace'), end='', flush=True)
    with console.open('ab') as output:
        process = subprocess.Popen(command, cwd=ROOT, env=isolated_env([gpu], port),
                                   stdout=output, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                mirror()
                time.sleep(.5)
            mirror()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if process.returncode:
        raise RuntimeError(f'{cell}/{phase} exited {process.returncode}; see {console}')
    print(f'FINISH GPU {gpu}: {cell} / {phase}', flush=True)


def worker(study, gpu, coordinator=None, inference=None):
    from src.experiments.rq_cross_validation.common import read_json, write_json, lock, now
    manifest = read_json(study/'study_manifest.json')
    cfg = manifest['config']
    assignments = {'0': ['beauty_rqkmeans', 'toys_rqkmeans'], '1': ['beauty_rqvae'], '2': ['toys_rqvae']}
    state = study/'tmux'/f'gpu{gpu}.state.json'
    def status(value, **extra):
        write_json(state, {'gpu': gpu, 'status': value, 'updated_at': now(), 'pid': os.getpid(), **extra})
    with lock(study/'tmux'/f'.gpu{gpu}.lock'):
        try:
            status('waiting_shared_inputs')
            if gpu == '0' and coordinator is not None:
                adopt_preparation(study, coordinator, inference)
            wait_for(study, 'shared_ready.json', peers=True)
            for cell in assignments[gpu]:
                status('running', cell=cell, phase='tokenizer')
                execute_cell(study, cfg, gpu, cell, 'tokenizer')
            write_json(study/'tmux'/f'sid_gpu{gpu}.done.json', {'finished_at': now()})
            status('waiting_sid_barrier')
            for peer in ('0', '1', '2'):
                wait_for(study, f'sid_gpu{peer}.done.json', peers=True)
            for cell in assignments[gpu]:
                status('running', cell=cell, phase='recommender_and_analysis')
                execute_cell(study, cfg, gpu, cell, 'all')
            write_json(study/'tmux'/f'all_gpu{gpu}.done.json', {'finished_at': now()})
            status('complete')
            if gpu == '0':
                for peer in ('0', '1', '2'):
                    wait_for(study, f'all_gpu{peer}.done.json', peers=True)
                from src.experiments.rq_cross_validation.report import report
                report(study)
                result = read_json(study/'study_manifest.json')
                result.update(status='complete', finished_at=now())
                write_json(study/'study_manifest.json', result)
        except BaseException as ex:
            status('failed', error=f'{type(ex).__name__}: {ex}')
            raise


def start(study, coordinator, inference):
    from src.experiments.rq_cross_validation.common import write_json, digest, now
    names = {'0': 'rqcv-beauty-rqkmeans-gpu0', '1': 'rqcv-beauty-rqvae-gpu1', '2': 'rqcv-toys-rqvae-gpu2'}
    for name in names.values():
        if subprocess.run(['tmux', 'has-session', '-t', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError(f'Session already exists: {name}')
    for gpu, name in names.items():
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--study', str(study), '--gpu', gpu]
        if gpu == '0' and coordinator is not None:
            command += ['--coordinator', str(coordinator), '--inference', str(inference)]
        subprocess.run(['tmux', 'new-session', '-d', '-s', name, '-c', str(ROOT), shlex.join(command)], check=True)
        subprocess.run(['tmux', 'set-option', '-w', '-t', name, 'remain-on-exit', 'on'], check=True)
        subprocess.run(['tmux', 'pipe-pane', '-t', name, '-o', f'cat >> {shlex.quote(str(study/"tmux"/f"gpu{gpu}.console.log"))}'], check=True)
    write_json(study/'tmux/sessions.json', {'sessions': names, 'started_at': now(),
               'runner_sha256': digest(__file__), 'original_coordinator': coordinator, 'original_inference': inference})
    print(names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['start', 'worker'])
    p.add_argument('--study', required=True)
    p.add_argument('--gpu', choices=['0', '1', '2'])
    p.add_argument('--coordinator', type=int)
    p.add_argument('--inference', type=int)
    args = p.parse_args()
    study = Path(args.study).resolve()
    (study/'tmux').mkdir(exist_ok=True)
    if args.command == 'start':
        start(study, args.coordinator, args.inference)
    else:
        worker(study, args.gpu, args.coordinator, args.inference)


if __name__ == '__main__':
    main()
