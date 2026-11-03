"""One isolated evaluator per physical GPU, then publish into a portable run."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from src.analysis.long_tail.io import (canonical_hash, commit_shard, completed_shards,
                                      read_json, write_json)
from .experiment import exclusive, now, validate_run

LOGGER = logging.getLogger(__name__)


def assignment(cells, gpus):
    devices = gpus.split(',')
    if len(devices) != len(cells) or len(set(devices)) != len(devices) or not all(x.isdigit() for x in devices):
        raise ValueError('Provide one distinct physical GPU index per selected cell')
    return dict(zip(cells, devices))


def probe_devices(devices):
    """Probe the same physical devices the isolated workers will actually use."""
    script = ('import json,torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
              'x=torch.ones(1,device="cuda:0"); torch.cuda.synchronize(); '
              'print(json.dumps(dict(name=torch.cuda.get_device_name(0), '
              'free_bytes=torch.cuda.mem_get_info(0)[0],allocation=x.item())))')
    import json
    result = {}
    for device in devices:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
        result[device] = json.loads(subprocess.check_output([sys.executable, '-c', script], env=env, text=True))
        if result[device]['free_bytes'] < 12*1024**3:
            raise RuntimeError(f'Physical GPU {device} has less than 12 GiB free')
    return result


def worker_run(run, manifest, cell):
    worker = run/'workers'/cell
    if worker.exists():
        previous = read_json(worker/'manifest.json')
        if previous.get('parent_identity') != manifest['identity']:
            raise ValueError('Worker belongs to a different prepared run')
        return worker
    # Each process owns its manifest, lock, snapshots and progress directory.
    for entry in manifest['snapshots']:
        if entry['path'].startswith(f'inputs/{cell}/'):
            dest = worker/entry['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run/entry['path'], dest)
    m = dict(manifest)
    m.update(cells=[cell], sources={cell:manifest['sources'][cell]},
             snapshots=[e for e in manifest['snapshots'] if e['path'].startswith(f'inputs/{cell}/')],
             cell_status={}, status='prepared', parent_identity=manifest['identity'],
             preflight={cell:manifest['preflight'][cell]})
    m['identity'] = canonical_hash({'parent':manifest['identity'], 'cell':cell})
    write_json(worker/'manifest.json', m)
    return worker


def publish(run, worker, cell):
    """Reference worker files without copying multi-GB score arrays."""
    m = read_json(worker/'manifest.json')
    if m['cell_status'].get(cell) != 'complete' or m['status'] not in ('evaluated', 'complete'):
        raise ValueError(f'Worker did not complete: {cell}')
    stage = f'scores_{cell}'
    count = completed_shards(worker, stage)
    existing = completed_shards(run, stage)
    if existing > count:
        raise ValueError('Parent contains excess worker shards')
    for index, marker in enumerate(sorted((worker/'progress'/stage).glob('*.json'))):
        info = read_json(marker)
        paths = [worker/entry['path'] for entry in info['artifacts']]
        if index < existing:
            old = read_json(run/'progress'/stage/marker.name)
            expected = [str(p.relative_to(run)) for p in paths]
            if old['sample_keys'] != info['sample_keys'] or [e['path'] for e in old['artifacts']] != expected:
                raise ValueError('Parent/worker committed identities disagree')
        else:
            commit_shard(run, stage, index, info['sample_keys'], paths)


def evaluate_parallel(run_dir, gpus, resume=False):
    run = Path(run_dir).resolve()
    with exclusive(run):
        m = read_json(run/'manifest.json')
        validate_run(run, m)
        if m['status'] != 'prepared' and not resume:
            raise ValueError('Existing evaluation requires --resume')
        assigned = assignment(m['cells'], gpus)
        if any(m['preflight'].get(c, {}).get('score_checks') != 'passed' for c in m['cells']):
            raise ValueError('All selected cells must pass preflight before parallel evaluation')
        if 'gpu_assignment' in m and m['gpu_assignment'] != assigned:
            raise ValueError('Resume GPU assignment differs')
        probes = probe_devices(list(assigned.values()))
        m.update(status='running', gpu_assignment=assigned, physical_gpu_probes=probes,
                 execution_backend='one_process_per_gpu', parallel_started_at=now())
        write_json(run/'manifest.json', m)
        jobs, completed, failures = {}, [], []
        started = time.monotonic()
        try:
            for cell, gpu in assigned.items():
                worker = worker_run(run, m, cell)
                previous = read_json(worker/'manifest.json')
                if previous['cell_status'].get(cell) == 'complete':
                    publish(run, worker, cell); completed.append(cell)
                    m['cell_status'][cell] = 'complete'
                    continue
                log = run/'logs'/f'{cell}_gpu{gpu}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-m', 'src.analysis.exact_catalog', 'evaluate',
                           '--run-dir', str(worker), '--device', 'cuda:0']
                if previous['status'] != 'prepared':
                    command.append('--resume')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                with log.open('a') as handle:
                    process = subprocess.Popen(command, env=env, stdout=handle,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                jobs[cell] = dict(process=process, worker=worker, log=log, started=time.monotonic())
                m.setdefault('workers', {})[cell] = dict(pid=process.pid, physical_gpu=gpu,
                    run_dir=str(worker.relative_to(run)), log=str(log.relative_to(run)), status='running')
                LOGGER.info('Started %s on physical GPU%s, PID %s', cell, gpu, process.pid)
                write_json(run/'manifest.json', m)
            while jobs:
                for cell, job in list(jobs.items()):
                    code = job['process'].poll()
                    if code is None:
                        continue
                    state = m['workers'][cell]
                    state.update(exit_code=code, wall_seconds=time.monotonic()-job['started'], finished_at=now())
                    if code == 0:
                        publish(run, job['worker'], cell)
                        state['status'] = 'complete'; m['cell_status'][cell] = 'complete'; completed.append(cell)
                    else:
                        state['status'] = 'failed'; failures.append(cell)
                    del jobs[cell]
                    write_json(run/'manifest.json', m)
                if jobs:
                    time.sleep(3)
            if failures:
                raise RuntimeError(f'Workers failed: {failures}; inspect their logs, then --resume')
            validate_run(run, m)
            m.update(status='evaluated', finished_at=now(), parallel_wall_seconds=time.monotonic()-started)
        except BaseException as ex:
            for job in jobs.values():
                job['process'].terminate()
            for job in jobs.values():
                try:
                    job['process'].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    job['process'].kill(); job['process'].wait()
            m.update(status='failed', error=f'{type(ex).__name__}: {ex}')
            raise
        finally:
            write_json(run/'manifest.json', m)
    return run
