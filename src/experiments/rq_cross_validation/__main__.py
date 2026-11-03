"""Four single-GPU experiments with a resource-aware queue for SID and GR stages."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .common import (EXPERIMENTS, ROOT, canonical_hash, environment, input_identity,
                     load_config, lock, now, read_json, validate_resume_model, save_yaml, source_hashes, stage, write_json)

LOGGER = logging.getLogger(__name__)


def gpu_ids(text):
    ids = [value.strip() for value in text.split(',')]
    if not ids or len(ids) != len(set(ids)) or any(not value.isdigit() for value in ids):
        raise ValueError('GPU IDs must be distinct physical device numbers')
    return ids


def schedule(cfg, experiments, gpu_text):
    """Assign experiments to the explicit GPU pool; shared allocations queue in order."""
    gpus = gpu_ids(gpu_text)
    ports = cfg['master_ports']
    if len(ports) != len(EXPERIMENTS) or len(set(ports)) != len(EXPERIMENTS) or any(not 1024 <= p <= 65535 for p in ports):
        raise ValueError('Configure four distinct unprivileged MASTER_PORT values')
    return [{'cell': cell, 'gpu': gpus[index % len(gpus)], 'master_port': ports[EXPERIMENTS.index(cell)]}
            for index, cell in enumerate(experiments)]


def sid_schedule(cfg, experiments, gpu_text):
    from .configuration import tokenizer_config
    jobs = schedule(cfg, experiments, gpu_text)
    for job in jobs:
        recipe = tokenizer_config(cfg, job['cell'], Path('/tmp/rq-plan'), cfg['beauty_embedding'])
        if int(recipe.trainer.devices) != 1:
            raise ValueError('SID jobs require one GPU per experiment')
    return jobs


def isolated_env(gpus, port=None):
    env = os.environ.copy()
    # Avoid inherited launch state from an unrelated DDP job.
    for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'NODE_RANK', 'GROUP_RANK'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpus), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', TF_CPP_MIN_LOG_LEVEL='3',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if port is not None:
        env.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    return env


def gpu_preflight(cfg, jobs):
    for job in jobs:
        with socket.socket() as s:
            s.bind(('127.0.0.1', job['master_port']))
    devices = list(dict.fromkeys(gpu for job in jobs for gpu in job['gpu'].split(',')))
    command = [cfg['python'], '-m', 'src.experiments.rq_cross_validation.worker', 'probe',
               '--gpus', ','.join(devices), '--minimum-gib', str(cfg['minimum_free_gpu_gib'])]
    if cfg.get('allow_gpu_sharing', False):
        command.append('--allow-gpu-sharing')
    result = subprocess.run(command, cwd=ROOT, env=isolated_env(devices), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError('GPU preflight failed; training was not started.\n' + result.stdout + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def shared_embeddings(cfg, study, job, dataset='toys'):
    from .configuration import embedding_config
    from .pipeline import run_model
    from .prepare import validate_embedding
    shared = study / 'shared' / dataset
    folder = shared / 'embedding_inference'
    n = read_json(shared / 'manifest.json')['catalog_items']
    def execute():
        reuse = cfg.get('beauty_embedding') if dataset == 'beauty' else None
        if reuse:
            path = Path(reuse)
            artifacts = [path]
        else:
            _, receipt = run_model(cfg, folder, embedding_config(cfg, folder, dataset), 'inference', False)
            path = folder / 'pickle/merged_predictions_tensor.pt'
            artifacts = [path, receipt]
        result = validate_embedding(path, n, cfg['embedding_dim'])
        result.update(model_id=cfg['embedding_model_id'], embedding_dim=cfg['embedding_dim'])
        write_json(shared / 'embedding.json', result)
        return result, [*artifacts, shared / 'embedding.json']
    previous = os.environ.copy()
    try:
        os.environ.update(isolated_env([job['gpu']], job['master_port']))
        stage(study, f'{dataset}_embeddings', {'study': read_json(study / 'study_manifest.json')['identity'],
              'model_id': cfg['embedding_model_id'], 'dimension': cfg['embedding_dim']}, execute)
    finally:
        os.environ.clear()
        os.environ.update(previous)


def start_job(cfg, study, job, selected_stage):
    directory = study / 'experiments' / job['cell']
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / 'pipeline.log').open('a')
    command = [cfg['python'], '-m', 'src.experiments.rq_cross_validation.pipeline',
               '--study', str(study), '--cell', job['cell'], '--stage', selected_stage]
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=isolated_env([job['gpu']], job['master_port']),
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        handle.close()
        raise
    write_json(directory / f'launch_{selected_stage}.json', {**job, 'argv': command, 'pid': process.pid, 'started_at': now()})
    LOGGER.info('Started %s on GPU %s, PID %s', job['cell'], job['gpu'], process.pid)
    return process, handle


def launch(cfg, study, jobs, selected_stage):
    """Start queued work as soon as its GPU is free, without waiting for other GPUs."""
    pending = list(jobs)
    active = {}
    failures = []
    try:
        while pending or active:
            for job in list(pending):
                if job['gpu'] in active:
                    continue
                process, handle = start_job(cfg, study, job, selected_stage)
                active[job['gpu']] = (job, process, handle)
                pending.remove(job)
            finished = []
            for gpu, (job, process, handle) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                handle.close()
                finished.append(gpu)
                if status:
                    failures.append(job['cell'])
                LOGGER.info('Finished %s on GPU %s, exit=%s', job['cell'], gpu, status)
            for gpu in finished:
                del active[gpu]
            if active and not finished:
                time.sleep(.25)
    except BaseException:
        import signal
        for _, process, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process, handle in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            handle.close()
        raise
    if failures:
        raise RuntimeError(f'Failed cells: {failures}; successful cells are retained for --resume')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/analysis/rq_cross_validation.yaml')
    parser.add_argument('--embedding-model', choices=['flan-t5-xl', 'flan-t5-base'], default=None)
    parser.add_argument('--gpus', help='GR/analysis GPU pool; each active experiment owns one GPU; default from config')
    parser.add_argument('--sid-gpus', help='SID GPU pool; queued experiments wait for their assigned GPU; default from config')
    parser.add_argument('--experiment', choices=['all', *EXPERIMENTS], default='all')
    parser.add_argument('--stage', choices=['all', 'prepare', 'tokenizer', 'recommender', 'analysis', 'summary'], default='all')
    parser.add_argument('--resume', help='An explicit existing study directory')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cfg = load_config(args.config, args.embedding_model)
    from .embedding import inspect_model
    model_status = inspect_model(cfg)
    experiments = list(EXPERIMENTS) if args.experiment == 'all' else [args.experiment]
    datasets = list(dict.fromkeys(cfg['experiments'][cell]['dataset'] for cell in experiments))
    jobs = schedule(cfg, experiments, args.gpus or cfg['gpus'])
    sid_jobs = sid_schedule(cfg, experiments, args.sid_gpus or cfg['sid_gpus']) if args.stage in ('all', 'tokenizer') else []
    if args.dry_run:
        print(json.dumps({'jobs': jobs, 'sid_jobs': sid_jobs, 'max_parallel_gr': len(set(j['gpu'] for j in jobs)), 'max_parallel_sid': len(set(j['gpu'] for j in sid_jobs)), 'stage': args.stage,
                          'steps': ['raw inputs/groups', 'shared dataset embeddings', 'tokenizer train', 'SID export',
                                    'TIGER train', 'long-tail smoke/full', 'second-layer smoke/full', 'four-cell report'],
                          'embedding': model_status,
                          'generate_embeddings': [d for d in datasets if d != 'beauty' or not cfg.get('beauty_embedding')],
                          'config': cfg}, ensure_ascii=False, indent=2))
        return
    if args.stage == 'summary':
        if not args.resume:
            parser.error('--stage summary requires --resume STUDY_DIR')
        from .report import report
        print(report(Path(args.resume)))
        return
    if args.resume:
        validate_resume_model(cfg, read_json(Path(args.resume).resolve() / 'study_manifest.json'))
    model_status = inspect_model(cfg, required=True)
    cfg['embedding_snapshot'] = model_status['snapshot']
    if Path(sys.executable).resolve() != Path(cfg['python']).resolve():
        raise ValueError('Run this study with the configured grid Python')
    if shutil.disk_usage(cfg['datasets']['beauty']).free < cfg['minimum_free_disk_gib'] * 1024**3:
        raise RuntimeError('Insufficient free disk space')
    from .configuration import recommender_config
    gr_template = recommender_config(cfg, experiments[0], Path('/tmp/rq-plan'), 'GENERATED_SID_PATH')
    # Validate all interpolations before creating a study or starting GPU jobs.
    from omegaconf import OmegaConf
    OmegaConf.to_container(gr_template, resolve=True)
    inputs = input_identity(cfg)
    code = source_hashes()
    identity = canonical_hash({'config': cfg, 'inputs': inputs, 'code': code})
    if args.resume:
        study = Path(args.resume).resolve()
        manifest = read_json(study / 'study_manifest.json')
        if manifest['identity'] != identity:
            raise ValueError('Resume refused: source code, configuration or input fingerprints differ')
    else:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        study = Path(cfg['output_root']) / stamp
        study.mkdir(parents=True, exist_ok=False)
        manifest = {'schema_version': 1, 'config': cfg, 'inputs': inputs, 'code': code,
                    'identity': identity, 'environment': environment(), 'status': 'created', 'started_at': now()}
        write_json(study / 'study_manifest.json', manifest)
        save_yaml(study / 'config.yaml', cfg)
        save_yaml(study / 'recommender_template.yaml', gr_template)
    (study / 'logs').mkdir(exist_ok=True)
    handler = logging.FileHandler(study / 'logs/study.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler)
    LOGGER.info('Study directory: %s', study)
    for cell in EXPERIMENTS:
        status_path = study / 'experiments' / cell / 'status.json'
        if not status_path.exists():
            write_json(status_path, {'status': 'pending', 'reason': 'Awaiting shared preparation and GPU preflight'})
    with lock(study / '.study.lock'), ExitStack() as resources:
        try:
            from .prepare import prepare_raw
            stage(study, 'raw_data', {'study': identity}, lambda: prepare_raw_action(cfg, study, prepare_raw))
            # Raw scans and frozen groups remain useful even if a GPU is unavailable.
            manifest.update(status='preflight', schedule=jobs, sid_schedule=sid_jobs)
            write_json(study / 'study_manifest.json', manifest)
            allocations = sid_jobs + jobs
            for gpu in dict.fromkeys(gpu for job in allocations for gpu in job['gpu'].split(',')):
                resources.enter_context(lock(Path(cfg['output_root']) / '.locks' / f'gpu-{gpu}.lock'))
            for port in dict.fromkeys(job['master_port'] for job in allocations):
                resources.enter_context(lock(Path(cfg['output_root']) / '.locks' / f'port-{port}.lock'))
            manifest['devices'] = gpu_preflight(cfg, allocations)
            write_json(study / 'study_manifest.json', manifest)
            for dataset in datasets:
                shared_embeddings(cfg, study, jobs[0], dataset)
            if args.stage != 'prepare':
                manifest['status'] = 'running'
                write_json(study / 'study_manifest.json', manifest)
                if sid_jobs:
                    launch(cfg, study, sid_jobs, 'tokenizer')
                if args.stage != 'tokenizer':
                    launch(cfg, study, jobs, args.stage)
            if args.stage in ('all', 'analysis') and all((study / 'experiments' / cell / 'progress/second_layer.json').exists() and read_json(study / 'experiments' / cell / 'progress/second_layer.json')['status'] == 'complete' for cell in EXPERIMENTS):
                from .report import report
                report(study)
                manifest['status'] = 'complete'
            else:
                manifest['status'] = 'stage_complete'
            manifest['finished_at'] = now()
            write_json(study / 'study_manifest.json', manifest)
        except BaseException as ex:
            LOGGER.exception('Study stopped before completion')
            manifest.update(status='failed', error=f'{type(ex).__name__}: {ex}', failed_at=now())
            write_json(study / 'study_manifest.json', manifest)
            raise
    print(study)


def prepare_raw_action(cfg, study, prepare):
    prepare(cfg, study)
    artifacts = [study / 'shared' / dataset / name for dataset in cfg['datasets'] for name in ('manifest.json', 'items.csv')]
    return {'datasets': list(cfg['datasets'])}, artifacts


if __name__ == '__main__':
    main()
