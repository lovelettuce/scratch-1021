"""An isolated GPU allocation: exact pipeline modules, explicit checkpoint artifacts."""
from __future__ import annotations

import argparse
import logging
import os
import csv
import subprocess
from pathlib import Path
import time

from omegaconf import OmegaConf

from .common import digest, now, save_yaml, write_json


def execute(kind, config_path, resume=False):
    import torch
    # Establish the project's import order and custom resolvers before model instantiation.
    from src.utils import custom_hydra_resolvers  # noqa: F401
    from src.utils.launcher_utils import pipeline_launcher
    cfg = OmegaConf.load(config_path)
    output = Path(cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / '.hydra/config.yaml'
    if archive.exists():
        previous = OmegaConf.load(archive)
        if OmegaConf.to_container(previous) != OmegaConf.to_container(cfg):
            raise ValueError('Archived training configuration differs')
    else:
        save_yaml(archive, cfg)
    # Persist a resolved configuration as additional provenance. No Hydra runtime dependency.
    save_yaml(output / '.hydra/resolved.yaml', OmegaConf.to_container(cfg, resolve=True))
    required = int(cfg.trainer.devices)
    if not torch.cuda.is_available() or torch.cuda.device_count() < required:
        raise RuntimeError(f'Worker requires {required} visible CUDA GPUs, found {torch.cuda.device_count()}')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.set_float32_matmul_precision('medium' if kind == 'train' else 'highest')  # match original entry points
    last = output / 'checkpoints/last.ckpt'
    if kind == 'train' and resume and last.is_file():
        cfg.ckpt_path = str(last)
    started = time.monotonic()
    with pipeline_launcher(cfg) as modules:
        if kind == 'train':
            modules.trainer.fit(model=modules.model, datamodule=modules.datamodule, ckpt_path=cfg.ckpt_path)
            callback = modules.trainer.checkpoint_callback
            best = Path(callback.best_model_path)
            if not callback.best_model_path or not best.is_file():
                raise RuntimeError('Training ended without a monitored best checkpoint; refusing current-weight fallback')
            checkpoint = torch.load(best, map_location='cpu', weights_only=False)
            initialized = checkpoint.get('layers_initialized')
            if initialized is not None and not all(initialized):
                raise RuntimeError('Best tokenizer checkpoint contains uninitialized codebooks')
            best_score = callback.best_model_score
            result = {'best_checkpoint': str(best), 'best_checkpoint_sha256': digest(best),
                      'last_checkpoint': str(last), 'epoch': checkpoint['epoch'],
                      'global_step': checkpoint['global_step'], 'monitor': callback.monitor,
                      'score': float(best_score) if best_score is not None else None,
                      'training_ended_epoch': modules.trainer.current_epoch,
                      'training_ended_global_step': modules.trainer.global_step,
                      'seconds': time.monotonic() - started, 'device': torch.cuda.get_device_name(0),
                      'precision': str(cfg.trainer.precision), 'world_size': modules.trainer.world_size,
                      'batch_size_per_device': cfg.data_loading.datamodule.train_dataloader_config.get('batch_size_per_device'),
                      'configured_max_steps': cfg.trainer.max_steps, 'configured_max_epochs': cfg.trainer.max_epochs,
                      'finished_at': now(),
                      'resumed_from': cfg.ckpt_path,
                      'resume_limit': 'Optimizer, epoch and callback state restored; iterable mid-epoch order is not bit-exact.'}
            if cfg.get('test'):
                modules.trainer.test(model=modules.model, datamodule=modules.datamodule, ckpt_path=str(best))
            if modules.trainer.is_global_zero:
                write_json(output / 'training_result.json', result)
        else:
            if cfg.ckpt_path:
                # Strictly validate the architecture before Lightning performs prediction loading.
                ckpt = torch.load(cfg.ckpt_path, map_location='cpu', weights_only=False)
                if not all(ckpt.get('layers_initialized', [True])):
                    raise ValueError('Cannot export uninitialized tokenizer')
                modules.model.load_state_dict(ckpt['state_dict'], strict=True)
            modules.trainer.predict(model=modules.model, datamodule=modules.datamodule,
                                    ckpt_path=cfg.ckpt_path, return_predictions=False)
            if modules.trainer.is_global_zero:
                write_json(output / 'inference_result.json', {'seconds': time.monotonic() - started,
                           'finished_at': now(), 'checkpoint': cfg.ckpt_path})


def gpu_process_snapshot(gpu_ids):
    """Record existing compute owners before this probe creates CUDA contexts."""
    try:
        inventory = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
        processes = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
            '--format=csv,noheader,nounits'], text=True)
    except (OSError, subprocess.CalledProcessError) as ex:
        raise RuntimeError('Cannot inspect GPU availability using nvidia-smi') from ex
    uuids = {row[0].strip(): row[1].strip() for row in csv.reader(inventory.splitlines()) if row}
    active = [row for row in csv.reader(processes.splitlines()) if row]
    snapshot = {}
    for physical in gpu_ids:
        if physical not in uuids:
            raise RuntimeError(f'Physical GPU {physical} does not exist')
        snapshot[physical] = [{'pid': row[0].strip(), 'used_memory_mib': row[2].strip() if len(row) > 2 else None}
                              for row in active if len(row) >= 2 and row[1].strip() == uuids[physical]]
    return snapshot


def ensure_idle_gpus(gpu_ids):
    snapshot = gpu_process_snapshot(gpu_ids)
    for physical, owners in snapshot.items():
        if owners:
            raise RuntimeError(f'GPU {physical} is occupied by compute process(es) {[owner["pid"] for owner in owners]}; '
                               'one exclusive GPU per experiment is required')
    return snapshot


def probe(gpu_ids, minimum_gib, allow_gpu_sharing=False):
    """Allocate and synchronize on every requested device; NVML alone is insufficient."""
    snapshot = gpu_process_snapshot(gpu_ids) if allow_gpu_sharing else ensure_idle_gpus(gpu_ids)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < len(gpu_ids):
        raise RuntimeError(f'CUDA unavailable: requested physical GPUs {gpu_ids}; '
                           f'PyTorch sees {torch.cuda.device_count()}. Check host/container GPU access.')
    devices = []
    for position, physical in enumerate(gpu_ids):
        with torch.cuda.device(position):
            free, total = torch.cuda.mem_get_info()
            if free < minimum_gib * 1024 ** 3:
                raise RuntimeError(f'GPU {physical} has only {free / 1024 ** 3:.1f} GiB free')
            torch.ones(1, device=f'cuda:{position}').sum().item()
            torch.cuda.synchronize()
            devices.append({'physical_id': physical, 'name': torch.cuda.get_device_name(position),
                            'free_bytes': free, 'total_bytes': total,
                            'sharing_allowed': allow_gpu_sharing, 'existing_compute_processes': snapshot[physical]})
    return devices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['train', 'inference', 'probe'])
    parser.add_argument('--config')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--minimum-gib', type=float, default=12)
    parser.add_argument('--allow-gpu-sharing', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.kind == 'probe':
        import json
        print(json.dumps(probe(args.gpus.split(','), args.minimum_gib, args.allow_gpu_sharing)))
    else:
        execute(args.kind, args.config, args.resume)


if __name__ == '__main__':
    main()
