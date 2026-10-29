"""Configuration, provenance, atomic artifacts and resumable shards."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from . import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[3]
PATH_KEYS = ('data_dir', 'training_run_dir', 'checkpoint', 'semantic_id_path', 'output_root')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_config(path, max_samples=None):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    for key in PATH_KEYS:
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    if max_samples is not None:
        cfg['max_samples'] = max_samples
    for key in ('batch_size', 'bootstrap_replicates'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg.get('max_samples') is not None and cfg['max_samples'] < 1:
        raise ValueError('max_samples must be positive')
    if not cfg['beam_widths'] or min(cfg['beam_widths']) < max(cfg['k_values']):
        raise ValueError('Every beam width must be >= largest K')
    if len(set(cfg['beam_widths'])) != len(cfg['beam_widths']):
        raise ValueError('beam_widths must be unique')
    cfg['schema_version'] = SCHEMA_VERSION
    return cfg


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.stem, suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(Path(tmp))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    atomic_write(path, lambda p: p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'))


def write_csv(path, frame):
    atomic_write(path, lambda p: frame.to_csv(p, index=False, compression='gzip' if str(path).endswith('.gz') else None))


def write_pt(path, obj):
    import torch
    atomic_write(path, lambda p: torch.save(obj, p))


def read_json(path):
    return json.loads(Path(path).read_text())


def file_inventory(folder):
    paths = sorted(Path(folder).glob('*.tfrecord.gz'))
    if not paths:
        raise FileNotFoundError(f'No TFRecords in {folder}')
    return [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths]


def provenance(cfg):
    logging.getLogger(__name__).info("Fingerprinting training/testing data, SID, checkpoint and archived config")
    return {
        'schema_version': SCHEMA_VERSION,
        'training': file_inventory(Path(cfg['data_dir']) / 'training'),
        'testing': file_inventory(Path(cfg['data_dir']) / 'testing'),
        'checkpoint_sha256': digest(cfg['checkpoint']),
        'sid_sha256': digest(cfg['semantic_id_path']),
        'training_config_sha256': digest(Path(cfg['training_run_dir']) / '.hydra/config.yaml'),
    }


def environment():
    from importlib.metadata import version, PackageNotFoundError
    import torch
    deps = {}
    for name in ('torch', 'tensorflow', 'hydra-core', 'numpy', 'pandas', 'matplotlib', 'pytest'):
        try:
            deps[name] = version(name)
        except PackageNotFoundError:
            deps[name] = None
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return {'python': platform.python_version(), 'executable': os.sys.executable,
            'dependencies': deps, 'git_commit': git('rev-parse', 'HEAD'),
            'git_status': git('status', '--short'),
            'cuda_available': torch.cuda.is_available()}


def validate_identity(manifest, cfg, inputs):
    if manifest['config'] != cfg or manifest['inputs'] != inputs:
        raise ValueError('Resume refused: configuration or input fingerprints differ')


def completed_shards(run_dir, stage):
    """A marker is committed only after all associated artifacts are durable."""
    markers = sorted((Path(run_dir) / 'progress' / stage).glob('*.json'))
    for expected, marker in enumerate(markers):
        if int(marker.stem) != expected:
            raise ValueError(f'Non-contiguous shard markers: {stage}')
        info = read_json(marker)
        for artifact in info['artifacts']:
            p = Path(run_dir) / artifact['path']
            if not p.is_file() or digest(p) != artifact['sha256']:
                raise ValueError(f'Corrupt or missing committed artifact: {p}')
    return len(markers)


def commit_shard(run_dir, stage, index, keys, paths):
    run_dir = Path(run_dir)
    write_json(run_dir / 'progress' / stage / f'{index:06d}.json', {
        'sample_keys': keys,
        'artifacts': [{'path': str(Path(p).relative_to(run_dir)), 'sha256': digest(p)} for p in paths],
    })
