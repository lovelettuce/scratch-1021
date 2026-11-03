"""Explicit paths, immutable study identities and verified stage commits."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf

from src.analysis.long_tail.io import (
    ROOT, atomic_write, canonical_hash, digest, environment, file_inventory,
    read_json, write_csv, write_json,
)

EXPERIMENTS = ('beauty_rqkmeans', 'beauty_rqvae', 'toys_rqvae', 'toys_rqkmeans')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def absolute(value):
    path = Path(value).expanduser()
    return str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(path, embedding_model=None):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    from .embedding import select_model
    select_model(cfg, embedding_model)
    for key in ('python', 'output_root'):
        cfg[key] = absolute(cfg[key])
    cfg['datasets'] = {k: absolute(v) for k, v in cfg['datasets'].items()}
    cfg['recommender']['config'] = absolute(cfg['recommender'].get('config', 'configs/analysis/rq_recommender.yaml'))
    # Historical paths in old configurations are inert; only new-study inputs are used.
    cfg.pop('reference', None)
    cfg.setdefault('allow_gpu_sharing', False)
    if not isinstance(cfg['allow_gpu_sharing'], bool):
        raise ValueError('allow_gpu_sharing must be boolean')
    if cfg.get('study_version') != 2 or cfg['seed'] != 42 or tuple(cfg['experiments']) != EXPERIMENTS:
        raise ValueError('This study requires version 2, seed 42 and four newly trained cells')
    if cfg.get('tokenizer') != {'use_recipe_defaults': True, 'devices_per_experiment': 1}:
        raise ValueError('SID training uses recipe defaults with one GPU per experiment; remove old budget/batch overrides')
    if (cfg['semantic_levels'], cfg['codebook_width']) != (3, 256):
        raise ValueError('Expected the registered 3 × 256 semantic tokenizer')
    if cfg['analysis']['batch_size'] != 8 or cfg['analysis']['beam_widths'] != [10, 20, 50]:
        raise ValueError('Replay requires batch 8 and beams 10/20/50')
    return cfg


def source_hashes():
    paths = sorted(set((ROOT / 'src').rglob('*.py')) | set((ROOT / 'configs').rglob('*.yaml')) |
                   set((ROOT / 'scripts').glob('run_rq_cross_validation.sh')))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def input_identity(cfg):
    identities = {'data': {name: {split: file_inventory(Path(folder) / split)
                    for split in ('items', 'training', 'evaluation', 'testing')}
                  for name, folder in cfg['datasets'].items()},
                  'beauty_embedding': digest(cfg['beauty_embedding']) if cfg['beauty_embedding'] else None,
                  'recommender_config': digest(cfg['recommender']['config'])}
    # Pin model weights and tokenization assets, including symlink targets in the HF cache.
    snapshot = Path(cfg['embedding_snapshot'])
    assets = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*'))
              if p.is_file() and '.cache' not in p.relative_to(snapshot).parts
              and (p.suffix in ('.json', '.model', '.safetensors', '.bin'))}
    if not assets or not any('model' in p or 'pytorch_model' in p for p in assets):
        raise ValueError('Incomplete local FLAN-T5 snapshot')
    identities['embedding_assets'] = assets
    identities['embedding_model'] = {'model_id': cfg['embedding_model_id'], 'dimension': cfg['embedding_dim']}
    return identities


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as ex:
            raise RuntimeError(f'Another process holds {path}') from ex
        yield


def stage(directory, name, identity, action):
    """Only a verified commit can skip execution; partial directories are resumable."""
    directory = Path(directory)
    path = directory / 'progress' / f'{name}.json'
    key = canonical_hash(identity)
    if path.exists():
        state = read_json(path)
        if state['identity'] != key:
            raise ValueError(f'Resume identity differs for {name}')
        if state['status'] == 'complete':
            for artifact in state['artifacts']:
                p = directory / artifact['path']
                if not p.is_file() or digest(p) != artifact['sha256']:
                    raise ValueError(f'Committed artifact is missing or corrupt: {p}')
            return state['result']
    state = {'status': 'running', 'identity': key, 'started_at': now()}
    write_json(path, state)
    try:
        result, artifacts = action()
        state.update(status='complete', result=result, finished_at=now(), artifacts=[
            {'path': os.path.relpath(p, directory), 'sha256': digest(p)} for p in artifacts])
        write_json(path, state)
        return result
    except BaseException as ex:
        state.update(status='failed', error=f'{type(ex).__name__}: {ex}', finished_at=now())
        write_json(path, state)
        raise


def save_yaml(path, value):
    atomic_write(path, lambda p: OmegaConf.save(OmegaConf.create(value), p))


def validate_resume_model(cfg, manifest):
    previous = manifest['config']
    old = (previous.get('embedding_model_id', 'google/flan-t5-xl'), previous['embedding_dim'])
    if old != (cfg['embedding_model_id'], cfg['embedding_dim']):
        raise ValueError('Resume refused: embedding model differs; create a new study')
