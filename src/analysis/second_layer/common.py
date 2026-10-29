"""Public shared IO plus second-experiment provenance and source validation."""
from pathlib import Path

from omegaconf import OmegaConf

from src.analysis.long_tail.io import (ROOT, canonical_hash, completed_shards, digest,
    environment, file_inventory, load_config as base_config, read_json, write_json)


def load_config(path, max_samples=None):
    cfg = base_config(path, max_samples)
    p = Path(cfg['source_run_dir']).expanduser()
    cfg['source_run_dir'] = str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    if cfg['beam_widths'] != [10, 20, 50] or cfg['batch_size'] != 8:
        raise ValueError('Replay must use source beam widths 10/20/50 and batch size 8')
    return cfg


def source_paths(run, stage, suffix):
    paths = []
    for marker in sorted((Path(run)/'progress'/stage).glob('*.json')):
        for item in read_json(marker)['artifacts']:
            if item['path'].endswith(suffix):
                paths.append(Path(run)/item['path'])
    return paths


def source_dataset(run, manifest):
    p = Path(manifest['dataset_dir'])
    return p if p.exists() else Path(run).parent.parent/'datasets'/p.name


def source_identity(cfg):
    """Validate all immutable source artifacts before accepting cached inputs."""
    run = Path(cfg['source_run_dir'])
    source = read_json(run/'manifest.json')
    if source['status'] != 'complete' or not source['is_full_evaluation']:
        raise ValueError('Source must be a complete full-dataset experiment')
    for key in ('batch_size', 'beam_widths', 'seed', 'device'):
        if cfg[key] != source['config'][key]:
            raise ValueError(f'Source replay configuration differs: {key}')
    if source['precision'] != 'float32':
        raise ValueError('Source precision is not float32')
    actual = {
        'checkpoint_sha256': digest(cfg['checkpoint']),
        'sid_sha256': digest(cfg['semantic_id_path']),
        'training_config_sha256': digest(Path(cfg['training_run_dir'])/'.hydra/config.yaml'),
        'training': file_inventory(Path(cfg['data_dir'])/'training'),
        'testing': file_inventory(Path(cfg['data_dir'])/'testing'),
    }
    for key, value in actual.items():
        if value != source['inputs'][key]:
            raise ValueError(f'Source fingerprint differs: {key}')
    for stage in ['test_inputs','teacher_forcing'] + [f'beam_{b}' for b in cfg['beam_widths']]:
        count = completed_shards(run, stage)
        if count != source['stages'][stage]['batches']:
            raise ValueError(f'Source stage incomplete: {stage}')
    dataset = source_dataset(run, source)
    meta = read_json(dataset/'manifest.json')
    if digest(dataset/'items.csv') != meta['items_sha256']:
        raise ValueError('Source popularity groups corrupted')
    return source, {'source_manifest_sha256': digest(run/'manifest.json'),
        'source_dataset_id': meta['dataset_id'], 'source_items_sha256': meta['items_sha256'], **actual}


def code_hashes():
    paths = list((ROOT/'src/analysis/second_layer').glob('*.py'))
    paths += [ROOT/'src/models/modules/semantic_id/tiger_generation_model.py']
    paths += list((ROOT/'src/analysis/long_tail').glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
