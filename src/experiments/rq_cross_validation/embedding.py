"""Offline model selection and asset validation without loading model weights."""
from pathlib import Path
import json

MODELS = {'flan-t5-xl': ('google/flan-t5-xl', 2048),
          'flan-t5-base': ('google/flan-t5-base', 768)}


def select_model(cfg, selected=None):
    from .common import absolute
    name = selected or cfg.get('embedding_model', 'flan-t5-xl')
    repo, dim = MODELS[name]
    legacy = {'model_id': repo, 'source': cfg.get('embedding_snapshot'),
              'embedding_dim': cfg.get('embedding_dim', dim),
              'beauty_embedding': cfg.get('beauty_embedding')}
    spec = cfg.get('embedding_models', {}).get(name, legacy if name == 'flan-t5-xl' else
              {'model_id': repo, 'source': repo, 'embedding_dim': dim, 'beauty_embedding': None})
    if spec['model_id'] != repo or spec['embedding_dim'] != dim:
        raise ValueError(f'{name} requires model_id={repo} and embedding_dim={dim}')
    cfg['embedding_model'] = name
    cfg['embedding_model_id'] = repo
    cfg['embedding_source'] = spec['source']
    cfg['embedding_dim'] = dim
    cfg['beauty_embedding'] = absolute(spec['beauty_embedding']) if spec.get('beauty_embedding') else None
    if name == 'flan-t5-base' and cfg['beauty_embedding']:
        raise ValueError('FLAN-T5-base must generate new Beauty embeddings')
    cfg['embedding_snapshot'] = (repo if spec['source'] == repo else absolute(spec['source']))
    return cfg


def inspect_model(cfg, required=False):
    """Resolve only local cache entries; never fetch metadata or load weights."""
    source = cfg['embedding_snapshot']
    try:
        if source == cfg['embedding_model_id']:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id=source, local_files_only=True))
        else:
            path = Path(source)
        config = json.loads((path / 'config.json').read_text())
        if config.get('model_type') != 't5' or config.get('d_model') != cfg['embedding_dim']:
            raise ValueError(f'Model architecture/dimension mismatch: {path}')
        # The encoder and tokenizer must both be usable before a study is created.
        weights = [p for p in (path / 'model.safetensors', path / 'pytorch_model.bin') if p.is_file()]
        for index in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
            if (path / index).is_file():
                shards = set(json.loads((path / index).read_text())['weight_map'].values())
                if shards and all((path / shard).is_file() for shard in shards):
                    weights.append(path / index)
        if not weights or not (path / 'tokenizer_config.json').is_file() or not any(
                (path / p).is_file() for p in ('tokenizer.json', 'spiece.model')):
            raise ValueError(f'Incomplete model weights/tokenizer: {path}')
        return {'available': True, 'snapshot': str(path.resolve()), 'model_id': cfg['embedding_model_id'],
                'embedding_dim': cfg['embedding_dim']}
    except (OSError, ValueError, KeyError) as ex:
        message = f'{ex}. Download separately: huggingface-cli download {cfg["embedding_model_id"]}'
        if required:
            raise ValueError(message) from ex
        return {'available': False, 'snapshot': source, 'error': message,
                'model_id': cfg['embedding_model_id'], 'embedding_dim': cfg['embedding_dim']}
