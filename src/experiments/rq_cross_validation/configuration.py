"""Compose standalone TIGER settings and existing method-specific tokenizer recipes."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from .common import ROOT


def base_config(experiment, entry='train'):
    # Register the project's resolvers before copying unresolved interpolations.
    from src.utils import custom_hydra_resolvers  # noqa: F401
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'configs')):
        return compose(config_name=entry, overrides=[f'experiment={experiment}'])


def paths_for(config, out, data, single_device=True):
    with open_dict(config):
        config.data_dir = str(data)
        config.paths.root_dir = str(ROOT)
        config.paths.output_dir = str(out)
        config.paths.work_dir = str(ROOT)
        config.paths.profile_dir = str(out / 'profile_output')
        config.paths.metadata_dir = str(out / 'metadata')
        if single_device:
            config.trainer.devices = 1
        config.trainer.num_nodes = 1
        config.id = out.name
        config.seed = 42
        config.ckpt_path = None
        config.should_skip_retry = True
        if config.get('callbacks') and 'restart_job' in config.callbacks:
            # Explicit study resume owns retries; the legacy callback calls os._exit.
            config.callbacks.restart_job = None
        config.extras.print_config = False
    return config


def tokenizer_config(cfg, cell, out, embedding):
    method = cfg['experiments'][cell]['tokenizer']
    config = base_config('rqvae_train_flat' if method == 'rqvae' else 'rkmeans_train_flat')
    paths_for(config, Path(out), cfg['datasets'][cfg['experiments'][cell]['dataset']], single_device=False)
    with open_dict(config):
        config.embedding_path = str(embedding)
        config.embedding_dim = cfg['embedding_dim']
        if int(config.num_hierarchies) != cfg['semantic_levels'] or int(config.codebook_width) != cfg['codebook_width']:
            raise ValueError('SID recipe defaults disagree with the registered hierarchy/codebook dimensions')
        # Explicit user scheduling choice: four concurrent jobs, one GPU per job.
        # Keep per-device batch, optimization budget and all other SID defaults intact.
        config.trainer.devices = cfg['tokenizer']['devices_per_experiment']
        # Saving last enables explicit resume.
        config.callbacks.model_checkpoint.save_last = True
    return config


def recommender_config(cfg, cell, out, sid):
    from src.utils import custom_hydra_resolvers  # noqa: F401
    config = base_config('tiger_train_flat')
    with open_dict(config):
        config.merge_with(OmegaConf.load(cfg['recommender']['config']))
    paths_for(config, Path(out), cfg['datasets'][cfg['experiments'][cell]['dataset']])
    with open_dict(config):
        config.semantic_id_path = str(sid)
        config.model.num_embeddings_per_hierarchy = cfg["codebook_width"]
        config.gpu_count = 1
        config.num_hierarchies = 4
        config.accumulation_steps = cfg['recommender']['accumulation_steps']
        config.trainer.accumulate_grad_batches = cfg['recommender']['accumulation_steps']
        config.trainer.max_epochs = cfg['recommender']['max_epochs']
        config.trainer.precision = cfg['recommender']['precision']
        config.callbacks.model_checkpoint.save_last = True
    return config


def embedding_config(cfg, out, dataset='toys'):
    config = base_config('sem_embeds_inference_flat', 'inference')
    paths_for(config, Path(out), cfg['datasets'][dataset])
    config.embedding_model = cfg['embedding_snapshot']
    config.trainer.precision = 32
    return config


def export_config(cfg, cell, out, training, checkpoint, embedding):
    method = cfg['experiments'][cell]['tokenizer']
    config = base_config('rqvae_inference_flat' if method == 'rqvae' else 'rkmeans_inference_flat', 'inference')
    paths_for(config, Path(out), cfg['datasets'][cfg['experiments'][cell]['dataset']])
    with open_dict(config):
        config.embedding_path = str(embedding)
        config.embedding_dim = cfg['embedding_dim']
        config.num_hierarchies = cfg['semantic_levels']
        config.codebook_width = cfg['codebook_width']
        # The full archived architecture is authoritative, including BN, encoder and decoder.
        config.model = OmegaConf.load(Path(training) / '.hydra/config.yaml').model
        config.optim = OmegaConf.load(Path(training) / '.hydra/config.yaml').optim
        config.ckpt_path = str(checkpoint)
        config.callbacks.bq_writer = None
    return config


def analysis_config(cfg, cell, out, training, checkpoint, sid, source=None, limit=None):
    result = {**cfg['analysis'], 'seed': cfg['seed'], 'dataset_name': cfg['experiments'][cell]['dataset'].title(),
              'tokenizer_name': cfg['experiments'][cell]['tokenizer'], 'device': 'cuda:0',
              'data_dir': cfg['datasets'][cfg['experiments'][cell]['dataset']],
              'training_run_dir': str(training), 'checkpoint': str(checkpoint),
              'semantic_id_path': str(sid), 'output_root': str(out),
              'max_samples': limit, 'run_id': 'smoke16' if limit else 'full'}
    result.pop('smoke_samples')
    if source is not None:
        result['source_run_dir'] = str(source)
    return result
