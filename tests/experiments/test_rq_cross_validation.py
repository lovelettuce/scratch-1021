"""Study invariants, real CPU quantizer execution, resume and offline comparisons."""
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
import pandas as pd
import pytest
import torch

from src.utils import custom_hydra_resolvers  # noqa: F401
from src.experiments.rq_cross_validation.common import load_config, read_json, stage
from src.experiments.rq_cross_validation.configuration import (
    tokenizer_config, recommender_config, export_config, base_config, paths_for,
)
from src.experiments.rq_cross_validation.__main__ import schedule, sid_schedule, isolated_env, launch
from src.experiments.rq_cross_validation.prepare import dataset_summary, validate_sid
from src.experiments.rq_cross_validation.report import aligned, ratio_interval, validate_cache


@pytest.fixture
def config():
    return load_config('configs/analysis/rq_cross_validation.yaml')


def test_schedule_requires_distinct_gpus_and_ports(config):
    jobs = schedule(config, list(config['experiments']), '0,1,2')
    assert [j['gpu'] for j in jobs] == ['0', '1', '2', '0']
    assert len({j['master_port'] for j in jobs}) == 4
    assert jobs[0]['cell'] == 'beauty_rqkmeans'
    with pytest.raises(ValueError, match='distinct'):
        schedule(config, list(config['experiments']), '0,0,1')
    env = isolated_env(['2'], 29542)
    assert env['CUDA_VISIBLE_DEVICES'] == '2' and env['MASTER_PORT'] == '29542'
    assert env['HF_HUB_OFFLINE'] == '1'


def test_default_schedule_runs_four_jobs_on_gpus_4_to_7(config):
    cells = list(config['experiments'])
    for pool in ('gpus', 'sid_gpus'):
        assert config[pool] == '4,5,6,7'
    for jobs in (schedule(config, cells, config['gpus']),
                 sid_schedule(config, cells, config['sid_gpus'])):
        assert [j['cell'] for j in jobs] == cells
        assert [j['gpu'] for j in jobs] == ['4', '5', '6', '7']
        assert len({j['master_port'] for j in jobs}) == 4
    assert isolated_env(['7'])['CUDA_VISIBLE_DEVICES'] == '7'


def test_sid_schedule_queues_four_jobs_on_three_gpus(config):
    jobs = sid_schedule(config, list(config['experiments']), '0,1,2')
    assert len(jobs) == 4
    assert [j['gpu'] for j in jobs] == ['0', '1', '2', '0']
    assert len(set(j['gpu'] for j in jobs)) == 3
    assert all(j['gpu'] != '3' for j in jobs)


def test_recipes_keep_standalone_tiger_parameters(config, tmp_path):
    assert 'reference' not in config
    template = base_config('tiger_train_flat')
    with open_dict(template):
        template.merge_with(OmegaConf.load(config['recommender']['config']))
    expected = OmegaConf.to_container(template, resolve=False)
    for cell in config['experiments']:
        rec = recommender_config(config, cell, tmp_path, '/tmp/new-sids.pt')
        assert rec.trainer.max_epochs == 150
        assert rec.trainer.accumulate_grad_batches == 16
        assert rec.callbacks.model_checkpoint.monitor == 'val/recall@5'
        assert rec.callbacks.early_stopping.patience == 10
        assert rec.model.num_embeddings_per_hierarchy == 256
        actual = OmegaConf.to_container(rec, resolve=False)
        for key in ('optim', 'loss', 'eval', 'sequence_length'):
            assert actual[key] == expected[key]
        for key in ('huggingface_model', 'decoder', 'mlp_layers', 'weight_tying'):
            assert actual['model'][key] == expected['model'][key]
        train = rec.data_loading.train_dataloader_config.dataloader
        assert train.batch_size_per_device == 32 and train.num_workers == 8
        assert train.collate_fn.max_batch_size == 512
        assert train.assign_all_files_per_worker is False
        OmegaConf.to_container(rec, resolve=True)
        token = tokenizer_config(config, cell, tmp_path, config['beauty_embedding'])
        defaults = base_config('rkmeans_train_flat' if cell.endswith('rqkmeans') else 'rqvae_train_flat')
        assert token.trainer.devices == 1
        assert defaults.trainer.devices == 8
        assert token.trainer.max_steps == defaults.trainer.max_steps == (3000 if cell.endswith('rqkmeans') else 5000)
        assert token.trainer.max_epochs == defaults.trainer.max_epochs == -1
        if cell.endswith('rqvae'):
            assert token.trainer.check_val_every_n_epoch == 10
        for field in ('precision', 'strategy', 'accumulate_grad_batches', 'sync_batchnorm', 'check_val_every_n_epoch'):
            assert token.trainer[field] == defaults.trainer[field]
        for field in ('batch_size_per_device', 'num_workers', 'assign_all_files_per_worker', 'drop_last', 'should_shuffle_rows'):
            assert token.data_loading.datamodule.train_dataloader_config[field] == defaults.data_loading.datamodule.train_dataloader_config[field]
        assert OmegaConf.to_container(token.model) == OmegaConf.to_container(defaults.model)
        assert OmegaConf.to_container(token.optim) == OmegaConf.to_container(defaults.optim)
        assert token.data_loading.datamodule.train_dataloader_config.batch_size_per_device == 2048
        assert token.callbacks.model_checkpoint.every_n_train_steps == token.trainer.max_steps
        assert token.callbacks.model_checkpoint.every_n_epochs is None
        OmegaConf.to_container(token, resolve=True)


def test_export_architecture_is_exact_training_archive(config, tmp_path):
    train = tokenizer_config(config, 'beauty_rqvae', tmp_path, config['beauty_embedding'])
    (tmp_path / '.hydra').mkdir()
    OmegaConf.save(train, tmp_path / '.hydra/config.yaml')
    recipe = export_config(config, 'beauty_rqvae', tmp_path / 'export', tmp_path,
                           tmp_path / 'best.ckpt', config['beauty_embedding'])
    assert OmegaConf.to_container(recipe.model) == OmegaConf.to_container(train.model)
    assert recipe.model.encoder.output_dim == 64
    assert 'VectorQuantization' in recipe.model.quantization_layer._target_
    OmegaConf.to_container(recipe, resolve=True)


def test_stage_commit_resume_and_corruption(tmp_path):
    calls = []
    def action():
        calls.append(True)
        artifact = tmp_path / 'data.txt'
        artifact.write_text('result')
        return {'answer': 1}, [artifact]
    assert stage(tmp_path, 'test', {'input': 1}, action) == {'answer': 1}
    stage(tmp_path, 'test', {'input': 1}, action)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='identity'):
        stage(tmp_path, 'test', {'input': 2}, action)
    (tmp_path / 'data.txt').write_text('changed')
    with pytest.raises(ValueError, match='corrupt'):
        stage(tmp_path, 'test', {'input': 1}, action)
    def failure():
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        stage(tmp_path, 'other', {'input': 1}, failure)
    assert read_json(tmp_path / 'progress/other.json')['status'] == 'failed'
    stage(tmp_path, 'other', {'input': 1}, action)
    assert read_json(tmp_path / 'progress/other.json')['status'] == 'complete'


def test_raw_groups_only_count_training_and_keep_zero(tmp_path):
    import tensorflow as tf
    def example(features):
        return tf.train.Example(features=tf.train.Features(feature={k: tf.train.Feature(
            int64_list=tf.train.Int64List(value=v)) for k, v in features.items()})).SerializeToString()
    values = {'items': [{'id': [i]} for i in range(4)],
              'training': [{'user_id': [1], 'sequence_data': [0, 0, 1]}, {'user_id': [2], 'sequence_data': [1, 2]}],
              'evaluation': [{'user_id': [3], 'sequence_data': [3, 3]}],
              'testing': [{'user_id': [3], 'sequence_data': [3, 3]}, {'user_id': [4], 'sequence_data': [0]}]}
    for split, records in values.items():
        folder = tmp_path / split
        folder.mkdir()
        with tf.io.TFRecordWriter(str(folder / 'x.tfrecord.gz'), options='GZIP') as writer:
            for record in records:
                writer.write(example(record))
    items, summary = dataset_summary(tmp_path)
    assert items.train_frequency.tolist() == [2, 2, 1, 0]
    assert items.train_users.tolist() == [1, 2, 1, 0]
    assert items.group.iloc[3] == 'Train-unseen'
    assert summary['splits']['testing']['valid_next_item_records'] == 1
    assert summary['splits']['training']['potential_target_windows'] == 4


def test_sid_does_not_silently_expand_vocabulary(tmp_path):
    sid = torch.tensor([[0, 1], [0, 0], [0, 0], [0, 0]])
    path = tmp_path / 'sid.pt'
    torch.save(sid, path)
    assert validate_sid(path, 2).shape == (4, 2)  # unused codes are allowed and recorded
    sid[3, 1] = 256
    torch.save(sid, path)
    with pytest.raises(ValueError, match='256-token'):
        validate_sid(path, 2)


def tiny_quantizer(config, tmp_path, layerwise=False):
    recipe = tokenizer_config(config, 'toys_rqkmeans' if layerwise else 'beauty_rqvae', tmp_path, '/tmp/unused.pt')
    recipe.embedding_dim = 4
    recipe.codebook_width = 2
    recipe.num_hierarchies = 2
    recipe.model.verbose = False
    recipe.model.track_residuals = False
    recipe.model.init_buffer_size = 4
    if not layerwise:
        recipe.model.encoder.hidden_dim_list = [4]
        recipe.model.encoder.output_dim = 2
        recipe.model.decoder.hidden_dim_list = [4]
        recipe.model.quantization_layer.initializer.max_iter = 5
    return hydra.utils.instantiate(recipe.model)


def test_rqvae_cpu_fit_resume_and_strict_export(config, tmp_path):
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from src.data.loading.components.interfaces import ItemData
    torch.set_num_threads(1)
    torch.manual_seed(42)
    data = torch.randn(24, 4)
    def collate(rows):
        return ItemData(item_ids=torch.arange(len(rows)), transformed_features={'input_embedding': torch.stack(rows)})
    loader = DataLoader(data, batch_size=8, collate_fn=collate)
    model = tiny_quantizer(config, tmp_path)
    model.track_residuals = True
    callback = ModelCheckpoint(dirpath=tmp_path / 'ckpts', save_last=True, save_top_k=0)
    trainer = Trainer(accelerator='cpu', devices=1, max_epochs=2, max_steps=4, logger=False,
                      enable_progress_bar=False, enable_model_summary=False, callbacks=[callback])
    trainer.fit(model, train_dataloaders=[loader], val_dataloaders=loader)
    assert torch.isfinite(trainer.callback_metrics['val/loss'])
    assert all(layer.is_initialized for layer in model.quantization_layer_list)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    path = tmp_path / 'resume.ckpt'
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False)
    restored = tiny_quantizer(config, tmp_path)
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    restored.on_load_checkpoint(checkpoint)
    next_trainer = Trainer(accelerator='cpu', devices=1, max_epochs=3, max_steps=5, logger=False,
                           enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False)
    next_trainer.fit(restored, train_dataloaders=[loader], ckpt_path=str(path))
    assert next_trainer.global_step == 5
    assert all(layer.is_initialized for layer in restored.quantization_layer_list)
    restored.eval()
    restored._trainer = SimpleNamespace(state=SimpleNamespace(fn='predict'))
    result = restored.predict_step(collate(list(data[:8])))
    assert result.predictions.shape == (8, 2)


def test_layerwise_resume_keeps_layer_and_initialization_buffer(config, tmp_path):
    model = tiny_quantizer(config, tmp_path, True)
    model.current_layer = 1
    model.quantization_layer_list[0].is_initialized = True
    model.quantization_layer_list[1].init_buffer = torch.ones(2, 4)
    model.quantization_layer_list[1].is_initial_step = False
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    other = tiny_quantizer(config, tmp_path, True)
    other.on_load_checkpoint(checkpoint)
    other._trainer = SimpleNamespace(estimated_stepping_batches=30)
    other.on_train_start()
    assert other.current_layer == 1 and other.steps_per_layer == 15
    assert other.quantization_layer_list[0].is_initialized
    assert torch.equal(other.quantization_layer_list[1].init_buffer, torch.ones(2, 4))
    # Older completed checkpoints without buffers remain loadable.
    checkpoint.pop('initialization_buffers')
    checkpoint.pop('layers_initial_step')
    other.on_load_checkpoint(checkpoint)
    other.on_train_start()
    assert other.current_layer == 1


def test_clustered_conditional_rate_and_pair_alignment():
    frame = pd.DataFrame({'user_id': [0, 0, 1, 2], 'alive': [1, 1, 0, 0], 'entered': [1, 1, 1, 0]})
    point, low, high = ratio_interval(frame, 'alive', 'entered', 2000, 42)
    assert point == pytest.approx(2/3) and 0 <= low <= point <= high <= 1
    empty = ratio_interval(frame[frame.entered == 0], 'alive', 'entered', 100, 42)
    assert empty == (None, None, None)
    a = pd.DataFrame({'sample_key': ['x', 'y'], 'user_id': [1, 2], 'target_item_id': [0, 1], 'group': ['Head', 'Tail']})
    b = a.iloc[::-1]
    aa, bb = aligned(a, b)
    pd.testing.assert_frame_equal(aa, bb)
    with pytest.raises(ValueError, match='sample sets'):
        aligned(a, b.iloc[:1])


def test_finite_workers_cover_each_file_once(monkeypatch):
    from src.data.loading.components.dataloading import UnboundedSequenceIterable
    import src.data.loading.components.dataloading as module
    class Iterator:
        def update_list_of_file_paths(self, paths):
            self.paths = paths
            self.list_of_file_paths = paths
        def count_rows(self):
            return len(self.paths)
        def iterrows(self):
            yield from ({'file': p} for p in self.paths)
    all_rows = []
    for worker in range(3):
        monkeypatch.setattr(module, 'get_worker_info', lambda w=worker: SimpleNamespace(id=w, num_workers=3))
        config = SimpleNamespace(data_iterator=Iterator(), iterate_per_row=True, preprocessing_functions=[])
        dataset = UnboundedSequenceIterable(config, 'unused', assign_all_files_per_worker=False)
        dataset.set_list_of_files([str(i) for i in range(8)])
        dataset.set_distributed_params(1, 0)
        first = list(dataset)
        assert first == list(dataset)  # persistent workers reset on a new epoch
        all_rows.extend(row['file'] for row in first)
    assert sorted(all_rows) == [str(i) for i in range(8)]


def test_four_cell_offline_report_round_trip(config, tmp_path):
    import shutil
    from src.analysis.long_tail.io import write_csv, write_json, digest
    from src.experiments.rq_cross_validation.report import CELLS, report
    config['analysis']['bootstrap_replicates'] = 32
    study = tmp_path / 'original'
    write_json(study / 'study_manifest.json', {'config': config})
    groups = ['Head', 'Mid', 'Tail', 'Train-unseen'] * 3
    n = len(groups)
    meta = pd.DataFrame({'sample_key': [f'file:{i}' for i in range(n)], 'user_id': np.arange(n),
                         'target_item_id': np.arange(n), 'group': groups})
    for cell in CELLS:
        folder = study / 'comparison/inputs' / cell
        base = meta.copy()
        for name, value in [('model_ce', 2.), ('raw_ce', 1.8), ('windows_ce', 1.9),
                            ('model_mrr', .3), ('raw_mrr', .2), ('windows_mrr', .25),
                            ('raw_ce_gain', -.2), ('windows_ce_gain', -.1)]:
            base[name] = value
        write_csv(folder / 'baselines.csv.gz', base)
        for beam in (10, 20, 50):
            f = meta.copy()
            f['recall_at_10'] = .1
            f['ndcg_at_10'] = .05
            for level, value in enumerate([1, 1, 0, 0], 1):
                f[f'survival_level_{level}'] = value
            write_csv(folder / f'recommendation_{beam}.csv.gz', f)
            s = meta.copy()
            s['entered_second'] = 1
            s['retained_second'] = 1
            write_csv(folder / f'search_{beam}.csv.gz', s)
        write_csv(folder / 'items.csv', pd.DataFrame({'item_id': np.arange(n), 'group': groups}))
        for filename in ('recommendation_metrics.csv', 'nll_by_level.csv', 'tree.csv', 'unseen.csv', 'support_joint.csv'):
            write_csv(folder / filename, pd.DataFrame({'group': ['Tail']}))
        attr = []
        for beam in (10, 20, 50):
            attr.append({'beam': beam, 'cohort': 'entered', 'group': 'Tail', 'samples': 3,
                         'retained_second': .5, 'cause_retained': .5, 'cause_local_capacity': .2,
                         'cause_cross_parent': .3, 'cause_cutoff_tie': 0.})
        write_csv(folder / 'attribution.csv', pd.DataFrame(attr))
        write_csv(folder / 'paired_beam.csv', pd.DataFrame({'beam': [50], 'group': ['Tail'],
                  'retained_second': [.1], 'retained_second_ci_low': [.05], 'retained_second_ci_high': [.2]}))
        artifacts = [{'path': p.name, 'sha256': digest(p)} for p in sorted(folder.iterdir())]
        write_json(folder / 'manifest.json', {'cell': cell, 'dataset': cell.split('_')[0],
                   'tokenizer': cell.split('_', 1)[1], 'cell_role': 'trained_in_study', 'samples': n, 'checkpoint': {'epoch': 1, 'global_step': 5},
                   'artifacts': artifacts})
    report(study)
    expected = {p.name: digest(p) for p in (study / 'comparison/tables').iterdir()}
    copied = tmp_path / 'copied'
    shutil.copytree(study, copied)
    shutil.rmtree(study)
    report(copied)  # all original source paths are absent; no model/GPU access
    actual = {p.name: digest(p) for p in (copied / 'comparison/tables').iterdir()}
    assert expected == actual
    assert read_json(copied / 'comparison/tables/summary.json')['status'] == 'complete'
    for name in ('conditional_survival', 'baseline_gains', 'attribution'):
        assert (copied / f'comparison/figures/{name}.pdf').is_file()
    (copied / 'comparison/inputs/beauty_rqvae/items.csv').write_text('corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        report(copied)


def test_new_study_does_not_fill_beauty_cell_from_historical_reference(config, tmp_path, monkeypatch):
    from src.analysis.long_tail.io import write_json
    from src.experiments.rq_cross_validation import report as module
    write_json(tmp_path / 'study_manifest.json', {'config': config})
    write_json(tmp_path / 'reference/beauty_rqkmeans.json', {'checkpoint': '/absent/historical.ckpt'})
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical checkpoint must not populate the trained Beauty cell')
    monkeypatch.setattr(module, 'cache_cell', forbidden)
    with pytest.raises(ValueError, match='No completed cells'):
        module.report(tmp_path)


def test_new_study_rejects_comparison_inputs_without_trained_role(config, tmp_path):
    from src.analysis.long_tail.io import write_json
    from src.experiments.rq_cross_validation.report import report
    write_json(tmp_path / 'study_manifest.json', {'config': config})
    write_json(tmp_path / 'comparison/inputs/beauty_rqkmeans/manifest.json', {
        'cell': 'beauty_rqkmeans', 'cell_role': 'historical_reference', 'artifacts': []})
    with pytest.raises(ValueError, match='Historical Beauty checkpoint'):
        report(tmp_path)


@pytest.mark.parametrize('training_stage', ['tokenizer', 'recommender'])
def test_default_gpu_pool_starts_all_four_before_waiting(config, tmp_path, monkeypatch, training_stage):
    import src.experiments.rq_cross_validation.__main__ as coordinator
    started = []
    class Process:
        def poll(self):
            assert len(started) == 4
            return 0
    class Handle:
        def close(self):
            pass
    def start(cfg, study, job, stage):
        started.append(job['gpu'])
        return Process(), Handle()
    monkeypatch.setattr(coordinator, 'start_job', start)
    jobs = (coordinator.sid_schedule if training_stage == 'tokenizer' else coordinator.schedule)(
        config, list(config['experiments']), config['sid_gpus' if training_stage == 'tokenizer' else 'gpus'])
    coordinator.launch(config, tmp_path, jobs, training_stage)
    assert started == ['4', '5', '6', '7']


@pytest.mark.parametrize('training_stage', ['tokenizer', 'recommender'])
@pytest.mark.parametrize('first_exit', [0, 1])
def test_gpu_queue_reuses_free_card_while_other_jobs_run(config, tmp_path, monkeypatch, training_stage, first_exit):
    import src.experiments.rq_cross_validation.__main__ as coordinator
    events, live, processes = [], {}, []
    class Process:
        def __init__(self, command, **kwargs):
            self.ordinal = len(processes)
            self.pid = 1000 + self.ordinal
            self.gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
            self.done = False
            assert self.gpu not in live, 'A GPU was assigned to overlapping jobs'
            live[self.gpu] = self
            assert len(live) <= 3
            processes.append(self)
            events.append(('start', self.ordinal, self.gpu))
        def poll(self):
            if self.done:
                return first_exit if self.ordinal == 0 else 0
            # First finishes quickly; GPU1/2 must still be running when fourth starts.
            if self.ordinal == 0 or len(processes) == 4:
                self.done = True
                del live[self.gpu]
                events.append(('finish', self.ordinal, self.gpu))
                return first_exit if self.ordinal == 0 else 0
            return None
        def wait(self, **kwargs):
            return self.poll()
    monkeypatch.setattr(coordinator.subprocess, 'Popen', Process)
    jobs = (coordinator.sid_schedule if training_stage == 'tokenizer' else coordinator.schedule)(
        config, list(config['experiments']), '0,1,2')
    if first_exit:
        with pytest.raises(RuntimeError, match='Failed cells'):
            coordinator.launch(config, tmp_path, jobs, training_stage)
    else:
        coordinator.launch(config, tmp_path, jobs, training_stage)
    assert [event[0] for event in events[:3]] == ['start']*3
    assert events[3] == ('finish', 0, '0')
    assert events[4] == ('start', 3, '0')  # before GPU1/2 finish, including on first-job failure
    assert len(processes) == 4 and not live


def test_gpu_preflight_rejects_foreign_compute_owners(monkeypatch):
    from src.experiments.rq_cross_validation import worker
    def inventory(command, **kwargs):
        if '--query-gpu=index,uuid' in command:
            return '0, GPU-A\n1, GPU-B\n2, GPU-C\n3, GPU-D\n'
        return '2516587, GPU-D, 65766\n'
    monkeypatch.setattr(worker.subprocess, 'check_output', inventory)
    worker.ensure_idle_gpus(['0', '1', '2'])
    with pytest.raises(RuntimeError, match='GPU 3 is occupied'):
        worker.ensure_idle_gpus(['0', '1', '2', '3'])
    with pytest.raises(RuntimeError, match='GPU 4 does not exist'):
        worker.ensure_idle_gpus(['4'])


def test_explicit_gpu_sharing_keeps_memory_and_cuda_checks(monkeypatch):
    from contextlib import nullcontext
    from src.experiments.rq_cross_validation import worker
    owners = {'0': [{'pid': '2875794', 'used_memory_mib': '2680'}]}
    monkeypatch.setattr(worker, 'gpu_process_snapshot', lambda ids: owners)
    with pytest.raises(RuntimeError, match='GPU 0 is occupied'):
        worker.probe(['0'], 12)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'device', lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (70*1024**3, 80*1024**3))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'fixture A100')
    monkeypatch.setattr(torch, 'ones', lambda *args, **kwargs: SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1)))
    data = worker.probe(['0'], 12, allow_gpu_sharing=True)
    assert data[0]['existing_compute_processes'] == owners['0']
    assert data[0]['sharing_allowed'] is True
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (10*1024**3, 80*1024**3))
    with pytest.raises(RuntimeError, match='only 10.0 GiB free'):
        worker.probe(['0'], 12, allow_gpu_sharing=True)
