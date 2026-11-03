"""Offline selection, dimensional propagation and shared embedding provenance."""
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.experiments.rq_cross_validation.common import load_config, validate_resume_model, write_json, read_json
from src.experiments.rq_cross_validation.embedding import inspect_model

CONFIG = 'configs/analysis/rq_cross_validation.yaml'


def model_fixture(tmp_path, dim=768):
    for name, data in {'config.json': {'model_type': 't5', 'd_model': dim},
                       'tokenizer_config.json': {}, 'tokenizer.json': {}}.items():
        (tmp_path / name).write_text(json.dumps(data))
    (tmp_path / 'model.safetensors').write_bytes(b'test fixture only')
    return str(tmp_path)


def test_default_explicit_and_legacy(tmp_path):
    default = load_config(CONFIG)
    explicit = load_config(CONFIG, 'flan-t5-xl')
    assert default == explicit
    assert default['embedding_dim'] == 2048
    assert 'flan-t5-xl/snapshots/' in default['embedding_snapshot']
    assert default['beauty_embedding'].endswith('2026-10-22/09-26-23/pickle/merged_predictions_tensor.pt')
    legacy = OmegaConf.load(CONFIG)
    del legacy.embedding_models
    del legacy.embedding_model
    path = tmp_path / 'legacy.yaml'
    OmegaConf.save(legacy, path)
    old = load_config(path)
    for key in ('embedding_snapshot', 'embedding_dim', 'beauty_embedding'):
        assert old[key] == default[key]


def test_cache_missing_and_dimension(tmp_path):
    cfg = load_config(CONFIG, 'flan-t5-base')
    cfg['embedding_snapshot'] = str(tmp_path)
    assert not inspect_model(cfg)['available']
    with pytest.raises(ValueError, match='huggingface-cli download google/flan-t5-base'):
        inspect_model(cfg, required=True)
    model_fixture(tmp_path, 2048)
    with pytest.raises(ValueError, match='dimension mismatch'):
        inspect_model(cfg, required=True)
    model_fixture(tmp_path)
    assert inspect_model(cfg, required=True)['available']
    (tmp_path / 'model.safetensors').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        inspect_model(cfg, required=True)


def test_hf_cache_resolution_is_offline(tmp_path, monkeypatch):
    import huggingface_hub
    cfg = load_config(CONFIG, 'flan-t5-base')
    cfg['embedding_snapshot'] = cfg['embedding_model_id']
    def resolve(**kwargs):
        assert kwargs == {'repo_id': 'google/flan-t5-base', 'local_files_only': True}
        return model_fixture(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', resolve)
    assert inspect_model(cfg, required=True)['snapshot'] == str(tmp_path)


def test_dimension_flows_to_both_tokenizers_and_export(tmp_path):
    from src.experiments.rq_cross_validation.configuration import tokenizer_config, export_config, embedding_config
    cfg = load_config(CONFIG, 'flan-t5-base')
    assert cfg['beauty_embedding'] is None
    for cell in cfg['experiments']:
        folder = tmp_path / cell
        (folder / '.hydra').mkdir(parents=True)
        recipe = tokenizer_config(cfg, cell, folder, '/tmp/new-embedding.pt')
        assert recipe.embedding_dim == 768
        OmegaConf.save(recipe, folder / '.hydra/config.yaml')
        exported = export_config(cfg, cell, folder / 'export', folder, '/tmp/test.ckpt', '/tmp/new-embedding.pt')
        assert exported.embedding_dim == 768
        assert OmegaConf.to_container(exported.model) == OmegaConf.to_container(recipe.model)
    for dataset in ('beauty', 'toys'):
        recipe = embedding_config(cfg, tmp_path, dataset)
        assert recipe.data_dir == cfg['datasets'][dataset]


def test_shared_embeddings_resume_and_corruption(tmp_path, monkeypatch):
    from src.experiments.rq_cross_validation import pipeline
    from src.experiments.rq_cross_validation.__main__ import shared_embeddings
    cfg = load_config(CONFIG, 'flan-t5-base')
    write_json(tmp_path / 'study_manifest.json', {'identity': 'base-study'})
    calls = []
    def run(cfg, folder, recipe, kind, resume):
        calls.append(recipe.data_dir)
        (folder / 'pickle').mkdir(parents=True)
        torch.save(torch.zeros(3, 768), folder / 'pickle/merged_predictions_tensor.pt')
        receipt = folder / 'inference_result.json'
        write_json(receipt, {})
        return {}, receipt
    monkeypatch.setattr(pipeline, 'run_model', run)
    for dataset in ('beauty', 'toys'):
        write_json(tmp_path / 'shared' / dataset / 'manifest.json', {'catalog_items': 3})
        for _ in range(2):
            shared_embeddings(cfg, tmp_path, {'gpu': '4', 'master_port': 29540}, dataset)
        result = read_json(tmp_path / 'shared' / dataset / 'embedding.json')
        assert result['shape'] == [3, 768]
        assert result['model_id'] == 'google/flan-t5-base'
    assert len(calls) == 2
    path = tmp_path / 'shared/beauty/embedding_inference/pickle/merged_predictions_tensor.pt'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        shared_embeddings(cfg, tmp_path, {'gpu': '4', 'master_port': 29540}, 'beauty')


def test_resume_model_mismatch():
    xl, base = load_config(CONFIG), load_config(CONFIG, 'flan-t5-base')
    validate_resume_model(base, {'config': base})
    with pytest.raises(ValueError, match='new study'):
        validate_resume_model(base, {'config': xl})


def test_tiny_local_encoder_and_mean_aggregation(tmp_path):
    from transformers import T5Config, T5EncoderModel
    from src.components.network_blocks.hf_language_model import HFLanguageModel
    from src.models.components.network_blocks.embedding_aggregator import EmbeddingAggregator
    from src.models.components.network_blocks.aggregation_strategy import MeanAggregation
    config = T5Config(vocab_size=20, d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8, dropout_rate=0)
    config.save_pretrained(tmp_path)
    torch.save(T5EncoderModel(config).state_dict(), tmp_path / 'pytorch_model.bin')
    encoder = T5EncoderModel.from_pretrained(tmp_path, local_files_only=True).eval()
    wrapper = HFLanguageModel(encoder, EmbeddingAggregator(MeanAggregation())).eval()
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = ids.ne(0).long()
    with torch.no_grad():
        hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        torch.testing.assert_close(wrapper(ids, mask), expected)


def test_dry_run_no_weights_download_or_study(tmp_path, monkeypatch, capsys):
    import sys
    from src.experiments.rq_cross_validation import __main__ as cli
    cfg = OmegaConf.load(CONFIG)
    cfg.output_root = str(tmp_path / 'outputs')
    cfg.embedding_models['flan-t5-base'].source = str(tmp_path / 'missing')
    path = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, path)
    monkeypatch.setattr(sys, 'argv', ['rqcv', '--config', str(path), '--embedding-model', 'flan-t5-base', '--dry-run'])
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('weights loaded'))
    cli.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan['generate_embeddings'] == ['beauty', 'toys']
    assert not plan['embedding']['available']
    assert plan['config']['embedding_dim'] == 768
    assert not Path(cfg.output_root).exists()
    monkeypatch.setattr(sys, 'argv', ['rqcv', '--config', str(path), '--embedding-model', 'flan-t5-base'])
    with pytest.raises(ValueError, match='huggingface-cli download'):
        cli.main()
    assert not Path(cfg.output_root).exists()


def test_xl_reuses_beauty_without_inference(tmp_path, monkeypatch):
    from src.experiments.rq_cross_validation import pipeline
    from src.experiments.rq_cross_validation.__main__ import shared_embeddings
    cfg = load_config(CONFIG)
    embedding = tmp_path / 'beauty.pt'
    torch.save(torch.zeros(3, 2048), embedding)
    cfg['beauty_embedding'] = str(embedding)
    write_json(tmp_path / 'study_manifest.json', {'identity': 'xl-study'})
    write_json(tmp_path / 'shared/beauty/manifest.json', {'catalog_items': 3})
    monkeypatch.setattr(pipeline, 'run_model', lambda *a: pytest.fail('Beauty should be reused'))
    shared_embeddings(cfg, tmp_path, {'gpu': '4', 'master_port': 29540}, 'beauty')
    result = read_json(tmp_path / 'shared/beauty/embedding.json')
    assert result['path'] == str(embedding)
    assert result['shape'] == [3, 2048]


def test_model_asset_fingerprint_changes(tmp_path):
    from src.experiments.rq_cross_validation.common import input_identity, digest
    cfg = load_config(CONFIG, 'flan-t5-base')
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    cfg['embedding_snapshot'] = model_fixture(model_dir)
    cfg['datasets'] = {}
    cfg['reference'] = {}
    for dataset in ('beauty', 'toys'):
        folder = tmp_path / dataset
        cfg['datasets'][dataset] = str(folder)
        for split in ('items', 'training', 'evaluation', 'testing'):
            (folder / split).mkdir(parents=True)
            (folder / split / 'one.tfrecord.gz').write_bytes(b'inventory fixture')
    # No historical checkpoint, config archive or analysis manifest exists.
    before = input_identity(cfg)
    (model_dir / 'model.safetensors').write_bytes(b'changed')
    after = input_identity(cfg)
    assert before['embedding_assets'] != after['embedding_assets']
    assert after['embedding_model'] == {'model_id': 'google/flan-t5-base', 'dimension': 768}


def test_raw_groups_need_no_reference_or_embeddings(tmp_path, monkeypatch):
    import pandas as pd
    from src.experiments.rq_cross_validation import prepare
    cfg = load_config(CONFIG, 'flan-t5-base')
    assert 'reference' not in cfg
    items = pd.DataFrame({'item_id': [0, 1], 'train_frequency': [0, 2], 'group': ['Cold', 'Head']})
    calls = []
    def scan(folder):
        calls.append(folder)
        return items, {'catalog_items': 2}
    monkeypatch.setattr(prepare, 'dataset_summary', scan)
    prepare.prepare_raw(cfg, tmp_path)
    prepare.prepare_raw(cfg, tmp_path)
    assert len(calls) == 2
    for dataset in cfg['datasets']:
        frozen = pd.read_csv(tmp_path / 'shared' / dataset / 'items.csv')
        pd.testing.assert_frame_equal(frozen, items)
        assert not (tmp_path / 'shared' / dataset / 'embedding.json').exists()
    (tmp_path / 'shared/beauty/items.csv').write_text('changed')
    with pytest.raises(ValueError, match='groups have changed'):
        prepare.prepare_raw(cfg, tmp_path)
