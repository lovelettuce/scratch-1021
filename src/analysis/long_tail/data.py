"""Raw item-level data and frozen popularity groups, before augmentation."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .io import canonical_hash, read_json, write_csv, write_json, digest

GROUPS = ['Head', 'Mid', 'Tail', 'Train-unseen']


def frequency_bucket(n):
    if n == 0:
        return '0'
    for hi, label in [(5, '1–5'), (10, '6–10'), (20, '11–20'), (50, '21–50')]:
        if n <= hi:
            return label
    return '>50'


def assign_groups(frequencies):
    f = np.asarray(frequencies, dtype=np.int64)
    if (f < 0).any():
        raise ValueError('Negative popularity count')
    seen = sorted(f[f > 0], reverse=True)
    head = seen[math.ceil(.2 * len(seen)) - 1] if seen else None
    mid = seen[math.ceil(.5 * len(seen)) - 1] if seen else None
    groups = ['Train-unseen' if n == 0 else 'Head' if n >= head else 'Mid' if n >= mid else 'Tail' for n in f]
    return groups, {'head_min_frequency': int(head) if head is not None else None,
                    'mid_min_frequency': int(mid) if mid is not None else None}


def raw_records(path):
    """Stream TFRecords with CRC checking; avoid a tf.data worker pool per file."""
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    for index, serialized in enumerate(tf.compat.v1.io.tf_record_iterator(str(path), options)):
        features = tf.train.Example.FromString(serialized).features.feature
        for field in ('sequence_data', 'user_id'):
            if field not in features or features[field].WhichOneof('kind') != 'int64_list':
                raise ValueError(f'{path}:{index}: missing/non-integer {field}')
        sequence = list(features['sequence_data'].int64_list.value)
        users = list(features['user_id'].int64_list.value)
        if len(users) != 1:
            raise ValueError(f'{path}:{index}: expected one user_id')
        yield index, users[0], sequence


def validate_ids(sequence, n_items, location):
    if any(i < 0 or i >= n_items for i in sequence):
        raise ValueError(f'{location}: item ID outside SID catalog [0,{n_items})')


def load_sids(path):
    sid = torch.load(path, map_location='cpu', weights_only=True).long()
    if sid.ndim != 2 or sid.shape[0] < 2 or sid.shape[1] == 0 or (sid < 0).any():
        raise ValueError('Expected nonnegative SID tensor [levels, items]')
    rows = [tuple(r) for r in sid.T.tolist()]
    if len(set(rows)) != len(rows):
        raise ValueError('Full SIDs must uniquely identify catalog items')
    return sid


def prepare(cfg, inputs):
    sid = load_sids(cfg['semantic_id_path'])
    identity = {'schema_version': cfg['schema_version'], 'training': inputs['training'],
                'catalog_item_ids': list(range(sid.shape[1])), 'grouping': 'count-20-30-50-ties-hot-v1'}
    dataset_id = canonical_hash(identity)
    folder = Path(cfg['output_root']) / 'datasets' / dataset_id
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        meta = read_json(manifest_path)
        if meta['identity'] != identity or digest(folder / 'items.csv') != meta['items_sha256']:
            raise ValueError('Prepared dataset failed integrity validation')
        return folder
    counts = np.zeros(sid.shape[1], dtype=np.int64)
    users_per_item = [set() for _ in counts]
    records = 0
    empty = 0
    for entry in inputs['training']:
        path = Path(cfg['data_dir']) / 'training' / entry['file']
        for index, user, sequence in raw_records(path):
            records += 1
            empty += not sequence
            validate_ids(sequence, len(counts), f'{path}:{index}')
            for item, n in Counter(sequence).items():
                counts[item] += n
                users_per_item[item].add(user)
    groups, thresholds = assign_groups(counts)
    items = pd.DataFrame({'item_id': np.arange(len(counts)), 'train_frequency': counts,
                          'train_users': [len(s) for s in users_per_item], 'group': groups,
                          'frequency_bucket': [frequency_bucket(n) for n in counts]})
    write_csv(folder / 'items.csv', items)
    write_json(manifest_path, {'identity': identity, 'dataset_id': dataset_id,
        'items_sha256': digest(folder / 'items.csv'), 'thresholds': thresholds,
        'training_records': records, 'empty_training_records': empty,
        'training_interactions': int(counts.sum()), 'catalog_size': len(counts)})
    return folder


def test_batches(cfg, sid, data_cfg, audit):
    """Use original collation/masking; each original test record yields one target."""
    from src.data.loading.components.pre_processing import map_sparse_id_to_semantic_id
    from src.export_sid_softmax import _collate_records
    pending = []
    selected = 0
    limit = cfg.get('max_samples')
    audit.update(raw_records=0, valid_records=0, selected_records=0, filtered_records=0, filtered=[])
    for path in sorted((Path(cfg['data_dir']) / 'testing').glob('*.tfrecord.gz')):
        for index, user, sequence in raw_records(path):
            audit['raw_records'] += 1
            validate_ids(sequence, sid.shape[1], f'{path}:{index}')
            if len(sequence) < 2:
                audit['filtered_records'] += 1
                audit['filtered'].append({'file': path.name, 'index': index, 'reason': 'fewer_than_two_items'})
                continue
            audit['valid_records'] += 1
            if limit is not None and selected >= limit:
                continue
            row = {'sequence_data': torch.tensor(sequence), 'user_id': torch.tensor([user])}
            row = map_sparse_id_to_semantic_id(row, dataset_config=data_cfg.dataset_config,
                                               features_to_apply=['sequence_data'], num_hierarchies=sid.shape[0])
            row.update(__target_item_id=torch.tensor(sequence[-1]), __source_file=path.name,
                       __source_record_index=index)
            pending.append(row)
            selected += 1
            audit['selected_records'] = selected
            if len(pending) == cfg['batch_size']:
                yield _collate_records(pending, data_cfg)
                pending = []
    if pending:
        yield _collate_records(pending, data_cfg)


def model_and_data(cfg):
    import hydra
    from omegaconf import OmegaConf
    from src.utils import custom_hydra_resolvers  # noqa: F401
    archive = OmegaConf.load(Path(cfg['training_run_dir']) / '.hydra/config.yaml')
    archive.data_dir = cfg['data_dir']
    archive.semantic_id_path = cfg['semantic_id_path']
    archive.paths.root_dir = str(Path(__file__).resolve().parents[3])
    # This reader supports the exact item mapping pipeline used by this experiment.
    node = archive.data_loading.test_dataloader_config.dataloader
    pipeline = [x['_target_'].rsplit('.', 1)[-1] for x in node.dataset_config.preprocessing_functions]
    expected = ['filter_features_to_consider', 'convert_to_dense_numpy_array',
                'convert_fields_to_tensors', 'map_sparse_id_to_semantic_id']
    if pipeline != expected or node.padding_token != -1 or node.oov_token is not None:
        raise ValueError('Unsupported archived preprocessing; explicit adapter required')
    model = hydra.utils.instantiate(archive.model)
    checkpoint = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(cfg['device']).eval()
    model._make_deterministic(False)
    model.should_check_prefix = False
    sid = load_sids(cfg['semantic_id_path'])
    data_cfg = SimpleNamespace(labels=OmegaConf.to_container(node.labels, resolve=True),
        sequence_length=int(node.sequence_length), masking_token=node.masking_token,
        padding_token=node.padding_token, oov_token=node.oov_token,
        dataset_config=SimpleNamespace(user_id_field=node.dataset_config.user_id_field,
                                      semantic_id_map={'sequence_data': sid}))
    return model, data_cfg, sid, {'epoch': checkpoint.get('epoch'), 'global_step': checkpoint.get('global_step')}
