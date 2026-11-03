"""One raw-data scan per dataset, with tokenizer-independent catalog groups."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.long_tail.data import assign_groups, frequency_bucket, raw_records, validate_ids
from .common import canonical_hash, digest, read_json, write_csv, write_json


def catalog_ids(folder):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    ids = []
    for path in sorted((Path(folder) / 'items').glob('*.tfrecord.gz')):
        for value in tf.compat.v1.io.tf_record_iterator(str(path), tf.io.TFRecordOptions(compression_type='GZIP')):
            features = tf.train.Example.FromString(value).features.feature
            item = list(features['id'].int64_list.value)
            if len(item) != 1:
                raise ValueError(f'{path}: expected one item ID per row')
            ids.append(item[0])
    if sorted(ids) != list(range(len(ids))) or not ids:
        raise ValueError('Catalog IDs must be unique, contiguous and include ID 0')
    return len(ids)


def dataset_summary(folder):
    n = catalog_ids(folder)
    counts = np.zeros(n, dtype=np.int64)
    users = [set() for _ in range(n)]
    summary = {'catalog_items': n, 'splits': {}}
    for split in ('training', 'evaluation', 'testing'):
        records = interactions = valid = windows = 0
        for path in sorted((Path(folder) / split).glob('*.tfrecord.gz')):
            for index, user, sequence in raw_records(path):
                validate_ids(sequence, n, f'{path}:{index}')
                records += 1
                interactions += len(sequence)
                valid += len(sequence) >= 2
                windows += len(sequence) * (len(sequence) - 1) // 2
                if split == 'training':
                    for item, count in Counter(sequence).items():
                        counts[item] += count
                        users[item].add(user)
        summary['splits'][split] = {'records': records, 'interactions': interactions,
                                    'valid_next_item_records': valid, 'potential_target_windows': windows}
    groups, cuts = assign_groups(counts)
    summary['thresholds'] = cuts
    items = pd.DataFrame({'item_id': np.arange(n), 'train_frequency': counts,
                          'train_users': [len(u) for u in users], 'group': groups,
                          'frequency_bucket': [frequency_bucket(c) for c in counts]})
    return items, summary


def validate_embedding(path, n, dim):
    embedding = torch.load(path, map_location='cpu', weights_only=True)
    if embedding.dtype != torch.float32 or tuple(embedding.shape) != (n, dim) or not torch.isfinite(embedding).all():
        raise ValueError(f'Invalid semantic embedding: {path}')
    return {'path': str(path), 'sha256': digest(path), 'shape': list(embedding.shape), 'dtype': 'float32'}


def prepare_raw(cfg, study):
    for dataset, folder in cfg['datasets'].items():
        out = Path(study) / 'shared' / dataset
        if (out / 'manifest.json').exists():
            meta = read_json(out / 'manifest.json')
            if digest(out / 'items.csv') != meta['items_sha256']:
                raise ValueError('Frozen study groups have changed')
            continue
        items, summary = dataset_summary(folder)
        write_csv(out / 'items.csv', items)
        write_json(out / 'manifest.json', {**summary, 'items_sha256': digest(out / 'items.csv'),
                   'grouping': 'count-20-30-50-ties-hot-v1', 'dataset': dataset})



def validate_sid(path, n, width=256):
    from src.analysis.long_tail.data import load_sids
    sid = load_sids(path)
    if tuple(sid.shape) != (4, n) or int(sid.max()) >= width:
        raise ValueError('SID must have four levels and fit the registered 256-token heads, including disambiguation')
    return sid
