"""Pure offline ranking aggregation and user-clustered uncertainty."""
import numpy as np
import pandas as pd

from .data import GROUPS


def cluster_interval(frame, columns, repeats=2000, seed=42):
    """Percentile CI for sample means, resampling entire users with replacement."""
    if frame.empty:
        return {c: (None, None) for c in columns}
    grouped = frame.groupby('user_id', sort=True)
    sums = grouped[columns].sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repeats, 32):
        draws = rng.integers(0, len(counts), size=(min(32, repeats-start), len(counts)))
        estimates.append(sums[draws].sum(1) / counts[draws].sum(1)[:, None])
    limits = np.quantile(np.concatenate(estimates), [.025, .975], axis=0)
    return {c: tuple(float(x) for x in limits[:, i]) for i, c in enumerate(columns)}


def grouped_metrics(frame, items, k_values, repeats=2000, seed=42, grouping='group'):
    columns = [f'{metric}_at_{k}' for k in k_values for metric in ('recall', 'ndcg')]
    labels = GROUPS if grouping == 'group' else ['0', '1–5', '6–10', '11–20', '21–50', '>50']
    result = []
    total_train = items.train_frequency.sum()
    for label in ['Overall'] + labels:
        subset = frame if label == 'Overall' else frame[frame[grouping] == label]
        catalog = items if label == 'Overall' else items[items[grouping] == label]
        interval = cluster_interval(subset, columns, repeats, seed)
        row = {'group': label, 'test_samples': len(subset), 'test_items': subset.target_item_id.nunique(),
               'catalog_items': len(catalog), 'frequency_min': int(catalog.train_frequency.min()) if len(catalog) else None,
               'frequency_max': int(catalog.train_frequency.max()) if len(catalog) else None,
               'training_interaction_share': float(catalog.train_frequency.sum()/total_train) if total_train else None}
        for column in columns:
            row[column] = float(subset[column].mean()) if len(subset) else None
            row['macro_' + column] = float(subset.groupby('target_item_id')[column].mean().mean()) if len(subset) else None
            row[column + '_ci_low'], row[column + '_ci_high'] = interval[column]
        result.append(row)
    return pd.DataFrame(result)


def paired_comparison(base, other, repeats=2000, seed=42):
    if base.sample_key.duplicated().any() or other.sample_key.duplicated().any():
        raise ValueError('Duplicate sample keys in paired comparison')
    a, b = base.set_index('sample_key'), other.set_index('sample_key')
    if set(a.index) != set(b.index):
        raise ValueError('Paired experiments cover different samples')
    b = b.reindex(a.index)
    for c in ('user_id', 'target_item_id', 'group'):
        if not a[c].equals(b[c]):
            raise ValueError(f'Paired metadata mismatch: {c}')
    diff = a[['user_id', 'group']].copy()
    cols = ['recall_at_10', 'ndcg_at_10']
    for col in cols:
        diff[col] = b[col] - a[col]
    rows = []
    for label in ['Overall'] + GROUPS:
        f = diff if label == 'Overall' else diff[diff.group == label]
        ci = cluster_interval(f, cols, repeats, seed)
        for col in cols:
            rows.append({'group': label, 'metric': col, 'samples': len(f),
                'difference': float(f[col].mean()) if len(f) else None,
                'ci_low': ci[col][0], 'ci_high': ci[col][1]})
    return pd.DataFrame(rows)
