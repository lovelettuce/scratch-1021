"""Validated pairing, mutually exclusive destinations and clustered uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.rq_cross_validation.report import aligned

BEAMS = (10, 20, 50)
PAIRS = ((10, 50), (10, 20), (20, 50))
GROUPS = ('Overall', 'Head', 'Mid', 'Tail', 'Train-unseen')
STATES = ('lost_l1', 'lost_l2', 'lost_l3', 'lost_l4', 'complete_outside_top10', 'hit_top10')
OUTCOMES = STATES[2:]
METRICS = ('recall_at_5', 'ndcg_at_5', 'recall_at_10', 'ndcg_at_10')
SURVIVAL = tuple(f'survival_level_{i}' for i in range(1, 5))


def validate_frame(frame):
    required = ['sample_key', 'user_id', 'target_item_id', 'group', 'first_lost_level', *SURVIVAL, *METRICS]
    if missing := set(required) - set(frame):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Empty data or missing values')
    if frame.sample_key.duplicated().any():
        raise ValueError('Duplicate sample keys')
    if not frame.group.isin(GROUPS[1:]).all():
        raise ValueError('Unknown frozen group')
    flags = frame[list(SURVIVAL)].to_numpy()
    if not np.isin(flags, [0, 1]).all() or (np.diff(flags, axis=1) > 0).any():
        raise ValueError('Survival must be binary and cannot recover after loss')
    expected = np.where(flags.all(axis=1), 0, flags.sum(axis=1) + 1)
    if not np.array_equal(expected, frame.first_lost_level.to_numpy()):
        raise ValueError('First lost level differs from survival')
    for k in (5, 10):
        hit, ndcg = frame[f'recall_at_{k}'].to_numpy(), frame[f'ndcg_at_{k}'].to_numpy()
        if not np.isin(hit, [0, 1]).all():
            raise ValueError('Recall must be a binary hit indicator')
        if not np.isfinite(ndcg).all() or (ndcg < 0).any() or (ndcg > hit + 1e-6).any():
            raise ValueError('Invalid NDCG')
        if not np.array_equal(hit > 0, ndcg > 0) or (hit > flags[:, 3]).any():
            raise ValueError('Hit must imply complete survival and positive NDCG')
    if (frame.recall_at_5 > frame.recall_at_10).any():
        raise ValueError('Top-5 hits must be Top-10 hits')


def destination(frame):
    return np.select([frame.first_lost_level.eq(i) for i in range(1, 5)] +
                     [frame.recall_at_10.eq(1)], STATES[:4] + ('hit_top10',),
                     default='complete_outside_top10')


def pair_frames(frames, small, large, cell):
    a, b = aligned(frames[small], frames[large])
    _, third = aligned(frames[small], frames[next(x for x in BEAMS if x not in (small, large))])
    f = a[['user_id', 'target_item_id', 'group']].copy()
    f.insert(0, 'sample_key', a.index)
    f.insert(0, 'dataset', cell.split('_')[0])
    f.insert(0, 'cell', cell)
    f['beam_small'], f['beam_large'] = small, large
    for field in (*SURVIVAL, 'first_lost_level', *METRICS):
        f[field + '_small'], f[field + '_large'] = a[field], b[field]
    common = a.survival_level_1.eq(1) & b.survival_level_1.eq(1)
    f['pair_common_first'] = common
    f['all_beams_common_first'] = common & third.survival_level_1.eq(1)
    f['l2_rescued'] = common & a.survival_level_2.eq(0) & b.survival_level_2.eq(1)
    f['l2_reverse_lost'] = common & a.survival_level_2.eq(1) & b.survival_level_2.eq(0)
    f['state_small'], f['state_large'] = destination(a), destination(b)
    gain = a.recall_at_10.eq(0) & b.recall_at_10.eq(1)
    loss = a.recall_at_10.eq(1) & b.recall_at_10.eq(0)
    f['hit_change'] = np.select([gain, loss, a.recall_at_10.eq(1)],
                               ['gained', 'lost', 'both_hit'], default='both_miss')
    # These are observed transition categories, not causal attribution.
    first_changed = a.survival_level_1.ne(b.survival_level_1)
    f['path_change'] = np.select([first_changed, f.l2_rescued, f.l2_reverse_lost,
                                 common & a.survival_level_2.eq(1) & b.survival_level_2.eq(1)],
                                ['first_layer_changed', 'l2_rescued', 'l2_reverse_lost',
                                 'both_l2_survive'], default='neither_reaches_l2_end')
    for metric in METRICS:
        f['delta_' + metric] = b[metric] - a[metric]
    return f.reset_index(drop=True)


def ratio_statistics(frame, specs, repeats=2000, seed=42):
    """Resample whole users; ratio denominators are recomputed in every draw.

    Identical user contribution vectors are coalesced with multiplicity. Drawing
    their multinomial counts is exactly equivalent to drawing individual users,
    while avoiding a large sample-by-bootstrap tensor.
    """
    if repeats < 1:
        raise ValueError('bootstrap_replicates must be positive')
    names, vectors = [], []
    for name, numerator, denominator in specs:
        names.append(name)
        vectors.extend([np.asarray(numerator, float), np.asarray(denominator, float)])
    if frame.empty:
        return {name: (None, None, None, 0) for name in names}
    values = np.stack(vectors, axis=1)
    users = pd.DataFrame(values).groupby(frame.user_id.to_numpy(), sort=True).sum().to_numpy()
    unique, counts = np.unique(users, axis=0, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, len(names)))
    for start in range(0, repeats, 64):
        weights = rng.multinomial(len(users), counts / counts.sum(), size=min(64, repeats-start))
        sums = weights @ unique
        draws[start:start+len(weights)] = np.divide(sums[:, ::2], sums[:, 1::2],
            out=np.full_like(sums[:, ::2], np.nan), where=sums[:, 1::2] != 0)
    totals = values.sum(axis=0)
    result = {}
    for i, name in enumerate(names):
        valid = draws[:, i][np.isfinite(draws[:, i])]
        point = float(totals[2*i] / totals[2*i+1]) if totals[2*i+1] else None
        low, high = np.quantile(valid, [.025, .975]).tolist() if len(valid) else (None, None)
        result[name] = point, low, high, len(valid)
    return result


def summarize(frame, repeats=2000, seed=42):
    """Long-form cohort, fate, transition and gain/loss tables."""
    identity = frame.iloc[0][['cell', 'dataset', 'beam_small', 'beam_large']].to_dict()
    rows = {name: [] for name in ('cohorts', 'destinations', 'hit_changes', 'transitions', 'metrics')}
    for group in GROUPS:
        f = frame if group == 'Overall' else frame[frame.group.eq(group)]
        f = f.reset_index(drop=True)
        ones = np.ones(len(f))
        specs, entries = [], []

        def add(table, key, num, den, **fields):
            num, den = np.asarray(num, float), np.asarray(den, float)
            specs.append((key, num, den))
            entries.append((table, key, {**identity, 'group': group, 'samples': len(f),
                'users': f.user_id.nunique(), 'numerator_sum': float(num.sum()),
                'denominator_sum': float(den.sum()),
                'denominator_users': f.loc[den != 0, 'user_id'].nunique(), **fields}))

        for cohort in ('pair_common_first', 'all_beams_common_first'):
            common = f[cohort].to_numpy(bool)
            for queue in ('l2_rescued', 'l2_reverse_lost'):
                member = common & f[queue].to_numpy(bool)
                key = cohort + '/' + queue
                add('cohorts', key, member, common, cohort=cohort, queue=queue,
                    queue_samples=int(member.sum()), queue_users=f.loc[member, 'user_id'].nunique())
                side = 'large' if queue == 'l2_rescued' else 'small'
                for outcome in OUTCOMES:
                    mask = member & f['state_' + side].eq(outcome).to_numpy()
                    for denominator, den in (('queue', member), ('all_samples', ones)):
                        add('destinations', key + '/' + outcome + '/' + denominator, mask, den,
                            cohort=cohort, queue=queue, outcome=outcome, denominator=denominator,
                            outcome_samples=int(mask.sum()), outcome_users=f.loc[mask, 'user_id'].nunique())
                for metric in METRICS:
                    add('metrics', key + '/' + metric, member * f[metric + '_' + side], member,
                        cohort=cohort, queue=queue, metric=metric, statistic='queue_mean', side=side)
            add('cohorts', cohort + '/net_l2',
                common * (f.survival_level_2_large - f.survival_level_2_small), common,
                cohort=cohort, queue='net_l2_change')
        for change in ('gained', 'lost', 'both_hit', 'both_miss'):
            member = f.hit_change.eq(change).to_numpy()
            add('hit_changes', change, member, ones, hit_change=change, path_change='all',
                count=int(member.sum()))
            if change in ('gained', 'lost'):
                for path in ('first_layer_changed', 'l2_rescued', 'l2_reverse_lost',
                             'both_l2_survive', 'neither_reaches_l2_end'):
                    mask = member & f.path_change.eq(path).to_numpy()
                    add('hit_changes', change + '/' + path, mask, ones, hit_change=change,
                        path_change=path, count=int(mask.sum()))
        for metric in METRICS:
            delta = f['delta_' + metric].to_numpy()
            expected = float(f[metric + '_large'].mean() - f[metric + '_small'].mean()) if len(f) else None
            if len(f) and not np.isclose(delta.mean(), expected, atol=1e-14, rtol=0):
                raise ValueError('Paired metric delta does not reproduce saved means')
            add('metrics', 'delta/' + metric, delta, ones, cohort='all_samples', queue='all_samples',
                metric=metric, statistic='paired_delta', side='large_minus_small')
        if len(f):
            net = (f.hit_change.eq('gained').sum() - f.hit_change.eq('lost').sum()) / len(f)
            if not np.isclose(net, f.delta_recall_at_10.mean(), atol=1e-14, rtol=0):
                raise ValueError('Gain/loss counts do not conserve Recall@10')
        stats = ratio_statistics(f, specs, repeats, seed)
        for table, key, entry in entries:
            value, low, high, valid = stats[key]
            entry.update(value=value, ci_low=low, ci_high=high, bootstrap_valid_replicates=valid,
                         status='no_samples' if value is None else 'ok')
            rows[table].append(entry)
        for a in STATES:
            for b in STATES:
                mask = f.state_small.eq(a) & f.state_large.eq(b)
                rows['transitions'].append({**identity, 'group': group, 'state_small': a, 'state_large': b,
                    'count': int(mask.sum()), 'users': f.loc[mask, 'user_id'].nunique(),
                    'samples': len(f), 'share_all_samples': float(mask.mean()) if len(f) else None})
    return {k: pd.DataFrame(v) for k, v in rows.items()}
