"""Deterministic ranking and paired diagnostic rows. No model dependencies."""
from __future__ import annotations

import numpy as np

KS = (5, 10)
BEAMS = (10, 20, 50)
GROUPS = ('Overall', 'Head', 'Mid', 'Tail', 'Train-unseen')
TOLERANCE = 1e-4
TRACKS = ('original', 'rescored', 'legal', 'exact')


def order(scores, sids):
    scores, sids = np.asarray(scores), np.asarray(sids)
    if len(scores) != len(sids) or not np.isfinite(scores).all():
        raise ValueError('Invalid ranking scores')
    # np.lexsort uses its last key as primary.
    return np.lexsort(tuple(sids[:, i] for i in range(sids.shape[1]-1, -1, -1)) + (-scores,))


def ranked(scores, sids, target):
    indices = order(scores, sids)
    matches = np.all(sids == target, axis=1)
    found = np.flatnonzero(matches)
    if len(found) > 1:
        raise ValueError('Duplicate complete SID in candidate set')
    rank = int(np.flatnonzero(matches[indices])[0] + 1) if len(found) else None
    value = float(scores[found[0]]) if len(found) else None
    result = {'rank': rank, 'rank_best': None, 'rank_worst': None}
    if value is not None:
        result.update(rank_best=int((scores > value).sum()+1), rank_worst=int((scores >= value).sum()))
    for k in KS:
        hit = float(rank is not None and rank <= k)
        result[f'recall_at_{k}'] = hit
        result[f'ndcg_at_{k}'] = hit / np.log2(rank+1) if hit else 0.
        cutoff = float(scores[indices[k-1]]) if len(indices) >= k else None
        result[f'cutoff_tie_at_{k}'] = bool(cutoff is not None and (scores == cutoff).sum() > 1)
        competitors = scores[indices[~matches[indices]]]
        margin = value-float(competitors[k-1]) if value is not None and len(competitors) >= k else None
        result[f'margin_at_{k}'] = margin
        # Also inspect target vs K-1 competitor: an already included target can
        # be numerically indistinguishable from a competitor on either side.
        neighbors = competitors[max(0, k-2):k]
        result[f'sensitive_at_{k}'] = bool(value is not None and len(neighbors) and
                                          np.min(np.abs(neighbors-value)) <= TOLERANCE)
    return result, indices


def diagnose(meta, scores, catalog, target_parts, candidates, rescored, native, old, mapping=None,
             native_orders=None):
    """Return one row per beam and exact Top-50. Input candidate maps by beam."""
    scores = np.asarray(scores, dtype=np.float32)
    target_id = int(meta['target_item_id'])
    target = catalog[target_id]
    exact, indices = ranked(scores, catalog, target)
    if abs(float(np.sum(target_parts))-float(scores[target_id])) > TOLERANCE:
        raise ValueError('Target score differs from catalog column')
    mapping = mapping if mapping is not None else {tuple(s): i for i, s in enumerate(catalog)}
    top = [{'rank': rank+1, 'item_id': int(i), 'score': float(scores[i])}
           for rank, i in enumerate(indices[:50])]
    rows = []
    for beam in BEAMS:
        sid = np.asarray(candidates[beam])
        raw = np.asarray(rescored[beam], dtype=np.float32)
        native_scores = np.asarray(native[beam], dtype=np.float32)
        if not np.isfinite(native_scores).all() or (native_scores < 0).any():
            raise ValueError('Invalid native probability')
        if len({tuple(x) for x in sid}) != len(sid):
            raise ValueError('Duplicate beam SID')
        ids = np.array([mapping.get(tuple(s), -1) for s in sid])
        legal = ids >= 0
        canonical = raw.copy()
        column_error = float(np.max(np.abs(raw[legal]-scores[ids[legal]]))) if legal.any() else 0.
        positive = native_scores > 0
        native_error = float(np.max(np.abs(raw[positive]-np.log(native_scores[positive])))) if positive.any() else 0.
        if max(column_error, native_error) > TOLERANCE:
            raise ValueError(f'Path score error exceeds {TOLERANCE}: catalog={column_error}, native={native_error}')
        # Eliminate chunk-dependent roundoff within the accepted tolerance.
        canonical[legal] = scores[ids[legal]]
        all_rank, _ = ranked(canonical, sid, target)
        legal_rank, _ = ranked(canonical[legal], sid[legal], target)
        row = {**meta, 'beam': beam, 'target_score': float(scores[target_id]),
               'target_nll': -float(scores[target_id]), 'legal_candidates': int(legal.sum()),
               'invalid_candidates': int((~legal).sum()), 'catalog_score_error': column_error,
               'native_log_score_error': native_error, 'native_zero_scores': int((~positive).sum()),
               **{f'target_logp_{i+1}': float(x) for i, x in enumerate(target_parts)},
               **{f'exact_{key}': value for key, value in exact.items()},
               **{f'rescored_{key}': value for key, value in all_rank.items()},
               **{f'legal_{key}': value for key, value in legal_rank.items()}}
        row.update({key: old[beam][key] for key in old[beam]
                    if key.startswith('survival_level_') or key == 'first_lost_level'})
        for k in KS:
            for metric in ('recall', 'ndcg'):
                name = f'{metric}_at_{k}'
                row[f'original_{name}'] = float(old[beam][name])
            e, b = exact[f'recall_at_{k}'], legal_rank[f'recall_at_{k}']
            category = 'both_hit' if e and b else 'search_miss' if e else 'incidental_hit' if b else 'both_miss'
            row[f'category_at_{k}'] = category
            if e and not b and legal_rank['rank'] is not None:
                raise ValueError('Exact Top-K target in beam must be a beam Top-K hit')
            row[f'exact_top_{k}_beam_coverage'] = float(np.isin(indices[:k], ids[legal]).mean())
            import torch
            native_order = (native_orders[beam][k] if native_orders is not None else
                torch.topk(torch.from_numpy(native_scores.copy()), min(k, len(sid))).indices.numpy())
            row[f'invalid_slots_at_{k}'] = int((~legal[native_order]).sum())
            row[f'sensitive_at_{k}'] = exact[f'sensitive_at_{k}'] or all_rank[f'sensitive_at_{k}']
        row['best_legal_score_gap'] = float(scores[indices[0]]-canonical[legal].max()) if legal.any() else None
        if row['best_legal_score_gap'] is not None and row['best_legal_score_gap'] < -TOLERANCE:
            raise ValueError('Beam score exceeds exhaustive catalog maximum')
        rows.append(row)
    return rows, top
