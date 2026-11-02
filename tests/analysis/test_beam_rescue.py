"""Regression tests for rescue selection, conservation and portable CPU replay."""
import shutil

import numpy as np
import pandas as pd
import pytest

from src.analysis.beam_rescue.core import (
    BEAMS, GROUPS, METRICS, OUTCOMES, STATES, pair_frames, ratio_statistics, summarize, validate_frame,
)
from src.analysis.beam_rescue.report import run
from src.analysis.long_tail.io import digest, read_json, write_csv, write_json
from src.experiments.rq_cross_validation.common import EXPERIMENTS


def frame(states):
    records = []
    for i, state in enumerate(states):
        lost = state + 1 if state < 4 else 0
        hit = int(state == 5)
        records.append({'sample_key': f'partition:record:{i}', 'user_id': str(i // 2),
            'target_item_id': str(i), 'group': 'Tail', 'first_lost_level': lost,
            **{f'survival_level_{level}': int(lost == 0 or level < lost) for level in range(1, 5)},
            'recall_at_5': 0, 'ndcg_at_5': 0., 'recall_at_10': hit, 'ndcg_at_10': hit / np.log2(8)})
    return pd.DataFrame(records)


@pytest.fixture
def frames():
    return {10: frame([1, 1, 1, 1, 5, 2, 3, 5, 5, 0, 5, 2]),
            20: frame([0, 2, 3, 5, 2, 5, 5, 5, 5, 5, 5, 2]),
            50: frame([2, 3, 4, 5, 1, 5, 5, 5, 0, 5, 5, 2])}


def test_pairing_fates_reverse_loss_and_sensitivity(frames):
    frames[50] = frames[50].sample(frac=1, random_state=3)
    p = pair_frames(frames, 10, 50, 'beauty_rqkmeans')
    rescue = p[p.l2_rescued]
    assert len(rescue) == 4
    assert set(rescue.state_large) == set(OUTCOMES)
    assert p.l2_reverse_lost.sum() == 1
    assert p.loc[p.l2_reverse_lost, 'state_small'].tolist() == ['hit_top10']
    assert (p.l2_rescued & p.all_beams_common_first).sum() == 3
    assert p[p.hit_change.eq('gained')].path_change.isin(['l2_rescued', 'both_l2_survive', 'first_layer_changed']).all()
    result = summarize(p, repeats=64)
    d = result['destinations']
    subset = d[(d.group == 'Tail') & (d.cohort == 'pair_common_first') &
               (d.queue == 'l2_rescued') & (d.denominator == 'queue')]
    assert subset.outcome_samples.sum() == 4
    assert subset.value.sum() == 1
    assert (subset.value == .25).all()
    transitions = result['transitions']
    assert transitions[transitions.group == 'Tail']['count'].sum() == len(p)
    assert len(transitions[transitions.group == 'Tail']) == len(STATES)**2
    gains = p.hit_change.eq('gained').sum()
    losses = p.hit_change.eq('lost').sum()
    m = result['metrics']
    for name in METRICS:
        value = m[(m.group == 'Tail') & (m.metric == name) & (m.statistic == 'paired_delta')].value.item()
        assert value == pytest.approx(frames[50][name].mean() - frames[10][name].mean())
    assert (gains - losses) / len(p) == pytest.approx(p.delta_recall_at_10.mean())


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'duplicate'), ('missing', 'sample sets'), ('user', 'metadata'),
    ('target', 'metadata'), ('group', 'metadata')])
def test_alignment_rejects_bad_identity(frames, mutation, match):
    f = frames[50]
    if mutation == 'duplicate':
        frames[50] = pd.concat([f, f.iloc[:1]])
    elif mutation == 'missing':
        frames[50] = f.iloc[1:]
    else:
        column = {'user': 'user_id', 'target': 'target_item_id', 'group': 'group'}[mutation]
        frames[50].loc[0, column] = 'Head' if mutation == 'group' else 'different'
    with pytest.raises(ValueError, match=match):
        pair_frames(frames, 10, 50, 'beauty_rqkmeans')


@pytest.mark.parametrize('change,match', [
    ({'survival_level_3': 1}, 'recover'), ({'first_lost_level': 4}, 'First lost'),
    ({'recall_at_10': 1, 'ndcg_at_10': .3}, 'Hit must imply'),
    ({'survival_level_1': 2}, 'binary'), ({'ndcg_at_10': float('nan')}, 'missing'),
    ({'recall_at_5': 1, 'ndcg_at_5': .5}, 'Hit must imply')])
def test_rejects_inconsistent_survival_and_metrics(change, match):
    f = frame([1])
    for column, value in change.items():
        f.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        validate_frame(f)


def test_empty_queues_zero_hits_single_user_and_cluster_ratio():
    frames = {beam: frame([1, 1]) for beam in BEAMS}
    p = pair_frames(frames, 10, 50, 'beauty_rqkmeans')
    tables = summarize(p, repeats=32)
    d = tables['destinations']
    assert d[d.denominator == 'queue'].value.isna().all()
    assert (d[d.denominator == 'queue'].status == 'no_samples').all()
    m = tables['metrics']
    assert (m[(m.group == 'Tail') & (m.statistic == 'paired_delta')].value == 0).all()
    assert tables['metrics'][tables['metrics'].group == 'Head'].value.isna().all()
    f = pd.DataFrame({'user_id': ['a', 'a', 'b']})
    result = ratio_statistics(f, [('ratio', [1, 1, 0], [1, 1, 1])], repeats=2000)['ratio']
    assert result[:3] == pytest.approx((2/3, 0, 1))
    result = ratio_statistics(f, [('conditional', [1, 0, 0], [1, 0, 0])], repeats=2000)['conditional']
    assert result[:3] == (1, 1, 1)
    assert 0 < result[3] < 2000  # denominator-free draws are excluded, not converted to zero


def build_study(folder, frames):
    for cell in EXPERIMENTS:
        base = folder / 'comparison/inputs' / cell
        summaries = []
        for beam, f in frames.items():
            write_csv(base / f'recommendation_{beam}.csv.gz', f)
            for group in GROUPS:
                part = f if group == 'Overall' else f[f.group == group]
                summaries.append({'beam': beam, 'group': group, 'test_samples': len(part),
                                  **{metric: part[metric].mean() for metric in METRICS}})
        write_csv(base / 'recommendation_metrics.csv', pd.DataFrame(summaries))
        write_json(base / 'manifest.json', {'cell': cell, 'dataset': cell.split('_')[0],
            'cell_role': 'trained_in_study', 'samples': len(frames[10]),
            'artifacts': [{'path': f.name, 'sha256': digest(f)} for f in sorted(base.iterdir())]})


def test_portable_cpu_only_round_trip_and_corruption(frames, tmp_path, monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline report must not touch CUDA, checkpoints or source inference')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    original = tmp_path / 'original'
    build_study(original, frames)
    counts = {'beauty': len(frames[10]), 'toys': len(frames[10])}
    out = run(original, repeats=16, expected_counts=counts)
    expected = {f.name: digest(f) for f in (out / 'tables').glob('*.csv')}
    assert read_json(out / 'manifest.json')['status'] == 'complete'
    assert len(list((out / 'figures').glob('*.pdf'))) == 4
    copied = tmp_path / 'copied'
    shutil.copytree(original / 'comparison/inputs', copied / 'comparison/inputs')
    shutil.rmtree(original)
    out = run(copied, repeats=16, expected_counts=counts)
    assert expected == {f.name: digest(f) for f in (out / 'tables').glob('*.csv')}
    state = read_json(out / 'manifest.json')
    for relative, fingerprint in state['input_sha256'].items():
        assert digest(copied / relative) == fingerprint
    with pytest.raises(FileExistsError, match='not empty'):
        run(copied, expected_counts=counts)
    corrupt = copied / 'comparison/inputs/beauty_rqvae/recommendation_50.csv.gz'
    corrupt.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        run(copied, output_dir=copied / 'new', expected_counts=counts)
    assert not (copied / 'new').exists()


def test_rejects_wrong_coverage_and_frozen_metrics(frames, tmp_path):
    build_study(tmp_path, frames)
    with pytest.raises(ValueError, match='coverage'):
        run(tmp_path, repeats=4)  # current production snapshot counts are checked
    source = tmp_path / 'comparison/inputs/beauty_rqkmeans'
    f = pd.read_csv(source / 'recommendation_metrics.csv')
    f.loc[0, 'recall_at_10'] = .123
    write_csv(source / 'recommendation_metrics.csv', f)
    manifest = read_json(source / 'manifest.json')
    for artifact in manifest['artifacts']:
        artifact['sha256'] = digest(source / artifact['path'])
    write_json(source / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Frozen metric'):
        run(tmp_path, repeats=4, expected_counts={'beauty': 12, 'toys': 12})


def test_rejects_output_in_inputs(frames, tmp_path):
    with pytest.raises(ValueError, match='separate'):
        run(tmp_path, output_dir=tmp_path / 'comparison/inputs/nested')
