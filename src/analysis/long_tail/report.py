"""Regenerate all tables, figures and a Chinese report without loading a model."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import GROUPS
from .io import completed_shards, read_json, write_csv, write_json, atomic_write
from .metrics import grouped_metrics, paired_comparison

LOGGER = logging.getLogger(__name__)


def concat_csv(paths):
    frames = [pd.read_csv(p, keep_default_na=False, dtype={'sample_key': str, 'frequency_bucket': str, 'branch_bucket': str}) for p in paths]
    if not frames:
        raise ValueError('No committed result tables')
    return pd.concat(frames, ignore_index=True)


def markdown_table(frame):
    def value(x):
        if pd.isna(x):
            return '—'
        return f'{x:.6f}' if isinstance(x, float) else str(x)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---']*len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(value(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def diagnostic_summary(frame, keys):
    values = ['illegal_mass', 'raw_nll', 'legality_nll', 'conditional_nll', 'legal_mrr',
              'legal_hit_at_1', 'legal_hit_at_5', 'legal_hit_at_10', 'legal_children', 'prefix_train_interactions']
    grouped = frame.groupby(keys, dropna=False, sort=True)
    summary = grouped[values].mean()
    summary['samples'] = grouped.size()
    summary['illegal_mass_p50'] = grouped.illegal_mass.quantile(.5)
    summary['illegal_mass_p90'] = grouped.illegal_mass.quantile(.9)
    return summary.reset_index()


def recommendation_coverage(shards, items, k_values):
    groups = items.set_index('item_id')['group'].to_dict()
    totals = {k: {'slots': 0, 'valid': 0, 'tail': 0, 'tail_items': set()} for k in k_values}
    for path in shards:
        batch = torch.load(path, map_location='cpu', weights_only=True)
        for k in k_values:
            rank = torch.topk(batch['scores'], k, dim=-1).indices
            ids = batch['candidate_item_ids'].gather(1, rank).flatten().tolist()
            t = totals[k]
            t['slots'] += len(ids)
            for item in ids:
                if item >= 0:
                    t['valid'] += 1
                    if groups[item] == 'Tail':
                        t['tail'] += 1
                        t['tail_items'].add(item)
    n_tail = int((items.group == 'Tail').sum())
    return pd.DataFrame([{'k': k, 'recommended_slots': t['slots'], 'legal_slots': t['valid'],
        'invalid_ratio': 1 - t['valid']/t['slots'] if t['slots'] else None,
        'tail_share_among_legal': t['tail']/t['valid'] if t['valid'] else None,
        'tail_unique_recommended': len(t['tail_items']), 'tail_catalog_items': n_tail,
        'tail_catalog_coverage': len(t['tail_items'])/n_tail if n_tail else None} for k, t in totals.items()])


def report(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / 'manifest.json')
    cfg = manifest['config']
    expected_stages = ['teacher_forcing'] + [f'beam_{b}' for b in cfg['beam_widths']]
    for stage in expected_stages:
        state = manifest['stages'].get(stage, {})
        count = completed_shards(run_dir, stage)
        if state.get('status') != 'complete' or count != state.get('batches'):
            raise ValueError(f'Cannot report incomplete stage: {stage}')
    # Use committed marker lists, not arbitrary uncommitted partial artifacts.
    def paths(stage, suffix):
        result = []
        for marker in sorted((run_dir/'progress'/stage).glob('*.json')):
            for item in read_json(marker)['artifacts']:
                if item['path'].endswith(suffix):
                    result.append(run_dir/item['path'])
        return result
    dataset = Path(manifest['dataset_dir'])
    if not dataset.exists():
        # A copied output_root remains sufficient for offline regeneration.
        dataset = run_dir.parent.parent/'datasets'/dataset.name
    dataset_meta = read_json(dataset/'manifest.json')
    from .io import digest
    if digest(dataset/'items.csv') != dataset_meta['items_sha256']:
        raise ValueError('Popularity artifacts changed after inference')
    items = pd.read_csv(dataset/'items.csv', keep_default_na=False)
    diagnostic = concat_csv(paths('teacher_forcing', '.csv.gz'))
    if diagnostic.duplicated(['sample_key', 'level']).any():
        raise ValueError('Duplicate diagnostic rows')
    if not np.allclose(diagnostic.raw_nll, diagnostic.legality_nll + diagnostic.conditional_nll, atol=1e-6):
        raise ValueError('NLL decomposition failed')
    if not diagnostic.groupby('sample_key').size().eq(manifest['num_levels']).all():
        raise ValueError('Incomplete per-sample diagnostic levels')
    tables = run_dir/'tables'
    write_csv(tables/'diagnostics_by_group_level.csv', diagnostic_summary(diagnostic, ['group', 'level']))
    write_csv(tables/'diagnostics_by_branch.csv', diagnostic_summary(diagnostic, ['group', 'level', 'branch_bucket']))
    write_csv(tables/'diagnostics_by_collision.csv', diagnostic_summary(diagnostic, ['group', 'level', 'collision_group']))
    diagnostic['prefix_mass_bucket'] = pd.cut(diagnostic.prefix_train_interactions, [-1, 0, 5, 20, 100, 1000, np.inf],
        labels=['0', '1–5', '6–20', '21–100', '101–1000', '>1000']).astype(str)
    write_csv(tables/'diagnostics_by_prefix_mass.csv', diagnostic_summary(diagnostic, ['group', 'level', 'prefix_mass_bucket']))
    frames, metric_tables, coverage_tables, survival_rows, lost_rows = {}, [], [], [], []
    diagnostic_keys = set(diagnostic.sample_key)
    for beam in cfg['beam_widths']:
        stage = f'beam_{beam}'
        frame = concat_csv(paths(stage, '.csv.gz'))
        if frame.sample_key.duplicated().any() or set(frame.sample_key) != diagnostic_keys:
            raise ValueError('Inference stages are not aligned')
        frames[beam] = frame
        kwargs = dict(k_values=cfg['k_values'], repeats=cfg['bootstrap_replicates'], seed=cfg['seed'])
        metrics = grouped_metrics(frame, items, **kwargs)
        metrics.insert(0, 'beam', beam)
        metric_tables.append(metrics)
        frequency = grouped_metrics(frame, items, grouping='frequency_bucket', **kwargs)
        write_csv(tables/f'beam_{beam}_frequency_metrics.csv', frequency)
        collision = diagnostic[diagnostic.level == 1][['sample_key', 'collision_group']]
        split = frame.merge(collision, on='sample_key', validate='one_to_one')
        extra = []
        for (group, coll), part in split.groupby(['group', 'collision_group']):
            row = {'beam': beam, 'group': group, 'collision_group': coll, 'samples': len(part)}
            row.update({c: part[c].mean() for c in part if c.startswith(('recall_at_', 'ndcg_at_'))})
            extra.append(row)
        write_csv(tables/f'beam_{beam}_collision_metrics.csv', pd.DataFrame(extra))
        coverage = recommendation_coverage(paths(stage, '.pt'), items, cfg['k_values'])
        coverage.insert(0, 'beam', beam)
        coverage_tables.append(coverage)
        for label in ['Overall']+GROUPS:
            part = frame if label == 'Overall' else frame[frame.group == label]
            for level in range(1, manifest['num_levels']+1):
                survival_rows.append({'beam': beam, 'group': label, 'level': level, 'samples': len(part),
                    'survival_rate': part[f'survival_level_{level}'].mean() if len(part) else None})
            for lost in range(manifest['num_levels']+1):
                lost_rows.append({'beam': beam, 'group': label, 'first_lost_level': lost, 'samples': len(part),
                    'proportion': (part.first_lost_level == lost).mean() if len(part) else None})
        LOGGER.info('Aggregated beam %d (%d samples)', beam, len(frame))
    metrics = pd.concat(metric_tables, ignore_index=True)
    coverage = pd.concat(coverage_tables, ignore_index=True)
    survival = pd.DataFrame(survival_rows)
    write_csv(tables/'recommendation_metrics.csv', metrics)
    write_csv(tables/'recommendation_coverage.csv', coverage)
    write_csv(tables/'beam_survival.csv', survival)
    write_csv(tables/'first_lost_level.csv', pd.DataFrame(lost_rows))
    base = 10 if 10 in frames else min(frames)
    comparisons = []
    for beam, frame in frames.items():
        if beam != base:
            table = paired_comparison(frames[base], frame, cfg['bootstrap_replicates'], cfg['seed'])
            table.insert(0, 'beam', beam)
            table.insert(1, 'baseline_beam', base)
            comparisons.append(table)
    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame(columns=['beam','baseline_beam','group','metric','samples','difference','ci_low','ci_high'])
    write_csv(tables/'paired_beam_comparison.csv', comparison)
    export = {'schema_version': cfg['schema_version'], 'run_id': run_dir.name,
              'dataset_id': dataset_meta['dataset_id'], 'metrics': metrics.replace({np.nan: None}).to_dict('records'),
              'coverage': coverage.replace({np.nan: None}).to_dict('records'),
              'comparisons': comparison.replace({np.nan: None}).to_dict('records')}
    write_json(tables/'summary.json', export)
    make_figures(run_dir, items, metrics, diagnostic, survival)
    audit = manifest['stages']['teacher_forcing']['audit']
    primary = metrics[['beam', 'group', 'test_samples', 'recall_at_10', 'ndcg_at_10', 'macro_recall_at_10']]
    group_diag = diagnostic_summary(diagnostic, ['group', 'level'])
    tail = metrics[metrics.group == 'Tail']
    conclusions = []
    for _, row in tail.iterrows():
        conclusions.append(f"- Beam {int(row.beam)}：Tail Recall@10={row.recall_at_10:.6f}，NDCG@10={row.ndcg_at_10:.6f}，物品宏平均 Recall@10={row.macro_recall_at_10:.6f}。" if row.test_samples else f"- Beam {int(row.beam)}：Tail 无测试目标，指标不可估计。")
    if (diagnostic.group == 'Tail').any():
        worst = group_diag[group_diag.group == 'Tail'].sort_values('raw_nll', ascending=False).iloc[0]
        conclusions.append(f"- Tail 真实前缀上的最大平均 NLL 出现在第 {int(worst.level)} 层：总 NLL={worst.raw_nll:.4f}，合法性部分={worst.legality_nll:.4f}，合法集合内部分={worst.conditional_nll:.4f}。这是局部诊断，不能单独作为推荐失败的因果解释。")
        tail_diag = group_diag[group_diag.group == 'Tail']
        total_nll = tail_diag.raw_nll.sum()
        legal_nll = tail_diag.legality_nll.sum()
        fraction = legal_nll / total_nll if total_nll else 0.
        conclusions.append(f"- Tail 平均完整路径 NLL={total_nll:.4f}，其中非法概率分配对应的合法性 NLL={legal_nll:.4f}（{fraction:.2%}）。该占比是损失分解，不是约束解码能够提升 Recall 的百分比。")
        for beam in cfg['beam_widths']:
            part = survival[(survival.group == 'Tail') & (survival.beam == beam)].set_index('level')
            if len(part) and part.samples.max():
                conclusions.append(f"- Beam {beam}：Tail 目标前缀首层生存率={part.loc[1, 'survival_rate']:.2%}，末层生存率={part.loc[manifest['num_levels'], 'survival_rate']:.2%}；末层生存表示进入整个 beam，不等于 @10 命中。")

    for _, row in comparison[(comparison.group == 'Tail') & (comparison.metric == 'recall_at_10')].iterrows():
        if row.samples:
            verdict = '区间包含 0，未显示明确改善或下降' if row.ci_low <= 0 <= row.ci_high else ('区间支持召回提高' if row.ci_low > 0 else '区间支持召回下降')
            conclusions.append(f"- Beam {int(row.beam)} 相对 beam {base} 的 Tail Recall@10 差值为 {row.difference:+.6f}，95% 区间 [{row.ci_low:+.6f}, {row.ci_high:+.6f}]；{verdict}。")
    text = f'''# {cfg.get('dataset_name', 'Beauty')} 长尾推荐与 SID 诊断报告

## 实验与覆盖

- 运行：`{run_dir.name}`；{'完整测试集' if manifest['is_full_evaluation'] else '小规模验证，不能视为全量结果'}。
- 原始测试记录 {audit['raw_records']}，有效记录 {audit['valid_records']}，实际评估 {audit['selected_records']}，过滤 {audit['filtered_records']}；过滤原因及位置见 manifest。
- checkpoint epoch={manifest['checkpoint']['epoch']}，global_step={manifest['checkpoint']['global_step']}；{manifest['num_levels']} 层，分类头大小 {manifest['vocab_size']}。
- 推理使用 float32、无前缀约束，beam={cfg['beam_widths']}，K={cfg['k_values']}。非法候选保留排名位置，未命中记零。

## 分组与统计口径

仅统计原始训练交互，不计增强后的样本、不读取验证或测试交互建立分组。按非零频次物品数划分 Head/Mid/Tail，目标比例 20%/30%/50%，边界同频物品归入较热门组；阈值为 {dataset_meta['thresholds']}。训练未见物品单列。分组文件：`{dataset/'items.csv'}`。

样本平均按测试请求计权；物品宏平均只对测试中出现过的目标物品等权。置信区间按用户聚类 bootstrap，{cfg['bootstrap_replicates']} 次，seed={cfg['seed']}，95% percentile；区间用于样本平均指标与配对差值，宏平均为点估计。空组保存为空值。

## 推荐结果

{markdown_table(primary)}

完整 @5/@10 指标及区间见 `tables/recommendation_metrics.csv`。频次桶、碰撞组、曝光及覆盖率另存同目录。尾部推荐占比分母为合法推荐槽位，非法比例分母为所有推荐槽位，尾部覆盖率分母为目录内全部 Tail 物品数。

## 本次观察

{chr(10).join(conclusions)}

## Beam 配对比较

{markdown_table(comparison)}

差值为当前 beam 减 baseline beam {base}，不是相对百分比。增大 beam 不保证每个样本的 @10 表现单调提高。

## SID 与搜索分析

![逐层 NLL](figures/nll_decomposition.png)

![非法概率质量](figures/illegal_mass.png)

![目标前缀生存率](figures/beam_survival.png)

诊断均基于真实前缀。raw NLL = 合法性 NLL + 合法集合内 NLL；唯一合法孩子时，合法集合内 NLL 理论上为零。逐层表按流行度、合法孩子数、碰撞状态和前缀训练交互量分别汇总。生存率基于实际生成前缀，首次丢失层 0 表示最终仍存活。

## 限制与复用

这是一个 checkpoint 的离线评估；未点击物品不是确定负例，结果不推断线上效果或因果关系。当前生成算法使用概率相乘，沿用原实现的浮点行为。manifest 中运行耗时包含诊断、读写与恢复扫描，不能当作纯推理延迟比较。bootstrap 区间不包含重新训练的随机性。

使用 `python -m src.analysis.long_tail report --run-dir <此运行目录>` 可离线重算所有表和图，无需 checkpoint 或 GPU。概率及 log 概率按 PT 分片保存；逐样本 CSV.gz 使用 source_file:record_index 关联。跨 tokenizer 比较必须保持同一训练频次分组与测试记录集合。
'''
    atomic_write(run_dir/'report.md', lambda p: p.write_text(text))
    manifest['status'] = 'complete'
    manifest['report'] = 'report.md'
    write_json(run_dir/'manifest.json', manifest)
    LOGGER.info('Report ready: %s', run_dir/'report.md')
    return run_dir/'report.md'


def make_figures(run_dir, items, metrics, diagnostics, survival):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = run_dir/'figures'
    folder.mkdir(exist_ok=True)
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            atomic_write(folder/f'{name}.{suffix}', lambda p: fig.savefig(p, dpi=160))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    f = np.sort(items.train_frequency.to_numpy())[::-1]
    ax.plot(np.arange(1,len(f)+1), np.maximum(f, .5))
    ax.set(xscale='log', yscale='log', xlabel='Item popularity rank', ylabel='Training interactions (zero shown as 0.5)')
    save(fig,'training_frequency')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax, col in zip(axes,['recall_at_10','ndcg_at_10']):
        metrics.pivot(index='group', columns='beam', values=col).reindex(['Overall']+GROUPS).plot.bar(ax=ax)
        ax.set(ylabel=col, xlabel='Target group')
    save(fig,'recommendation_effect')
    d = diagnostics.groupby(['group','level'])[['illegal_mass','legality_nll','conditional_nll']].mean()
    fig, axes = plt.subplots(1,4,figsize=(14,4))
    for ax, group in zip(axes,GROUPS):
        if group in d.index.get_level_values(0):
            d.loc[group][['legality_nll','conditional_nll']].plot.bar(stacked=True,ax=ax)
        ax.set(title=group, ylabel='Mean NLL', xlabel='SID level')
    save(fig,'nll_decomposition')
    fig, ax = plt.subplots(figsize=(7,4))
    for group in GROUPS:
        if group in d.index.get_level_values(0):
            part=d.loc[group]
            ax.plot(part.index,part.illegal_mass,marker='o',label=group)
    ax.set(xlabel='SID level',ylabel='Mean illegal probability mass')
    ax.legend()
    save(fig,'illegal_mass')
    fig, axes = plt.subplots(1,4,figsize=(14,4))
    for ax, group in zip(axes,GROUPS):
        for beam, part in survival[survival.group == group].groupby('beam'):
            ax.plot(part.level,part.survival_rate,marker='o',label=str(beam))
        ax.set(title=group,xlabel='SID level',ylabel='Target prefix survival',ylim=(0,1))
        ax.legend(title='Beam')
    save(fig,'beam_survival')
