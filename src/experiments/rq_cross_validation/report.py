"""Portable, offline four-cell comparisons; missing cells remain explicitly missing."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.analysis.long_tail.data import GROUPS
from src.analysis.long_tail.io import completed_shards
from src.analysis.long_tail.metrics import cluster_interval, paired_comparison
from src.analysis.long_tail.report import markdown_table
from src.analysis.second_layer.common import source_paths
from .common import EXPERIMENTS, digest, read_json, write_csv, write_json, atomic_write

CELLS = EXPERIMENTS


def concatenate(paths):
    if not paths:
        raise ValueError('No committed source rows')
    return pd.concat([pd.read_csv(p, dtype={'sample_key': str, 'group': str,
                         'branch_bucket': str, 'edge_support_bucket': str}) for p in paths], ignore_index=True)


def cache_cell(study, cell, first, second):
    """Freeze only the comparison inputs; no original artifact is rewritten."""
    out = Path(study) / 'comparison/inputs' / cell
    if (out / 'manifest.json').exists():
        validate_cache(out)
        return out
    first, second = Path(first), Path(second)
    a, b = read_json(first / 'manifest.json'), read_json(second / 'manifest.json')
    if any(m['status'] != 'complete' or not m['is_full_evaluation'] for m in (a, b)):
        raise ValueError('Comparison requires complete full-data runs')
    if b['inputs']['source_manifest_sha256'] != digest(first / 'manifest.json'):
        raise ValueError('Second-layer source does not match the long-tail experiment')
    dataset = first.parent.parent / 'datasets' / Path(a['dataset_dir']).name
    for beam in (10, 20, 50):
        for run in (first, second):
            completed_shards(run, f'beam_{beam}')
        write_csv(out / f'recommendation_{beam}.csv.gz', concatenate(source_paths(first, f'beam_{beam}', '.csv.gz')))
        write_csv(out / f'search_{beam}.csv.gz', concatenate(source_paths(second, f'beam_{beam}', '.csv.gz')))
    baseline_dir = second.parent.parent / 'datasets' / b['dataset_id'] / b['baseline_relative_dir']
    completed_shards(baseline_dir, 'baselines')
    write_csv(out / 'baselines.csv.gz', concatenate(source_paths(baseline_dir, 'baselines', '.csv.gz')))
    for src, name in [(first / 'tables/recommendation_metrics.csv', 'recommendation_metrics.csv'),
                      (first / 'tables/diagnostics_by_group_level.csv', 'nll_by_level.csv'),
                      (first / 'tables/tree_structure.csv', 'tree.csv'),
                      (dataset / 'items.csv', 'items.csv'),
                      (second / 'tables/attribution_summary.csv', 'attribution.csv'),
                      (second / 'tables/paired_common_cohort.csv', 'paired_beam.csv'),
                      (second / 'tables/unseen_support_vs_loss.csv', 'unseen.csv'),
                      (second / 'tables/beam_50_by_branch_and_support.csv', 'support_joint.csv')]:
        atomic_write(out / name, lambda p, source=src: shutil.copyfile(source, p))
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'manifest.json')
    write_json(out / 'manifest.json', {'cell': cell, 'dataset': cell.split('_')[0],
               'tokenizer': cell.split('_', 1)[1], 'cell_role': 'trained_in_study', 'samples': b['num_samples'],
               'long_tail_source_sha256': digest(first / 'manifest.json'),
               'second_layer_source_sha256': digest(second / 'manifest.json'),
               'artifacts': [{'path': p.name, 'sha256': digest(p)} for p in files],
               'checkpoint': a['checkpoint'], 'vocab_size': a['vocab_size']})
    return out


def validate_cache(folder):
    m = read_json(folder / 'manifest.json')
    for item in m['artifacts']:
        p = folder / item['path']
        if not p.is_file() or digest(p) != item['sha256']:
            raise ValueError(f'Offline comparison input corrupt: {p}')
    return m


def interval_rows(frame, columns, cfg, **identity):
    rows = []
    for label in ('Overall', *GROUPS):
        f = frame if label == 'Overall' else frame[frame.group == label]
        intervals = cluster_interval(f, columns, cfg['analysis']['bootstrap_replicates'], cfg['seed'])
        row = {**identity, 'group': label, 'samples': len(f)}
        for c in columns:
            row[c] = float(f[c].mean()) if len(f) else None
            row[c + '_ci_low'], row[c + '_ci_high'] = intervals[c]
        rows.append(row)
    return rows


def ratio_interval(frame, numerator, denominator, repeats, seed):
    """Clustered ratio of sums: conditional survival uses the resampled denominator."""
    if frame.empty or frame[denominator].sum() == 0:
        return None, None, None
    sums = frame.groupby('user_id', sort=True)[[numerator, denominator]].sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, repeats, 32):
        draws = rng.integers(0, len(sums), (min(32, repeats - start), len(sums)))
        totals = sums[draws].sum(1)
        valid = totals[:, 1] > 0
        values.extend((totals[valid, 0] / totals[valid, 1]).tolist())
    lo, hi = np.quantile(values, [.025, .975])
    return float(frame[numerator].sum() / frame[denominator].sum()), float(lo), float(hi)


def aligned(a, b):
    if a.sample_key.duplicated().any() or b.sample_key.duplicated().any() or set(a.sample_key) != set(b.sample_key):
        raise ValueError('Different or duplicate sample sets in paired comparison')
    a = a.set_index('sample_key').sort_index()
    b = b.set_index('sample_key').reindex(a.index)
    for field in ('user_id', 'target_item_id', 'group'):
        if not a[field].equals(b[field]):
            raise ValueError(f'Paired metadata differs: {field}')
    return a, b


def report(study):
    study = Path(study).resolve()
    manifest = read_json(study / 'study_manifest.json')
    cfg = manifest['config']
    out = study / 'comparison'
    # Once frozen, these files are sufficient even after moving the entire study.
    for cell in EXPERIMENTS:
        base = study / 'experiments' / cell
        marker = base / 'progress/second_layer.json'
        if marker.exists() and read_json(marker)['status'] == 'complete' and not (out / 'inputs' / cell / 'manifest.json').exists():
            cache_cell(study, cell, base / 'analysis/long_tail/runs/full', base / 'analysis/second_layer/runs/full')
    available = [cell for cell in CELLS if (out / 'inputs' / cell / 'manifest.json').exists()]
    missing = [cell for cell in CELLS if cell not in available]
    if not available:
        raise ValueError('No completed cells available to report')
    all_tables = {name: [] for name in ('recommendation', 'nll', 'tree', 'attribution', 'paired_beam', 'unseen', 'support_joint')}
    baselines, survival, pairing, costs = [], [], [], []
    rec_frames, search_frames = {}, {}
    for cell in available:
        folder = out / 'inputs' / cell
        metadata = validate_cache(folder)
        if cfg.get('study_version') == 2 and metadata.get('cell_role') != 'trained_in_study':
            raise ValueError('Historical Beauty checkpoint cannot serve as a newly trained study cell')
        for key, filename in [('recommendation', 'recommendation_metrics.csv'), ('nll', 'nll_by_level.csv'),
                              ('tree', 'tree.csv'), ('attribution', 'attribution.csv'), ('paired_beam', 'paired_beam.csv'),
                              ('unseen', 'unseen.csv'), ('support_joint', 'support_joint.csv')]:
            f = pd.read_csv(folder / filename)
            f.insert(0, 'cell', cell)
            f.insert(1, 'dataset', metadata['dataset'])
            f.insert(2, 'tokenizer', metadata['tokenizer'])
            all_tables[key].append(f)
        baseline = pd.read_csv(folder / 'baselines.csv.gz')
        if baseline.sample_key.duplicated().any() or len(baseline) != metadata['samples']:
            raise ValueError(f'Baseline coverage invalid for {cell}')
        for name in ('raw', 'windows'):
            baseline[name + '_mrr_gain'] = baseline.model_mrr - baseline[name + '_mrr']
        cols = ['model_ce', 'raw_ce', 'windows_ce', 'model_mrr', 'raw_mrr', 'windows_mrr',
                'raw_ce_gain', 'windows_ce_gain', 'raw_mrr_gain', 'windows_mrr_gain']
        baselines.extend(interval_rows(baseline, cols, cfg, cell=cell, dataset=metadata['dataset']))
        rec_frames[cell], search_frames[cell] = {}, {}
        for beam in (10, 20, 50):
            f = pd.read_csv(folder / f'recommendation_{beam}.csv.gz')
            s = pd.read_csv(folder / f'search_{beam}.csv.gz')
            aligned(baseline, f)
            aligned(baseline, s)
            rec_frames[cell][beam] = f
            search_frames[cell][beam] = s
            f = f.copy()
            f['all_samples'] = 1
            for group in ('Overall', *GROUPS):
                part = f if group == 'Overall' else f[f.group == group]
                for level in range(1, 5):
                    numerator = f'survival_level_{level}'
                    denominator = 'all_samples' if level == 1 else f'survival_level_{level-1}'
                    value, low, high = ratio_interval(part, numerator, denominator,
                        cfg['analysis']['bootstrap_replicates'], cfg['seed'])
                    survival.append({'cell': cell, 'beam': beam, 'group': group, 'level': level,
                        'samples': len(part), 'entering_samples': int(part[denominator].sum()),
                        'absolute_survival': float(part[numerator].mean()) if len(part) else None,
                        'conditional_survival': value, 'ci_low': low, 'ci_high': high,
                        'top10_hit_rate': float(part.recall_at_10.mean()) if len(part) else None})
        cost_path = study / 'experiments' / cell / 'recommender_train/training_result.json'
        token_cost_path = study / 'experiments' / cell / 'tokenizer_train/training_result.json'
        cost = read_json(cost_path) if cost_path.exists() else metadata['checkpoint']
        token_cost = read_json(token_cost_path) if token_cost_path.exists() else {}
        costs.append({'cell': cell, 'best_epoch': cost.get('epoch'), 'best_global_step': cost.get('global_step'),
                      'ended_epoch': cost.get('training_ended_epoch'), 'ended_steps': cost.get('training_ended_global_step'),
                      'recommender_seconds': cost.get('seconds'), 'tokenizer_seconds': token_cost.get('seconds'),
                      'tokenizer_steps': token_cost.get('training_ended_global_step')})
    for dataset in ('beauty', 'toys'):
        a_name, b_name = dataset + '_rqkmeans', dataset + '_rqvae'
        if a_name not in available or b_name not in available:
            continue
        a_folder, b_folder = [out / 'inputs' / cell for cell in (a_name, b_name)]
        pd.testing.assert_frame_equal(pd.read_csv(a_folder / 'items.csv'), pd.read_csv(b_folder / 'items.csv'))
        for beam in (10, 20, 50):
            result = paired_comparison(rec_frames[a_name][beam], rec_frames[b_name][beam],
                                       cfg['analysis']['bootstrap_replicates'], cfg['seed'])
            result.insert(0, 'beam', beam)
            result.insert(0, 'dataset', dataset)
            result['difference_direction'] = 'RQ-VAE minus RQ-kmeans'
            pairing.extend(result.to_dict('records'))
            a, b = aligned(search_frames[a_name][beam], search_frames[b_name][beam])
            common = (a.entered_second == 1) & (b.entered_second == 1)
            f = a.loc[common, ['user_id', 'group']].copy()
            f['second_survival_difference'] = b.loc[common, 'retained_second'] - a.loc[common, 'retained_second']
            pairing.extend(interval_rows(f, ['second_survival_difference'], cfg, dataset=dataset, beam=beam,
                                         cohort='both_tokenizers_first_layer_survives',
                                         difference_direction='RQ-VAE minus RQ-kmeans'))
    tables = out / 'tables'
    merged = {key: pd.concat(value, ignore_index=True) for key, value in all_tables.items()}
    for name, f in merged.items():
        write_csv(tables / f'{name}.csv', f)
    baseline = pd.DataFrame(baselines)
    survival = pd.DataFrame(survival)
    write_csv(tables / 'baseline_gains_with_intervals.csv', baseline)
    write_csv(tables / 'layer_survival.csv', survival)
    write_csv(tables / 'paired_tokenizers.csv', pd.DataFrame(pairing))
    write_csv(tables / 'training_costs.csv', pd.DataFrame(costs))
    findings = []
    for cell in available:
        tail_base = baseline[(baseline.cell == cell) & (baseline.group == 'Tail')].iloc[0]
        joint = merged['support_joint']
        joint = joint[(joint.cell == cell) & (joint.group == 'Tail')]
        for beam in (10, 20, 50):
            p = survival[(survival.cell == cell) & (survival.group == 'Tail') & (survival.beam == beam)]
            active = p[(p.level >= 2) & p.conditional_survival.notna()]
            minimum = int(active.loc[active.conditional_survival.idxmin(), 'level']) if len(active) else None
            findings.append({'cell': cell, 'phenomenon': 'L2 has largest conditional drop among L2-L4',
                             'beam': beam, 'supported': minimum == 2 if minimum else None, 'limiting_level': minimum})
        findings.append({'cell': cell, 'phenomenon': 'Tail MRR improves but raw-frequency CE worsens',
                         'supported': bool(tail_base.raw_mrr_gain_ci_low > 0 and tail_base.raw_ce_gain_ci_high < 0),
                         'raw_mrr_gain': tail_base.raw_mrr_gain, 'raw_ce_gain': tail_base.raw_ce_gain})
        pair = merged['paired_beam']
        p = pair[(pair.cell == cell) & (pair.group == 'Tail') & (pair.beam == 50)]
        if len(p):
            row = p.iloc[0]
            findings.append({'cell': cell, 'phenomenon': 'Wider beam rescues L2 on common first-layer survivors',
                             'supported': bool(row.retained_second_ci_low > 0), 'difference': row.retained_second,
                             'ci_low': row.retained_second_ci_low, 'ci_high': row.retained_second_ci_high})
    findings = pd.DataFrame(findings)
    write_csv(tables / 'phenomenon_checks.csv', findings)
    verdicts = []
    for (phenomenon, beam), f in findings.groupby(['phenomenon', 'beam'], dropna=False, sort=True):
        evidence = int(f.supported.fillna(False).sum())
        checked = len(f)
        if missing:
            verdict = '证据不足：补充组合未完成'
        elif evidence == 4:
            verdict = '四组合重复'
        elif evidence:
            verdict = '部分组合重复'
        else:
            verdict = '未满足预定义重复标准'
        verdicts.append({'phenomenon': phenomenon, 'beam': beam, 'completed_cells': checked,
                         'supporting_cells': evidence, 'verdict': verdict})
    verdicts = pd.DataFrame(verdicts)
    write_csv(tables / 'replication_verdicts.csv', verdicts)
    write_json(tables / 'summary.json', {'status': 'complete' if not missing else 'incomplete',
               'available_cells': available, 'missing_cells': missing,
               'findings': findings.replace({np.nan: None}).to_dict('records'),
               'confidence': '95% percentile, user-clustered, seed 42; one training seed only'})
    figures(out, baseline, survival, merged['attribution'], merged['support_joint'])
    ce = baseline[baseline.group == 'Tail'][['cell', 'samples', 'raw_ce_gain', 'raw_ce_gain_ci_low',
                                          'raw_ce_gain_ci_high', 'raw_mrr_gain', 'raw_mrr_gain_ci_low', 'raw_mrr_gain_ci_high']]
    selection = merged['attribution']
    selection = selection[(selection.group == 'Tail') & (selection.cohort == 'entered')][
        ['cell', 'beam', 'samples', 'retained_second', 'cause_local_capacity', 'cause_cross_parent', 'cause_cutoff_tie']]
    missing_text = '全部四个组合已完成。' if not missing else f'待完成：{", ".join(missing)}。目前不足以判断现象的跨数据集、跨 RQ 方法普遍性。'
    text = f'''# Beauty / Toys × RQ tokenizer 交叉验证

{missing_text}

## 实验条件与覆盖

嵌入模型：{cfg.get('embedding_model_id', 'google/flan-t5-xl')}，维度：{cfg['embedding_dim']}。seed=42；TIGER 使用项目内独立配置 rq_recommender.yaml，最多150 epoch、验证 Recall@5 选优、patience=10、单 GPU、梯度累积16。四组均重新训练 SID 和 TIGER。SID 的训练预算和验证频率以本轮各组保存的训练配置为准；每组1 GPU、batch2,048；其余SID参数沿用默认值。相对八卡默认配置，全局batch由16,384改为2,048。两者预算和结构不同，不能声称计算成本相同。推理 float32，batch8，beam10/20/50，不施加前缀约束。GPU共享策略为 allow_gpu_sharing={cfg.get('allow_gpu_sharing', False)}；预检已有进程与剩余显存记录在study_manifest，耗时不能视为独占GPU下的纯运行性能。分组从本轮各数据集的原始训练交互重新计算并冻结，不依赖历史checkpoint或分析产物。

每个数据集共享语义输入和冻结的原始训练频次分组，各 tokenizer 的 SID 前缀与测试编码分别生成。样本以 dataset + sample_key 唯一关联；每个有效目标在分母中保留。Toys 的有效测试数量由原始数据扫描确定，不套用 Beauty 的22,363。不同数据集不作配对推断。

## 第二层搜索现象

{markdown_table(selection)}

条件存活率分母是该 beam 的第一层存活样本。比较 beam 收益时使用三个 beam 共同存活的第一层样本，见 paired_beam.csv；跨 tokenizer 的共同第一层样本见 paired_tokenizers.csv。layer_survival.csv 同时列出所有层的无条件和条件生存率，避免事先假定第二层一定是最大瓶颈。

![条件生存](../figures/conditional_survival.png)

局部容量、跨父竞争、截止并列仅描述剪枝机制。较大的跨父竞争占比不能证明局部预测已经准确，也不能单凭其随 beam 改变判断普遍性。各组合原报告保留局部/全局排名、累计分数和阈值分析。

## 相对频次基线的预测收益

{markdown_table(ce)}

CE 收益为基线减模型；MRR 收益为模型减基线。MRR 的差值也逐样本配对并按用户聚类 bootstrap。两种频次基线均加一平滑；同时报告原始交互和潜在目标窗口基线。跨 tokenizer 的合法孩子空间不同，原始 token CE 不能单独作为优劣结论，须结合各自的基线收益和 tree.csv 结构。

![收益](../figures/baseline_gains.png)

## 训练支持和未见路径

support_joint.csv 固定组、合法孩子桶后展示训练支持与困难的关系，保留每桶样本量；空桶不记零。unseen.csv 逐组合展示首次无训练支持层与实际丢失层，区分物品未见但前缀见过的情形。潜在目标窗口是增强前可能的标签窗口数量，不是历史真实抽样次数。这些关联不足以识别 tokenizer、监督分布或模型校准的因果作用。

## 证据判定与下一步

{markdown_table(verdicts)}

{markdown_table(findings)}

supported=true 表示该项预定义方向在该组合中成立；false 表示没有满足该项标准，不能一概解释为反向结论。最大条件损失为描述性比较；MRR/CE 与 beam 收益采用95%区间判断，其余支持关系保留描述性表格。若四个组合方向一致，得到“两类 RQ 方法、两个数据集、单 seed 下的重复证据”；部分成立时逐条报告适用范围，不宣称所有 RQ tokenizer 的普遍定理。CI 不包含重新训练随机性，多个探索性比较未经多重检验校正。

在补充组合完成并检查例外之前，不进入新算法实验。随后优先复核不一致组合的训练收敛、码本利用和树形差异；一致现象再用于设计受控的 tokenizer/训练干预实验。

## 离线复算

comparison/inputs 内保存本报告需要的冻结逐样本数据及校验和；复制整个 study 后执行 --stage summary --resume STUDY_DIR 即可重建 CSV/JSON、PNG/PDF，无需 checkpoint、GPU 或原目录。完整候选和真实前缀概率保留于各组合分析目录。训练耗时含初始化及I/O；training_costs.csv 记录实际停止和最佳 epoch/step，不能用每个数据集相同 epoch 数推断计算量相同。
'''
    # report.md is in comparison/, figures are alongside it.
    text = text.replace('../figures/', 'figures/')
    atomic_write(out / 'report.md', lambda p: p.write_text(text))
    return out / 'report.md'


def figures(out, baseline, survival, attribution, support):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target = out / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            atomic_write(target / f'{name}.{extension}', lambda p: fig.savefig(p, dpi=180))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for cell, f in survival[(survival.group == 'Tail') & (survival.beam == 50)].groupby('cell', sort=True):
        ax.plot(f.level, f.conditional_survival, marker='o', label=cell)
    ax.set(xticks=[1, 2, 3, 4], xlabel='SID level (beam 50)', ylabel='Tail conditional survival', ylim=(0, 1))
    ax.legend(fontsize=8)
    save(fig, 'conditional_survival')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    f = baseline[baseline.group == 'Tail'].sort_values('cell')
    for ax, metric in zip(axes, ['raw_ce_gain', 'raw_mrr_gain']):
        y = f[metric].to_numpy(float)
        ax.bar(range(len(f)), y)
        ax.errorbar(range(len(f)), y, yerr=np.array([y-f[metric+'_ci_low'], f[metric+'_ci_high']-y]), fmt='none', c='black')
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(xticks=range(len(f)), xticklabels=f.cell, ylabel=metric)
        ax.tick_params(axis='x', labelrotation=35)
    save(fig, 'baseline_gains')
    fig, ax = plt.subplots(figsize=(8, 4))
    f = attribution[(attribution.group == 'Tail') & (attribution.cohort == 'entered') & (attribution.beam == 50)]
    columns = ['cause_retained', 'cause_local_capacity', 'cause_cross_parent', 'cause_cutoff_tie']
    f.set_index('cell')[columns].plot.bar(stacked=True, ax=ax)
    ax.set_ylabel('Fraction of first-layer survivors')
    ax.legend(fontsize=7)
    save(fig, 'attribution')
