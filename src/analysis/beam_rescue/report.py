"""Standalone CPU report from frozen comparison CSVs, with source audit."""
from __future__ import annotations

from datetime import datetime
from importlib.metadata import version
import logging
from pathlib import Path
import platform
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.analysis.long_tail.io import ROOT, atomic_write, digest, read_json, write_csv, write_json
from src.analysis.long_tail.report import markdown_table
from src.experiments.rq_cross_validation.common import EXPERIMENTS
from src.experiments.rq_cross_validation.report import aligned, validate_cache
from .core import BEAMS, PAIRS, GROUPS, METRICS, OUTCOMES, pair_frames, summarize, validate_frame

LOGGER = logging.getLogger(__name__)
EXPECTED_COUNTS = {'beauty': 22363, 'toys': 19412}
LABELS = {'lost_l3': 'Lost at L3', 'lost_l4': 'Lost at L4',
          'complete_outside_top10': 'Complete, outside Top-10', 'hit_top10': 'Top-10 hit'}


def load_inputs(study, expected_counts):
    """Validate all four cells before creating any new report artifacts."""
    frames, inventory, checks = {}, {}, []
    for cell in EXPERIMENTS:
        folder = study / 'comparison/inputs' / cell
        metadata = validate_cache(folder)
        dataset = cell.split('_')[0]
        if (metadata['cell'], metadata['dataset'], metadata.get('cell_role')) != (cell, dataset, 'trained_in_study'):
            raise ValueError(f'Wrong cache identity: {cell}')
        files = {entry['path'] for entry in metadata['artifacts']}
        required = {f'recommendation_{beam}.csv.gz' for beam in BEAMS} | {'recommendation_metrics.csv'}
        if not required <= files:
            raise ValueError(f'Uncommitted recommendation inputs: {cell}')
        inventory[str((folder / 'manifest.json').relative_to(study))] = digest(folder / 'manifest.json')
        for entry in metadata['artifacts']:
            inventory[str((folder / entry['path']).relative_to(study))] = entry['sha256']
        frozen = pd.read_csv(folder / 'recommendation_metrics.csv')
        frames[cell] = {}
        for beam in BEAMS:
            f = pd.read_csv(folder / f'recommendation_{beam}.csv.gz',
                            dtype={'sample_key': str, 'user_id': str, 'target_item_id': str})
            validate_frame(f)
            if len(f) != metadata['samples'] or len(f) != expected_counts[dataset]:
                raise ValueError(f'Sample coverage differs: {cell}, beam {beam}, {len(f)}')
            if beam != BEAMS[0]:
                aligned(frames[cell][BEAMS[0]], f)
            for group in GROUPS:
                part = f if group == 'Overall' else f[f.group.eq(group)]
                saved = frozen[frozen.beam.eq(beam) & frozen.group.eq(group)]
                if len(saved) != 1 or int(saved.iloc[0].test_samples) != len(part):
                    raise ValueError(f'Frozen group coverage differs: {cell}/{beam}/{group}')
                for metric in METRICS:
                    actual, reference = part[metric].mean(), saved.iloc[0][metric]
                    if not (pd.isna(actual) and pd.isna(reference)) and not np.isclose(actual, reference, atol=1e-12, rtol=0):
                        raise ValueError(f'Frozen metric differs: {cell}/{beam}/{group}/{metric}')
            frames[cell][beam] = f
            checks.append({'cell': cell, 'beam': beam, 'samples': len(f), 'users': f.user_id.nunique(),
                           'schema_survival_alignment_and_frozen_metrics': 'passed'})
    return frames, inventory, checks


def make_figures(out, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = out / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    destinations = tables['destinations']
    changes = tables['hit_changes']
    for group in ('Overall', 'Tail'):
        d = destinations[(destinations.group == group) & (destinations.beam_small == 10) &
            (destinations.beam_large == 50) & (destinations.cohort == 'pair_common_first') &
            (destinations.queue == 'l2_rescued') & (destinations.denominator == 'queue')]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bottom = np.zeros(len(EXPERIMENTS))
        for state in OUTCOMES:
            part = d[d.outcome == state].set_index('cell').reindex(EXPERIMENTS)
            values = part.value.to_numpy(float) * 100
            ax.bar(EXPERIMENTS, values, bottom=bottom, label=LABELS[state])
            bottom += np.nan_to_num(values)
        for i, cell in enumerate(EXPERIMENTS):
            n = d[(d.cell == cell) & (d.outcome == OUTCOMES[0])].denominator_sum.iloc[0]
            ax.text(i, 102, f'n={int(n)}' if n else 'no samples', ha='center', fontsize=9)
        ax.set(ylabel='Share of L2-rescued samples (%)', ylim=(0, 116),
               title=f'{group}: destinations after L2 rescue, beam 10 to 50')
        ax.tick_params(axis='x', labelrotation=12)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(folder / f'rescue_destinations_{group.lower()}.{extension}', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(EXPERIMENTS))
        for change, offset, sign in [('gained', -.18, 1), ('lost', .18, -1)]:
            part = changes[(changes.group == group) & (changes.beam_small == 10) &
                (changes.beam_large == 50) & (changes.path_change == 'all') &
                (changes.hit_change == change)].set_index('cell').reindex(EXPERIMENTS)
            ax.bar(x + offset, sign * part.value.to_numpy(float) * 100, width=.36, label=change)
        delta = tables['metrics']
        delta = delta[(delta.group == group) & (delta.beam_small == 10) & (delta.beam_large == 50) &
                      (delta.statistic == 'paired_delta') & (delta.metric == 'recall_at_10')]
        delta = delta.set_index('cell').reindex(EXPERIMENTS)
        ax.plot(x, delta.value * 100, 'ko', label='net Recall@10 change')
        ax.axhline(0, color='grey', linewidth=.8)
        ax.set(xticks=x, xticklabels=EXPERIMENTS, ylabel='Percentage points of all group samples',
               title=f'{group}: Top-10 gains and losses, beam 10 to 50')
        ax.tick_params(axis='x', labelrotation=12)
        ax.legend()
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(folder / f'hit_gains_losses_{group.lower()}.{extension}', dpi=160)
        plt.close(fig)


def write_report(out, tables, repeats, seed):
    d, h, m = (tables[key] for key in ('destinations', 'hit_changes', 'metrics'))
    sections = ['# 第二层救回目标去向与最终推荐收益',
        '四组固定模型的离线配对诊断。主比较 beam10→50，辅助比较10→20、20→50；不重训、不重新推理。',
        '## 口径与证据范围',
        '共同首层队列要求比较双方均通过第一层；第二层救回指小 beam 在第二层丢失而大 beam 保留，'
        '反向丢失定义相反。另存三个 beam 共同首层队列作为敏感性分析。两类队列的去向分别读取大、小 beam。'
        '空队列为无样本，不解释为零收益。',
        f'主比例和指标差值按用户聚类 bootstrap {repeats} 次，seed={seed}，95% percentile 区间。'
        '每次重采样重新计算条件分母；记录有效重采样数，零分母的抽样不参与区间。'
        '状态转移矩阵仅作描述性计数。区间不包含重新训练随机性，探索性比较未经多重检验校正。',
        '保留已有 Top-K 并列与非法槽位口径；不从 CSV 推算精确排名。路径变化类别描述观测关联，不表示因果归因。',
        '## 长尾主比较：救回目标去了哪里']
    summary = []
    for cell in EXPERIMENTS:
        part = d[(d.cell == cell) & (d.group == 'Tail') & (d.beam_small == 10) & (d.beam_large == 50) &
                 (d.cohort == 'pair_common_first') & (d.queue == 'l2_rescued') & (d.denominator == 'queue')]
        row = {'组合': cell, '救回样本数': int(part.denominator_sum.iloc[0])}
        for outcome, label in LABELS.items():
            value = part[part.outcome == outcome].value.iloc[0]
            row[label] = f'{value:.2%}' if pd.notna(value) else '无样本'
        summary.append(row)
    sections += [markdown_table(pd.DataFrame(summary)),
                 '上述比例以第二层救回样本为分母；占全部组内样本的比例及区间另见 destinations.csv。',
                 '![救回去向](figures/rescue_destinations_tail.png)',
                 '## 长尾主比较：最终收益与损失']
    summary = []
    conclusions = []
    for cell in EXPERIMENTS:
        hit = h[(h.cell == cell) & (h.group == 'Tail') & (h.beam_small == 10) & (h.beam_large == 50) &
                (h.path_change == 'all')].set_index('hit_change')
        metric = m[(m.cell == cell) & (m.group == 'Tail') & (m.beam_small == 10) & (m.beam_large == 50) &
                   (m.statistic == 'paired_delta') & (m.metric == 'recall_at_10')].iloc[0]
        ndcg = m[(m.cell == cell) & (m.group == 'Tail') & (m.beam_small == 10) & (m.beam_large == 50) &
                 (m.statistic == 'paired_delta') & (m.metric == 'ndcg_at_10')].iloc[0]
        summary.append({'组合': cell, '新增命中': int(hit.loc['gained', 'count']),
            '丢失命中': int(hit.loc['lost', 'count']), 'Recall差值_pp': metric.value * 100,
            'Recall差值95%区间_pp': f'[{metric.ci_low*100:.4f}, {metric.ci_high*100:.4f}]',
            'NDCG差值': ndcg.value})
        part = d[(d.cell == cell) & (d.group == 'Tail') & (d.beam_small == 10) & (d.beam_large == 50) &
                 (d.cohort == 'pair_common_first') & (d.queue == 'l2_rescued') & (d.denominator == 'queue')].set_index('outcome')
        n = int(part.denominator_sum.iloc[0])
        if n:
            later = int(part.loc[['lost_l3', 'lost_l4'], 'outcome_samples'].sum())
            outside = int(part.loc['complete_outside_top10', 'outcome_samples'])
            if later > n / 2:
                explanation = f'{later}/{n} 个救回目标在第三、四层丢失，优先检查后续预测与搜索。'
            elif outside > n / 2:
                explanation = f'{outside}/{n} 个救回目标完整生成但未进 Top-10，优先检查路径得分与最终排序。'
            elif part.loc['hit_top10', 'outcome_samples'] > n / 2:
                explanation = '多数第二层救回目标最终命中，需结合其他样本的命中损失解释总体变化。'
            else:
                explanation = '救回目标去向没有单一非命中类别占过半，应联合检查后续丢失与最终排序。'
        else:
            explanation = '没有第二层救回样本，不能判断救回后的主要瓶颈。'
        gained, lost = int(hit.loc['gained', 'count']), int(hit.loc['lost', 'count'])
        if gained and lost:
            explanation += f' 全部长尾新增命中{gained}条、丢失命中{lost}条，存在收益与损失抵消。'
        if metric.ci_low <= 0 <= metric.ci_high:
            explanation += ' Recall差值区间包含0，不支持宣称有明确方向的改善或退化。'
        conclusions.append(f'- **{cell}**：{explanation}')
    sections += [markdown_table(pd.DataFrame(summary)),
                 '![收益与损失](figures/hit_gains_losses_tail.png)',
                 '## 分组合判断与下一步', *conclusions,
                 '这些结果定位搜索宽度收益的流失位置，不证明 tokenizer、概率校准或训练支持是因果来源。',
                 '## 产物与校验',
                 '- `paired/*.csv.gz`：逐样本身份、层状态、队列、去向和指标差值。',
                 '- `tables/cohorts.csv`：救回、反向损失及净第二层变化；包含三 beam 共同队列敏感性结果。',
                 '- `tables/destinations.csv`：两类队列去向，分别以队列和全部组内样本为分母。',
                 '- `tables/hit_changes.csv`：新增/丢失命中、双方命中/未命中；按首层、第二层变化细分。',
                 '- `tables/transitions.csv`：全部六状态间的完整转移矩阵，含零计数。',
                 '- `tables/metrics.csv`：队列指标均值与全样本配对差值，含 @5/@10 及区间。',
                 '- `manifest.json`：输入/源码哈希、参数、样本覆盖和验收状态。',
                 '比例列 value 使用0～1单位，差值使用原指标单位；报告中 pp 表示百分点。所有统计表均含样本数与用户数。'
                 '输入校验、元数据配对、生存一致性、原始指标复算、命中增减守恒与输入未修改检查均通过。']
    atomic_write(out / 'report.md', lambda p: p.write_text('\n\n'.join(sections) + '\n'))


def run(study_dir, output_dir=None, repeats=2000, seed=42, expected_counts=None):
    study = Path(study_dir).resolve()
    if repeats < 1:
        raise ValueError('bootstrap_replicates must be positive')
    out = Path(output_dir).resolve() if output_dir else study / 'comparison/beam_rescue'
    inputs = study / 'comparison/inputs'
    # Neither output nesting nor an ancestor may overwrite the immutable inputs.
    if out == inputs or out in inputs.parents or inputs in out.parents:
        raise ValueError('Output must be separate from frozen inputs')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty; choose a new --output-dir: {out}')
    frames, inventory, checks = load_inputs(study, expected_counts or EXPECTED_COUNTS)
    source_files = sorted(Path(__file__).parent.glob('*.py')) + [ROOT / relative for relative in (
        'src/experiments/rq_cross_validation/report.py', 'src/experiments/rq_cross_validation/common.py',
        'src/analysis/long_tail/io.py', 'src/analysis/long_tail/report.py', 'src/analysis/long_tail/data.py')]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit, dirty = None, 'git unavailable'
    manifest = {'schema_version': 1, 'status': 'running',
        'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'parameters': {'beams': BEAMS, 'pairs': PAIRS, 'bootstrap_replicates': repeats, 'seed': seed,
                       'device': 'cpu', 'expected_counts': expected_counts or EXPECTED_COUNTS},
        'input_sha256': inventory, 'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in source_files},
        'runtime': {'python': platform.python_version(),
                    'dependencies': {name: version(name) for name in ('numpy', 'pandas', 'matplotlib')}},
        'git_commit': commit, 'git_status': dirty, 'coverage_checks': checks}
    write_json(out / 'manifest.json', manifest)
    collected = {key: [] for key in ('cohorts', 'destinations', 'hit_changes', 'transitions', 'metrics')}
    try:
        for cell in EXPERIMENTS:
            for small, large in PAIRS:
                LOGGER.info('Analyzing %s: beam %s -> %s', cell, small, large)
                paired = pair_frames(frames[cell], small, large, cell)
                write_csv(out / f'paired/{cell}_{small}_{large}.csv.gz', paired)
                for key, table in summarize(paired, repeats, seed).items():
                    collected[key].append(table)
        tables = {key: pd.concat(parts, ignore_index=True) for key, parts in collected.items()}
        for key, table in tables.items():
            write_csv(out / f'tables/{key}.csv', table)
        for relative, fingerprint in inventory.items():
            if digest(study / relative) != fingerprint:
                raise ValueError(f'Frozen input changed during analysis: {relative}')
        make_figures(out, tables)
        write_report(out, tables, repeats, seed)
        manifest.update(status='complete', finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                        validation={'input_unchanged': True, 'saved_metrics_reproduced': True,
                                    'paired_delta_and_hit_conservation': True},
                        artifacts=[{'path': str(p.relative_to(out)), 'sha256': digest(p)}
                                   for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'])
    except BaseException as ex:
        manifest.update(status='failed', error=f'{type(ex).__name__}: {ex}')
        raise
    finally:
        write_json(out / 'manifest.json', manifest)
    return out
