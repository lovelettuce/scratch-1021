"""Portable CPU aggregation: committed outputs suffice, no source model access."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.beam_rescue.core import ratio_statistics
from src.analysis.long_tail.io import atomic_write, completed_shards, digest, read_json, write_csv, write_json
from src.analysis.long_tail.report import markdown_table
from .core import BEAMS, GROUPS, KS, TRACKS
from .experiment import exclusive, now, validate_run

CATEGORIES = ('both_hit', 'search_miss', 'incidental_hit', 'both_miss')
CONTRASTS = (('exact', 'legal', 'search'), ('legal', 'rescored', 'illegal_slots'),
             ('rescored', 'original', 'numeric_and_ties'), ('exact', 'original', 'total'))


def summarize(frame, repeats=2000, seed=42):
    metrics, pairs, diagnostics, cohorts = [], [], [], []
    for cell in frame.cell.unique():
        for beam in BEAMS:
            base = frame[(frame.cell == cell) & (frame.beam == beam)]
            for group in GROUPS:
                f = (base if group == 'Overall' else base[base.group == group]).reset_index(drop=True)
                identity = dict(cell=cell, dataset=cell.split('_')[0], beam=beam, group=group,
                                samples=len(f), users=f.user_id.nunique())
                entries, specs = [], []
                ones = np.ones(len(f))

                def add(table, key, values, denominator=ones, **fields):
                    values, den = np.asarray(values, float), np.asarray(denominator, float)
                    specs.append((key, values, den))
                    entries.append((table, key, dict(**identity, **fields,
                        numerator_sum=float(values.sum()), denominator_sum=float(den.sum()),
                        denominator_users=int(f.loc[den != 0, 'user_id'].nunique()))))

                for k in KS:
                    for track in TRACKS:
                        for metric in ('recall', 'ndcg'):
                            name = f'{metric}_at_{k}'
                            add(metrics, f'{track}/{name}', f[f'{track}_{name}'], track=track, metric=name)
                    for other, reference, label in CONTRASTS:
                        for metric in ('recall', 'ndcg'):
                            name = f'{metric}_at_{k}'
                            values = f[f'{other}_{name}']-f[f'{reference}_{name}']
                            add(metrics, f'{label}/{name}', values, track='paired_delta',
                                contrast=label, metric=name)
                    for category in CATEGORIES:
                        values = f[f'category_at_{k}'].eq(category).to_numpy()
                        add(pairs, f'{k}/{category}', values, k=k, category=category,
                            count=int(values.sum()))
                    for field in (f'exact_top_{k}_beam_coverage', f'sensitive_at_{k}',
                                  f'exact_cutoff_tie_at_{k}'):
                        add(diagnostics, field, f[field], diagnostic=field)
                    for layer in range(1, 5):
                        values = f[f'category_at_{k}'].eq('search_miss') & f.first_lost_level.eq(layer)
                        den = f[f'category_at_{k}'].eq('search_miss')
                        add(diagnostics, f'{k}/lost/{layer}', values, den,
                            diagnostic='search_miss_layer', k=k, first_lost_level=layer)
                    if len(f):
                        gain = f[f'category_at_{k}'].eq('search_miss').sum()
                        loss = f[f'category_at_{k}'].eq('incidental_hit').sum()
                        delta = (f[f'exact_recall_at_{k}']-f[f'legal_recall_at_{k}']).mean()
                        if not np.isclose(delta, (gain-loss)/len(f), atol=1e-14, rtol=0):
                            raise ValueError('Paired Recall conservation failed')
                        # Every metric's additive bridge must hold per sample.
                        for metric in ('recall', 'ndcg'):
                            name = f'{metric}_at_{k}'
                            bridge = sum((f[f'{other}_{name}'] - f[f'{reference}_{name}'])
                                for other, reference, _ in CONTRASTS[:3])
                            if not np.allclose(bridge, f[f'exact_{name}']-f[f'original_{name}'], atol=1e-14, rtol=0):
                                raise ValueError('Metric bridge decomposition failed')
                for cohort in ('l2_rescued_from_10', 'complete_outside_top10', 'original_hit_lost_from_10'):
                    den = f[cohort].to_numpy(bool)
                    for k in KS:
                        add(cohorts, f'{cohort}/{k}', den*f[f'exact_recall_at_{k}'], den,
                            cohort=cohort, k=k, cohort_samples=int(den.sum()))
                stats = ratio_statistics(f, specs, repeats, seed)
                for table, key, row in entries:
                    value, low, high, valid = stats[key]
                    row.update(value=value, ci_low=low, ci_high=high,
                               bootstrap_valid_replicates=valid,
                               status='no_samples' if value is None else 'ok')
                    table.append(row)
                # Continuous diagnostics are descriptive, not surrogate evidence for ranking gain.
                for column in ('target_nll', 'exact_rank', 'best_legal_score_gap',
                               'catalog_score_error', 'native_log_score_error', 'invalid_candidates',
                               'invalid_slots_at_5', 'invalid_slots_at_10'):
                    diagnostics.append({**identity, 'diagnostic': column,
                        'value': float(f[column].mean()) if len(f) else None,
                        'status': 'ok' if len(f) else 'no_samples'})
    return {k: pd.DataFrame(v) for k, v in dict(metrics=metrics, pairs=pairs,
                                               diagnostics=diagnostics, cohorts=cohorts).items()}


def figures(out, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group in ('Overall', 'Head', 'Mid', 'Tail', 'Train-unseen'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        m = tables['metrics']
        m = m[(m.group == group) & (m.beam == 50) & (m.metric == 'recall_at_10')]
        cells = list(m.cell.unique())
        x = np.arange(len(cells))
        for j, track in enumerate(TRACKS):
            part = m[m.track == track].set_index('cell').reindex(cells)
            axes[0].bar(x+(j-1.5)*.18, part.value*100, width=.18, label=track)
        axes[0].set(title=f'{group}: beam 50 vs exact', ylabel='Recall@10 (%)', xticks=x, xticklabels=cells)
        p = tables['pairs']
        p = p[(p.group == group) & (p.beam == 50) & (p.k == 10)]
        for category, offset, sign in [('search_miss', -.18, 1), ('incidental_hit', .18, -1)]:
            part = p[p.category == category].set_index('cell').reindex(cells)
            axes[1].bar(x+offset, sign*part.value*100, width=.36, label=category)
        axes[1].axhline(0, color='grey', linewidth=.8)
        axes[1].set(title='Exact minus legal beam: gains and losses', ylabel='Percentage points', xticks=x, xticklabels=cells)
        for ax in axes:
            ax.tick_params(axis='x', labelrotation=18)
            ax.legend(fontsize=8)
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            path = out/'figures'/f'{group.lower()}.{extension}'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)
        plt.close(fig)


def report(run_dir, repeats=None):
    run = Path(run_dir).resolve()
    with exclusive(run):
        manifest = read_json(run/'manifest.json')
        validate_run(run, manifest, external=False)
        parts = []
        for cell in manifest['cells']:
            stage = f'scores_{cell}'
            count = completed_shards(run, stage)
            if not count or manifest['cell_status'].get(cell) != 'complete':
                raise ValueError(f'Incomplete cell: {cell}')
            keys = []
            for marker in sorted((run/'progress'/stage).glob('*.json')):
                info = read_json(marker); keys.extend(info['sample_keys'])
                for entry in info['artifacts']:
                    if entry['path'].endswith('.paired.csv.gz'):
                        parts.append(pd.read_csv(run/entry['path'],
                            dtype={'sample_key':str, 'user_id':str, 'target_item_id':str}))
                array_path = next(x['path'] for x in info['artifacts'] if x['path'].endswith('.npy'))
                scores = np.load(run/array_path, mmap_mode='r', allow_pickle=False)
                if scores.shape != (len(info['sample_keys']), manifest['sources'][cell]['catalog_items']):
                    raise ValueError('Score matrix shard dimensions differ')
            if len(keys) != manifest['sources'][cell]['selected_samples'] or len(set(keys)) != len(keys):
                raise ValueError('Report coverage is incomplete or duplicate')
            if keys != manifest['sources'][cell]['keys'][:len(keys)]:
                raise ValueError('Committed score sample order differs')
        frame = pd.concat(parts, ignore_index=True)
        if frame.duplicated(['cell', 'sample_key', 'beam']).any():
            raise ValueError('Duplicate paired report rows')
        for cell in manifest['cells']:
            for beam in BEAMS:
                if len(frame[(frame.cell == cell)&(frame.beam == beam)]) != manifest['sources'][cell]['selected_samples']:
                    raise ValueError('Missing paired report rows')
        tables = summarize(frame, repeats or manifest['parameters']['bootstrap_replicates'],
                           manifest['parameters']['seed'])
        out = run/'report'
        for name, table in tables.items():
            write_csv(out/'tables'/f'{name}.csv', table)
        figures(out, tables)
        text = ['# 全目录精确排序：搜索遗漏与模型评分限制',
            '本报告固定模型、SID和测试历史，对全部合法物品累加原始全词表条件 log 概率。'
            '不做合法孩子内归一化，不过滤训练未见物品或历史物品。精确排序指穷举目录，不指无限精度。',
            '**覆盖：**'+('四组/所选组全量测试。' if manifest['is_full_evaluation'] else '小规模验证，不可作为全量结论。'),
            '## 比较口径',
            'original 为原始 torch.topk；rescored 对原始候选统一重评分；legal 再删除非法候选且不补充；'
            'exact 枚举合法目录。后三者按分数降序、SID字典序处理精确并列。合法候选分数统一引用目录列，'
            '独立重评分差异已校验不超过1e-4。截止点附近1e-4范围样本单列敏感性比例。',
            '主比较 exact−legal 表示搜索候选范围变化；legal−rescored 表示最终非法槽位影响；'
            'rescored−original 表示数值及并列规则影响。三者之和逐样本还原 exact−original。',
            f"主要比例与配对差值按用户聚类 bootstrap {repeats or manifest['parameters']['bootstrap_replicates']} 次，"
            'seed=42，95%区间；仅反映固定模型的测试抽样不确定性，未经多重比较校正。空队列为无样本。',
            '## Beam50 与精确 Top-10：各组结果']
        summary = []
        for cell in manifest['cells']:
            for group in GROUPS:
                m = tables['metrics']; p = tables['pairs']
                m = m[(m.cell == cell)&(m.group == group)&(m.beam == 50)&(m.metric == 'recall_at_10')]
                p = p[(p.cell == cell)&(p.group == group)&(p.beam == 50)&(p.k == 10)].set_index('category')
                delta = m[(m.track == 'paired_delta')&(m.contrast == 'search')].iloc[0]
                def val(track):
                    value = m[m.track == track].value.iloc[0]
                    return '无样本' if pd.isna(value) else f'{value:.3%}'
                summary.append({'组合':cell,'分组':group,'样本':int(delta.samples),
                    '原始':val('original'),'合法beam':val('legal'),'精确':val('exact'),
                    '搜索遗漏':int(p.loc['search_miss','count']),
                    '原有偶然命中':int(p.loc['incidental_hit','count']),
                    '差值_pp':f'{delta.value*100:.4f}' if pd.notna(delta.value) else '无样本',
                    '区间_pp':f'[{delta.ci_low*100:.4f}, {delta.ci_high*100:.4f}]' if pd.notna(delta.value) else '无样本'})
        text.append(markdown_table(pd.DataFrame(summary)))
        for group in ('Overall', 'Tail'):
            text.append(f'![{group}](figures/{group.lower()}.png)')
        text += ['## 结论边界与判读',
            'search_miss 表示模型已将目标排入精确Top-K，但beam未保留它；incidental_hit表示原beam遗漏了'
            '更高分竞争物品，使真实目标获得命中。二者可能抵消，因此不能只看净Recall差值。',
            'both_miss 表示模型全目录排序未将真实目标排入Top-K，不能据此确定训练支持、tokenizer或概率校准的因果作用。'
            '精确排序不是实际推荐效果的上限，更准确地搜索当前模型得分可能降低推荐命中。',
            '完整生成但未进Top-10、第二层救回和原有命中丢失队列的精确排名去向见 cohorts.csv；'
            '逐样本NLL、目标排名、分数差、搜索首次丢失层和数值审计保存在cells下的paired分片。',
            '验收：提交分片校验和、样本覆盖、目录矩阵维度、命中增减守恒及四轨指标桥接均通过。'
            '报告可仅由本次目录的快照和提交产物重建，无需GPU、checkpoint或原始数据。']
        atomic_write(out/'report.md', lambda p:p.write_text('\n\n'.join(text)+'\n'))
        write_json(out/'manifest.json', dict(status='complete', generated_at=now(),
            parameters=dict(bootstrap_replicates=repeats or manifest['parameters']['bootstrap_replicates'],seed=42),
            artifacts=[{'path':str(p.relative_to(out)),'sha256':digest(p)}
                       for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json']))
        manifest.update(status='complete', report_generated_at=now())
        write_json(run/'manifest.json', manifest)
    return out
