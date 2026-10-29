"""Offline aggregation of saved target diagnostics and compact beam snapshots."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.long_tail.data import GROUPS
from src.analysis.long_tail.io import (atomic_write,completed_shards,digest,read_json,write_csv,write_json)
from src.analysis.long_tail.metrics import cluster_interval
from src.analysis.long_tail.report import markdown_table
from .common import source_paths
from .diagnostics import CAUSES

LOGGER=logging.getLogger(__name__)


def frames(paths):
    if not paths:raise ValueError('No committed diagnostic shards')
    return pd.concat([pd.read_csv(p,dtype={'sample_key':str,'branch_bucket':str,
        'edge_support_bucket':str,'window_support_bucket':str}) for p in paths],ignore_index=True)


def cohort_summary(frame,cohort,columns,cfg,identity=None,point_columns=()):
    rows=[]
    for group in ['Overall']+GROUPS:
        part=frame if group=='Overall' else frame[frame.group==group]
        ci=cluster_interval(part,list(columns),cfg['bootstrap_replicates'],cfg['seed'])
        row={**(identity or {}),'cohort':cohort,'group':group,'samples':len(part)}
        for c in columns:
            row[c]=float(part[c].mean()) if len(part) else None
            row[c+'_ci_low'],row[c+'_ci_high']=ci[c]
        for c in point_columns:
            row[c]=float(part[c].mean()) if len(part) else None
        rows.append(row)
    return rows


def bucket_summary(frame,keys):
    numeric=['retained_second','model_ce','model_mrr','raw_ce_gain','windows_ce_gain',
             'local_rank_best','global_rank_best','log_score_margin','edge_interaction_share',
             'parent_train_interactions','edge_train_interactions','edge_train_users','edge_potential_target_windows']
    entered=frame[frame.entered_second==1]
    g=entered.groupby(keys,observed=True,dropna=False)
    f=g[numeric].mean()
    f['samples']=g.size()
    for cause in CAUSES[1:]:f['cause_'+cause]=g['cause_'+cause].mean()
    f['local_rank_p50']=g.local_rank_best.quantile(.5)
    f['local_rank_p90']=g.local_rank_best.quantile(.9)
    f['margin_p50']=g.log_score_margin.quantile(.5)
    f['margin_p90']=g.log_score_margin.quantile(.9)
    return f.reset_index()


def report(run_dir):
    folder=Path(run_dir).resolve()
    manifest=read_json(folder/'manifest.json');cfg=manifest['config']
    dataset=folder.parent.parent/'datasets'/manifest['dataset_id']
    meta=read_json(dataset/'manifest.json')
    if digest(dataset/'prefix_support.csv')!=meta['prefix_support_sha256']:
        raise ValueError('Support artifacts changed')
    baseline_dir=dataset/manifest['baseline_relative_dir']
    completed_shards(baseline_dir,'baselines')
    baseline=frames(source_paths(baseline_dir,'baselines','.csv.gz'))
    if baseline.sample_key.duplicated().any() or len(baseline)!=manifest['num_samples']:
        raise ValueError('Baseline coverage invalid')
    beam_frames={}
    for beam in cfg['beam_widths']:
        stage=f'beam_{beam}'
        count=completed_shards(folder,stage)
        if manifest['stages'].get(stage,{}).get('status')!='complete' or count!=manifest['stages'][stage]['shards']:
            raise ValueError(f'Incomplete replay stage: {stage}')
        f=frames(source_paths(folder,stage,'.csv.gz'))
        if f.sample_key.duplicated().any() or set(f.sample_key)!=set(baseline.sample_key):
            raise ValueError('Replay stages do not cover the same sample set')
        if f.cause.isna().any() or not f.cause.isin(CAUSES).all():raise ValueError('Missing pruning attribution')
        if not np.array_equal((f.cause=='first_layer_lost').to_numpy(),f.entered_second.to_numpy()==0):
            raise ValueError('First-layer attribution inconsistent')
        if not np.array_equal((f.cause=='retained').to_numpy(),f.retained_second.to_numpy()==1):
            raise ValueError('Second-layer attribution inconsistent')
        for cause in CAUSES:f['cause_'+cause]=(f.cause==cause).astype(float)
        if not f[['cause_'+c for c in CAUSES]].sum(1).eq(1).all():raise ValueError('Attribution is not exhaustive')
        for c in ('local_rank_best','global_rank_best','cutoff_probability','numeric_state'):
            if f.loc[f.entered_second==1,c].isna().any():raise ValueError(f'Entered sample has no {c}')
        beam_frames[beam]=f
    common=set.intersection(*(set(f.loc[f.entered_second==1,'sample_key']) for f in beam_frames.values()))
    write_json(folder/'tables/cohorts.json',{'samples':len(baseline),'common_entered_sample_keys':sorted(common),
        'entered_sample_counts':{str(b):int(f.entered_second.sum()) for b,f in beam_frames.items()}})
    baseline_rows=[]
    cohorts={'all':baseline,'common_entered':baseline[baseline.sample_key.isin(common)]}
    for beam,f in beam_frames.items():cohorts[f'entered_beam_{beam}']=baseline[baseline.sample_key.isin(f.loc[f.entered_second==1,'sample_key'])]
    ci_cols=['model_ce','raw_ce','windows_ce','raw_ce_gain','windows_ce_gain']
    points=[f'{method}_{metric}' for method in ('model','raw','windows') for metric in ('mrr','hit_at_1','hit_at_5','hit_at_10')]
    for name,f in cohorts.items():baseline_rows.extend(cohort_summary(f,name,ci_cols,cfg,point_columns=points))
    baseline_summary=pd.DataFrame(baseline_rows)
    write_csv(folder/'tables/baseline_comparison.csv',baseline_summary)
    attribution=[];detail=[];cross=[];continuous=[]
    main=['entered_second','retained_second']+['cause_'+c for c in CAUSES]
    for beam,f in beam_frames.items():
        for name,part in [('all',f),('entered',f[f.entered_second==1]),('common_entered',f[f.sample_key.isin(common)])]:
            attribution.extend(cohort_summary(part,name,main,cfg,{'beam':beam}))
        for bucket in ('edge_support_bucket','window_support_bucket','branch_bucket'):
            out=bucket_summary(f,['group',bucket]);out.insert(0,'beam',beam)
            write_csv(folder/f'tables/beam_{beam}_by_{bucket}.csv',out)
        joint=bucket_summary(f,['group','branch_bucket','edge_support_bucket'])
        write_csv(folder/f'tables/beam_{beam}_by_branch_and_support.csv',joint)
        entered=f[f.entered_second==1]
        score=entered.groupby(['group','cause']).agg(samples=('sample_key','size'),
            local_rank_p50=('local_rank_best','median'),global_rank_p50=('global_rank_best','median'),
            parent_probability_p50=('parent_probability','median'),conditional_probability_p50=('target_conditional_probability','median'),
            margin_p50=('log_score_margin','median'),same_parent_higher_mean=('same_parent_higher','mean'),
            other_parent_higher_mean=('other_parent_higher','mean'),illegal_other_higher_mean=('other_parent_illegal_higher','mean')).reset_index()
        score.insert(0,'beam',beam);continuous.append(score)
        for support in ('raw','window'):
            unseen=f[f.group=='Train-unseen']
            column=f'first_unsupported_{support}_level'
            grouped=unseen.groupby([column,'first_lost_level']).size()
            for (unsupported,lost),count in grouped.items():
                cross.append({'beam':beam,'support':support,'first_unsupported_level':int(unsupported),
                    'first_lost_level':int(lost),'count':int(count),'unseen_samples':len(unseen),
                    'proportion':count/len(unseen)})
        # Descriptive conditional rates for each value of prefix support/margins, not causal estimates.
        for column,bins,labels in [('local_rank_best',[0,1,5,10,20,50,np.inf],['1','2–5','6–10','11–20','21–50','>50']),
                                   ('log_score_margin',[-np.inf,-5,-2,-1,0,1,np.inf],['≤−5','−5–−2','−2–−1','−1–0','0–1','>1'])]:
            part=entered.copy();part['score_bucket']=pd.cut(part[column],bins,labels=labels,include_lowest=True).astype('object')
            grouped=part.groupby(['group','score_bucket'],observed=True,dropna=False)
            out=grouped.agg(samples=('sample_key','size'),retained_rate=('retained_second','mean')).reset_index()
            out.insert(0,'variable',column);out.insert(0,'beam',beam);detail.append(out)
        LOGGER.info('Offline aggregation beam %d complete',beam)
    attr=pd.DataFrame(attribution)
    write_csv(folder/'tables/attribution_summary.csv',attr)
    write_csv(folder/'tables/competition_scores.csv',pd.concat(continuous,ignore_index=True))
    write_csv(folder/'tables/rank_margin_retention.csv',pd.concat(detail,ignore_index=True))
    cross=pd.DataFrame(cross,columns=['beam','support','first_unsupported_level','first_lost_level','count','unseen_samples','proportion'])
    write_csv(folder/'tables/unseen_support_vs_loss.csv',cross)
    paired=[]
    base_beam=cfg['beam_widths'][0]
    a=beam_frames[base_beam].set_index('sample_key').loc[sorted(common)]
    for beam,f in beam_frames.items():
        if beam==base_beam:continue
        b=f.set_index('sample_key').loc[a.index]
        for c in ('user_id','target_item_id','group'):
            if not a[c].equals(b[c]):raise ValueError('Common cohort paired metadata mismatch')
        difference=a[['user_id','group']].copy()
        cols=['retained_second','cause_local_capacity','cause_cross_parent']
        for c in cols:difference[c]=b[c]-a[c]
        paired.extend(cohort_summary(difference,'common_entered',cols,cfg,{'beam':beam,'baseline_beam':base_beam}))
    paired=pd.DataFrame(paired)
    write_csv(folder/'tables/paired_common_cohort.csv',paired)
    summary={'schema_version':1,'run_id':folder.name,'num_samples':len(baseline),'common_entered_samples':len(common),
        'attribution':attr.replace({np.nan:None}).to_dict('records'),
        'baselines':baseline_summary.replace({np.nan:None}).to_dict('records'),
        'paired':paired.replace({np.nan:None}).to_dict('records')}
    write_json(folder/'tables/summary.json',summary)
    make_figures(folder,attr,baseline_summary,beam_frames,cross)
    observed=[]
    for group in GROUPS:
        b=baseline_summary[(baseline_summary.cohort=='all') & (baseline_summary.group==group)].iloc[0]
        if b.samples:
            observed.append(f"- {group}：TIGER 合法集合 CE={b.model_ce:.4f}；原始频次基线 CE={b.raw_ce:.4f}，收益={b.raw_ce_gain:+.4f}，95% 区间 [{b.raw_ce_gain_ci_low:+.4f}, {b.raw_ce_gain_ci_high:+.4f}]。")
    for _,r in attr[(attr.group=='Tail') & (attr.cohort=='entered')].iterrows():
        if r.samples:
            observed.append(f"- Tail / beam {int(r.beam)}：第一层存活样本 {int(r.samples)} 条；第二层保留 {r.retained_second:.2%}，局部容量淘汰 {r.cause_local_capacity:.2%}，跨父竞争淘汰 {r.cause_cross_parent:.2%}，截止并列淘汰 {r.cause_cutoff_tie:.2%}。")
    selected=attr[attr.cohort=='entered'][['beam','group','samples','retained_second','cause_local_capacity','cause_cross_parent','cause_cutoff_tie']]
    text=f'''# {cfg.get('dataset_name', 'Beauty')} 第二层路径淘汰归因与训练支持报告

## 实验、来源和覆盖

运行 `{folder.name}`，{'完整实验' if manifest['is_full_evaluation'] else '小规模验证'}，共 {len(baseline)} 条样本；来源实验 `{manifest['source_run_id']}`。checkpoint epoch={manifest['checkpoint']['epoch']}，global_step={manifest['checkpoint']['global_step']}。使用 grid 环境、float32、无前缀约束、batch size={cfg['batch_size']}、beam={cfg['beam_widths']}。

来源输入、冻结分组、缓存与完整候选均经指纹校验；重放的每层目标生存状态与第一次实验逐样本一致。容差内候选并列差异：{sum(s['tie_difference_rows'] for s in manifest['stages'].values())} 行。当前结果只描述该 checkpoint 的搜索和预测，不将某类淘汰直接归因于 tokenizer。

## 第二层淘汰机制

下表分母为各 beam 中第一层仍存活的该组样本。所有样本口径及共同进入第二层口径另见 attribution_summary.csv；三个 beam 的共同进入样本有 {len(common)} 条。

{markdown_table(selected)}

局部容量不足：至少 B 个同父候选严格高于目标；跨父竞争：全局至少 B 个候选严格更高、同父不足 B 个；截止点并列：严格更高候选不足 B 个但目标实际被剪掉。各类互斥穷尽。局部排名良好但跨父淘汰不能单独证明第一层有错，也可能涉及多个父分支的预测和分支质量。

![淘汰构成](figures/attribution.png)

## 本次观察与频次基线

{chr(10).join(observed)}

CE 收益 = 基线 CE − TIGER 合法集合 CE，正值表示 TIGER 优于不看用户历史的该频次基线；收益不全部归因于某一历史建模能力。原始交互频次和潜在目标窗口两种基线均加一平滑，零支持父节点为合法孩子上的均匀分布，不用测试数据调参。

![频次基线](figures/baseline_ce.png)

![局部与全局排名](figures/local_global_rank.png)

## 训练支持、潜在监督和未见路径

支持数据仅来自原始训练记录：{meta['statistics']['training_records']} 条记录，{meta['statistics']['training_interactions']} 次交互，{meta['statistics']['potential_target_windows']} 个潜在目标窗口。序列第 t 个物品贡献 t−1 个潜在目标窗口，t 从 1 开始；各序列总窗口为 n(n−1)/2。该量不是历史实际采样曝光，未模拟一个任意训练顺序冒充原训练过程。

各层前缀包含物品数、交互量、独立用户数、潜在窗口量、分支占比、孩子交互 HHI 和熵。支持桶与合法孩子桶联合表控制一部分结构差异，但仍是描述性分析。

![支持量与存活](figures/support_survival.png)

![零支持位置](figures/unseen_support_loss.png)

Train-Unseen 的 first_unsupported_raw_level 与 first_unsupported_window_level 分别表示第一个无原始交互支持、无潜在目标窗口支持的前缀层；0 表示所有层均有支持。first_lost_level=0 表示最终仍在 beam。交叉表比例分母为该 beam 的所有 Train-Unseen 测试样本。

## 统计与后续实验

主要比例、CE 收益和共同进入队列上的配对差值采用用户聚类 bootstrap，{cfg['bootstrap_replicates']} 次，seed={cfg['seed']}，95% percentile 区间。MRR/Top-K、细分桶和交叉表为描述性点估计；空组指标为空。并列合法排名取平均名次；实际剪枝按原 torch.topk 行为判断。

先依据第二层竞争机制和频次基线收益选择后续方向：训练支持不足时研究训练采样与共享；合法分支 CE 仍高时研究分支划分与可预测性；局部可选但跨父竞争失败时继续检查累计得分和竞争分支。当前实验未重训 tokenizer、未训练预测探针，也未加入语义或协同关系对照，不能确认这些解释的因果作用。

## 离线复用

逐样本诊断见 diagnostics/，剪枝前父 beam 和剪枝后子 beam 见 traces/；未保存所有第二层条件概率矩阵。本次所有表和图均可用 `python -m src.analysis.second_layer report --run-dir <此目录>` 离线重建，不读取 checkpoint、原始 TFRecords 或第一次实验目录。迁移时复制当前输出根目录中的 datasets 和 runs。

numeric_state 标注 parent_zero、conditional_zero、product_underflow 或 cutoff_zero；不存在合法有限对数差时保存空值，未用 epsilon 改写搜索分数。rank_best/rank_worst 保留精确并列区间。
'''
    atomic_write(folder/'report.md',lambda p:p.write_text(text))
    manifest['status']='complete';manifest['report']='report.md'
    write_json(folder/'manifest.json',manifest)
    LOGGER.info('Report ready: %s',folder/'report.md')
    return folder/'report.md'


def make_figures(folder,attr,baseline,beam_frames,cross):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):atomic_write(folder/f'figures/{name}.{ext}',lambda p:fig.savefig(p,dpi=160))
        plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,group in zip(axes,GROUPS):
        part=attr[(attr.group==group)&(attr.cohort=='entered')].set_index('beam')
        part[['cause_'+c for c in CAUSES[1:]]].plot.bar(stacked=True,ax=ax,legend=False)
        ax.set(title=group,ylabel='Fraction of entered targets',ylim=(0,1))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,[s.replace('cause_','') for s in labels],loc='upper center',ncol=4)
    save(fig,'attribution')
    fig,ax=plt.subplots(figsize=(8,4))
    baseline[(baseline.cohort=='all')&(baseline.group!='Overall')].set_index('group')[['model_ce','raw_ce','windows_ce']].plot.bar(ax=ax)
    ax.set(ylabel='Legal-set cross entropy',xlabel='Target group')
    save(fig,'baseline_ce')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(beam,f) in zip(axes,beam_frames.items()):
        entered=f[f.entered_second==1]
        if len(entered):
            ax.hexbin(entered.local_rank_best,entered.global_rank_best,gridsize=35,bins='log',mincnt=1,xscale='log',yscale='log')
        else:
            ax.text(.5,.5,'No entered targets',ha='center',transform=ax.transAxes)
        ax.axhline(beam,color='red',linestyle='--');ax.axvline(beam,color='red',linestyle='--')
        ax.set(title=f'Beam {beam}',xlabel='Local best rank',ylabel='Global best rank')
    save(fig,'local_global_rank')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    order=['0','1–5','6–20','21–100','101–1000','>1000']
    for ax,(beam,f) in zip(axes,beam_frames.items()):
        for group,part in f[f.entered_second==1].groupby('group'):
            g=part.groupby('edge_support_bucket').retained_second.mean().reindex(order)
            ax.plot(range(len(order)),g,marker='o',label=group)
        ax.set_xticks(range(len(order)),order,rotation=45)
        ax.set(title=f'Beam {beam}',ylabel='Conditional survival',ylim=(0,1));ax.legend(fontsize=7)
    save(fig,'support_survival')
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for i,mode in enumerate(('raw','window')):
        for j,beam in enumerate(beam_frames):
            a=np.zeros((5,5))
            for r in cross[(cross.support==mode)&(cross.beam==beam)].itertuples():a[r.first_unsupported_level,r.first_lost_level]=r.count
            axes[i,j].imshow(a,origin='lower',aspect='auto');axes[i,j].set(title=f'{mode}, beam {beam}',xlabel='First lost level',ylabel='First unsupported level')
    save(fig,'unseen_support_loss')
