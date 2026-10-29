"""Offline, teacher-forced second-level baselines using frozen source probabilities."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.long_tail.io import commit_shard,completed_shards,read_json,write_csv
from .common import source_paths


def distribution_metrics(probabilities,target_index,log_probabilities=None):
    p=np.asarray(probabilities)
    ranking=np.asarray(log_probabilities) if log_probabilities is not None else p
    value=ranking[target_index]
    rank=1+int((ranking>value).sum())+.5*(int((ranking==value).sum())-1)
    ce=-float(log_probabilities[target_index]) if log_probabilities is not None else -float(np.log(value))
    return {'ce':ce,'mrr':1/rank,**{f'hit_at_{k}':float(rank<=k) for k in (1,5,10)}}


def teacher_records(batch,support):
    records=[]
    for n,meta in enumerate(batch['metadata']):
        target=tuple(batch['target_sids'][n].tolist())
        parent=target[:1]
        legal=sorted(support.children[parent])
        position=legal.index(target[1])
        logp=batch['log_probabilities'][n,1,legal].float()
        normalized=logp-torch.logsumexp(logp,dim=0)
        metrics=distribution_metrics(normalized.exp().numpy(),position,normalized.numpy())
        row={**meta,**support.features(target),
             'tf_first_probability':float(batch['probabilities'][n,0,target[0]]),
             'tf_second_probability':float(batch['probabilities'][n,1,target[1]]),
             **{'model_'+k:v for k,v in metrics.items()}}
        for label,mode in [('raw','train_interactions'),('windows','potential_target_windows')]:
            children,q=support.baseline(parent,mode)
            assert children==legal
            base=distribution_metrics(q,position)
            row.update({f'{label}_{k}':v for k,v in base.items()})
            row[f'{label}_ce_gain']=base['ce']-metrics['ce']
        records.append(row)
    return pd.DataFrame(records)


def prepare_baselines(cfg,folder,support):
    folder=Path(folder)
    done=completed_shards(folder,'baselines')
    all_keys=[]
    for shard,path in enumerate(source_paths(cfg['source_run_dir'],'teacher_forcing','.pt')):
        batch=torch.load(path,map_location='cpu',weights_only=True)
        keys=batch['sample_keys']
        if cfg.get('max_samples') is not None:
            remaining=cfg['max_samples']-len(all_keys)
            if remaining<=0:break
            keep=min(remaining,len(keys))
            batch={k:(v[:keep] if isinstance(v,(list,torch.Tensor)) else v) for k,v in batch.items()}
            keys=batch['sample_keys']
        if shard<done:
            if read_json(folder/'progress/baselines'/f'{shard:06d}.json')['sample_keys']!=keys:
                raise ValueError('Baseline resume alignment differs')
        else:
            out=folder/'baselines'/f'{shard:06d}.csv.gz'
            write_csv(out,teacher_records(batch,support))
            commit_shard(folder,'baselines',shard,keys,[out])
        all_keys.extend(keys)
    if len(set(all_keys))!=len(all_keys) or not all_keys:
        raise ValueError('Empty/duplicated baseline sample keys')
    return all_keys
