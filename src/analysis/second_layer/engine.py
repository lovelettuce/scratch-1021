"""Validated source reuse, restartable second-level search replay."""
import logging
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.analysis.long_tail.data import model_and_data
from src.analysis.long_tail.engine import batch_groups,cached_batches,merge_payloads
from src.analysis.long_tail.io import (commit_shard,completed_shards,environment,read_json,
    validate_identity,write_csv,write_json,write_pt)
from .common import code_hashes,source_identity,source_paths,source_dataset
from .support import prepare_support
from .baselines import prepare_baselines
from .diagnostics import SearchCollector

LOGGER=logging.getLogger(__name__)


def prepare(cfg):
    LOGGER.info('Validating first experiment and current inputs')
    source,identity=source_identity(cfg)
    folder,support=prepare_support(cfg,identity)
    frozen=pd.read_csv(source_dataset(cfg['source_run_dir'],source)/'items.csv').set_index('item_id')
    for item,sid in enumerate(support.rows):
        leaf=support.get(sid)
        if leaf['train_interactions']!=frozen.loc[item,'train_frequency'] or leaf['train_users']!=frozen.loc[item,'train_users']:
            raise ValueError('Training support does not reproduce frozen item statistics')
    from src.analysis.long_tail.io import canonical_hash
    baseline_id=canonical_hash({'source':identity['source_manifest_sha256'],'max_samples':cfg['max_samples']})
    baseline_dir=folder/'teacher'/baseline_id
    keys=prepare_baselines(cfg,baseline_dir,support)
    expected=source['stages']['teacher_forcing']['samples']
    expected=min(expected,cfg['max_samples']) if cfg['max_samples'] is not None else expected
    if len(keys)!=expected:
        raise ValueError('Selected teacher-forcing sample count differs from source')
    LOGGER.info('Prepared support and baselines for %d samples',len(keys))
    return source,identity,folder,baseline_dir,support,keys


def input_batches(cfg):
    selected=0
    for batch in cached_batches(Path(cfg['source_run_dir'])):
        n=len(batch[3])
        limit=cfg['max_samples']
        if limit is not None:
            n=min(n,limit-selected)
        if n<=0:break
        if n<len(batch[3]):
            from types import SimpleNamespace
            inp,labels,users,ids,locations=batch
            # Labels are flattened by the original collator.
            levels=labels.labels['sequence_data'].numel()//len(ids)
            batch=(SimpleNamespace(mask=inp.mask[:n],transformed_sequences={k:v[:n] for k,v in inp.transformed_sequences.items()}),
                   SimpleNamespace(labels={k:v.reshape(len(ids),levels)[:n].reshape(-1) for k,v in labels.labels.items()}),
                   users[:n],ids[:n],locations[:n])
        selected+=n
        yield batch
        if limit is not None and selected>=limit:break


def compatibility(reference,generated,scores,survival,keys):
    n=len(keys)
    if reference['sample_keys'][:n]!=keys:
        raise ValueError('Replay source candidate key mismatch')
    old=reference['generated_sids'][:n]
    old_scores=reference['scores'][:n]
    if old.shape!=generated.shape or not torch.allclose(old_scores,scores,atol=1e-7,rtol=1e-6):
        raise ValueError('Replay score or shape differs from source')
    mismatch=(old!=generated).any(-1)
    tied_rows=0
    for row in torch.where(mismatch.any(-1))[0].tolist():
        # Accept only numerical ties in the candidate set/ordering, never different target survival.
        a={tuple(s):float(v) for s,v in zip(old[row].tolist(),old_scores[row])}
        b={tuple(s):float(v) for s,v in zip(generated[row].tolist(),scores[row])}
        for sid in a.keys() & b.keys():
            if not np.isclose(a[sid],b[sid],atol=1e-7,rtol=1e-6):
                raise ValueError('Replay shared candidate score differs')
        missing=list(a.keys()-b.keys());added=list(b.keys()-a.keys())
        cutoff=min(float(old_scores[row].min()),float(scores[row].min()))
        if len(missing)!=len(added) or any(not np.isclose(v,cutoff,atol=1e-7,rtol=1e-6)
                                          for v in [a[s] for s in missing]+[b[s] for s in added]):
            raise ValueError('Replay candidates differ beyond cutoff ties')
        tied_rows+=1
    return tied_rows


def evaluate(cfg,resume=None):
    import fcntl
    source,identity,dataset,baseline_dir,support,expected_keys=prepare(cfg)
    hashes=code_hashes()
    if resume:
        folder=Path(resume).resolve()
        manifest=read_json(folder/'manifest.json')
        validate_identity(manifest,cfg,identity)
        if manifest['code_hashes']!=hashes:
            raise ValueError('Resume refused: analysis or generation source code changed')
    else:
        name=cfg.get('run_id') or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        if Path(name).name!=name:raise ValueError('Invalid run_id')
        folder=Path(cfg['output_root'])/'runs'/name
        folder.mkdir(parents=True,exist_ok=False)
        OmegaConf.save(OmegaConf.create(cfg),folder/'config.yaml')
        manifest={'schema_version':1,'config':cfg,'inputs':identity,'code_hashes':hashes,
            'environment':environment(),'dataset_id':dataset.name,
            'baseline_relative_dir':str(baseline_dir.relative_to(dataset)),
            'source_run_id':Path(cfg['source_run_dir']).name,'status':'created','stages':{},
            'num_samples':len(expected_keys),'is_full_evaluation':cfg['max_samples'] is None,
            'precision':'float32','batches_per_output_shard':16,
            'score_tolerances':{'atol':1e-7,'rtol':1e-6}}
        write_json(folder/'manifest.json',manifest)
    lock=(folder/'.evaluation.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Run is locked by another evaluator')
    handler=None
    try:
        (folder/'logs').mkdir(exist_ok=True)
        handler=logging.FileHandler(folder/'logs/evaluate.log')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logging.getLogger().addHandler(handler)
        torch.set_num_threads(1)
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        model,data_cfg,sid,checkpoint=model_and_data(cfg)
        manifest.update(status='evaluating',checkpoint=checkpoint,
            device_name=torch.cuda.get_device_name(cfg['device']) if str(cfg['device']).startswith('cuda') else 'cpu')
        write_json(folder/'manifest.json',manifest)
        base=pd.concat([pd.read_csv(p,keep_default_na=False) for p in source_paths(baseline_dir,'baselines','.csv.gz')]).set_index('sample_key')
        for width in cfg['beam_widths']:
            stage=f'beam_{width}'
            done=completed_shards(folder,stage)
            reference_paths=source_paths(cfg['source_run_dir'],stage,'.pt')
            reference_csv=source_paths(cfg['source_run_dir'],stage,'.csv.gz')
            model.top_k_for_generation=width
            all_keys=[]
            total_ties=0
            chunks=0
            LOGGER.info('Replay %s, committed shards=%d',stage,done)
            for shard,group in enumerate(batch_groups(input_batches(cfg))):
                chunks+=1
                keys=[f'{file}:{index}' for batch in group for file,index in batch[4]]
                if shard<done:
                    marker=read_json(folder/'progress'/stage/f'{shard:06d}.json')
                    if marker['sample_keys']!=keys:raise ValueError('Resume sample alignment differs')
                    saved=pd.read_csv(folder/'diagnostics'/stage/f'{shard:06d}.csv.gz')
                    total_ties+=int(saved.replay_tie_difference.sum())
                    all_keys.extend(keys)
                    continue
                generated_batches=[];score_batches=[];trace_batches=[];records=[];survival_batches=[]
                for batch in group:
                    inp,labels,users,item_ids,locations=batch
                    targets=labels.labels['sequence_data'].reshape(len(item_ids),-1).to(cfg['device'])
                    if not torch.equal(targets.cpu(),sid[:,item_ids].T):raise ValueError('Target/SID mismatch')
                    collector=SearchCollector(targets,support)
                    kwargs={model.feature_to_model_input_map.get(k,k):v.to(cfg['device']) for k,v in inp.transformed_sequences.items()}
                    with torch.inference_mode():
                        generated,scores=model.generate(attention_mask=inp.mask.to(cfg['device']),
                            step_observer=collector.step,candidate_observer=collector.candidates,**kwargs)
                    rows,trace=collector.finish()
                    batch_keys=[f'{file}:{index}' for file,index in locations]
                    for i,(key,row) in enumerate(zip(batch_keys,rows)):
                        meta=base.loc[key].to_dict()
                        if int(meta['user_id'])!=int(users[i]) or int(meta['target_item_id'])!=int(item_ids[i]):
                            raise ValueError('Frozen source metadata mismatch')
                        records.append({'sample_key':key,**meta,'beam':width,**row})
                    trace_batches.append(trace)
                    generated_batches.append(generated.cpu());score_batches.append(scores.cpu())
                    survival_batches.append(np.stack(collector.survival,axis=1))
                generated=torch.cat(generated_batches);scores=torch.cat(score_batches)
                alive=np.concatenate(survival_batches)
                ref=torch.load(reference_paths[shard],map_location='cpu',weights_only=True)
                ties=compatibility(ref,generated,scores,alive,keys)
                original=pd.read_csv(reference_csv[shard]).iloc[:len(keys)]
                if original.sample_key.tolist()!=keys:raise ValueError('Source survival key mismatch')
                expected_alive=original[[f'survival_level_{l}' for l in range(1,sid.shape[0]+1)]].to_numpy(bool)
                if not np.array_equal(alive,expected_alive):raise ValueError('Replay target survival changed')
                # Attach baseline final outcome as well, so the report is independent of source files.
                for i,row in enumerate(records):
                    row['source_recall_at_10']=float(original.iloc[i].recall_at_10)
                    row['replay_tie_difference']=0
                if ties:
                    # Count differing rows precisely, including harmless candidate-order ties.
                    mismatch=(ref['generated_sids'][:len(keys)]!=generated).any(-1).any(-1).tolist()
                    for row,diff in zip(records,mismatch):row['replay_tie_difference']=int(diff)
                csv=folder/'diagnostics'/stage/f'{shard:06d}.csv.gz'
                pt=folder/'traces'/stage/f'{shard:06d}.pt'
                write_csv(csv,pd.DataFrame(records))
                write_pt(pt,{'sample_keys':keys,**merge_payloads(trace_batches)})
                commit_shard(folder,stage,shard,keys,[csv,pt])
                all_keys.extend(keys);total_ties+=ties
                if shard%10==0:LOGGER.info('%s shard=%d samples=%d',stage,shard,len(all_keys))
            if all_keys!=expected_keys or chunks<done:raise ValueError('Replay coverage differs from source selection')
            manifest['stages'][stage]={'status':'complete','shards':chunks,'samples':len(all_keys),'tie_difference_rows':total_ties}
            write_json(folder/'manifest.json',manifest)
            LOGGER.info('Completed %s: %d samples, harmless tie differences=%d',stage,len(all_keys),total_ties)
        manifest['status']='inference_complete'
        write_json(folder/'manifest.json',manifest)
        return folder
    except Exception:
        manifest['status']='interrupted';write_json(folder/'manifest.json',manifest)
        raise
    finally:
        if handler:
            logging.getLogger().removeHandler(handler);handler.close()
        lock.close()
