from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils import custom_hydra_resolvers  # noqa: F401; initialize project utilities before model imports

from src.analysis.second_layer.support import build_support,SupportIndex,support_bucket
from src.analysis.second_layer.baselines import distribution_metrics,teacher_records
from src.analysis.second_layer.diagnostics import SearchCollector,classify,rank_bounds
from src.analysis.second_layer.engine import compatibility


def support_fixture():
    rows=[(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(2,0)]
    frame,stats=build_support(rows,[(7,[0,0,1]),(8,[1,4])])
    return SupportIndex(torch.tensor(rows).T,frame),frame,stats


def test_exact_support_and_window_opportunities():
    support,frame,stats=support_fixture()
    assert stats['training_interactions']==5
    assert stats['potential_target_windows']==4  # 3*2/2 + 2*1/2
    assert support.get(())['train_users']==2
    assert support.get((0,))['train_users']==2  # distinct users, not sum of child users
    assert support.get((0,0))['train_interactions']==2
    assert support.get((0,0))['potential_target_windows']==1
    assert support.get((0,1))['potential_target_windows']==2
    assert support.get((2,0))['train_interactions']==0
    assert support.features((2,0))['first_unsupported_raw_level']==1
    assert support.features((0,2))['first_unsupported_raw_level']==2
    assert [support_bucket(n) for n in (0,1,6,21,101,1001)]==['0','1–5','6–20','21–100','101–1000','>1000']


def test_smoothed_baselines_and_log_probability_stability():
    support,_,_=support_fixture()
    children,q=support.baseline((0,))
    assert children==[0,1,2,3]
    np.testing.assert_allclose(q,[3/8,3/8,1/8,1/8])
    _,zero=support.baseline((2,))
    np.testing.assert_allclose(zero,[1])
    tie=distribution_metrics([.5,.5],0)
    assert tie['mrr']==pytest.approx(1/1.5)
    assert tie['ce']==pytest.approx(np.log(2))
    stable=distribution_metrics([0.,0.,.5,.5],0,[-1000.,-1001.,-0.693,-0.693])
    assert stable['mrr']==pytest.approx(1/3)
    # CE must use log probabilities, even if exp(logp) underflows.
    logits=torch.tensor([[[0.,0.,0.,0.],[-1000.,0.,0.,0.]]])
    logs=logits.log_softmax(-1)
    batch={'metadata':[{'sample_key':'x','user_id':1,'target_item_id':0,'group':'Tail'}],
           'target_sids':torch.tensor([[0,0]]),'log_probabilities':logs,'probabilities':logs.exp()}
    out=teacher_records(batch,support).iloc[0]
    assert np.isfinite(out.model_ce) and out.model_ce>1000
    assert out.raw_ce_gain==pytest.approx(out.raw_ce-out.model_ce)


def collect(q,p,target,selected,parents=(0,1)):
    support,_,_=support_fixture()
    c=SearchCollector(torch.tensor([target]),support)
    prefixes=torch.tensor(parents).reshape(1,2,1)
    scores=torch.tensor([p],dtype=torch.float32)
    c.step(0,prefixes,scores)
    conditional=torch.tensor([q],dtype=torch.float32)
    cumulative=conditional*scores[:,:,None]
    c.candidates({'hierarchy':1,'parent_prefixes':prefixes,'parent_scores':scores,
        'next_token_ids':torch.arange(4).repeat(1,2),'conditional_probabilities':conditional.reshape(1,-1),
        'cumulative_scores':cumulative.reshape(1,-1)})
    child=torch.tensor([selected])
    c.step(1,child,torch.ones(1,2))
    rows,trace=c.finish()
    return rows[0],trace


def test_local_capacity_and_cross_parent_displacement():
    row,_=collect([[.5,.3,.1,.1],[.4,.3,.2,.1]],[.8,.2],[0,2],[[0,0],[0,1]])
    assert row['cause']=='local_capacity' and row['same_parent_higher']==2
    row,_=collect([[.1,.6,.2,.1],[.45,.4,.1,.05]],[.1,.9],[0,1],[[1,0],[1,1]])
    assert row['cause']=='cross_parent' and row['local_rank_best']==1
    assert row['same_parent_higher']==0 and row['other_parent_higher']>=2
    assert row['global_rank_best']>2 and row['log_score_margin']<0
    assert row['same_parent_legal_higher']+row['same_parent_illegal_higher']==row['same_parent_higher']
    assert row['other_parent_legal_higher']+row['other_parent_illegal_higher']==row['other_parent_higher']


def test_retention_cutoff_ties_and_missing_parent():
    row,_=collect([[.6,.2,.1,.1],[.4,.3,.2,.1]],[.8,.2],[0,0],[[0,0],[0,1]])
    assert row['cause']=='retained'
    row,_=collect([[.5,.5,0,0],[.5,.5,0,0]],[.5,.5],[0,1],[[0,0],[1,0]])
    assert row['cause']=='cutoff_tie' and row['global_rank_best']==1 and row['global_rank_worst']==4
    row,_=collect([[.5,.5,0,0],[1.,0,0,0]],[.5,.5],[0,1],[[1,0],[2,0]],parents=(1,2))
    assert row['cause']=='first_layer_lost' and row['target_cumulative_probability'] is None
    assert row['first_lost_level']==1
    with pytest.raises(ValueError):classify(False,True)


def test_native_underflow_and_illegal_competitors():
    row,_=collect([[1e-20,0,0,0],[0,0,0,0]],[1e-30,1e-30],[0,0],[[0,1],[1,0]])
    assert row['target_cumulative_probability']==0
    assert 'product_underflow' in row['numeric_state'] and 'cutoff_zero' in row['numeric_state']
    assert row['log_score_margin'] is None and row['target_log_score'] is not None
    row,_=collect([[.1,.6,.2,.1],[.05,.05,.5,.4]],[.1,.9],[0,1],[[1,2],[1,3]])
    assert row['cause']=='cross_parent' and row['other_parent_illegal_higher']==2


def test_candidate_observer_does_not_change_generation():
    from src.models.modules.semantic_id.tiger_generation_model import SemanticIDEncoderDecoder,SemanticIDGenerativeRecommender
    model=SimpleNamespace(num_hierarchies=2,num_embeddings_per_hierarchy=4,top_k_for_generation=2,should_check_prefix=False)
    model.encoder_forward_pass=lambda **kw:(torch.zeros(1,1,2),torch.ones(1,1))
    model.decoder_forward_pass=lambda **kw:(torch.ones(kw['encoder_output'].size(0),1,2),kw['past_key_values'])
    heads=torch.nn.ModuleList([torch.nn.Linear(2,4,bias=False) for _ in range(2)])
    for head in heads:head.weight.data.copy_(torch.arange(8).reshape(4,2))
    model.decoder=SimpleNamespace(decoder_mlp=heads)
    model._beam_search_one_step=SemanticIDGenerativeRecommender._beam_search_one_step.__get__(model)
    generate=SemanticIDEncoderDecoder.generate.__get__(model)
    kwargs={'attention_mask':torch.ones(1,2),'input_ids':torch.ones(1,2,dtype=torch.long)}
    a=generate(**kwargs)
    events=[]
    def observe(event):
        events.append(event['hierarchy'])
        for value in event.values():
            if isinstance(value,torch.Tensor):value.zero_()
    b=generate(**kwargs,candidate_observer=observe)
    assert events==[1]
    torch.testing.assert_close(a[0],b[0]);torch.testing.assert_close(a[1],b[1])


def test_replay_compatibility_accepts_only_cutoff_ties():
    ref={'sample_keys':['a'],'generated_sids':torch.tensor([[[0],[1]]]),'scores':torch.tensor([[.5,.5]])}
    assert compatibility(ref,torch.tensor([[[1],[2]]]),torch.tensor([[.5,.5]]),None,['a'])==1
    with pytest.raises(ValueError):
        compatibility(ref,torch.tensor([[[1],[2]]]),torch.tensor([[.6,.5]]),None,['a'])
