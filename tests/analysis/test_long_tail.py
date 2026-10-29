from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.long_tail.data import assign_groups, validate_ids, frequency_bucket
from src.analysis.long_tail.diagnostics import SIDIndex
from src.analysis.long_tail.engine import first_lost, recommendation_values
from src.analysis.long_tail.io import (commit_shard, completed_shards, read_json,
    validate_identity, write_json, write_pt)
from src.analysis.long_tail.metrics import grouped_metrics, paired_comparison, cluster_interval


def test_popularity_ties_zero_and_tiny_catalog():
    groups, thresholds = assign_groups([0, 100, 100, 20, 10, 5])
    assert groups == ['Train-unseen', 'Head', 'Head', 'Mid', 'Tail', 'Tail']
    assert thresholds == {'head_min_frequency': 100, 'mid_min_frequency': 20}
    assert assign_groups([1])[0] == ['Head']
    assert assign_groups([0])[0] == ['Train-unseen']
    assert assign_groups([3,3,3])[0] == ['Head']*3
    validate_ids([0,1], 2, 'test')
    with pytest.raises(ValueError):
        validate_ids([-1], 2, 'test')
    assert [frequency_bucket(n) for n in (0,1,6,11,21,51)] == ['0','1–5','6–10','11–20','21–50','>50']


def test_nll_decomposition_and_underflow():
    sid = torch.tensor([[0,0,1],[0,1,0]])
    index = SIDIndex(sid, [100,2,1])
    logits = torch.tensor([[[0.,0.,-1000.],[-1000.,0.,1.]], [[1.,0.,-20.],[0.,-10.,-20.]]])
    target = torch.tensor([[0,0],[1,0]])
    f = index.diagnose(logits.log_softmax(-1), target, [{'sample_key':'a'},{'sample_key':'b'}])
    np.testing.assert_allclose(f.raw_nll, f.legality_nll + f.conditional_nll, atol=1e-6)
    assert np.isfinite(f.raw_nll).all()
    assert f.iloc[-1].conditional_nll == pytest.approx(0,abs=1e-6)
    assert f.iloc[0].legal_rank == 1.5
    assert f.iloc[0].collision_size == 2
    assert index.mass[()] == 103
    full = SIDIndex(torch.tensor([[0,1],[0,0]]), [1,1])
    f = full.diagnose(torch.zeros(1,2,2).log_softmax(-1),torch.tensor([[0,0]]),[{}])
    assert f.iloc[0].illegal_mass == pytest.approx(0,abs=1e-6)
    assert first_lost([True,False,False]) == 2
    assert first_lost([True,True]) == 0


def test_metrics_invalid_slots_and_macro():
    # Candidate 0 is illegal but consumes the first rank; target is rank 2.
    scores = torch.tensor([[.9,.8,.7],[.9,.8,.7]])
    generated = torch.tensor([[[9],[1],[2]], [[9],[2],[3]]])
    values = recommendation_values(scores,generated,torch.tensor([[1],[1]]),[1,2])
    np.testing.assert_allclose(values['recall_at_1'],[0,0])
    np.testing.assert_allclose(values['ndcg_at_2'],[1/np.log2(3),0])
    f = pd.DataFrame({'sample_key':['a','b','c'],'user_id':[1,1,2], 'target_item_id':[0,0,1],
        'group':['Tail']*3, 'recall_at_10':[1.,1.,0.], 'ndcg_at_10':[1.,1.,0.]})
    items = pd.DataFrame({'item_id':[0,1,2], 'group':['Tail','Tail','Head'], 'train_frequency':[1,1,10]})
    result = grouped_metrics(f,items,[10],repeats=20).set_index('group')
    assert result.loc['Tail','recall_at_10'] == pytest.approx(2/3)
    assert result.loc['Tail','macro_recall_at_10'] == .5
    assert np.isnan(result.loc['Head','recall_at_10'])
    assert result.loc['Overall','recall_at_10'] == result.loc['Tail','recall_at_10']
    paired = paired_comparison(f,f.iloc[::-1].copy(),repeats=20)
    assert (paired[paired.samples > 0].difference == 0).all()
    assert cluster_interval(f,['recall_at_10'],20,42) == cluster_interval(f,['recall_at_10'],20,42)


def test_resume_validates_artifacts_and_identity(tmp_path):
    path=tmp_path/'x.pt'
    write_pt(path, {'a':torch.tensor([1])})
    commit_shard(tmp_path,'tf',0,['file:0'],[path])
    assert completed_shards(tmp_path,'tf') == 1
    validate_identity({'config':{'x':1},'inputs':{'y':2}},{'x':1},{'y':2})
    with pytest.raises(ValueError):
        validate_identity({'config':{},'inputs':{}},{'x':2},{})
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Corrupt'):
        completed_shards(tmp_path,'tf')


def test_repository_evaluator_agrees():
    from src.components.eval_metrics import SIDRetrievalEvaluator, Recall, NDCG
    scores=torch.arange(20,0,-1,dtype=torch.float32).reshape(2,10)
    generated=torch.arange(20).reshape(2,10,1)
    targets=torch.tensor([[2],[19]])
    values=recommendation_values(scores,generated,targets,[5,10])
    evaluator=SIDRetrievalEvaluator({'recall':Recall,'ndcg':NDCG},[5,10])
    evaluator(scores,generated,targets.flatten())
    for key,metric in evaluator.metrics.items():
        assert metric.compute().item() == pytest.approx(values[key.replace('@','_at_')].mean(),abs=1e-6)


def test_generate_observer_keeps_results():
    from src.models.modules.semantic_id.tiger_generation_model import SemanticIDEncoderDecoder, SemanticIDGenerativeRecommender
    model=SimpleNamespace(num_hierarchies=2,num_embeddings_per_hierarchy=4,
                          top_k_for_generation=2,should_check_prefix=False)
    model.encoder_forward_pass=lambda **kw: (torch.zeros(1,1,2),torch.ones(1,1))
    model.decoder_forward_pass=lambda **kw: (torch.ones(kw['encoder_output'].size(0),1,2),kw['past_key_values'])
    heads=torch.nn.ModuleList([torch.nn.Linear(2,4,bias=False) for _ in range(2)])
    for head in heads:
        head.weight.data.copy_(torch.arange(8).reshape(4,2))
    model.decoder=SimpleNamespace(decoder_mlp=heads)
    model._beam_search_one_step=SemanticIDGenerativeRecommender._beam_search_one_step.__get__(model)
    generate=SemanticIDEncoderDecoder.generate.__get__(model)
    kwargs={'attention_mask':torch.ones(1,2),'input_ids':torch.ones(1,2,dtype=torch.long)}
    a=generate(**kwargs)
    seen=[]
    def observer(level,prefix,scores):
        seen.append(prefix.clone())
        prefix.zero_()  # Mutation must not touch live beams.
        scores.zero_()
    b=generate(**kwargs,step_observer=observer)
    assert len(seen)==2
    torch.testing.assert_close(a[0],b[0])
    torch.testing.assert_close(a[1],b[1])


def test_prepare_counts_original_training_only(tmp_path, monkeypatch):
    from src.analysis.long_tail import data
    sid_path=tmp_path/'sids.pt'
    torch.save(torch.tensor([[0,0,1],[0,1,0]]),sid_path)
    cfg={'semantic_id_path':str(sid_path),'output_root':str(tmp_path/'out'),
         'data_dir':str(tmp_path),'schema_version':1}
    calls=[]
    def original_records(path):
        calls.append(str(path))
        yield 0, 7, [0,0,1]
        yield 1, 8, [1]
    monkeypatch.setattr(data,'raw_records',original_records)
    folder=data.prepare(cfg,{'training':[{'file':'train.tfrecord.gz'}]})
    items=pd.read_csv(folder/'items.csv')
    assert items.train_frequency.tolist()==[2,2,0]
    assert items.train_users.tolist()==[1,2,0]
    assert items.group.iloc[2]=='Train-unseen'
    assert all('/training/' in p for p in calls)
    assert data.prepare(cfg,{'training':[{'file':'train.tfrecord.gz'}]}) == folder
    assert len(calls)==1


def test_coverage_keeps_invalid_slots(tmp_path):
    from src.analysis.long_tail.report import recommendation_coverage
    path=tmp_path/'batch.pt'
    torch.save({'scores':torch.tensor([[.9,.8,.7]]),
                'candidate_item_ids':torch.tensor([[-1,0,1]])},path)
    items=pd.DataFrame({'item_id':[0,1,2],'group':['Tail','Head','Tail']})
    f=recommendation_coverage([path],items,[2]).iloc[0]
    assert f.invalid_ratio==.5
    assert f.tail_share_among_legal==1
    assert f.tail_catalog_coverage==.5


def test_input_cache_interruption_resume(tmp_path, monkeypatch):
    from src.analysis.long_tail import engine
    audit_counts={'raw_records':3,'valid_records':3,'selected_records':3,'filtered_records':0,'filtered':[]}
    def batch(i):
        return (SimpleNamespace(mask=torch.ones(1,2),transformed_sequences={'sequence_data':torch.tensor([[i,i]])}),
                SimpleNamespace(labels={'sequence_data':torch.tensor([0,0])}),
                torch.tensor([i]),torch.tensor([0]), [('file',i)])
    def failing(cfg,sid,data_cfg,audit):
        audit.update(audit_counts)
        yield batch(0)
        raise RuntimeError('simulated interruption')
    manifest={'stages':{}}
    monkeypatch.setattr(engine,'test_batches',failing)
    with pytest.raises(RuntimeError,match='simulated'):
        engine.cache_test_inputs(tmp_path,{},None,None,manifest)
    assert completed_shards(tmp_path,'test_inputs')==1
    def successful(cfg,sid,data_cfg,audit):
        audit.update(audit_counts)
        yield from (batch(i) for i in range(3))
    monkeypatch.setattr(engine,'test_batches',successful)
    engine.cache_test_inputs(tmp_path,{},None,None,manifest)
    assert completed_shards(tmp_path,'test_inputs')==3
    loaded=list(engine.cached_batches(tmp_path))
    assert [b[2].item() for b in loaded]==[0,1,2]
    assert manifest['stages']['test_inputs']['samples']==3
    # Complete cache is reusable without re-reading original TFRecords.
    monkeypatch.setattr(engine,'test_batches',failing)
    engine.cache_test_inputs(tmp_path,{},None,None,manifest)


def test_evaluate_resume_skips_committed_inference(tmp_path, monkeypatch):
    from src.analysis.long_tail import engine
    dataset=tmp_path/'dataset'
    dataset.mkdir()
    pd.DataFrame({'item_id':[0,1,2], 'group':['Head','Mid','Tail'],
        'frequency_bucket':['>50','11–20','1–5'],'train_frequency':[100,20,1]}).to_csv(dataset/'items.csv',index=False)
    sid=torch.tensor([[0,0,1],[0,1,0]])
    cfg={'seed':42,'device':'cpu','beam_widths':[10],'k_values':[5,10],
         'output_root':str(tmp_path/'outputs'),'max_samples':None,'run_id':'resume-test'}
    monkeypatch.setattr(engine,'provenance',lambda cfg: {'fixture':True})
    monkeypatch.setattr(engine,'prepare',lambda cfg, inputs: dataset)
    monkeypatch.setattr(engine,'environment',lambda: {})
    model=SimpleNamespace(num_hierarchies=2,num_embeddings_per_hierarchy=16,
                          decoder=SimpleNamespace(decoder_mlp=[SimpleNamespace(out_features=16)]*2))
    monkeypatch.setattr(engine,'model_and_data',lambda cfg:(model,None,sid,{'epoch':0,'global_step':0}))
    def records(cfg,sid,data_cfg,audit):
        audit.update(raw_records=3,valid_records=3,selected_records=3,filtered_records=0,filtered=[])
        for i in range(3):
            yield (SimpleNamespace(mask=torch.ones(1,2),transformed_sequences={}),
                   SimpleNamespace(labels={'sequence_data':sid[:,i]}),torch.tensor([i]),torch.tensor([i]),[('file',i)])
    monkeypatch.setattr(engine,'test_batches',records)
    calls=[]
    fail=[True]
    def infer(cfg,stage,batch,meta,model,sid,index):
        calls.append(stage)
        if stage=='beam_10' and fail[0]:
            fail[0]=False
            raise RuntimeError('inference interruption')
        keys=[r['sample_key'] for r in meta]
        return {'sample_keys':keys,'values':torch.tensor([meta[0]['target_item_id']])},pd.DataFrame(meta)
    monkeypatch.setattr(engine,'infer_batch',infer)
    with pytest.raises(RuntimeError,match='inference interruption'):
        engine.evaluate(cfg)
    run=Path(cfg['output_root'])/'runs/resume-test'
    assert completed_shards(run,'teacher_forcing')==1
    assert read_json(run/'manifest.json')['status']=='interrupted'
    assert calls.count('teacher_forcing')==3
    engine.evaluate(cfg,resume=run)
    assert calls.count('teacher_forcing')==3
    assert completed_shards(run,'beam_10')==1
    assert read_json(run/'manifest.json')['status']=='inference_complete'
    payload=torch.load(run/'predictions/beam_10/000000.pt',weights_only=True)
    assert payload['sample_keys']==['file:0','file:1','file:2']
    assert payload['values'].tolist()==[0,1,2]
