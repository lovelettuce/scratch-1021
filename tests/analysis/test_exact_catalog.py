"""Exhaustive ranking, probability semantics, provenance and portable replay."""
from itertools import product
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.analysis.exact_catalog.core import BEAMS, TOLERANCE, diagnose, ranked
from src.analysis.exact_catalog.scoring import candidate_scores, catalog_scores, encode
from src.analysis.exact_catalog.experiment import save_array, validate_run
from src.analysis.exact_catalog.report import report, summarize
from src.analysis.long_tail.io import commit_shard, digest, read_json, write_csv, write_json
from src.analysis.long_tail.diagnostics import teacher_forcing


class CausalModel(nn.Module):
    num_hierarchies = 4
    feature_to_model_input_map = {'sequence_data': 'input_ids'}

    def __init__(self):
        super().__init__()
        torch.manual_seed(17)
        self.embedding = nn.Embedding(4, 8)
        self.heads = nn.ModuleList(nn.Linear(8, 4) for _ in range(4))
        self.decoder = SimpleNamespace(decoder_mlp=self.heads)
        self.encoder_calls = 0

    def encoder_forward_pass(self, input_ids, attention_mask):
        self.encoder_calls += 1
        hidden = (self.embedding(input_ids)*attention_mask[..., None]).sum(1, keepdim=True)
        return hidden, torch.ones(len(input_ids), 1)

    def decoder_forward_pass(self, future_ids, encoder_output, **kwargs):
        zero = torch.zeros(len(future_ids), 1, 8)
        # Each predicted position sees exactly the preceding candidate tokens.
        return torch.cat([zero, self.embedding(future_ids).cumsum(1)], 1)+encoder_output

    def forward(self, attention_mask_encoder, input_ids, future_ids):
        enc, _ = self.encoder_forward_pass(input_ids, attention_mask_encoder)
        return self.decoder_forward_pass(future_ids, enc)


def test_exhaustive_scores_match_independent_full_forward_and_chunks():
    model = CausalModel().eval()
    inp = SimpleNamespace(mask=torch.ones(2, 3),
        transformed_sequences={'sequence_data':torch.tensor([[0,1,2],[1,2,3]])})
    catalog = torch.tensor(list(product(range(3), repeat=4)))
    with torch.inference_mode():
        encoded, mask = encode(model, inp)
        scores = catalog_scores(model, encoded, mask, catalog, 7)
        assert model.encoder_calls == 1
        for chunk in [1, 16, 128]:
            torch.testing.assert_close(scores, catalog_scores(model, encoded, mask, catalog, chunk), atol=1e-5, rtol=0)
        permutation = torch.randperm(len(catalog))
        shuffled = catalog_scores(model, encoded, mask, catalog[permutation], 13)
        torch.testing.assert_close(shuffled, scores[:, permutation], atol=1e-5, rtol=0)
        for i in [0, 4, 80]:
            targets = catalog[i].expand(2, -1)
            logs, _ = teacher_forcing(model, inp, SimpleNamespace(labels={'sequence_data':targets}))
            independent = logs.gather(-1, targets[..., None]).squeeze(-1).sum(-1)
            torch.testing.assert_close(scores[:,i], independent, atol=1e-5, rtol=0)
        assert scores.exp().sum(-1).lt(1).all()  # Illegal token mass is not renormalized away.
    with pytest.raises(ValueError, match='positive'):
        candidate_scores(model, encoded, mask, catalog[None], 0)


def synthetic(target_id=0, kind='search_miss'):
    catalog = np.array(list(product(range(3), repeat=4)), dtype=np.int64)[:32]
    scores = np.arange(32, dtype=np.float32)*-.1-2
    if kind == 'incidental_hit':
        target_id = 20
    if kind == 'extreme':
        scores -= 1000
    all_sids = np.array(list(product(range(4), repeat=4)))
    mapping = {tuple(s): i for i,s in enumerate(catalog)}
    cs, rs, native, old = {}, {}, {}, {}
    for beam in BEAMS:
        if kind == 'incidental_hit':
            ids = [20] + list(range(21,32))
        elif kind == 'both_hit':
            ids = list(range(32))
        else:
            ids = list(range(1,32))
        rows = [catalog[i] for i in ids]
        rows += [s for s in all_sids if tuple(s) not in mapping]
        cs[beam] = np.array(rows[:beam])
        rs[beam] = np.array([scores[mapping[tuple(s)]] if tuple(s) in mapping else -1.
                             for s in cs[beam]], np.float32)
        native[beam] = np.exp(rs[beam]).astype(np.float32)
        rank = next((i+1 for i,j in enumerate(np.argsort(-native[beam]))
                     if np.array_equal(cs[beam][j], catalog[target_id])), None)
        old[beam] = {'first_lost_level':0 if rank else 2,
                    **{f'survival_level_{i}':int(rank is not None or i < 2) for i in range(1,5)},
                    **{f'{m}_at_{k}':float(rank is not None and rank<=k) *
                       (1/np.log2(rank+1) if m=='ndcg' and rank is not None else 1)
                       for k in [5,10] for m in ['recall','ndcg']}}
    meta=dict(cell='beauty_rqkmeans',dataset='beauty',sample_key='p:0',user_id='7',
              target_item_id=str(target_id),group='Tail')
    parts = np.full(4, scores[target_id]/4, dtype=np.float32)
    rows, top = diagnose(meta,scores,catalog,parts,cs,rs,native,old)
    for row in rows:
        row.update(l2_rescued_from_10=False,complete_outside_top10=False,original_hit_lost_from_10=False)
    return rows, top, (meta,scores,catalog,parts,cs,rs,native,old)


@pytest.mark.parametrize('kind,category', [('search_miss','search_miss'),
    ('incidental_hit','incidental_hit'),('both_hit','both_hit')])
def test_search_misses_incidental_hits_and_invalid_slots(kind,category):
    rows,_,_=synthetic(kind=kind)
    assert all(r['category_at_10']==category for r in rows)
    if kind=='both_hit':
        assert rows[-1]['invalid_candidates']>0
        assert rows[-1]['rescored_recall_at_10']==0
        assert rows[-1]['legal_recall_at_10']==1
    tables=summarize(pd.DataFrame(rows),repeats=32)
    assert set(tables)=={'metrics','pairs','diagnostics','cohorts'}
    assert tables['cohorts'].value.isna().all()
    assert (tables['metrics'][tables['metrics'].group=='Head'].status=='no_samples').all()


def test_ties_sid_order_missing_targets_and_empty_candidates():
    sids=np.array([[1,0],[0,1],[0,0]])
    values=np.array([-1000.,-1000.,-1001.])
    r,order=ranked(values,sids,sids[0])
    assert order.tolist()==[1,0,2]
    assert (r['rank'],r['rank_best'],r['rank_worst'])==(2,1,2)
    assert r['recall_at_5']==1
    empty,_=ranked(np.array([]),np.empty((0,2),int),sids[0])
    assert empty['rank'] is None and empty['recall_at_10']==0
    with pytest.raises(ValueError,match='Duplicate'):
        ranked(np.ones(2),np.array([[0],[0]]),np.array([0]))
    with pytest.raises(ValueError,match='Invalid'):
        ranked(np.array([np.nan]),np.array([[0]]),np.array([0]))


def test_numeric_audit_extreme_scores_and_duplicate_candidates():
    rows,_,args=synthetic(kind='extreme')
    assert rows[0]['native_zero_scores']>0 and np.isfinite(rows[0]['target_nll'])
    _,_,args=synthetic()
    values=list(args)
    values[5]={k:v.copy() for k,v in values[5].items()}
    values[5][10][0]+=.01
    with pytest.raises(ValueError,match='score error'):
        diagnose(*values)
    values=list(args); values[4]={k:v.copy() for k,v in values[4].items()}
    values[4][10][0]=values[4][10][1]
    with pytest.raises(ValueError,match='Duplicate'):
        diagnose(*values)


def build_run(path):
    rows,top,args=synthetic(kind='both_hit')
    npy=path/'inputs/beauty_rqkmeans/catalog.npy'
    save_array(npy,args[2])
    frame=pd.DataFrame(rows)
    files=[path/'cells/beauty_rqkmeans/000000.npy',path/'cells/beauty_rqkmeans/000000.paired.csv.gz']
    save_array(files[0],args[1][None])
    write_csv(files[1],frame)
    commit_shard(path,'scores_beauty_rqkmeans',0,['p:0'],files)
    write_json(path/'manifest.json',dict(status='evaluated',cells=['beauty_rqkmeans'],
        cell_status={'beauty_rqkmeans':'complete'},is_full_evaluation=False,
        snapshots=[{'path':str(npy.relative_to(path)),'sha256':digest(npy)}],
        parameters={'bootstrap_replicates':32,'seed':42},
        sources={'beauty_rqkmeans':{'selected_samples':1,'catalog_items':32,'keys':['p:0']}},
        input_sha256={'/missing/checkpoint':'irrelevant_to_portable_report'}))


def test_portable_report_without_gpu_checkpoint_or_external_inputs(tmp_path,monkeypatch):
    original=tmp_path/'a';build_run(original)
    report(original)
    hashes={p.name:digest(p) for p in (original/'report/tables').glob('*.csv')}
    copied=tmp_path/'b';shutil.copytree(original,copied);shutil.rmtree(original)
    def forbidden(*a,**kw):raise AssertionError('Report touched GPU/model')
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    report(copied)
    assert hashes=={p.name:digest(p) for p in (copied/'report/tables').glob('*.csv')}
    assert len(list((copied/'report/figures').glob('*.pdf')))==5
    (copied/'cells/beauty_rqkmeans/000000.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):
        report(copied)


def test_snapshot_corruption_is_rejected(tmp_path):
    build_run(tmp_path)
    m=read_json(tmp_path/'manifest.json')
    validate_run(tmp_path,m,external=False)
    (tmp_path/m['snapshots'][0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='snapshot'):
        validate_run(tmp_path,m,external=False)


def test_prefix_pruning_can_miss_globally_best_sequence():
    first=np.array([.6,.4])
    conditional=np.array([[.4,.3,.3],[.9,.05,.05]])
    joint=first[:,None]*conditional
    greedy=joint[first.argmax()].max()
    assert joint.max()==pytest.approx(.36) and greedy==pytest.approx(.24)
    assert first.argmax()!=np.unravel_index(joint.argmax(),joint.shape)[0]


def test_real_scoring_round_trip_interruption_and_resume(tmp_path, monkeypatch):
    from src.analysis.exact_catalog import experiment as exp
    from src.analysis.long_tail.engine import recommendation_values
    model=CausalModel().eval()
    source=tmp_path/'source';source.mkdir()
    catalog=torch.tensor(list(product(range(3),repeat=4)))[:32]
    sid_path=source/'sid.pt';torch.save(catalog.T,sid_path)
    n=17;keys=[f'p:{i}' for i in range(n)]
    target_ids=torch.tensor([i%32 for i in range(n)])
    histories=torch.tensor([[i%3,(i+1)%3,(i+2)%3] for i in range(n)])
    input_files=[]
    for start in range(0,n,8):
        count=min(8,n-start);p=source/f'input{start}.pt'
        torch.save({'mask':torch.ones(count,3), 'transformed_sequences':{'sequence_data':histories[start:start+count]},
            'item_ids':target_ids[start:start+count], 'labels':{'sequence_data':catalog[target_ids[start:start+count]].flatten()},
            'users':torch.tensor([i//2 for i in range(start,start+count)]),
            'locations':[('p',i) for i in range(start,start+count)]},p)
        input_files.append(str(p))
    all_sids=torch.tensor(list(product(range(4),repeat=4)))
    inp=SimpleNamespace(mask=torch.ones(n,3),transformed_sequences={'sequence_data':histories})
    with torch.inference_mode():
        encoded,mask=encode(model,inp)
        scores=catalog_scores(model,encoded,mask,all_sids,23)
    predictions={}
    for b in BEAMS:
        ids=scores.topk(b).indices
        candidates=all_sids[ids]
        probs=scores.gather(1,ids).exp()
        mapping={tuple(s):i for i,s in enumerate(catalog.tolist())}
        items=torch.tensor([[mapping.get(tuple(s),-1) for s in row] for row in candidates.tolist()])
        path=source/f'beam{b}.pt'
        torch.save(dict(sample_keys=keys,generated_sids=candidates,scores=probs,
            target_sids=catalog[target_ids],candidate_valid=items>=0,candidate_item_ids=items),path)
        predictions[str(b)]=[str(path)]
        values=recommendation_values(probs,candidates,catalog[target_ids],[5,10])
        rows=[]
        for i in range(n):
            complete=bool((items[i]==target_ids[i]).any())
            rows.append(dict(sample_key=keys[i],user_id=str(i//2),target_item_id=str(int(target_ids[i])),group='Tail',
                first_lost_level=0 if complete else 2,
                **{f'survival_level_{l}':int(complete or l<2) for l in range(1,5)},
                **{k:float(v[i]) for k,v in values.items()}))
        write_csv(source/f'recommendation_{b}.csv.gz',pd.DataFrame(rows))
    spec={'beauty_rqkmeans':dict(config={'semantic_id_path':str(sid_path)},cache=str(source),
        input_files=input_files,prediction_files=predictions,samples=n,catalog_items=32,keys=keys)}
    monkeypatch.setattr(exp,'sources',lambda *a:(spec,{}))
    monkeypatch.setattr(exp,'environment',lambda:{'fixture':True})
    monkeypatch.setattr(exp,'model_and_data',lambda cfg:(model,None,None,None))
    run=exp.prepare(tmp_path,cell='beauty_rqkmeans',candidate_chunk_size=13)
    exp.evaluate(run,device='cpu',preflight_only=True)
    assert read_json(run/'manifest.json')['preflight']['beauty_rqkmeans']['samples']==16
    original_commit=exp.commit_shard
    def interrupted(*args,**kw):
        original_commit(*args,**kw)
        raise RuntimeError('simulated interruption after durable commit')
    monkeypatch.setattr(exp,'commit_shard',interrupted)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        exp.evaluate(run,device='cpu',resume=True)
    marker=run/'progress/scores_beauty_rqkmeans/000000.json'
    previous=digest(marker)
    monkeypatch.setattr(exp,'commit_shard',original_commit)
    exp.evaluate(run,device='cpu',resume=True)
    assert digest(marker)==previous
    m=read_json(run/'manifest.json')
    assert m['cell_status']['beauty_rqkmeans']=='complete'
    assert len(read_json(marker)['sample_keys'])==17
    report(run,repeats=16)
    with pytest.raises(ValueError,match='requires --resume'):
        exp.evaluate(run,device='cpu')
    # Offline migration does not need the model, source predictions or checkpoint.
    copied=tmp_path/'portable';shutil.copytree(run,copied);shutil.rmtree(source)
    report(copied,repeats=16)


def test_parallel_assignment_worker_isolation_and_idempotent_publish(tmp_path):
    from src.analysis.exact_catalog.parallel import assignment, worker_run, publish
    assert assignment(['a','b'],'4,5')=={'a':'4','b':'5'}
    for devices in ('4,4','4','x,5'):
        with pytest.raises(ValueError,match='distinct'):
            assignment(['a','b'],devices)
    parent=tmp_path/'parent';build_run(parent)
    manifest=read_json(parent/'manifest.json')
    manifest.update(identity='frozen-parent',status='preflight_complete',cell_status={},
                    preflight={'beauty_rqkmeans':{'score_checks':'passed'}})
    # Remove parent score markers, retaining immutable source snapshots.
    shutil.rmtree(parent/'progress');shutil.rmtree(parent/'cells')
    worker=worker_run(parent,manifest,'beauty_rqkmeans')
    assert worker_run(parent,manifest,'beauty_rqkmeans')==worker
    m=read_json(worker/'manifest.json')
    assert m['cells']==['beauty_rqkmeans'] and m['parent_identity']=='frozen-parent'
    build_run(worker)
    publish(parent,worker,'beauty_rqkmeans')
    marker=parent/'progress/scores_beauty_rqkmeans/000000.json'
    before=digest(marker)
    publish(parent,worker,'beauty_rqkmeans')
    assert digest(marker)==before
    assert all(x['path'].startswith('workers/beauty_rqkmeans/') for x in read_json(marker)['artifacts'])
    with pytest.raises(ValueError,match='different'):
        worker_run(parent,dict(manifest,identity='other'),'beauty_rqkmeans')
