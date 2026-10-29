"""Exact original-interaction and pre-sampling target-window support."""
from collections import Counter, defaultdict
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.long_tail.data import load_sids, raw_records, validate_ids
from src.analysis.long_tail.diagnostics import branch_bucket
from src.analysis.long_tail.io import (canonical_hash, digest, read_json, write_csv, write_json)


def support_bucket(n):
    if n == 0:
        return '0'
    for maximum, label in [(5,'1–5'),(20,'6–20'),(100,'21–100'),(1000,'101–1000')]:
        if n <= maximum:
            return label
    return '>1000'


class SupportIndex:
    def __init__(self, sid, frame):
        self.rows = [tuple(s) for s in sid.T.tolist()]
        self.children = defaultdict(set)
        for row in self.rows:
            for level, token in enumerate(row):
                self.children[row[:level]].add(token)
        self.records = {tuple(map(int, r.prefix.split(','))) if r.prefix else ():r._asdict()
                        for r in frame.itertuples(index=False)}

    def get(self, prefix):
        return self.records[tuple(prefix)]

    def baseline(self, parent, mode='train_interactions'):
        children = sorted(self.children[tuple(parent)])
        counts = np.array([self.get(tuple(parent)+(v,))[mode] for v in children],dtype=float)
        return children, (counts+1)/(counts.sum()+len(children))

    def features(self, target):
        row=tuple(target)
        parent, edge=self.get(row[:1]),self.get(row[:2])
        first_raw=next((l for l in range(1,len(row)+1) if self.get(row[:l])['train_interactions']==0),0)
        first_windows=next((l for l in range(1,len(row)+1) if self.get(row[:l])['potential_target_windows']==0),0)
        return {'parent_train_interactions':parent['train_interactions'],
            'edge_train_interactions':edge['train_interactions'], 'edge_train_users':edge['train_users'],
            'edge_item_count':edge['item_count'], 'parent_potential_target_windows':parent['potential_target_windows'],
            'edge_potential_target_windows':edge['potential_target_windows'],
            'edge_interaction_share':edge['interaction_share'],
            'parent_child_hhi':parent['child_interaction_hhi'],
            'parent_child_entropy':parent['child_interaction_entropy'],
            'legal_children':parent['legal_children'], 'branch_bucket':branch_bucket(parent['legal_children']),
            'edge_support_bucket':support_bucket(edge['train_interactions']),
            'window_support_bucket':support_bucket(edge['potential_target_windows']),
            'first_unsupported_raw_level':first_raw,'first_unsupported_window_level':first_windows}


def build_support(rows, records):
    items, freq, windows = Counter(), Counter(), Counter()
    users=defaultdict(set)
    children=defaultdict(set)
    for row in rows:
        for level in range(len(row)+1):
            items[row[:level]] += 1
            if level < len(row):
                children[row[:level]].add(row[level])
    n_records=0
    total_windows=0
    for user, sequence in records:
        n_records+=1
        total_windows += len(sequence)*(len(sequence)-1)//2
        for position,item in enumerate(sequence):
            for level in range(len(rows[item])+1):
                prefix=rows[item][:level]
                freq[prefix]+=1
                windows[prefix]+=position
                users[prefix].add(user)
    if windows[()] != total_windows:
        raise AssertionError('Window support does not sum to n(n-1)/2')
    result=[]
    for prefix in sorted(items, key=lambda x:(len(x),x)):
        child_counts=np.array([freq[prefix+(v,)] for v in sorted(children[prefix])],dtype=float)
        q=child_counts/child_counts.sum() if child_counts.sum() else np.array([])
        result.append({'prefix':','.join(map(str,prefix)),'level':len(prefix),'item_count':items[prefix],
            'train_interactions':freq[prefix],'train_users':len(users[prefix]),
            'potential_target_windows':windows[prefix],
            'interaction_share':freq[prefix]/freq[prefix[:-1]] if prefix and freq[prefix[:-1]] else None,
            'legal_children':len(children[prefix]),'child_interaction_hhi':float((q*q).sum()) if len(q) else None,
            'child_interaction_entropy':float(-(q[q>0]*np.log(q[q>0])).sum()) if len(q) else None})
    return pd.DataFrame(result),{'training_records':n_records,'training_interactions':freq[()],
                                  'potential_target_windows':total_windows}


def prepare_support(cfg, identity):
    fingerprint=canonical_hash({'schema_version':1,'training':identity['training'],
        'sid':identity['sid_sha256'],'source_items':identity['source_items_sha256'],'support':'raw-and-potential-windows-v1'})
    folder=Path(cfg['output_root'])/'datasets'/fingerprint
    manifest_path=folder/'manifest.json'
    sid=load_sids(cfg['semantic_id_path'])
    if manifest_path.exists():
        meta=read_json(manifest_path)
        if digest(folder/'prefix_support.csv')!=meta['prefix_support_sha256']:
            raise ValueError('Support dataset corrupted')
    else:
        def records():
            for entry in identity['training']:
                path=Path(cfg['data_dir'])/'training'/entry['file']
                for index,user,sequence in raw_records(path):
                    validate_ids(sequence,sid.shape[1],f'{path}:{index}')
                    yield user,sequence
        frame,stats=build_support([tuple(s) for s in sid.T.tolist()],records())
        write_csv(folder/'prefix_support.csv',frame)
        write_json(manifest_path,{'dataset_id':fingerprint,'statistics':stats,
            'source_items_sha256':identity['source_items_sha256'],
            'prefix_support_sha256':digest(folder/'prefix_support.csv')})
    frame=pd.read_csv(folder/'prefix_support.csv',keep_default_na=False,dtype={'prefix':str})
    # Restore nullable numerical fields explicitly, without turning the root prefix into NaN.
    for col in ('interaction_share','child_interaction_hhi','child_interaction_entropy'):
        frame[col]=pd.to_numeric(frame[col],errors='coerce')
    return folder,SupportIndex(sid,frame)
