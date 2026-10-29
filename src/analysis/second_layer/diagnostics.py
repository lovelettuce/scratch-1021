"""Exact native-float search ranks and mechanism-level pruning attribution."""
import math

import numpy as np

CAUSES=['first_layer_lost','retained','local_capacity','cross_parent','cutoff_tie']
SEARCH_FIELDS=['parent_probability','parent_beam_position','target_conditional_probability',
    'target_cumulative_probability','cutoff_probability','log_score_margin','target_log_score',
    'local_rank_best','local_rank_worst','legal_rank_best','legal_rank_worst',
    'global_rank_best','global_rank_worst','same_parent_higher','other_parent_higher',
    'same_parent_legal_higher','same_parent_illegal_higher','other_parent_legal_higher',
    'other_parent_illegal_higher','global_equal','numeric_state']


def rank_bounds(values,target):
    values=np.asarray(values)
    return int((values>target).sum())+1,int((values>=target).sum())


def classify(entered, retained, same_higher=0, other_higher=0, width=10):
    if not entered:
        if retained:
            raise ValueError('A lost parent cannot reappear')
        return 'first_layer_lost'
    if retained:
        if same_higher+other_higher>=width:
            raise ValueError('Retained target has >=B strictly better competitors')
        return 'retained'
    if same_higher>=width:
        return 'local_capacity'
    if same_higher+other_higher>=width:
        return 'cross_parent'
    return 'cutoff_tie'


class SearchCollector:
    def __init__(self, targets, support):
        self.targets=targets.detach().cpu().numpy()
        self.support=support
        self.rows=None
        self.trace={}
        self.survival=[]

    def candidates(self,event):
        if event['hierarchy']!=1:
            raise ValueError('Expected second-level snapshot')
        parents=event['parent_prefixes'].detach().cpu()
        parent_scores=event['parent_scores'].detach().cpu()
        ids=event['next_token_ids'].detach().cpu().numpy()
        conditional=event['conditional_probabilities'].detach().cpu().numpy()
        cumulative=event['cumulative_scores'].detach().cpu().numpy()
        n,width=parent_scores.shape
        vocab=ids.shape[1]//width
        ids=ids.reshape(n,width,vocab)
        conditional=conditional.reshape(n,width,vocab)
        cumulative=cumulative.reshape(n,width,vocab)
        if not np.isfinite(cumulative).all() or (cumulative<0).any():
            raise ValueError('Invalid native cumulative scores')
        self.trace['parent_prefixes']=parents
        self.trace['parent_scores']=parent_scores
        self.rows=[]
        for i,(c1,c2,*_) in enumerate(self.targets):
            row={key:None for key in SEARCH_FIELDS}
            positions=np.flatnonzero(parents[i,:,0].numpy()==c1)
            row['entered_second']=int(bool(len(positions)))
            if not len(positions):
                self.rows.append(row)
                continue
            if len(positions)!=1:
                raise ValueError('Duplicated parent prefix')
            parent=int(positions[0])
            token_positions=np.flatnonzero(ids[i,parent]==c2)
            if len(token_positions)!=1:
                raise ValueError('Token permutation invalid')
            token=int(token_positions[0])
            score=float(cumulative[i,parent,token])
            q=float(conditional[i,parent,token])
            p=float(parent_scores[i,parent])
            flat=cumulative[i].reshape(-1)
            cutoff=float(np.partition(flat,-width)[-width])
            high=cumulative[i]>score
            if not hasattr(self.support, 'second_legal_masks'):
                self.support.second_legal_masks={}
            if vocab not in self.support.second_legal_masks:
                mask=np.zeros((vocab,vocab),dtype=bool)
                for prefix,children in self.support.children.items():
                    if len(prefix)==1:
                        mask[prefix[0],list(children)]=True
                self.support.second_legal_masks[vocab]=mask
            legal=self.support.second_legal_masks[vocab][parents[i,:,0].numpy()[:,None],ids[i]]
            same=high[parent]
            global_best,global_worst=rank_bounds(flat,score)
            local_best,local_worst=rank_bounds(conditional[i,parent],q)
            legal_best,legal_worst=rank_bounds(conditional[i,parent][legal[parent]],q)
            same_higher=int(same.sum())
            other_higher=int(high.sum())-same_higher
            same_legal=int((same & legal[parent]).sum())
            other_legal=int((high & legal).sum())-same_legal
            log_score=math.log(p)+math.log(q) if p>0 and q>0 else None
            margin=math.log(score)-math.log(cutoff) if score>0 and cutoff>0 else None
            flags=[]
            if p==0: flags.append('parent_zero')
            if q==0: flags.append('conditional_zero')
            if score==0 and p>0 and q>0: flags.append('product_underflow')
            if cutoff==0: flags.append('cutoff_zero')
            row.update(parent_probability=p,parent_beam_position=parent+1,
                target_conditional_probability=q,target_cumulative_probability=score,
                cutoff_probability=cutoff,log_score_margin=margin,target_log_score=log_score,
                local_rank_best=local_best,local_rank_worst=local_worst,
                legal_rank_best=legal_best,legal_rank_worst=legal_worst,
                global_rank_best=global_best,global_rank_worst=global_worst,
                same_parent_higher=same_higher,other_parent_higher=other_higher,
                same_parent_legal_higher=same_legal,same_parent_illegal_higher=same_higher-same_legal,
                other_parent_legal_higher=other_legal,other_parent_illegal_higher=other_higher-other_legal,
                global_equal=global_worst-global_best+1,numeric_state=';'.join(flags) or 'normal')
            self.rows.append(row)

    def step(self,level,prefixes,scores):
        prefixes=prefixes.detach().cpu()
        target=self.targets[:,:level+1]
        alive=(prefixes.numpy()==target[:,None,:]).all(-1).any(-1)
        self.survival.append(alive)
        if level==1:
            if self.rows is None:
                raise ValueError('Missing pre-pruning snapshot')
            self.trace['child_prefixes']=prefixes
            self.trace['child_scores']=scores.detach().cpu()
            width=prefixes.shape[1]
            for i,row in enumerate(self.rows):
                row['retained_second']=int(alive[i])
                row['cause']=classify(row['entered_second'],alive[i],row['same_parent_higher'] or 0,
                                     row['other_parent_higher'] or 0,width)

    def finish(self):
        survival=np.stack(self.survival,axis=1)
        if (np.diff(survival.astype(int),axis=1)>0).any():
            raise ValueError('Target path resurrected')
        for i,row in enumerate(self.rows):
            lost=np.flatnonzero(~survival[i])
            row['first_lost_level']=int(lost[0]+1) if len(lost) else 0
            row['final_in_beam']=int(survival[i,-1])
        return self.rows,self.trace
