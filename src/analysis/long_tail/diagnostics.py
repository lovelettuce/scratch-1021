"""Trie statistics and stable teacher-forced probability diagnostics."""
from collections import Counter, defaultdict

import pandas as pd
import torch


def branch_bucket(n):
    if n == 1:
        return '1'
    for hi, label in [(4, '2–4'), (16, '5–16'), (64, '17–64')]:
        if n <= hi:
            return label
    return '≥65'


class SIDIndex:
    def __init__(self, sid, frequencies):
        self.rows = [tuple(x) for x in sid.T.tolist()]
        self.item_by_sid = {s: i for i, s in enumerate(self.rows)}
        self.children = defaultdict(set)
        self.mass = Counter()
        self.group_sizes = Counter(s[:-1] for s in self.rows)
        for item, row in enumerate(self.rows):
            for level, token in enumerate(row):
                self.children[row[:level]].add(token)
                self.mass[row[:level]] += int(frequencies[item])

    def tree_table(self):
        return pd.DataFrame([{'level': len(prefix) + 1, 'prefix': ','.join(map(str, prefix)),
                              'legal_children': len(children), 'prefix_train_interactions': self.mass[prefix]}
                             for prefix, children in self.children.items()])

    def diagnose(self, log_probs, targets, metadata):
        log_probs, targets = log_probs.cpu(), targets.cpu()
        rows = []
        for n, target in enumerate(targets.tolist()):
            sid = tuple(target)
            for level, token in enumerate(target):
                prefix = sid[:level]
                legal = sorted(self.children[prefix])
                logp = log_probs[n, level]
                if token not in legal:
                    raise ValueError('Target outside legal SID tree')
                logz = torch.logsumexp(logp[legal], dim=0)
                # Clamp tiny floating roundoff; decomposition still uses the same logZ.
                logz = torch.minimum(logz, torch.zeros_like(logz))
                rank = 1. + (logp[legal] > logp[token]).sum().item() + .5 * ((logp[legal] == logp[token]).sum().item() - 1)
                raw_nll = -logp[token].item()
                legality = -logz.item()
                rows.append({**metadata[n], 'level': level + 1, 'target_token': token,
                    'legal_children': len(legal), 'branch_bucket': branch_bucket(len(legal)),
                    'illegal_mass': -torch.expm1(logz).item(), 'raw_nll': raw_nll,
                    'legality_nll': legality, 'conditional_nll': raw_nll - legality,
                    'legal_rank': rank, 'legal_mrr': 1. / rank,
                    **{f'legal_hit_at_{k}': float(rank <= k) for k in (1, 5, 10)},
                    'prefix_train_interactions': self.mass[prefix],
                    'collision_size': self.group_sizes[sid[:-1]],
                    'collision_group': 'collision' if self.group_sizes[sid[:-1]] > 1 else 'singleton'})
        return pd.DataFrame(rows)


def teacher_forcing(model, model_input, labels):
    targets = labels.labels['sequence_data'].reshape(model_input.mask.size(0), model.num_hierarchies).long()
    kwargs = {model.feature_to_model_input_map.get(k, k): v for k, v in model_input.transformed_sequences.items()}
    output = model(attention_mask_encoder=model_input.mask, future_ids=targets, **kwargs)[:, :-1]
    logits = torch.stack([head(output[:, level]) for level, head in enumerate(model.decoder.decoder_mlp)], dim=1).float()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits')
    return logits.log_softmax(-1), targets
