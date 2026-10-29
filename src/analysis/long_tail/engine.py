"""Staged inference with validated, atomic, restartable batches."""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from .data import model_and_data, prepare, test_batches
from .diagnostics import SIDIndex, teacher_forcing
from .io import (commit_shard, completed_shards, environment, provenance, read_json,
                 validate_identity, write_csv, write_json, write_pt)

LOGGER = logging.getLogger(__name__)


def recommendation_values(scores, generated, targets, k_values):
    """Match the repository evaluator's torch.topk behavior, including score ties."""
    matches = (generated == targets[:, None, :]).all(-1)
    result = {}
    for k in k_values:
        indices = torch.topk(scores, k, dim=-1).indices
        hits = matches.gather(1, indices).float()
        result[f'recall_at_{k}'] = hits.sum(1).clamp(max=1).cpu().numpy()
        discounts = torch.log2(torch.arange(2, k + 2, device=scores.device).float())
        result[f'ndcg_at_{k}'] = (hits / discounts).sum(1).cpu().numpy()
    return result


def first_lost(survival):
    return next((i + 1 for i, alive in enumerate(survival) if not alive), 0)


def metadata_for_batch(batch, items):
    model_input, labels, users, item_ids, locations = batch
    metadata = []
    for user, item, (file, index) in zip(users.tolist(), item_ids.tolist(), locations):
        info = items.loc[item]
        metadata.append({'sample_key': f'{file}:{index}', 'source_file': file,
            'source_record_index': index, 'user_id': user, 'target_item_id': item,
            'group': info['group'], 'frequency_bucket': str(info['frequency_bucket']),
            'train_frequency': int(info['train_frequency'])})
    return metadata


def cache_test_inputs(run_dir, cfg, sid, data_cfg, manifest):
    """Parse large TFRecords once; subsequent stages reuse compact CPU batches."""
    done = completed_shards(run_dir, 'test_inputs')
    state = manifest['stages'].get('test_inputs', {})
    if state.get('status') == 'complete':
        if state['batches'] != done:
            raise ValueError('Test input cache is incomplete')
        return
    audit = {}
    batches = 0
    for shard, batch in enumerate(test_batches(cfg, sid, data_cfg, audit)):
        batches += 1
        model_input, labels, users, item_ids, locations = batch
        keys = [f'{file}:{index}' for file, index in locations]
        if shard < done:
            if read_json(run_dir/'progress/test_inputs'/f'{shard:06d}.json')['sample_keys'] != keys:
                raise ValueError('Cached test input alignment differs')
            continue
        path = run_dir/'test_inputs'/f'{shard:06d}.pt'
        write_pt(path, {'mask': model_input.mask, 'transformed_sequences': model_input.transformed_sequences,
                       'labels': labels.labels, 'users': users, 'item_ids': item_ids,
                       'locations': locations})
        commit_shard(run_dir, 'test_inputs', shard, keys, [path])
    if not batches or batches < done:
        raise ValueError('Empty or truncated test input cache')
    manifest['stages']['test_inputs'] = {'status': 'complete', 'batches': batches,
                                        'samples': audit['selected_records'], 'audit': audit}
    write_json(run_dir/'manifest.json', manifest)
    LOGGER.info('Cached %d test inputs; filtered=%d', audit['selected_records'], audit['filtered_records'])


def cached_batches(run_dir):
    from types import SimpleNamespace
    for marker in sorted((run_dir/'progress/test_inputs').glob('*.json')):
        artifact = read_json(marker)['artifacts'][0]['path']
        batch = torch.load(run_dir/artifact, map_location='cpu', weights_only=True)
        yield (SimpleNamespace(mask=batch['mask'], transformed_sequences=batch['transformed_sequences']),
               SimpleNamespace(labels=batch['labels']), batch['users'], batch['item_ids'], batch['locations'])


def infer_batch(cfg, stage, batch, metadata, model, sid, index):
    keys = [r['sample_key'] for r in metadata]
    model_input, labels, users, item_ids, locations = batch
    model_input.mask = model_input.mask.to(cfg['device'])
    model_input.transformed_sequences = {k: v.to(cfg['device']) for k, v in model_input.transformed_sequences.items()}
    labels.labels = {k: v.to(cfg['device']) for k, v in labels.labels.items()}
    targets = labels.labels['sequence_data'].reshape(-1, sid.shape[0]).long()
    if not torch.equal(targets.cpu(), sid[:, item_ids].T):
        raise ValueError('Target item and SID do not agree')
    with torch.inference_mode():
        if stage == 'teacher_forcing':
            log_probs, targets = teacher_forcing(model, model_input, labels)
            payload = {'sample_keys': keys, 'metadata': metadata, 'target_sids': targets.cpu(),
                'probabilities': log_probs.exp().cpu(), 'log_probabilities': log_probs.cpu()}
            frame = index.diagnose(log_probs, targets, metadata)
        else:
            width = int(stage.split('_')[1])
            model.top_k_for_generation = width
            survival = []
            def observer(level, prefixes, scores):
                survival.append((prefixes == targets[:, None, :level + 1]).all(-1).any(-1).cpu())
            kwargs = {model.feature_to_model_input_map.get(k, k): v for k, v in model_input.transformed_sequences.items()}
            generated, scores = model.generate(attention_mask=model_input.mask, step_observer=observer, **kwargs)
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite generation score')
            values = recommendation_values(scores, generated, targets, cfg['k_values'])
            alive = torch.stack(survival, dim=1).tolist()
            candidate_ids = [[index.item_by_sid.get(tuple(s), -1) for s in row] for row in generated.cpu().tolist()]
            valid = [[i >= 0 for i in row] for row in candidate_ids]
            records = []
            for n, meta in enumerate(metadata):
                records.append({**meta, **{k: float(v[n]) for k, v in values.items()},
                    **{f'survival_level_{l+1}': int(a) for l, a in enumerate(alive[n])},
                    'first_lost_level': first_lost(alive[n])})
            payload = {'sample_keys': keys, 'generated_sids': generated.cpu(),
                'scores': scores.cpu(), 'candidate_item_ids': torch.tensor(candidate_ids),
                'candidate_valid': torch.tensor(valid), 'target_sids': targets.cpu()}
            frame = pd.DataFrame(records)
    return payload, frame


def batch_groups(iterator, size=16):
    pending = []
    for batch in iterator:
        pending.append(batch)
        if len(pending) == size:
            yield pending
            pending = []
    if pending:
        yield pending


def merge_payloads(payloads):
    return {key: torch.cat([p[key] for p in payloads], dim=0) if isinstance(payloads[0][key], torch.Tensor)
            else [value for payload in payloads for value in payload[key]] for key in payloads[0]}


def evaluate(cfg, resume=None):
    import fcntl
    inputs = provenance(cfg)
    dataset_dir = prepare(cfg, inputs)
    if resume:
        run_dir = Path(resume).resolve()
        manifest = read_json(run_dir / 'manifest.json')
        validate_identity(manifest, cfg, inputs)
    else:
        name = cfg.get('run_id') or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        if Path(name).name != name:
            raise ValueError('run_id must be a single directory name')
        run_dir = Path(cfg['output_root']) / 'runs' / name
        run_dir.mkdir(parents=True, exist_ok=False)
        from omegaconf import OmegaConf
        OmegaConf.save(OmegaConf.create(cfg), run_dir / 'config.yaml')
        manifest = {'config': cfg, 'inputs': inputs, 'dataset_dir': str(dataset_dir),
            'environment': environment(), 'status': 'created', 'stages': {},
            'precision': 'float32', 'is_full_evaluation': cfg.get('max_samples') is None,
            'ranking': 'repository torch.topk at each K; invalid candidates retain slots'}
        write_json(run_dir / 'manifest.json', manifest)
    lock = (run_dir / '.evaluation.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another process is evaluating this run')
    handler = None
    try:
        (run_dir / 'logs').mkdir(exist_ok=True)
        handler = logging.FileHandler(run_dir / 'logs/evaluate.log')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logging.getLogger().addHandler(handler)
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.set_num_threads(1)
        if str(cfg['device']).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        model, data_cfg, sid, checkpoint_info = model_and_data(cfg)
        vocab = model.num_embeddings_per_hierarchy
        if model.num_hierarchies != sid.shape[0] or any(h.out_features != vocab for h in model.decoder.decoder_mlp):
            raise ValueError('SID/head shape mismatch')
        if max(cfg['beam_widths']) > vocab:
            raise ValueError('Beam width exceeds first-level vocabulary')
        items = pd.read_csv(dataset_dir / 'items.csv', keep_default_na=False).set_index('item_id')
        index = SIDIndex(sid, items.train_frequency.tolist())
        write_csv(run_dir / 'tables/tree_structure.csv', index.tree_table())
        manifest.update(checkpoint=checkpoint_info, vocab_size=vocab, num_levels=sid.shape[0],
                        status='evaluating', device_name=torch.cuda.get_device_name(cfg['device']) if str(cfg['device']).startswith('cuda') else 'cpu')
        write_json(run_dir / 'manifest.json', manifest)
        cache_test_inputs(run_dir, cfg, sid, data_cfg, manifest)
        for stage in ['teacher_forcing'] + [f'beam_{b}' for b in cfg['beam_widths']]:
            done = completed_shards(run_dir, stage)
            audit = manifest['stages']['test_inputs']['audit']
            keys_seen = set()
            stage_start = time.monotonic()
            LOGGER.info('Stage %s: %d committed batches', stage, done)
            batches = 0
            for shard, group in enumerate(batch_groups(cached_batches(run_dir))):
                batches += 1
                group_metadata = [metadata_for_batch(batch, items) for batch in group]
                keys = [r['sample_key'] for metadata in group_metadata for r in metadata]
                if keys_seen.intersection(keys) or len(keys) != len(set(keys)):
                    raise ValueError('Duplicate test sample key')
                keys_seen.update(keys)
                if shard < done:
                    marker = read_json(run_dir / 'progress' / stage / f'{shard:06d}.json')
                    if marker['sample_keys'] != keys:
                        raise ValueError('Resume sample alignment differs')
                    continue
                payloads, frames = [], []
                for batch, metadata in zip(group, group_metadata):
                    payload, frame = infer_batch(cfg, stage, batch, metadata, model, sid, index)
                    payloads.append(payload)
                    frames.append(frame)
                if stage == 'teacher_forcing':
                    pt = run_dir/'teacher_forcing'/f'{shard:06d}.pt'
                    csv = run_dir/'diagnostics'/f'{shard:06d}.csv.gz'
                else:
                    pt = run_dir/'predictions'/stage/f'{shard:06d}.pt'
                    csv = pt.with_suffix('.csv.gz')
                write_pt(pt, merge_payloads(payloads))
                write_csv(csv, pd.concat(frames, ignore_index=True))
                commit_shard(run_dir, stage, shard, keys, [pt, csv])
                if shard % 10 == 0:
                    LOGGER.info('%s shard=%d samples=%d', stage, shard, len(keys_seen))
            if batches < done or not keys_seen:
                raise ValueError('Unexpected empty/truncated test stream')
            if audit['valid_records'] != audit['selected_records'] and cfg.get('max_samples') is None:
                raise ValueError('Full evaluation did not cover valid records')
            manifest['stages'][stage] = {'status': 'complete', 'batches': batches,
                'samples': len(keys_seen), 'audit': audit, 'last_invocation_seconds': time.monotonic() - stage_start}
            write_json(run_dir / 'manifest.json', manifest)
            LOGGER.info('Completed %s: %d samples; filtered=%d', stage, len(keys_seen), audit['filtered_records'])
        stage_audits = [v['audit'] for v in manifest['stages'].values()]
        if any(a != stage_audits[0] for a in stage_audits):
            raise ValueError('Stage test coverage differs')
        manifest['status'] = 'inference_complete'
        write_json(run_dir / 'manifest.json', manifest)
        return run_dir
    except Exception:
        manifest['status'] = 'interrupted'
        write_json(run_dir / 'manifest.json', manifest)
        raise
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
        lock.close()
