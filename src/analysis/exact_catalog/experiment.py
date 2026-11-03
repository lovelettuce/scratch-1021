"""Source audit, immutable preparation, resumable sequential GPU evaluation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from src.analysis.long_tail.io import (ROOT, atomic_write, canonical_hash, commit_shard,
    completed_shards, digest, environment, read_json, write_csv, write_json)
from src.analysis.long_tail.data import load_sids, model_and_data
from src.analysis.long_tail.engine import recommendation_values
from src.analysis.beam_rescue.core import validate_frame
from src.experiments.rq_cross_validation.common import EXPERIMENTS
from src.experiments.rq_cross_validation.report import aligned, validate_cache
from .core import BEAMS, TOLERANCE, diagnose
from .scoring import candidate_scores, catalog_scores, configure_precision, encode

LOGGER = logging.getLogger(__name__)
COUNTS = {'beauty': (22363, 12101), 'toys': (19412, 11924)}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def save_array(path, array):
    def save(p):
        with p.open('wb') as handle:
            np.save(handle, array, allow_pickle=False)
    atomic_write(path, save)


def sources(study, cells):
    """Validate committed sources without opening raw TFRecords or retraining."""
    inventory, specs = {}, {}
    for cell in cells:
        source = study / 'experiments' / cell / 'analysis/long_tail/runs/full'
        manifest = read_json(source / 'manifest.json')
        cache = study / 'comparison/inputs' / cell
        meta = validate_cache(cache)
        if manifest['status'] != 'complete' or not manifest['is_full_evaluation'] or manifest['precision'] != 'float32':
            raise ValueError(f'Incomplete/non-float32 source: {cell}')
        if digest(source / 'manifest.json') != meta['long_tail_source_sha256']:
            raise ValueError('Comparison/source manifests disagree')
        cfg = manifest['config'].copy()
        sid = load_sids(cfg['semantic_id_path'])
        expected, n_items = COUNTS[cell.split('_')[0]]
        if tuple(sid.shape) != (4, n_items) or manifest['stages']['test_inputs']['samples'] != expected:
            raise ValueError(f'Source sample/catalog coverage differs: {cell}')
        if (sid >= manifest['vocab_size']).any():
            raise ValueError('SID token outside classifier vocabulary')
        for name, path in [('checkpoint', cfg['checkpoint']), ('sid', cfg['semantic_id_path']),
                           ('training_config', str(Path(cfg['training_run_dir']) / '.hydra/config.yaml'))]:
            h = digest(path)
            if h != manifest['inputs'][name + '_sha256']:
                raise ValueError(f'Frozen {name} differs: {cell}')
            inventory[str(Path(path).resolve())] = h
        inventory[str(source / 'manifest.json')] = digest(source / 'manifest.json')
        inventory[str(cache / 'manifest.json')] = digest(cache / 'manifest.json')
        for entry in meta['artifacts']:
            inventory[str(cache / entry['path'])] = entry['sha256']
        keys, input_files, predictions = [], [], {}
        for stage in ('test_inputs', 'teacher_forcing', *(f'beam_{b}' for b in BEAMS)):
            count = completed_shards(source, stage)
            if count != manifest['stages'][stage]['batches']:
                raise ValueError(f'Source stage truncated: {cell}/{stage}')
            stage_keys, paths = [], []
            for marker in sorted((source / 'progress' / stage).glob('*.json')):
                info = read_json(marker)
                stage_keys.extend(info['sample_keys'])
                inventory[str(marker)] = digest(marker)
                for entry in info['artifacts']:
                    p = source / entry['path']
                    inventory[str(p)] = entry['sha256']
                    if p.suffix == '.pt':
                        paths.append(str(p))
            if stage == 'test_inputs':
                keys, input_files = stage_keys, paths
                if len(keys) != expected or len(set(keys)) != expected:
                    raise ValueError('Duplicate/missing cached test identity')
            elif stage_keys != keys:
                raise ValueError('Prediction and cached input ordering disagree')
            if stage.startswith('beam_'):
                predictions[stage.split('_')[1]] = paths
        frames = {}
        for beam in BEAMS:
            frame = pd.read_csv(cache / f'recommendation_{beam}.csv.gz',
                                dtype={'sample_key': str, 'user_id': str, 'target_item_id': str})
            validate_frame(frame)
            if len(frame) != expected or set(frame.sample_key) != set(keys):
                raise ValueError('Frozen CSV coverage differs')
            if frames:
                aligned(next(iter(frames.values())), frame)
            frames[beam] = frame.set_index('sample_key').loc[keys].reset_index()
        specs[cell] = dict(config=cfg, source=str(source), cache=str(cache),
                           input_files=input_files, prediction_files=predictions,
                           samples=expected, catalog_items=n_items, keys=keys)
    return specs, inventory


def code_inventory():
    paths = list(Path(__file__).parent.glob('*.py')) + [
        ROOT / 'src/models/modules/semantic_id/tiger_generation_model.py',
        ROOT / 'src/analysis/long_tail/data.py', ROOT / 'src/analysis/long_tail/io.py',
        ROOT / 'src/analysis/long_tail/diagnostics.py', ROOT / 'src/analysis/long_tail/engine.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def prepare(study_dir, cell='all', max_samples=None, candidate_chunk_size=128, run_dir=None):
    study = Path(study_dir).resolve()
    if candidate_chunk_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError('Sample limit and chunk size must be positive')
    cells = list(EXPERIMENTS) if cell == 'all' else [cell]
    if not set(cells) <= set(EXPERIMENTS):
        raise ValueError('Unknown experiment cell')
    specs, inventory = sources(study, cells)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    run = Path(run_dir).resolve() if run_dir else study / 'comparison/exact_catalog/runs' / stamp
    if run.exists():
        raise FileExistsError(f'Run already exists: {run}')
    if run == study or study in run.parents and not (study / 'comparison/exact_catalog/runs') in run.parents:
        raise ValueError('Custom run must not overwrite study artifacts')
    snapshot = []
    for name, spec in specs.items():
        folder = run / 'inputs' / name
        sid = load_sids(spec['config']['semantic_id_path']).T.numpy()
        save_array(folder / 'catalog.npy', sid)
        catalog = pd.DataFrame({'item_id': np.arange(len(sid)),
                                **{f'c{i+1}': sid[:, i] for i in range(sid.shape[1])}})
        write_csv(folder / 'catalog.csv', catalog)
        for beam in BEAMS:
            frame = pd.read_csv(Path(spec['cache']) / f'recommendation_{beam}.csv.gz',
                dtype={'sample_key': str, 'user_id': str, 'target_item_id': str})
            frame = frame.set_index('sample_key').loc[spec['keys']].reset_index()
            if max_samples is not None:
                frame = frame.iloc[:max_samples]
            write_csv(folder / f'beam_{beam}.csv.gz', frame)
        spec['selected_samples'] = min(max_samples or spec['samples'], spec['samples'])
        for p in sorted(folder.iterdir()):
            snapshot.append({'path': str(p.relative_to(run)), 'sha256': digest(p)})
    manifest = dict(schema_version=1, status='prepared', started_at=now(), study_dir=str(study),
        cells=cells, sources=specs, input_sha256=inventory, source_sha256=code_inventory(),
        snapshots=snapshot, is_full_evaluation=max_samples is None,
        parameters=dict(max_samples=max_samples, candidate_chunk_size=candidate_chunk_size,
                        batch_size=8, bootstrap_replicates=2000, seed=42, smoke_samples=16,
                        precision='float32', tf32=False, ranking='score_desc_sid_lexicographic'),
        environment=environment(), cell_status={}, preflight={})
    manifest['identity'] = canonical_hash({k: manifest[k] for k in
        ('cells', 'input_sha256', 'source_sha256', 'parameters')})
    write_json(run / 'manifest.json', manifest)
    LOGGER.info('Prepared %s', run)
    return run


def validate_run(run, manifest, external=True):
    for entry in manifest['snapshots']:
        if digest(run / entry['path']) != entry['sha256']:
            raise ValueError('Corrupt frozen experiment snapshot')
    if external:
        if code_inventory() != manifest['source_sha256']:
            raise ValueError('Resume refused: scoring source code changed; prepare a new run')
        for path, fingerprint in manifest['input_sha256'].items():
            if digest(path) != fingerprint:
                raise ValueError(f'Resume refused: input changed: {path}')


@contextmanager
def exclusive(run):
    with (run / '.evaluation.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another evaluator/report is using this run') from None
        yield


def load_predictions(spec, frames, device='cpu'):
    result = {}
    for beam in BEAMS:
        parts = [torch.load(p, map_location='cpu', weights_only=True)
                 for p in spec['prediction_files'][str(beam)]]
        keys = [x for part in parts for x in part['sample_keys']]
        if keys != spec['keys']:
            raise ValueError('Beam payload identity differs')
        data = {key: torch.cat([p[key] for p in parts]) for key in
                ('generated_sids', 'scores', 'target_sids', 'candidate_valid', 'candidate_item_ids')}
        data['sample_keys'] = keys
        original = frames[beam]
        n = len(original)
        # CPU and CUDA log2 differ by an ULP at some ranks. Reproduce both
        # native topk ties and NDCG discounts on the original device type.
        score_tensor = data['scores'][:n].to(device)
        values = recommendation_values(score_tensor, data['generated_sids'][:n].to(device),
                                       data['target_sids'][:n].to(device), [5, 10])
        for k in (5, 10):
            data[f'native_indices_{k}'] = torch.topk(score_tensor, k, dim=-1).indices.cpu()
        for key, value in values.items():
            if not np.array_equal(value, original[key].to_numpy(dtype=np.float32)):
                raise ValueError(f'Native torch.topk metric does not reproduce {key}')
        result[beam] = data
    return result


def input_batch(path, limit=None):
    cached = torch.load(path, map_location='cpu', weights_only=True)
    n = len(cached['item_ids']) if limit is None else min(limit, len(cached['item_ids']))
    inp = SimpleNamespace(mask=cached['mask'][:n],
        transformed_sequences={k: v[:n] for k, v in cached['transformed_sequences'].items()})
    targets = cached['labels']['sequence_data'].reshape(len(cached['item_ids']), -1)[:n].long()
    return inp, cached['item_ids'][:n].long(), targets, cached['users'][:n], cached['locations'][:n]


def score_batch(model, path, catalog, prediction, frames, offset, count, chunk, check_forward=False):
    inp, target_ids, targets, users, locations = input_batch(path, count)
    n = len(target_ids)
    if not torch.equal(targets, torch.from_numpy(catalog[target_ids.numpy()])):
        raise ValueError('Cached target does not match catalog SID')
    encoded, mask = encode(model, inp)
    all_scores = catalog_scores(model, encoded, mask, torch.from_numpy(catalog), chunk).numpy()
    target_parts = candidate_scores(model, encoded, mask, targets[:, None], chunk)[:, 0].numpy()
    # Independent full-forward audit runs in the pilot; production encodes once.
    if check_forward:
        from src.analysis.long_tail.diagnostics import teacher_forcing
        logs, _ = teacher_forcing(model, SimpleNamespace(mask=inp.mask.to(encoded.device),
            transformed_sequences={k: v.to(encoded.device) for k, v in inp.transformed_sequences.items()}),
            SimpleNamespace(labels={'sequence_data': targets.to(encoded.device)}))
        reference = logs.gather(-1, targets.to(encoded.device)[..., None]).squeeze(-1).cpu().numpy()
        if np.max(np.abs(reference-target_parts)) > TOLERANCE:
            raise ValueError('Encoder-reuse scoring disagrees with existing full forward')
    rescores = {b: candidate_scores(model, encoded, mask,
        prediction[b]['generated_sids'][offset:offset+n], chunk).sum(-1).numpy() for b in BEAMS}
    rows, tops, rescored_rows = [], [], []
    mapping = {tuple(s): i for i, s in enumerate(catalog)}
    for i in range(n):
        base = frames[10].iloc[offset+i]
        if int(base.target_item_id) != int(target_ids[i]):
            raise ValueError('Target metadata differs')
        if str(base.user_id) != str(int(users[i])) or base.sample_key != f'{locations[i][0]}:{locations[i][1]}':
            raise ValueError('Cached user/sample metadata differs')
        meta = {key: base[key] for key in ('sample_key', 'user_id', 'target_item_id', 'group')}
        old = {b: frames[b].iloc[offset+i].to_dict() for b in BEAMS}
        candidates = {b: prediction[b]['generated_sids'][offset+i].numpy() for b in BEAMS}
        native = {b: prediction[b]['scores'][offset+i].numpy() for b in BEAMS}
        rs = {b: rescores[b][i] for b in BEAMS}
        for b in BEAMS:
            expected_ids = np.array([mapping.get(tuple(s), -1) for s in candidates[b]])
            if not np.array_equal(expected_ids, prediction[b]['candidate_item_ids'][offset+i].numpy()):
                raise ValueError('Frozen candidate item mapping differs from catalog')
            if not np.array_equal(expected_ids >= 0, prediction[b]['candidate_valid'][offset+i].numpy()):
                raise ValueError('Frozen candidate validity differs from catalog')
        native_orders = {b: {k:prediction[b][f'native_indices_{k}'][offset+i].numpy()
                              for k in (5, 10)} for b in BEAMS}
        result, top = diagnose(meta, all_scores[i], catalog, target_parts[i], candidates, rs, native, old,
                               mapping, native_orders)
        # Attach the preceding rescue experiment's cohorts directly from frozen flags.
        a = old[10]
        for row in result:
            b = old[row['beam']]
            common = bool(a['survival_level_1'] and b['survival_level_1'])
            row.update(l2_rescued_from_10=common and not a['survival_level_2'] and bool(b['survival_level_2']),
                complete_outside_top10=bool(b['survival_level_4']) and not bool(b['recall_at_10']),
                original_hit_lost_from_10=bool(a['recall_at_10']) and not bool(b['recall_at_10']))
        rows.extend(result)
        tops.extend({**meta, **entry} for entry in top)
        for b in BEAMS:
            for j, sid in enumerate(candidates[b]):
                rescored_rows.append({'sample_key': meta['sample_key'], 'beam': b, 'slot': j,
                    'sid': ','.join(map(str, sid)), 'native_score': float(native[b][j]),
                    'rescored_log_score': float(rs[b][j]),
                    'item_id': int(prediction[b]['candidate_item_ids'][offset+i, j])})
    return all_scores, rows, tops, rescored_rows


def evaluate(run_dir, device='cuda:0', resume=False, preflight_only=False):
    run = Path(run_dir).resolve()
    with exclusive(run):
        m = read_json(run / 'manifest.json')
        validate_run(run, m)
        if m['status'] != 'prepared' and not resume:
            raise ValueError('Existing evaluation requires --resume')
        configure_precision()
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable: no CPU fallback; source preparation remains valid')
            probe = torch.ones(1, device=device)
            torch.cuda.synchronize(device)
            m['device_probe'] = {'device': device, 'name': torch.cuda.get_device_name(device),
                'free_bytes': torch.cuda.mem_get_info(device)[0], 'allocation': float(probe.item())}
        m.update(status='running', device=device)
        write_json(run / 'manifest.json', m)
        try:
            for cell in m['cells']:
                spec = m['sources'][cell]
                folder = run / 'inputs' / cell
                frames = {b: pd.read_csv(folder / f'beam_{b}.csv.gz',
                    dtype={'sample_key': str, 'user_id': str, 'target_item_id': str}) for b in BEAMS}
                catalog = np.load(folder / 'catalog.npy', allow_pickle=False)
                prediction = load_predictions(spec, frames, device)
                cfg = dict(spec['config'], device=device)
                model, _, _, _ = model_and_data(cfg)
                model.float().eval()
                chunk = m['parameters']['candidate_chunk_size']
                stage = f'scores_{cell}'
                done = completed_shards(run, stage)
                if m['cell_status'].get(cell) == 'complete':
                    count = sum(len(read_json(p)['sample_keys']) for p in (run/'progress'/stage).glob('*.json'))
                    if count != spec['selected_samples']:
                        raise ValueError('Completed cell has truncated score shards')
                    del model
                    continue
                # Each cell must pass a separate, real full-catalog 16-sample preflight.
                if cell not in m['preflight']:
                    started = time.monotonic()
                    smoke_rows, smoke_tops = [], []
                    offset = 0
                    with torch.inference_mode():
                        for path in spec['input_files'][:2]:
                            scores, rows, tops, _ = score_batch(model, path, catalog, prediction, frames,
                                offset, min(8, spec['selected_samples']-offset), chunk, check_forward=True)
                            offset += len(scores)
                            smoke_rows.extend(rows); smoke_tops.extend(tops)
                            if offset >= min(16, spec['selected_samples']):
                                break
                    seconds = time.monotonic()-started
                    write_csv(run/'preflight'/cell/'paired.csv.gz', pd.DataFrame(smoke_rows))
                    write_csv(run/'preflight'/cell/'top50.csv.gz', pd.DataFrame(smoke_tops))
                    m['preflight'][cell] = dict(samples=offset, seconds=seconds,
                        estimated_full_seconds=seconds/offset*spec['selected_samples'],
                        peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                        score_checks='passed')
                    write_json(run/'manifest.json', m)
                    LOGGER.info('%s preflight %d samples %.1fs; estimated full %.1fh', cell, offset, seconds,
                                seconds/offset*spec['selected_samples']/3600)
                if preflight_only:
                    del model
                    continue
                offset = 0
                buffer_scores, buffer_rows, buffer_tops, buffer_rescores, buffer_keys = [], [], [], [], []
                shard = 0
                with torch.inference_mode():
                    for batch_index, path in enumerate(spec['input_files']):
                        if offset >= spec['selected_samples']:
                            break
                        count = min(8, spec['selected_samples']-offset)
                        keys = spec['keys'][offset:offset+count]
                        if shard < done:
                            buffer_keys.extend(keys)
                        else:
                            scores, rows, tops, rescores = score_batch(model, path, catalog, prediction,
                                frames, offset, count, chunk)
                            buffer_scores.append(scores); buffer_rows.extend(rows)
                            buffer_tops.extend(tops); buffer_rescores.extend(rescores); buffer_keys.extend(keys)
                        offset += count
                        if (batch_index+1) % 16 == 0 or offset == spec['selected_samples']:
                            if shard < done:
                                previous = read_json(run/'progress'/stage/f'{shard:06d}.json')
                                if previous['sample_keys'] != buffer_keys:
                                    raise ValueError('Resume shard identities differ')
                            else:
                                base = run/'cells'/cell/f'{shard:06d}'
                                files = [base.with_suffix('.npy'), base.with_suffix('.paired.csv.gz'),
                                         base.with_suffix('.top50.csv.gz'), base.with_suffix('.beam.csv.gz')]
                                save_array(files[0], np.concatenate(buffer_scores))
                                for p, records in zip(files[1:], [buffer_rows, buffer_tops, buffer_rescores]):
                                    f = pd.DataFrame(records)
                                    f.insert(0, 'dataset', cell.split('_')[0]); f.insert(0, 'cell', cell)
                                    write_csv(p, f)
                                commit_shard(run, stage, shard, buffer_keys, files)
                                LOGGER.info('%s committed %d/%d samples', cell, offset, spec['selected_samples'])
                            shard += 1
                            buffer_scores, buffer_rows, buffer_tops, buffer_rescores, buffer_keys = [], [], [], [], []
                if shard < done:
                    raise ValueError('Resume contains excess committed shards')
                m['cell_status'][cell] = 'complete'
                write_json(run/'manifest.json', m)
                del model
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
            # Verify external inputs were unchanged throughout inference.
            validate_run(run, m)
            m.update(status='preflight_complete' if preflight_only else 'evaluated', finished_at=now())
        except BaseException as ex:
            m.update(status='failed', error=f'{type(ex).__name__}: {ex}')
            raise
        finally:
            write_json(run/'manifest.json', m)
    return run
