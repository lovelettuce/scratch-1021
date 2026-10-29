"""Sequential stages for one matrix cell; cells share a queued GPU pool, each active cell owns one GPU."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from omegaconf import OmegaConf

from .common import ROOT, digest, lock, now, read_json, save_yaml, stage, write_json
from .configuration import analysis_config, export_config, recommender_config, tokenizer_config
from .prepare import validate_sid

LOGGER = logging.getLogger(__name__)


def subprocess_logged(command, log_path, environment=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment or os.environ.copy())
        try:
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    write_json(log_path.with_suffix('.command.json'), {'argv': command, 'started_at': started,
               'finished_at': now(), 'returncode': returncode,
               'cuda_visible_devices': (environment or os.environ).get('CUDA_VISIBLE_DEVICES'),
               'master_port': (environment or os.environ).get('MASTER_PORT')})
    if returncode:
        raise RuntimeError(f'Command exited {returncode}; see {log_path}')


def run_model(cfg, out, recipe, kind, resume):
    out = Path(out)
    recipe_path = out / 'launch_config.yaml'
    if recipe_path.exists() and OmegaConf.to_container(OmegaConf.load(recipe_path)) != OmegaConf.to_container(recipe):
        raise ValueError(f'Launch configuration changed: {recipe_path}')
    save_yaml(recipe_path, recipe)
    command = [cfg['python'], '-m', 'src.experiments.rq_cross_validation.worker', kind, '--config', str(recipe_path)]
    if resume:
        command.append('--resume')
    subprocess_logged(command, out / 'logs/process.log')
    result_path = out / ('training_result.json' if kind == 'train' else 'inference_result.json')
    return read_json(result_path), result_path


def run_analysis(cfg, cell, out, training, checkpoint, sid, source=None, limit=None):
    module = 'long_tail' if source is None else 'second_layer'
    recipe = analysis_config(cfg, cell, out, training, checkpoint, sid, source, limit)
    path = Path(out) / 'analysis_config.yaml'
    save_yaml(path, recipe)
    run = Path(out) / 'runs' / recipe['run_id']
    command = [cfg['python'], '-m', f'src.analysis.{module}', 'run', '--config', str(path)]
    if (run / 'manifest.json').exists():
        command += ['--resume', str(run)]
    subprocess_logged(command, Path(out) / 'logs/process.log')
    manifest = read_json(run / 'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError(f'Analysis did not complete: {run}')
    return run


def run_cell(study, cell, selected_stage='all'):
    study = Path(study).resolve()
    root_manifest = read_json(study / 'study_manifest.json')
    cfg = root_manifest['config']
    out = study / 'experiments' / cell
    dataset = cfg['experiments'][cell]['dataset']
    shared = read_json(study / 'shared' / dataset / 'manifest.json')
    embedding = read_json(study / 'shared' / dataset / 'embedding.json')
    if digest(embedding['path']) != embedding['sha256']:
        raise ValueError('Shared embedding has changed')
    identity = {'study': root_manifest['identity'], 'cell': cell, 'embedding': embedding['sha256']}
    with lock(out / '.pipeline.lock'):
        write_json(out / 'status.json', {'status': 'running', 'stage': selected_stage, 'started_at': now(),
                   'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'master_port': os.environ.get('MASTER_PORT')})
        try:
            def train_tokenizer():
                folder = out / 'tokenizer_train'
                result, receipt = run_model(cfg, folder, tokenizer_config(cfg, cell, folder, embedding['path']), 'train', True)
                return result, [receipt, result['best_checkpoint'], folder / '.hydra/config.yaml']
            if selected_stage in ('all', 'tokenizer'):
                token = stage(out, 'tokenizer_train', identity, train_tokenizer)
                def export():
                    folder = out / 'sid_export'
                    recipe = export_config(cfg, cell, folder, out / 'tokenizer_train', token['best_checkpoint'], embedding['path'])
                    _, receipt = run_model(cfg, folder, recipe, 'inference', False)
                    path = folder / 'pickle/merged_predictions_tensor.pt'
                    sid = validate_sid(path, shared['catalog_items'], cfg['codebook_width'])
                    result = {'path': str(path), 'sha256': digest(path), 'shape': list(sid.shape),
                              'semantic_code_usage': [int(sid[i].unique().numel()) for i in range(3)],
                              'disambiguation_max': int(sid[3].max())}
                    write_json(folder / 'sid_validation.json', result)
                    return result, [path, receipt, folder / 'sid_validation.json']
                stage(out, 'sid_export', {**identity, 'tokenizer_checkpoint': token['best_checkpoint_sha256']}, export)
            sid_state = out / 'progress/sid_export.json'
            if selected_stage in ('all', 'recommender', 'analysis'):
                if not sid_state.exists() or read_json(sid_state)['status'] != 'complete':
                    raise ValueError('Run --stage tokenizer before recommender/analysis')
                sid = read_json(sid_state)['result']['path']
                validate_sid(sid, shared['catalog_items'], cfg['codebook_width'])
                if digest(sid) != read_json(sid_state)['result']['sha256']:
                    raise ValueError('Exported SID changed')
                rec_identity = {**identity, 'sid': digest(sid)}
                def train_recommender():
                    folder = out / 'recommender_train'
                    result, receipt = run_model(cfg, folder, recommender_config(cfg, cell, folder, sid), 'train', True)
                    return result, [receipt, result['best_checkpoint'], folder / '.hydra/config.yaml']
                if selected_stage in ('all', 'recommender'):
                    stage(out, 'recommender_train', rec_identity, train_recommender)
                if selected_stage in ('all', 'analysis'):
                    rec_state = read_json(out / 'progress/recommender_train.json')
                    if rec_state['status'] != 'complete':
                        raise ValueError('Recommender training is incomplete')
                    best = rec_state['result']['best_checkpoint']
                    if digest(best) != rec_state['result']['best_checkpoint_sha256']:
                        raise ValueError('Best recommender checkpoint changed')
                    training = out / 'recommender_train'
                    analysis_identity = {**rec_identity, 'checkpoint': digest(best)}
                    def long_tail(smoke=False):
                        folder = out / 'analysis' / ('smoke_long_tail' if smoke else 'long_tail')
                        run = run_analysis(cfg, cell, folder, training, best, sid, limit=cfg['analysis']['smoke_samples'] if smoke else None)
                        m = read_json(run / 'manifest.json')
                        frozen = pd.read_csv(Path(m['dataset_dir']) / 'items.csv', keep_default_na=False)
                        expected = pd.read_csv(study / 'shared' / dataset / 'items.csv', keep_default_na=False)
                        pd.testing.assert_frame_equal(frozen, expected, check_dtype=False)
                        if not smoke and m['stages']['test_inputs']['samples'] != shared['splits']['testing']['valid_next_item_records']:
                            raise ValueError('Full test coverage differs from raw dataset scan')
                        return {'run': str(run)}, [run / 'manifest.json', run / 'tables/summary.json', run / 'report.md']
                    stage(out, 'smoke_long_tail', analysis_identity, lambda: long_tail(True))
                    first = stage(out, 'long_tail', analysis_identity, long_tail)['run']
                    def second_layer(smoke=False):
                        folder = out / 'analysis' / ('smoke_second_layer' if smoke else 'second_layer')
                        run = run_analysis(cfg, cell, folder, training, best, sid, source=first,
                                           limit=cfg['analysis']['smoke_samples'] if smoke else None)
                        return {'run': str(run)}, [run / 'manifest.json', run / 'tables/summary.json', run / 'report.md']
                    stage(out, 'smoke_second_layer', analysis_identity, lambda: second_layer(True))
                    stage(out, 'second_layer', analysis_identity, second_layer)
            write_json(out / 'status.json', {'status': 'complete' if selected_stage == 'all' else 'stage_complete',
                       'stage': selected_stage, 'finished_at': now()})
        except BaseException as ex:
            write_json(out / 'status.json', {'status': 'failed', 'stage': selected_stage,
                       'error': f'{type(ex).__name__}: {ex}', 'finished_at': now()})
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    p.add_argument('--cell', required=True)
    p.add_argument('--stage', default='all')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run_cell(args.study, args.cell, args.stage)


if __name__ == '__main__':
    main()
