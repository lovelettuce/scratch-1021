"""python -m src.analysis.exact_catalog {prepare,evaluate,report,run}."""
import argparse
import logging

from .experiment import evaluate, prepare
from .report import report
from .parallel import evaluate_parallel
from src.experiments.rq_cross_validation.common import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--study-dir', required=True)
        p.add_argument('--cell', choices=['all', *EXPERIMENTS], default='all')
        p.add_argument('--max-samples', type=int)
        p.add_argument('--candidate-chunk-size', type=int, default=128)
        p.add_argument('--run-dir', help='New output directory (must not exist)')
        if name == 'run':
            p.add_argument('--device', default='cuda:0')
            p.add_argument('--gpus', help='Distinct physical GPUs, one per selected cell; overrides --device')
    p = sub.add_parser('evaluate')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--gpus', help='Parallel full scoring after all preflights; overrides --device')
    p = sub.add_parser('report')
    p.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    values = vars(args).copy(); command = values.pop('command')
    if command == 'prepare':
        print(prepare(**values))
    elif command == 'evaluate':
        gpus = values.pop('gpus')
        if gpus:
            if values.pop('preflight_only'):
                parser.error('--gpus is for full scoring; run preflight separately')
            values.pop('device')
            print(evaluate_parallel(gpus=gpus, **values))
        else:
            print(evaluate(**values))
    elif command == 'report':
        print(report(**values))
    else:
        device = values.pop('device')
        gpus = values.pop('gpus')
        run = prepare(**values)
        # Gate the full evaluation on all selected models passing the real pilot.
        if gpus:
            import os
            import subprocess
            import sys
            from .parallel import assignment
            from src.analysis.long_tail.io import read_json
            assigned = assignment(read_json(run/'manifest.json')['cells'], gpus)
            subprocess.run([sys.executable, '-m', 'src.analysis.exact_catalog', 'evaluate',
                '--run-dir', str(run), '--device', 'cuda:0', '--preflight-only'],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=next(iter(assigned.values()))), check=True)
            evaluate_parallel(run, gpus=gpus, resume=True)
        else:
            evaluate(run, device=device, preflight_only=True)
            evaluate(run, device=device, resume=True)
        print(report(run))


if __name__ == '__main__':
    main()
