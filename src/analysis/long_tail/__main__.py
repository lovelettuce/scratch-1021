"""Run with python -m src.analysis.long_tail --help."""
import argparse
import logging
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')


def main():
    parser = argparse.ArgumentParser(description='Frozen long-tail groups, resumable inference and offline reports')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'evaluate', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--max-samples', type=int)
        if name != 'prepare':
            p.add_argument('--resume', help='Existing run directory; requires identical config and fingerprints')
    p = sub.add_parser('report')
    p.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.command == 'report':
        from .report import report
        report(args.run_dir)
    else:
        from .io import load_config, provenance
        cfg = load_config(args.config, args.max_samples)
        if args.command == 'prepare':
            from .data import prepare
            print(prepare(cfg, provenance(cfg)))
        else:
            from .engine import evaluate
            folder = evaluate(cfg, args.resume)
            if args.command == 'run':
                from .report import report
                report(folder)
            print(folder)


if __name__ == '__main__':
    main()
