"""Run with python -m src.analysis.beam_rescue --study-dir STUDY."""
import argparse
import logging

from .report import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', required=True)
    parser.add_argument('--output-dir', help='New, empty output directory; default: STUDY/comparison/beam_rescue')
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    print(run(args.study_dir, args.output_dir, args.bootstrap_replicates, args.seed))


if __name__ == '__main__':
    main()
