"""CLI for support preparation, replay and offline reports."""
import argparse
import logging
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','evaluate','run'):
        q=sub.add_parser(name);q.add_argument('--config',required=True);q.add_argument('--max-samples',type=int)
        if name!='prepare':q.add_argument('--resume')
    q=sub.add_parser('report');q.add_argument('--run-dir',required=True)
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.command=='report':
        from .report import report
        print(report(args.run_dir))
        return
    from .common import load_config
    from .engine import prepare,evaluate
    cfg=load_config(args.config,args.max_samples)
    if args.command=='prepare':
        _,_,folder,baseline,_,keys=prepare(cfg)
        print(f'support={folder}\nbaselines={baseline}\nsamples={len(keys)}')
    else:
        folder=evaluate(cfg,args.resume)
        if args.command=='run':
            from .report import report
            report(folder)
        print(folder)


if __name__=='__main__':main()
