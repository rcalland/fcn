#!/usr/bin/env python

import argparse
import datetime
import os.path as osp

import chainer

import fcn

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='jsk', choices=('jsk', 'mit'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-o', '--out')
    parser.add_argument('--resume')
    args = parser.parse_args()

    if args.dataset == 'jsk':
        dataset_class = datasets.APC2016JSKDataset
    else:
        dataset_class = datasets.APC2016MITDataset
    out = args.out
    resume = args.resume
    if out is None:
        if resume:
            out = osp.dirname(resume)
        else:
            timestamp = datetime.datetime.now().isoformat()
            out = osp.join('logs', timestamp)
    gpu = args.gpu
    max_iter = 100000

    trainer = fcn.trainers.fcn32s.get_trainer(
        dataset_class=dataset_class,
        gpu=gpu,
        max_iter=max_iter,
        out=out,
        resume=resume,
        interval_log=10,
        interval_eval=100,
        optimizer=chainer.optimizers.Adam(alpha=1e-5),
    )
    trainer.run()


if __name__ == '__main__':
    main()
