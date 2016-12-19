#!/usr/bin/env python

import os.path as osp

import chainer

import fcn.data
import fcn.util


def main():

    dataset_dir = chainer.dataset.get_dataset_directory('apc2016')

    path = osp.join(dataset_dir, 'APC2016rbo.tgz')
    fcn.data.cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vSV9oLTd1U2I3TDg',
        path=path,
    )
    fcn.util.extract_file(path, to_directory=dataset_dir)

    path = osp.join(dataset_dir, 'APC2016JSKseg/annotated.tgz')
    fcn.data.cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vaExFU1AxWHlMdTg',
        path=path,
    )
    fcn.util.extract_file(path, to_directory=dataset_dir)


    dataset_dir = chainer.dataset.get_dataset_directory('apc2016mit')
    path = osp.join(dataset_dir, 'training.tgz')
    fcn.data.cached_download(
        url='https://drive.google.com/uc?id=0B4mCa-2YGnp7ZEMwcW5rcVBpeG8',
        path=path,
        md5='14298453fcdbcaaa6b3fa8439a486e6d',
    )
    path = osp.join(dataset_dir, 'benchmark.tgz')
    fcn.data.cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vOHhlbENra2REZDA'
        path=path
        md5='15a6ff714fafb3950c0b0ff0161d6ef6',
    )


if __name__ == '__main__':
    main()
