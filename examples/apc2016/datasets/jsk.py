import glob
import os
import os.path as osp
import re

import chainer
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

from base import APC2016DatasetBase


class APC2016JSKDataset(APC2016DatasetBase):

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        ids = self._get_ids()
        iter_train, iter_val = train_test_split(
            ids, test_size=0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if data_type == 'train' else iter_val

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        # APC2016rbo
        dataset_dir = chainer.dataset.get_dataset_directory(
            'apc2016/APC2016rbo')
        for img_file in os.listdir(dataset_dir):
            if not re.match(r'^.*_[0-9]*_bin_[a-l].jpg$', img_file):
                continue
            data_id = osp.splitext(img_file)[0]
            ids.append(osp.join('rbo', data_id))
        # APC2016seg
        dataset_dir = chainer.dataset.get_dataset_directory(
            'apc2016/APC2016JSKseg/annotated')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join('JSKseg', data_id))
        return ids

    def get_example(self, i):
        ann_id, data_id = self.ids[i].split('/')
        assert ann_id in ('rbo', 'JSKseg')

        if ann_id == 'rbo':
            dataset_dir = chainer.dataset.get_dataset_directory(
                'apc2016/APC2016rbo')

            img_file = osp.join(dataset_dir, data_id + '.jpg')
            img = scipy.misc.imread(img_file)
            datum = self.img_to_datum(img)

            label = np.zeros(img.shape[:2], dtype=np.int32)

            shelf_bin_mask_file = osp.join(
                dataset_dir, data_id + '.pbm')
            shelf_bin_mask = scipy.misc.imread(shelf_bin_mask_file, mode='L')
            label[shelf_bin_mask < 127] = -1

            mask_glob = osp.join(dataset_dir, data_id + '_*.pbm')
            for mask_file in glob.glob(mask_glob):
                mask_id = osp.splitext(osp.basename(mask_file))[0]
                mask = scipy.misc.imread(mask_file, mode='L')
                label_name = mask_id[len(data_id + '_'):]
                label_value = self.label_names.index(label_name)
                label[mask > 127] = label_value
        else:
            dataset_dir = chainer.dataset.get_dataset_directory(
                'apc2016/APC2016JSKseg/annotated')

            img_file = osp.join(dataset_dir, data_id, 'image.png')
            img = scipy.misc.imread(img_file)
            datum = self.img_to_datum(img)

            label_file = osp.join(dataset_dir, data_id, 'label.png')
            label = scipy.misc.imread(label_file, mode='L')
            label = label.astype(np.int32)
            label[label == 255] = -1
        return datum, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = APC2016JSKDataset('val')
    for i in xrange(len(dataset)):
        labelviz = dataset.visualize_example(i)
        plt.imshow(labelviz)
        plt.show()
