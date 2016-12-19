import glob
import itertools
import os
import os.path as osp

import chainer
import numpy as np
import scipy.misc

from base import APC2016DatasetBase


class APC2016MITDataset(APC2016DatasetBase):

    img_axis_size = 216

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        self.dataset_dir = chainer.dataset.get_dataset_directory('apc2016mit')
        if data_type == 'train':
            self.data_ids = self._get_data_ids_training()
        else:
            self.data_ids = self._get_data_ids_benchmark()

    def _get_data_ids_benchmark(self):
        def yield_data_id_from_scene_dir(scene_dir):
            for i_frame in itertools.count():
                rgb_file = osp.join(
                    scene_dir, 'frame-{:06}.color.png'.format(i_frame))
                segm_file = osp.join(
                    scene_dir, 'segm/frame-{:06}.segm.png'.format(i_frame))
                if not (osp.exists(rgb_file) and osp.exists(segm_file)):
                    break
                data_id = (rgb_file, segm_file)
                yield data_id

        data_ids = []
        # office
        contain_dir = osp.join(self.dataset_dir, 'benchmark/office/test')
        for loc in ['shelf', 'tote']:
            loc_dir = osp.join(contain_dir, loc)
            for scene_dir in os.listdir(loc_dir):
                scene_dir = osp.join(loc_dir, scene_dir)
                data_ids += list(yield_data_id_from_scene_dir(scene_dir))
        # warehouse
        contain_dir = osp.join(self.dataset_dir, 'benchmark/warehouse')
        for sub in ['practice', 'competition']:
            sub_contain_dir = osp.join(contain_dir, sub)
            for loc in ['shelf', 'tote']:
                loc_dir = osp.join(sub_contain_dir, loc)
                for scene_dir in os.listdir(loc_dir):
                    scene_dir = osp.join(loc_dir, scene_dir)
                    data_ids += list(yield_data_id_from_scene_dir(scene_dir))
        return data_ids

    def _get_data_ids_training(self):
        def yield_data_id_from_scene_dir(scene_dir):
            for i_frame in itertools.count():
                rgb_file = osp.join(
                    scene_dir, 'frame-{:06}.color.png'.format(i_frame))
                mask_file = osp.join(
                    scene_dir,
                    'masks/frame-{:06}.mask.png'.format(i_frame))
                if not (osp.exists(rgb_file) and
                        osp.exists(mask_file)):
                    break
                data_id = (rgb_file, mask_file, cls_id)
                yield data_id

        data_ids = []
        for loc in ['shelf', 'tote']:
            for cls_id, cls in enumerate(self.label_names):
                if cls == 'background':
                    continue
                cls_dir = osp.join(self.dataset_dir, 'training', loc, cls)
                for scene_dir in os.listdir(cls_dir):
                    scene_dir = osp.join(cls_dir, scene_dir)
                    data_ids += list(yield_data_id_from_scene_dir(scene_dir))
        return data_ids

    def __len__(self):
        return len(self.data_ids)

    def get_example(self, i):
        data_id = self.data_ids[i]

        def expand_to_imsize(img, imsize, fill=0):
            shape = imsize
            if img.ndim == 3:
                shape = (imsize[0], imsize[1], img.shape[-1])
            h, w = img.shape[:2]
            img_exp = np.zeros(shape, dtype=img.dtype)
            img_exp.fill(fill)
            img_exp[:h, :w] = img
            return img_exp

        rgb_file = data_id[0]
        img = scipy.misc.imread(rgb_file, mode='RGB')
        height, width = img.shape[:2]
        scale = 1. * self.img_axis_size / max(height, width)
        img = scipy.misc.imresize(img, size=scale, interp='bilinear')
        imsize = (self.img_axis_size, self.img_axis_size)
        img = expand_to_imsize(img, imsize)

        if len(data_id) == 3:
            # annotation by mask file
            _, mask_file, cls_id = data_id
            mask = scipy.misc.imread(mask_file, mode='L')
            assert mask.shape == (height, width)
            mask = scipy.misc.imresize(mask, size=scale, interp='nearest')
            # mask -> label
            label = np.zeros(mask.shape, dtype=np.int32)
            label[mask > 127] = cls_id
            label = expand_to_imsize(label, imsize)
        else:
            # annotation by label file
            _, segm_file = data_id
            # Label value is multiplied by 9:
            #   ex) 0: 0/9=0 (background), 54: 54/9=6 (dasani_bottle_water)
            label = scipy.misc.imread(segm_file, mode='L') / 9
            assert label.shape == (height, width)
            label = scipy.misc.imresize(label, size=scale, interp='nearest')
            label = label.astype(np.int32)
            label = expand_to_imsize(label, imsize, fill=-1)

        return self.img_to_datum(img), label


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    dataset = APC2016MITDataset('val')
    for i in xrange(len(dataset)):
        fig = plt.figure()
        img_viz = dataset.visualize_example(i)
        plt.imshow(img_viz)
        fig.canvas.manager.window.attributes('-topmost', 1)
        fig.canvas.manager.window.attributes('-topmost', 0)
        plt.show()
