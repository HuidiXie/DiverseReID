from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile
import os
from ..utils.data import BaseImageDataset


class DG_CUHK02(BaseImageDataset):
    dataset_dir = "cuhk02"
    dataset_name = "cuhk02"
    cam_pairs = ['P1', 'P2', 'P3', 'P4', 'P5']

    def __init__(self, root='', verbose=True, **kwargs):
        super(DG_CUHK02, self).__init__()
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)
        query = []
        gallery = []

        self.train = train
        self.query = query
        self.gallery = gallery
        self.mix_dataset = self.train

        self.num_mix_pids, self.num_mix_imgs, self.num_mix_cams = self.get_imagedata_info(self.train)

        if verbose:
            print("=> CUHK02 loaded")
            self.print_dataset_statistics(train, query, gallery)

    def process_train(self, train_path):

        num_train_pids, camid = 0, 0
        data, query, gallery = [], [], []

        for cam_pair in self.cam_pairs:
            cam_pair_dir = osp.join(train_path, cam_pair)

            cam1_dir = osp.join(cam_pair_dir, 'cam1')
            cam2_dir = osp.join(cam_pair_dir, 'cam2')

            impaths1 = glob.glob(osp.join(cam1_dir, '*.png'))
            impaths2 = glob.glob(osp.join(cam2_dir, '*.png'))

            pids1 = [
                osp.basename(impath).split('_')[0] for impath in impaths1
            ]
            pids2 = [
                osp.basename(impath).split('_')[0] for impath in impaths2
            ]
            pids = set(pids1 + pids2)
            pid2label = {
                pid: label + num_train_pids
                for label, pid in enumerate(pids)
            }

            # add images to train from cam1
            for impath in impaths1:
                pid = osp.basename(impath).split('_')[0]
                pid = pid2label[pid]
                data.append((impath, pid, camid))
            camid += 1

            # add images to train from cam2
            for impath in impaths2:
                pid = osp.basename(impath).split('_')[0]
                pid = pid2label[pid]
                data.append((impath, pid, camid))
            camid += 1
            num_train_pids += len(pids)

        return data

