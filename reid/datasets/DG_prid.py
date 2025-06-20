from __future__ import print_function, absolute_import
import os.path as osp
import re
import urllib
import zipfile
import os
from scipy.io import loadmat
from glob import glob
import random

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class DG_PRID(BaseImageDataset):
    dataset_dir = "prid_2011"
    dataset_name = 'prid'
    _junk_pids = list(range(201, 750))

    def __init__(self, root='', verbose=True, split_id=0, **kwargs):
        super(DG_PRID, self).__init__()

        if isinstance(root, list):
            split_id = root[1]
            self.root = root[0]
        else:
            self.root = root
            split_id = 0
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        self.cam_a_dir = os.path.join(
            self.dataset_dir, 'single_shot', 'cam_a'
        )
        self.cam_b_dir = os.path.join(
            self.dataset_dir, 'single_shot', 'cam_b'
        )
        self.split_path = os.path.join(self.dataset_dir, 'splits_single_shot.json')

        required_files = [
            self.dataset_dir,
            self.cam_a_dir,
            self.cam_b_dir
        ]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = self.read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                    .format(split_id,
                            len(splits) - 1)
            )
        split = splits[split_id]

        train, query, gallery = self.process_split(split)

        if verbose:
            print("=> PRID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)




    def prepare_split(self):
        if not os.path.exists(self.split_path):
            print('Creating splits ...')

            splits = []
            for _ in range(10):
                # randomly sample 100 IDs for train and use the rest 100 IDs for test
                # (note: there are only 200 IDs appearing in both views)
                pids = [i for i in range(1, 201)]
                train_pids = random.sample(pids, 100)
                train_pids.sort()
                test_pids = [i for i in pids if i not in train_pids]
                split = {'train': train_pids, 'test': test_pids}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            self.write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    def process_split(self, split):
        train_pids = split['train']
        test_pids = split['test']

        train_pid2label = {pid: label for label, pid in enumerate(train_pids)}

        # train
        train = []
        for pid in train_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            pid = train_pid2label[pid]
            img_a_path = os.path.join(self.cam_a_dir, img_name)
            train.append((img_a_path, pid, 0))
            img_b_path = os.path.join(self.cam_b_dir, img_name)
            train.append((img_b_path, pid, 1))

        # query and gallery
        query, gallery = [], []
        for pid in test_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_a_path = os.path.join(self.cam_a_dir, img_name)
            query.append((img_a_path, pid, 0))
            img_b_path = os.path.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
        for pid in range(201, 750):
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_b_path = os.path.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))

        return train, query, gallery

    def read_json(self, fpath):
        import json
        """Reads json file from a path."""
        with open(fpath, 'r') as f:
            obj = json.load(f)
        return obj

    def write_json(self, obj, fpath):
        import json
        """Writes to a json file."""
        self.mkdir_if_missing(os.path.dirname(fpath))
        with open(fpath, 'w') as f:
            json.dump(obj, f, indent=4, separators=(',', ': '))

    def mkdir_if_missing(self, dirname):
        import errno
        """Creates dirname if it is missing."""
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise