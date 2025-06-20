from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob
import os
import re
import urllib
import zipfile
from scipy.io import loadmat
from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json



class DG_GRID(BaseImageDataset):
    dataset_dir = "GRID"
    dataset_name = 'grid'

    def __init__(self, root='', verbose=True, split_id = 0, **kwargs):
        super(DG_GRID, self).__init__()

        if isinstance(root, list):
            split_id = root[1]
            self.root = root[0]
        else:
            self.root = root
            split_id = 0
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        self.probe_path = os.path.join(
            self.dataset_dir, 'probe'
        )
        self.gallery_path = os.path.join(
            self.dataset_dir, 'gallery'
        )
        self.split_mat_path = os.path.join(
            self.dataset_dir, 'features_and_partitions.mat'
        )
        self.split_path = os.path.join(self.dataset_dir, 'splits.json')

        required_files = [
            self.dataset_dir, self.probe_path, self.gallery_path,
            self.split_mat_path
        ]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = self.read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        if verbose:
            print("=> GRID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)




    def prepare_split(self):
        if not os.path.exists(self.split_path):
            print('Creating 10 random splits')
            split_mat = loadmat(self.split_mat_path)
            trainIdxAll = split_mat['trainIdxAll'][0] # length = 10
            probe_img_paths = sorted(
                glob(os.path.join(self.probe_path, '*.jpeg'))
            )
            gallery_img_paths = sorted(
                glob(os.path.join(self.gallery_path, '*.jpeg'))
            )

            splits = []
            for split_idx in range(10):
                train_idxs = trainIdxAll[split_idx][0][0][2][0].tolist()
                assert len(train_idxs) == 125
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, query, gallery = [], [], []

                # processing probe folder
                for img_path in probe_img_paths:
                    img_name = os.path.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx], camid))
                    else:
                        query.append((img_path, img_idx, camid))

                # process gallery folder
                for img_path in gallery_img_paths:
                    img_name = os.path.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx], camid))
                    else:
                        gallery.append((img_path, img_idx, camid))

                split = {
                    'train': train,
                    'query': query,
                    'gallery': gallery,
                    'num_train_pids': 125,
                    'num_query_pids': 125,
                    'num_gallery_pids': 900
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            self.write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))


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