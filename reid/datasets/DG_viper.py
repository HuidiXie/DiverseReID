from glob import glob
import os.path as osp
import re
import warnings
import pdb
from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
import pdb
import os

class DG_VIPeR(BaseImageDataset):
    dataset_dir = "viper"
    dataset_name = "viper"

    def __init__(self, root='',verbose=True, **kwargs):
        super(DG_VIPeR, self).__init__()

        self.root = root
        type = 'split_1a'

        self.train_dir = os.path.join(self.root, self.dataset_dir, type, 'train')
        self.query_dir = os.path.join(self.root, self.dataset_dir, type, 'query')
        self.gallery_dir = os.path.join(self.root, self.dataset_dir, type, 'gallery')

        required_files = [
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_train(self.train_dir, is_train = True)
        query = self.process_train(self.query_dir, is_train = False)
        gallery = self.process_train(self.gallery_dir, is_train = False)

        if verbose:
            print("=> VIPeR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)



    def process_train(self, path, is_train = True):
        data = []
        img_list = glob(os.path.join(path, '*.png'))
        for img_path in img_list:
            img_name = img_path.split('/')[-1] # p000_c1_d045.png
            split_name = img_name.split('_')
            pid = int(split_name[0][1:])
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            camid = int(split_name[1][1:])
            # dirid = int(split_name[2][1:-4])
            data.append((img_path, pid, camid))

        return data