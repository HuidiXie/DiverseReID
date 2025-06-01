from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob


class RandPerson(object):
    # dataset_dir='datasets/randoerson_subset/randoerson_subset'
    # 初始化函数，root 是数据集的根目录，combine_all 是一个标志，默认值为 True
    def __init__(self, root, combine_all=True):

        self.images_dir = osp.join(root,'randperson_subset')#图像数据所在的根目录路径
        self.img_path = 'randperson_subset'## 图像文件夹的子路径，假设图像存放在 randperson_subset 目录下
        self.train_path = self.img_path

        self.train = []#you

        self.num_mix_pids = 0
        self.has_time_info = True
        self.load()
        self.mix_dataset = self.train

    def preprocess(self):
        # 获取所有训练图像路径，使用通配符匹配以 '*.jpg' 结尾的文件
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*g')))

        data = []# 用于存储图像数据的列表
        all_pids = {}# 用于存储每个 ID 的映射关系
        # 摄像头编号的偏移量
        camera_offset = [0, 2, 4, 6, 8, 9, 10, 12, 13, 14, 15]
        # 每个相机序列的帧数偏移量
        frame_offset = [0, 160000, 340000,490000, 640000, 1070000, 1330000, 1590000, 1890000, 3190000, 3490000]
        fps = 24# 帧率，假设是 24 fps
        # 遍历所有的图像路径
        for fpath in fpaths:
            # 获取图像文件名
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            fields = fname.split('_')
            pid = int(fields[0])# 获取图像中的人员 ID（即文件名的第一个字段）
            # 如果该人员 ID 不在映射表中，给它分配一个新的 ID
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            # 更新人员 ID，映射到新的 ID
            pid = all_pids[pid]  # relabel
            # 获取摄像头 ID（通过解析文件名中的相机字段）
            camid = camera_offset[int(fields[1][1:])] + int(fields[2][1:])  # make it starting from 0
            # 获取时间信息，时间等于（帧偏移量 + 当前帧的偏移量）除以帧率
            # time = (frame_offset[int(fields[1][1:])] + int(fields[3][1:7])) / fps
            # 将文件名、人员 ID、摄像头 ID、时间信息加入数据列表
            # data.append((fname, pid, camid, time))
            # 将文件名、人员 ID、摄像头 ID加入数据列表
            data.append((fpath, pid, camid))
            # data.append(fpath, pid, camid)
            # 打印每张图像的相关信息
            # print(fname, pid, camid, time)
        return data, int(len(all_pids)) # train(images),num_mix_pids(ids)

    def load(self):
        self.train, self.num_mix_pids = self.preprocess()

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  all    | {:5d} | {:8d}"
              .format(self.num_mix_pids, len(self.train)))
