from __future__ import print_function, absolute_import
import argparse
import os.path as osp
from glob import glob
import random
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from reid import datasets
from reid import models
from reid.models.memory import MemoryClassifier
from reid.trainers import Trainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.solver import WarmupMultiStepLR

start_epoch = best_mAP = 0
best_mAP_ema = 0


def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset


def get_mix_train_loader(args, dataset, height, width, batch_size, workers,
                         num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer])

    train_set = sorted(dataset.mix_dataset) if trainset is None else sorted(trainset)
    sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
    rmgs_flag = False
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=rmgs_flag, pin_memory=True, drop_last=True), length=None)
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)

def main_worker(args):
    global start_epoch, best_mAP, best_mAP_ema
    start_time = time.monotonic()

    cudnn.benchmark = True
    logPath = osp.join(args.logs_dir,
                       args.dataset_src1 + '+' + args.dataset_src2 + '+' + args.dataset_src3  + '->' + args.dataset)

    sys.stdout = Logger(osp.join(logPath, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load datasets")
    dataset_src1 = get_data(args.dataset_src1, args.data_dir)
    dataset_src2 = get_data(args.dataset_src2, args.data_dir)
    dataset_src3 = get_data(args.dataset_src3, args.data_dir)
    # dataset_src4 = get_data(args.dataset_src4, args.data_dir)  # If four source domains are trained
    dataset = get_data(args.dataset, args.data_dir)

    num_classes1 = dataset_src1.num_mix_pids
    num_classes2 = dataset_src2.num_mix_pids
    num_classes3 = dataset_src3.num_mix_pids
    # num_classes4 = dataset_src4.num_mix_pids

    # num_classes = [num_classes1, num_classes2, num_classes3,num_classes4] # If four source domains
    num_classes = [num_classes1, num_classes2, num_classes3]
    args.num_classes = num_classes

    print(' number classes = ', num_classes)
    # datasets_src = [dataset_src1, dataset_src2, dataset_src3,dataset_src4]
    datasets_src = [dataset_src1, dataset_src2, dataset_src3]

    print('Using train set and test set for training!')
    train_loader_src1 = get_mix_train_loader(args, dataset_src1, args.height, args.width,
                                             args.batch_size, args.workers, args.num_instances, iters)
    train_loader_src2 = get_mix_train_loader(args, dataset_src2, args.height, args.width,
                                             args.batch_size, args.workers, args.num_instances, iters)
    train_loader_src3 = get_mix_train_loader(args, dataset_src3, args.height, args.width,
                                             args.batch_size, args.workers, args.num_instances, iters)

    # train_loader_src4 = get_mix_train_loader(args, dataset_src4, args.height, args.width,
    #                                          args.batch_size, args.workers, args.num_instances, iters)
    # train_loader = [train_loader_src1, train_loader_src2, train_loader_src3,train_loader_src4]
    train_loader = [train_loader_src1, train_loader_src2, train_loader_src3]
    test_loader = get_test_loader(dataset, args.height, args.width, args.test_batch_size, args.workers)
    # Create model
    model = create_model(args)
    print(model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.module.parameters()) / 1000000.0))

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        checkpoint = load_checkpoint(osp.join(logPath, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
        return

    print("==> Initialize source-domain class centroids and memorys")
    source_centers_all = []
    memories = []

    for dataset_i in range(len(datasets_src)):
        dataset_source = datasets_src[dataset_i]
        sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                              args.test_batch_size, args.workers,
                                              testset=sorted(dataset_source.mix_dataset))
        source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
        sour_fea_dict = collections.defaultdict(list)

        for f, pid, _ in sorted(dataset_source.mix_dataset):                        # img_path, pid, camid
            sour_fea_dict[pid].append(source_features[f].unsqueeze(0))

        source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in sorted(sour_fea_dict.keys())]
        source_centers = torch.stack(source_centers, 0)  # pid,2048
        source_centers = F.normalize(source_centers, dim=1).cuda()
        source_centers_all.append(source_centers)

        curMemo = MemoryClassifier(2048, source_centers.shape[0],
                                   temp=args.temp, momentum=args.momentum).cuda()

        curMemo.features = source_centers.repeat(2,1)
        curMemo.labels = torch.arange(num_classes[dataset_i]).cuda()
        curMemo = nn.DataParallel(curMemo)
        memories.append(curMemo)

        del source_centers, sour_cluster_loader, sour_fea_dict

    params = [{"params": [value]} for value in model.module.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=[30, 50], gamma=0.1, warmup_factor=0.1,
                                     warmup_iters=10, warmup_method="linear")
    trainer = Trainer(args, model, memories)

    for epoch in range(args.epochs):
        # Calculate distance
        print('==> start training epoch {} \t ==> learning rate = {}'.format(epoch, optimizer.param_groups[0]['lr']))
        torch.cuda.empty_cache()
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=args.iters)

        if (epoch + 1) % 1 == 0 or (epoch == args.epochs - 1):
            if (epoch + 1) <= 0:
                pass
            else:
                mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
                is_best = (mAP > best_mAP)
                best_mAP = max(mAP, best_mAP)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(logPath, 'checkpoint.pth.tar'))

                print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(logPath, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")

    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--dataset_src1', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('--dataset_src2', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('--dataset_src3', type=str, default='msmt17v1',
                        choices=datasets.names())
    # parser.add_argument('--dataset_src4', type=str, default='cuhk02',
    #                     choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--mem_type', type=str, default='cos',
                        help="type of softmax for mc")
    parser.add_argument('--margin', type=float, default=0.35,
                        help="margin for softmax type")
    parser.add_argument('--lamda1', type=float, default=0.7,
                        help="weight for loss item 1")
    parser.add_argument('--lamda2', type=float, default=0.1,
                        help="weight for orth loss")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=5)
    parser.add_argument('--eval-step', type=int, default=1)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='./data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--resume', type=str, default='')

    # Style settings
    parser.add_argument('--updateStyle', action='store_true')
    main()

