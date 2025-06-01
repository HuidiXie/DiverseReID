from __future__ import print_function, absolute_import, division
import time
import torch
from .utils.meters import AverageMeter

class Trainer(object):
    def __init__(self, args, model, memory):
        super(Trainer, self).__init__()
        self.model = model
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args

    def train(self, epoch, data_loaders, optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        source_count = len(data_loaders)
        end = time.time()

        for i in range(train_iters):

            batch_data = [data_loaders[i].next() for i in range(source_count)]
            data_time.update(time.time() - end)
            inputs_list = []
            targets_list = []

            for ith in range(len(batch_data)):
                inputs = batch_data[ith][0].cuda()
                targets = batch_data[ith][2].cuda() # why 2? # img_path, pid, camid   0 is data, 1 is image path,2 is pid, 3 is camid
                inputs_list.append(inputs)
                targets_list.append(targets)

            loss_id = 0.
            output_list = []
            for j in range(source_count):
                true_bn_x = self.model(inputs_list[j], style=self.args.updateStyle)
                loss_id += self.memory[j](true_bn_x, targets_list[j]).mean()    # Forward
                output_list.append(true_bn_x)

            loss_orth = 0.0
            for m in range(source_count):
                for n in range(m + 1, source_count):
                    feat_former = output_list[m]
                    feat_latter = output_list[n]
                    orth = torch.mm(feat_former.transpose(0, 1), feat_latter)
                    loss_orth += self.args.lamda2 * torch.sum(torch.pow(torch.diagonal(orth, dim1=-2, dim2=-1), 2))

            loss_final = loss_id / source_count + loss_orth / (source_count * (source_count - 1) / 2)
            optimizer.zero_grad()
            loss_final.backward()     # Backward
            optimizer.step()

            losses.update(loss_final.item())

            with torch.no_grad():
                for m_ind in range(source_count):
                    self.memory[m_ind].module.MomentumUpdate(output_list[m_ind], targets_list[m_ind])  # memory bank update

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'loss {:.3f} ({:.3f})'.format(epoch, i + 1, train_iters,
                                                    batch_time.val, batch_time.avg,
                                                    losses.val, losses.avg))