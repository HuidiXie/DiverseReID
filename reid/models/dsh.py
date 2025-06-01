import torch
import random
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

class StyleRandomization(nn.Module):
    def __init__(self, p=1.0, eps=1e-6, mode ='dsh', mix='random'):
        super().__init__()
        self.p = p
        self.eps = eps
        self.mode = mode
        self.mix = mix

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        std = (var + self.eps).sqrt()

        x = (x - mean) / std

        if self.mix == 'random':
            idx_swap = torch.randperm(N)
        elif self.mix == 'crossdomain':
            perm = torch.arange(N - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(N // 2)]
            perm_a = perm_a[torch.randperm(N // 2)]
            idx_swap = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError

        if self.mode == 'random':
            alpha = torch.rand(N, 1, 1)
        elif self.mode == 'beta':
            beta = torch.distributions.Beta(0.1, 0.1)  # beta
            alpha = beta.sample((N, 1, 1))    # torch.Size([96, 1, 1])
        elif self.mode == 'dir':
            dirichlet = tdist.dirichlet.Dirichlet(concentration=torch.tensor([0.00390625] * 256, device='cuda'))
            alpha= dirichlet.sample((N, ))  # 96, 256
            alpha =alpha.mean(dim=1, keepdim=True)
            alpha = alpha.unsqueeze(2)
        elif self.mode == 'dsh':
            # dwass
            distance = torch.pow((mean - mean[idx_swap]), 2) + torch.pow(std, 2) + \
                       torch.pow(std[idx_swap], 2) - 2 * std * std[idx_swap]
            alpha = 1.0 / (1.0 + distance)
            alpha = F.softmax(alpha, dim=1)
        else:
            raise NotImplementedError

        alpha = alpha.cuda()
        mean = alpha * mean + (1 - alpha) * mean[idx_swap]
        var = alpha * var + (1 - alpha) * var[idx_swap]
        x = x * (var + self.eps).sqrt() + mean
        x = x.view(N, C, H, W)
        return x
