from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os

w_rate = 1e-4

#通过SVDO实现约束正交性，提高特征多样性
class SVDORegularizer(nn.Module):

    def __init__(self, controller=1e-4):
        super().__init__()

        os_beta = None

        # try:
        #     os_beta = float(os.environ.get('beta'))
        # except (ValueError, TypeError):
        #     raise RuntimeError('No beta specified. ABORTED.')
        # self.beta = os_beta

        self.param_controller = controller #控制正则化强度的参数
    #计算矩阵A的主导特征值（最大特征值）
    def dominant_eigenvalue(self, A: 'N x N'):

        N, _ = A.size()
        x = torch.rand(N, 1, device='cuda')

        # Ax = (A @ x).squeeze()
        # AAx = (A @ Ax).squeeze()

        # return torch.norm(AAx, p=2) / torch.norm(Ax, p=2)

        Ax = (A @ x)#矩阵和随机向量的乘积
        AAx = (A @ Ax)#进上一个结果进行矩阵乘法的结果

        return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

        # for _ in range(1):
        #     x = A @ x
        # numerator = torch.bmm(
        #     torch.bmm(A, x).permute(0, 2, 1),
        #     x
        # ).squeeze()
        # denominator = torch.bmm(
        #     x.permute(0, 2, 1),
        #     x
        # ).squeeze()

        # return numerator / denominator
    #用于计算输入矩阵A的最大和最小奇异值，使用到上面函数
    def get_singular_values(self, A: 'M x N, M >= N'):

        ATA = A.permute(1, 0) @ A
        N, _ = ATA.size()
        largest = self.dominant_eigenvalue(ATA)#计算最大特征值
        I = torch.eye(N, device='cuda')  # noqa
        I = I * largest  # noqa
        tmp = self.dominant_eigenvalue(ATA - I)#计算的到的矩阵的最大特征值来近似最小特征值
        return tmp + largest, largest #最小、最大

    def forward(self, W: 'C x S x H x W'):

        # old_W = W   #W通常是卷积层的权重
        old_size = W.size()

        W = W.view(old_size[0], -1).permute(1, 0)

        smallest, largest = self.get_singular_values(W)
        return (self.param_controller* (largest / smallest - 1) ** 2).squeeze()#.get_value() 不再使用
