import torch
import torch.nn as nn

# from .NR import NoneRegularizer
# from .SVMO import SVMORegularizer
from .SVDO import SVDORegularizer
# from .SO import SORegularizer

# mapping = {
#     False: NoneRegularizer,
#     True: SVMORegularizer,
# }


class ConvRegularizer(nn.Module):

    def __init__(self):#klass,controller 控制器参数我已固定，除非再次传入
        super().__init__()
        # self.reg_instance = klass(controller)
        # 直接使用 SVDORegularizer 实例化一个对象
        self.reg_instance = SVDORegularizer()#controller

    # def get_all_conv_layers(self, module):
    #     # 遍历模型，递归地找到所有的 Conv2D 层
    #     if isinstance(module, (nn.Sequential, list)):
    #         for m in module:
    #             yield from self.get_all_conv_layers(m)
    #
    #     if isinstance(module, nn.Conv2d):
    #         yield module
    #
    #     for child in module.children():
    #         yield from self.get_all_conv_layers(child)

    def get_all_conv_layers(self, module):
        if isinstance(module, nn.Conv2d):
            yield module

        for child in module.children():
            yield from self.get_all_conv_layers(child)

    def forward(self, net, ignore=False):

        accumulator = torch.tensor(0.0).cuda()

        if ignore:
            return accumulator
        # 遍历所有卷积层，对其权重进行正则化
        for conv in self.get_all_conv_layers(net):#拿出来.module.backbone_modules()
            accumulator += self.reg_instance(conv.weight)

        # print(accumulator.data)
        return accumulator


def get_regularizer():
    """返回 SVDO 正则化器实例"""
    # 直接返回 SVDO 正则化器的实例
    return ConvRegularizer()#args不传入任何参数
