import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import numpy as np
import collections
import pdb
from .softmaxs import cosSoftmax, arcSoftmax, circleSoftmax


class MC(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        return grad_inputs, None, None, None


def mc(inputs, indexes, features, momentum=0.5):
    return MC.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MemoryClassifier(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, mem_type='cos',  margin=0.35, lamda1=0.7, lamda2=1.0):
        super(MemoryClassifier, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp

        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.type = mem_type
        self.margin = margin

        self.register_buffer('features', torch.zeros(2 * num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())
    def MomentumUpdate(self, inputs,  indexes):#inputs2,移除

        for x, y in zip(inputs, indexes):

            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y] / self.features[y].norm()

    def forward(self, inputs, indexes):

        logits = mc(inputs, indexes, self.features, self.momentum)
        if self.type == 'cos':
            if self.margin > 0:
                logits = cosSoftmax(logits, indexes, self.margin)
        elif self.type == 'arc':
            logits = arcSoftmax(logits, indexes, self.margin)
        elif self.type == 'circle':
            logits = circleSoftmax(logits, indexes, self.margin)
        else:
            assert False, "invalid type {}".format(self.type)

        logits = logits / self.temp
        loss = F.cross_entropy(logits, indexes)
        return loss








