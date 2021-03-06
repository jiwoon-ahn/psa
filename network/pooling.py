import sys

import torch
import torch.nn as nn
from torch.autograd import Function


class WildcatPool2dFunction(Function):

    @staticmethod
    def get_positive_k(k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    @staticmethod
    def forward(ctx, input, kmax, kmin, alpha):
        ctx.kmax = kmax
        ctx.kmin = kmin
        ctx.alpha = alpha
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        kmax = WildcatPool2dFunction.get_positive_k(ctx.kmax, n)
        kmin = WildcatPool2dFunction.get_positive_k(ctx.kmin, n)

        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))

        ctx.indices_max = indices.narrow(2, 0, kmax)
        output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

        if kmin > 0 and ctx.alpha != 0:
            ctx.indices_min = indices.narrow(2, n - kmin, kmin)
            output.add_(sorted.narrow(2, n - kmin, kmin).sum(2).mul_(ctx.alpha / kmin)).div_(2)

        ctx.save_for_backward(input)
        return output.view(batch_size, num_channels)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors

        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        kmax = WildcatPool2dFunction.get_positive_k(ctx.kmax, n)
        kmin = WildcatPool2dFunction.get_positive_k(ctx.kmin, n)

        grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmax)

        grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, ctx.indices_max,
                                                                                              grad_output_max).div_(
            kmax)

        if kmin > 0 and ctx.alpha != 0:
            grad_output_min = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmin)
            grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2,
                                                                                                      ctx.indices_min,
                                                                                                      grad_output_min).mul_(
                ctx.alpha / kmin)
            grad_input.add_(grad_input_min).div_(2)

        return grad_input.view(batch_size, num_channels, h, w), None, None, None


class WildcatPool2d(nn.Module):
    def __init__(self, kmax=1, kmin=None, alpha=1):
        super(WildcatPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha

    def forward(self, input):
        return WildcatPool2dFunction.apply(input, self.kmax, self.kmin, self.alpha)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ', kmin=' + str(self.kmin) + ', alpha=' + str(
            self.alpha) + ')'


class ClassWisePoolFunction(Function):

    @staticmethod
    def forward(ctx, input, num_maps):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % num_maps != 0:
            print(
                'Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / num_maps)
        x = input.view(batch_size, num_outputs, num_maps, h, w)
        output = torch.sum(x, 2)
        ctx.save_for_backward(input)
        ctx.num_maps = num_maps
        return output.view(batch_size, num_outputs, h, w) / num_maps

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, ctx.num_maps,
                                                                               h, w).contiguous()

        return grad_input.view(batch_size, num_channels, h, w), None


class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction.apply(input, self.num_maps)

    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps={num_maps})'.format(num_maps=self.num_maps)
