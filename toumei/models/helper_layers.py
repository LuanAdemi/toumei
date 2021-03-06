import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The MIT License (MIT)

Copyright (c) 2020 ProGamerGov

Copyright (c) 2015 Justin Johnson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class AdditionLayer(nn.Module):
    def forward(self, x, y):
        return x + y


class MaxPool2dLayer(nn.Module):
    def forward(self, x, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False):
        return F.max_pool2d(x, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)


class PadLayer(nn.Module):
    def forward(self, x, padding=(1, 1, 1, 1), value=None):
        if value == None:
            return F.pad(x, padding)
        else:
            return F.pad(x, padding, value=value)


class ReluLayer(nn.Module):
    def forward(self, x):
        return F.relu(x)


class RedirectedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = grad_input[input_tensor < 0] * 1e-1
        return grad_input


class RedirectedReluLayer(nn.Module):
    def forward(self, tensor):
        return RedirectedReLU.apply(tensor)


class SoftMaxLayer(nn.Module):
    def forward(self, x, dim=1):
        return F.softmax(x, dim=dim)


class DropoutLayer(nn.Module):
    def forward(self, x, p=0.4000000059604645, training=False, inplace=True):
        return F.dropout(input=x, p=p, training=training, inplace=inplace)


class CatLayer(nn.Module):
    def forward(self, x, dim=1):
        return torch.cat(x, dim)


class LocalResponseNormLayer(nn.Module):
    def forward(self, x, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0):
        return F.local_response_norm(x, size=size, alpha=alpha, beta=beta, k=k)


class AVGPoolLayer(nn.Module):
    def forward(self, x, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False):
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                            count_include_pad=count_include_pad)
