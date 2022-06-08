import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride = 1):
        ## bn => relu => conv
        super(ResidualBlock, self).__init__()
        self.ch_in = ch_in;self.ch_out = ch_out;self.stride = stride;self.ch_mid = ch_in//4
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.conv1 = nn.Conv2d(ch_in,self.ch_mid, kernel_size = 1,stride = 1,padding = 0, bias = False)
        self.bn2 = nn.BatchNorm2d(self.ch_mid)
        self.conv2 = nn.Conv2d(self.ch_mid, self.ch_mid, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(self.ch_mid)
        self.conv3 = nn.Conv2d(self.ch_mid, ch_out, kernel_size = 1, stride = 1, padding = 0, bias = False)
        ## downsizing을 위해서 사용하는 conv1x1 layer
        self.conv4 = nn.Conv2d(ch_in, ch_out, kernel_size = 1, stride = stride, padding=0, bias = False)
        ## 공통으로 사용하는 ReLU activation function
        self.relu = nn.ReLU(inplace = True)

    def forward(self,x):
        res = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(self.bn3(out))
        out = self.conv3(out)
        if (self.ch_in != self.out_ch) or (self.stride != 1):
            res = self.conv4(out1)
        out += res
        return out

