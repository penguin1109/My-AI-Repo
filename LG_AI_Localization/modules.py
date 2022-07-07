import torch.nn.functional as F
import torch.nn as nn


def conv1x1(ch_in, ch_out, stride, padding):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size = 1, padding = padding, stride = stride),
        nn.BatchNorm2d(ch_out), nn.ReLU(inplace = True)
    )
def conv3x3(ch_in, ch_out, stride,padding):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, stride = stride, kernel_size = 3, padding = padding),
        nn.BatchNorm2d(ch_out), nn.ReLU(inplace = True)
    )
class ResBlock(nn.Module):
    def __init__(self,ch_in, ch_mid, ch_out, downsample = False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.layer = nn.Sequential(
                conv1x1(ch_in, ch_mid, stride = 2, padding = 0),
                conv3x3(ch_mid, ch_mid, stride = 1, padding = 1),
                conv1x1(ch_mid, ch_out, stride = 1, padding = 0)
            )
            self.downsize = conv1x1(ch_in, ch_out, stride = 2, padding = 0)
        else:
            self.layer = nn.Sequential(
                conv1x1(ch_in, ch_mid, stride = 1, padding = 0),
                conv3x3(ch_mid, ch_mid, stride = 1, padding = 1),
                conv1x1(ch_mid, ch_out, stride = 1, padding = 0)
            )
            self.make_equal_channel = conv1x1(ch_in, ch_out, stride = 1, padding = 0)
    def forward(self, x, mask = None):
        if mask is None:
            if self.downsample:
                out = self.layer(x)
                x = self.downsize(x)
                return out + 1
            else:
                out = self.layer(x)
                if out.size() != x.size():
                    x = self.make_equal_channel(x)
                return out + x
        else:
            if self.downsample:
                out = self.layer(x)
                masked = self.downsize(out * mask)
                return out + masked
            else:
                out = self.layer(x)
                masked = out * mask
                if masked.size != out.size():
                    masked = self.make_equal_channel(masked)
                return out + masked

class SoftMask(nn.Module):
    """
    - bottom-up top-down fully convolutional structure
    - upsampling by linear interpolation
    - res1 ->
    """
    def __init__(self, ch_in, ch_mid, ch_out):
        super(SoftMask, self).__init__()
        self.final = nn.Sequential(
            conv1x1(ch_out, ch_out, stride=1, padding=0),
            conv1x1(ch_out, ch_out, stride = 1, padding = 0)
        )
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # x = self.downsample(x)
        upsample = F.interpolate(x, mode = 'linear')
        out = self.activation(self.final(upsample))
