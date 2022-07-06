"""
contains the code for common layers to implement 
ASPP / SSA Module / ResBlock etc..
"""
import torch
import torch.nn as nn

## Common Layers for all Networks ##
def conv1x1(ch_in, ch_out, stride):
    model = nn.Conv2d(ch_in, ch_out, kernel_size = (1,1), stride = stride, padding = 0)
    return model

def conv3x3(ch_in, ch_out, stride):
    model = nn.Conv2d(ch_in, ch_out, kernel_size = (3, 3), stride = stride, padding = 1)
    return model

class residual_block(nn.Module):
    ## Common Residual Block
    def __init__(self, ch_in, ch_mid, ch_out, downsample = False):
        super(residual_block, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.layer = nn.Sequential(
                conv1x1(ch_in, ch_mid, stride = 2),
                conv3x3(ch_mid, ch_mid, stride = 1),
                conv3x3(ch_mid, ch_out, stride = 1),
            )
            self.identity = conv1x1(ch_in, ch_out, stride = 2)
        else:
            self.layer = nn.Sequential(
                conv1x1(ch_in, ch_mid, stride = 1),
                conv3x3(ch_mid, ch_mid, stride = 1),
                conv3x3(ch_mid, ch_out, stride = 1),
            )
            self.identity = conv1x1(ch_in, ch_out, sride = 1)
    def forward(self, x):
        out = self.layer(x)
        if self.downsample == False:
            if out.size() is not x.size():
                x = self.identity(x)
        return out + x



class upsample(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride):
        super(upsample, self).__init__()
        self.layer = nn.ConvTranspose2d(ch_in, ch_out, kernel_size = kernel, stride = stride)
    def forward(self, x):
        return self.layer(x)

## Layers only for NBNet ##
class nbnet_conv_block(nn.Module):
    ## Convolution Block for NBNet
    def __init__(self, in_ch, out_ch, kernel_size = 3, padding = 1, stride = 1, dilation,  downsample = False):
        super(conv_block, self).__init__()
        layers = [
                nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size,stride = stride, padding = padding,),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(in_channels = out_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding,),
                nn.LeakyReLU(inplace = True)
            ]
        self.skip = not downsample ## downsample = False -> skip = True
        if downsample:
            self.downsample = nn.Conv2d(in_channels = out_ch, out_channels = out_ch,
                kernel_size = 4, stride = 2)
        self.conv_block = nn.Sequential(*layers)
        self.identity = nn.Conv2d(in_channels = in_ch, out_channels = out_ch,
            kernel_size = 1, stride = 2, padding = 0)
    def forward(self, x):
        out = self.conv_block(x)
        x = self.identity(x)
        if self.skip == False: # downsample = True
            return self.downsample(out + x)
        else: # downsample = False
            return out + x




    
