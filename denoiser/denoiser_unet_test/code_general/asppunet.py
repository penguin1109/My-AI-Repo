## denoiser_unet_test/code/asppED.py
# Encoder-Decoder Shaped Network with the ASPP Module

import torch
import torch.nn as nn
import torch.nn.functional as F

## ASPP Layer
class ASPP(nn.Module):
  def __init__(self, ch_in, ch_mid,ch_out, rate = [6, 12, 18]):
    super(ASPP, self).__init__()
    self.conv = nn.Conv2d(ch_in, ch_mid, kernel_size = 1, stride = 1, padding = 0)
    self.aspp1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size = 3, stride = 1, padding = rate[0], dilation = rate[0]),
            nn.ReLU(inplace = True), nn.BatchNorm2d(ch_mid),
        )
    self.aspp2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size = 3, stride = 1, padding = rate[1], dilation = rate[1]),
            nn.ReLU(inplace = True), nn.BatchNorm2d(ch_mid),
        )
    self.aspp3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size = 3, stride = 1, padding = rate[2], dilation = rate[2]),
            nn.ReLU(inplace = True), nn.BatchNorm2d(ch_mid)
        )
    self.last = nn.Conv2d(5 * ch_mid, ch_out, kernel_size = 1)

    self.pool = nn.Sequential(
        nn.AvgPool2d(kernel_size = 2, stride = 2),
        nn.Upsample(scale_factor = 2, mode = 'bilinear'))

    
  def forward(self, x):
    x1 = self.conv(x)
    x2 = self.aspp1(x)
    x3 = self.aspp2(x)
    x4 = self.aspp3(x)
    x5 = self.pool(x)
    out = torch.cat([x1,x2,x3, x4, x5], dim = 1)

    return self.last(out)

class SepConv2d(nn.Module):
  def __init__(self, ch_in, ch_out, kernel_size = 1, stride = 1,padding = 0, dilation = 1, bias = True):
    super(SepConv2d, self).__init__()
    self.depthwise = nn.Conv2d(ch_in, ch_in, kernel_size, stride, padding, dilation,groups = ch_in, bias = bias)
    self.pointwise = nn.Conv2d(ch_in, ch_out, 1, 1, 0, 1, 1, bias = bias)

  def forward(self, x):
    return self.pointwise(self.depthwise(x))

class EncBlock(nn.Module):
  ## ReLU6 = min(max(0, x), 6)
  def __init__(self, ch_in, ch_mid, ch_out, downsample = True):
    super(EncBlock, self).__init__()
    self.identity = nn.Conv2d(ch_in, ch_out, kernel_size = 1, stride = 2, padding = 0, bias = True)

    self.block1 = nn.Sequential(
        SepConv2d(ch_in = ch_in, ch_out = ch_mid, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(ch_mid),
        nn.ReLU6(inplace = True))
    
    self.block2 = nn.Sequential(
        SepConv2d(ch_in = ch_mid, ch_out = ch_mid, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(ch_mid),
        nn.ReLU6(inplace = True))
    
    self.block3 = nn.Sequential(
        SepConv2d(ch_in = ch_mid, ch_out = ch_out, kernel_size = 3, stride = 2, padding = 1),
        nn.BatchNorm2d(ch_out),
        nn.ReLU6(inplace = True)
    )

  def forward(self, x):
    identity = self.identity(x)
    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)

    return out + identity

class MiddleBlock(nn.Module):
  def __init__(self):
    super(MiddleBlock, self).__init__()
    self.layer1 = SepConv2d(728, 728, 3, 1, 1)
    self.layer2 = SepConv2d(728, 728, 3, 1, 1)
    self.layer3 = SepConv2d(728, 728, 3, 1, 1)
  def forward(self, x):
    out = self.layer3(self.layer2(self.layer1(x)))
    return out + x

class DecBlock(nn.Module):
  def __init__(self, ch_in, ch_mid, ch_out):
    super(DecBlock, self).__init__()
    self.identity = nn.Conv2d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)
    self.layer1 = SepConv2d(ch_in, ch_mid, kernel_size = 3, stride = 1, padding = 1)
    self.layer2 = SepConv2d(ch_mid, ch_out, kernel_size = 3, stride = 1, padding = 1)
  
  def forward(self, x):
    identity = self.identity(x)
    x = self.layer1(x)
    x = self.layer2(x)
    return (identity + x)

class ASPPUNet(nn.Module):
  def __init__(self, ch_in = 1):
    super(ASPPUNet, self).__init__()
    ## Entry Flow ##
    self.entry1 = EncBlock(ch_in, 64, 128)
    self.entry2 = EncBlock(128, 128, 128)
    self.entry3 = EncBlock(128, 256, 256)
    self.entry4 = EncBlock(256, 728, 728)

    ## Middle Flow ##
    middle = [MiddleBlock() for i in range(12)]
    self.middle = nn.Sequential(*middle)

    ## Atrous Spatial Pyramid Pooling ##
    self.aspp = ASPP(728, 728, 256)

    ## Bilinear Upsample ##
    self.upsample = nn.Upsample(scale_factor = 4, mode = 'bilinear')
    ## Decoder ##
    self.dec1 = DecBlock(384, 256, 256)
    self.dec2 = DecBlock(384, 128, 128)
    self.dec3 = DecBlock(128, 64, 64)
    ## Trans Conv ##
    self.trans_conv1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride= 2, padding = 1, output_padding = 1)
    self.trans_conv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
    ## Tail ##
    self.tail = nn.Conv2d(64, 1, kernel_size = 3, stride = 1, padding = 1)





  def forward(self, x):
    ## Entry Flow ##
    e1 = self.entry1(x)
    e2 = self.entry2(e1)
    e3 = self.entry3(e2)
    e4 = self.entry4(e3)
    ## Middle Flow ##
    middle = self.middle(e4) + e4

    ## Atrous Spatial Pyramid Pooling ##
    aspp = self.aspp(middle)
    upsample = self.upsample(aspp)
    ## Decoder ##
    cat1 = torch.cat((e2, upsample), dim = 1)
    d1 = self.dec1(cat1)
    d1 = self.trans_conv1(d1)

    cat2 = torch.cat((e1, d1), dim = 1)
    d2 = self.dec2(cat2)
    d2 = self.trans_conv2(d2)
    d3 = self.dec3(d2)
    out = self.tail(d3)
    out = torch.clip(out, min = 0.0, max = 1.0)

    return out



if __name__ == "__main__":
  sample = torch.zeros((2, 1, 512, 512))
  net = ASPPUNet()
  out = net(sample)

