## denoiser_unet_test/generative_model/generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDown(nn.Module):
  def __init__(self, ch_in, ch_out, normalize = True, dropout = 0.5):
    super(UNetDown, self).__init__()
    layers = [nn.Conv2d(ch_in, ch_out, kernel_size = 5, stride = 2, padding = 2, bias = False)]
    if normalize:
      layers.append(nn.InstanceNorm2d(ch_out))
    layers.append(nn.LeakyReLU(0.2, inplace = True))
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)
  def forward(self, x):
    return self.model(x)

class UNetUp(nn.Module):
  def __init__(self, ch_in, ch_out, dropout = 0.5):
    super(UNetUp, self).__init__()
    layers = [
              nn.ConvTranspose2d(ch_in, ch_out, kernel_size = 5, stride = 2, padding = 2, bias = False),
              nn.InstanceNorm2d(ch_out),
              nn.LeakyReLU(0.2, inplace = True)
    ]
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)
    
  def forward(self, x, skip):
    x = self.model(x)
    x = torch.cat((x, skip), dim = 1)

    return x

class Generator(nn.Module):
  def __init__(self, ch_in = 1):
    super(Generator, self).__init__()

    self.down1 = UNetDown(ch_in, 64, normalize = False)
    self.down2 = UNetDown(64, 128)
    self.down3 = UNetDown(128, 256)
    self.down4 = UNetDown(256, 512, dropout=0.5)
    self.down5 = UNetDown(512, 512, dropout=0.5)
    self.down6 = UNetDown(512, 512, dropout=0.5)
    self.down7 = UNetDown(512, 512, dropout=0.5)
    self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

    self.up1 = UNetUp(512, 512, dropout=0.5)
    self.up2 = UNetUp(1024, 512, dropout=0.5)
    self.up3 = UNetUp(1024, 512, dropout=0.5)
    self.up4 = UNetUp(1024, 512, dropout=0.5)
    self.up5 = UNetUp(1024, 256)
    self.up6 = UNetUp(512, 128)
    self.up7 = UNetUp(256, 64)

    self.tail = nn.Sequential(
        nn.Upsample(scale_factor = 2),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(128, 1, kernel_size = 5, stride = 2, padding = 2),
        nn.Tanh()
    )
  
  def forward(self, x):
    d1 = self.down1(x)
    d2 = self.down1(d1)
    d3 = self.down1(d2)
    d4 = self.down1(d3)
    d5 = self.down1(d4)
    d6 = self.down1(d5)
    d7 = self.down1(d6)
    d8 = self.down1(d7)

    u1 - self.up1(d8, d7)
    u2 = self.up2(u1, d6)
    u3 = self.up3(u2, d5)
    u4 = self.up4(u3, d4)
    u5 = self.up5(u4, d3)
    u6 = self.up6(u5, d2)
    u7 = self.up7(u6, d1)

    return self.tail(u7)
