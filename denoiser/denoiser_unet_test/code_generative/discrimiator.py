## denoiser_unest_test/generative_model/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Single_Discriminator(nn.Module):
  def __init__(self, ch_in = 1):
    super(Single_Discriminator, self).__init__()
    def discriminator_block(ch_in, ch_out, normalization = True):
      layers = [nn.Conv2d(ch_in, ch_out, kernel_size = 5, stride = 2, padding = 2)]
      if normalization:
        layers.append(nn.InstanceNorm2d(ch_out))
      layers.append(nn.LeakyReLU(0.2, inplace = True))
      return layers
  
    self.model = nn.Sequential(
        *discriminator_block(ch_in, 64, normalization = False),
        *discriminator_block(64, 128),
        *discriminator_block(128, 256),
        *discriminator_block(256, 512),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(512, 1, kernel_size = 5, stride = 1, padding = 2, bias = False),
        nn.Tanh()
    )

  def forward(self, x):
    return self.model(x)

class Discriminator(nn.Module):
  def __init__(self, down = False):
    super(Discriminator, self).__init__()
    self.img_size = 512 * 512 * 1
    if down:
      self.img_size /= 2
    self.model = nn.Sequential(
        nn.Linear(int(self.img_size), 512),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Linear(256, 1),
    )
  def forward(self, x):
    flat = x.view(x.size(0), -1)
    out = self.model(flat)
    return out

class Double_Discriminator(nn.Module):
  def __init__(self):
    super(Double_Discriminator, self).__init__()
    self.D1 = Discriminator(down = False)
    self.D2 = Discriminator(down = True)

  def forward(self, x1, x2):
    o1 = self.D1(x1)
    o2 = self.D2(x2)
  
    return o1, o2
