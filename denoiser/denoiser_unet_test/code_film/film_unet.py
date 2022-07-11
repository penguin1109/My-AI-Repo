## denoiser_unet_test/code_film/film_unet.py
import torch
import torch.nn as nn

class UNetConvBlock(nn.Module):
  ## Convolution block for the encoder of the UNet
  def __init__(self, ch_in, ch_out, film , dec, final = False):
    super(UNetConvBlock, self).__init__()
    layers = []
    self.final = final
    self.film = film
    self.dec = dec ## decoder이면 concat = True / 아니면 concat = False
    if self.film:
      if self.dec:
        self.film_layer = FiLMLayer(ch_out, concat = self.dec)
      else:
        self.film_layer = FiLMLayer(ch_in, concat = self.dec)
    self.layer1 = nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.InstanceNorm2d(ch_out),
        nn.Tanh()
    )
    if self.final:
      self.layer2 = nn.Sequential(
          nn.Conv2d(ch_out, 1, kernel_size = 3, stride = 1, padding = 1, bias = True),
          nn.InstanceNorm2d(1),
          nn.Tanh()
      )
    else:
      self.layer2 = nn.Sequential(
        nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.InstanceNorm2d(ch_out),
        nn.Tanh()
      )
  def forward(self, x, skip_x):
    if self.film:
      x = self.film_layer(x, skip_x)
    
    return self.layer2(self.layer1(x))

class filmUnet(nn.Module):
  def __init__(self, ch_in):
    super(filmUnet, self).__init__()
    self.ch_in = ch_in
    self.first_conv_block = nn.Sequential(
        nn.Conv2d(ch_in, 32, kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.InstanceNorm2d(32),
        nn.Tanh(),
        nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.InstanceNorm2d(32),
        nn.Tanh()
    )
    self.first_downsample = nn.MaxPool2d(kernel_size = 2)
    self.second_down_block = UNetConvBlock(32, 64, film = True, dec = False)
    self.second_downsample = nn.MaxPool2d(kernel_size = 2)
    
    self.bridge = UNetConvBlock(64, 128, film = True, dec = False)

    self.first_upsample = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
    self.first_up_bridge = UNetConvBlock(128, 64, film = True, dec = True)
    self.second_upsample = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2)
    
    self.tail = UNetConvBlock(64, 32, film = True, dec = True, final = True)

  def forward(self, x):
    ## Encoder ##
    first_conv = self.first_conv_block(x)
    d1 = self.first_downsample(first_conv)
    o1 = self.second_down_block(d1, None)
    d2 = self.second_downsample(o1)
    ## Bridge ##
    o2 = self.bridge(d2, None)
    ## Decoder ##
    u1 = self.first_upsample(o2)
    o3 = self.first_up_bridge(u1, o1)
    u2 = self.second_upsample(o3)
    o4 = self.tail(u2, first_conv)
    ## Denoised Output ##
    return o4







