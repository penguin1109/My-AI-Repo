## denoiser_unet_test/code/ratunet.py (without FiLM)

import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
  def __init__(self, ch_in, ch_1, ch_2, global_pool = False):
    super(AttentionBlock, self).__init__()
    self.conv1 = nn.Conv2d(ch_in, ch_1, kernel_size = 1, stride = 1, padding = 0)
    self.conv2 = nn.Conv2d(ch_in, ch_2, kernel_size = 1, stride = 1, padding = 0)
    self.softmax = nn.Softmax()
    self.sigmoid = nn.Sigmoid()
    self.conv3 = nn.Conv2d(ch_2, ch_in, kernel_size = 1, stride = 1, padding = 0)
    self.global_pool = global_pool
    if global_pool:
      self.pool = nn.AdaptiveAvgPool2d(1)
    else:
      self.pool = None
  def forward(self, x):
    conv1 = self.conv1(x)
    conv2 = self.conv2(x)
    if self.pool:
      conv1 = self.pool(conv1)
      conv1 = conv1.view(x.size(0),1, conv1.size(1))
    else:
      conv1 = conv1.view(x.size(0), conv1.size(2) * conv1.size(3), 1)
    conv2 = conv2.view(x.size(0), conv2.size(1), conv2.size(2) * conv2.size(3)) ## (B, C/2, H*W)
    conv1 = self.softmax(conv1) ## (B, C/2, 1)

    out = []
    for b in range(x.size(0)):
      c1 = conv1[b, :, :]
      c2 = conv2[b, :, :]
      if self.global_pool:
        out.append(torch.tensordot(c1, c2, dims = ([1], [0])))
      else:
        out.append(torch.tensordot(c1, c2, dims = ([0], [1])))
    out = torch.stack(out, dim = 0)
    if self.global_pool == False:
      out = out.view(out.size(0), out.size(2), out.size(1), 1)
      out = self.conv3(out)

    else:
      out = out.view(out.size(0), 1, x.size(2), x.size(3))
    
    out = self.sigmoid(out)

    return out ## x랑 dot poduct 하기 전에 output으로 만드는 것

class SelfAttention(nn.Module):
  def __init__(self, ch_in):
    super(SelfAttention, self).__init__()
    self.block1 = AttentionBlock(ch_in, 1, ch_in // 2)
    self.block2 = AttentionBlock(ch_in, ch_in // 2, ch_in // 2, global_pool = True)

  def forward(self, x):
    out1 = self.block1(x) ## (B, C, 1, 1)
    out1 = torch.mul(x, out1)
    out2 = self.block2(out1)

    return torch.mul(out1, out2)

class EncBlock(nn.Module):
  def __init__(self, ch):
    super(EncBlock, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(ch), nn.PReLU(),
        nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(ch), nn.PReLU())
    self.identity = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)
  def forward(self, x):
    out = self.conv(x)
    skip = self.identity(x)
    return out, skip

class DecBlock(nn.Module):
  def __init__(self, ch):
    super(DecBlock, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(ch, ch, kernel_size = 3, stride =1, padding = 1),
        nn.BatchNorm2d(ch), nn.PReLU(),
        nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(ch), nn.PReLU(),
        nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(ch), nn.PReLU())
  def forward(self, x, skip):
    out = self.conv(x)
    return torch.cat((out, x), dim = 1)

class RatUNet(nn.Module):
  def __init__(self, ch_in = 1):
    super(RatUNet, self).__init__()
    self.head = nn.Conv2d(ch_in, 64, kernel_size = 3, stride = 1, padding = 1)
    self.down1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1)
    self.enc1 = EncBlock(128)
    self.down2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)
    self.enc2 = EncBlock(256)
    self.down3 = nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1)

    self.up1 = nn.ConvTranspose2d(512, 256, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
    self.dec1 = DecBlock(256)
    self.up2 = nn.ConvTranspose2d(512, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
    self.dec2 = DecBlock(128)
    self.up3 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)

    self.tail = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
    )
    self.noise = nn.Conv2d(192, 1, kernel_size = 3, stride = 1, padding = 1)
                        
    self.attention = SelfAttention(192)

  def forward(self, x):
    head = self.head(x)

    down1 = self.down1(head)
    enc1, skip1 = self.enc1(down1)
    down2 = self.down2(enc1)
    enc2, skip2 = self.enc2(down2)
    down3 = self.down3(enc2)

    up1 = self.up1(down3)
    dec1 = self.dec1(up1, skip2)
    up2 = self.up2(dec1)
    dec2 = self.dec2(up2, skip1)
    up3 = self.up3(dec2)

    tail = self.tail(up3)
    tail = torch.cat((head, tail), dim = 1)
    noise = self.noise(self.attention(tail))
    clean = x - noise
    return clean



if __name__ == "__main__":
  sample = torch.zeros((2, 1, 512, 512))
  at =  RatUNet()
  out = at(sample)
  print(out.shape)
