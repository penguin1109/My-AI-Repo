## denoiser_unet_test/nafnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
  def forward(self, x):
    x1, x2 = x.chunk(2, dim = 1)
    return x1 * x2
  
class NAFBlock(nn.Module):
  def __init__(self, ch_in, dw_exp=2, ffn_exp=2):
    super(NAFBlock, self).__init__()
    dw_ch = dw_exp * ch_in
    self.conv1 = nn.Conv2d(ch_in, dw_ch, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True)
    self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size = 3, padding = 1, stride = 1, groups = dw_ch, bias = True)
    self.conv3 = nn.Conv2d(dw_ch // 2, ch_in, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True)

    ## Simplified Channel Attention
    self.sca = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dw_ch // 2, dw_ch // 2, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True)
    )
    ## Simple Gate
    self.sg = SimpleGate()
    
    ffn_ch = ffn_exp * ch_in
    self.conv4 = nn.Conv2d(ch_in, ffn_ch, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True)
    self.conv5 = nn.Conv2d(ffn_ch // 2, ch_in, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True)
    self.norm1 = LayerNorm2d(ch_in)
    self.norm2 = LayerNorm2d(ch_in)

    self.dropout1 = nn.Identity()
    self.dropout2 = nn.Identity()

    ## Trainable Parameters for Layer Normalization
    self.beta = nn.Parameter(torch.zeros((1, ch_in, 1, 1)), requires_grad = True)
    self.gamma = nn.Parameter(torch.zeros((1, ch_in, 1, 1)), requires_grad = True)

  def forward(self, img):
    x = img
    ## Pahse 1
    x = self.norm1(x)
    x = self.conv2(self.conv1(x))

    x = self.sg(x)
    x = x * self.sca(x)
    x = self.conv3(x)

    x = self.dropout1(x)

    y = img + x * self.beta
    ## Phase 2
    x = self.conv4(self.norm2(x))
    x = self.sg(x)
    x = self.conv5(x)
    x = self.dropout2(x)

    return y + x * self.gamma


class NAFNet(nn.Module):
  def __init__(self, ch_in, scale, middle_block_num = 1, 
               enc_block_num = [], dec_block_num = []):
    super(NAFNet, self).__init__()
    self.head = nn.Conv2d(ch_in, scale, kernel_size = 3, padding = 1, stride = 1, groups = 1, bias = True)
    self.tail = nn.Conv2d(scale, ch_in, kernel_size = 3, padding = 1, stride = 1, groups = 1, bias = True)

    self.encoder = nn.ModuleList()
    self.decoder = nn.ModuleList()

    self.up = nn.ModuleList()
    self.down = nn.ModuleList()

    temp = scale
    for n in enc_block_num:
      self.encoder.append(
          nn.Sequential(
              *[NAFBlock(temp) for _ in range(n)]
          )
      )
      self.down.append(nn.Conv2d(temp, temp*2, kernel_size = 2, stride = 2))
      temp *= 2
    self.mid = nn.Sequential(*[NAFBlock(temp) for _ in range(middle_block_num)])
    for n in dec_block_num:
      self.up.append(
          nn.Sequential(
              nn.Conv2d(temp, temp * 2, kernel_size = 1, bias = False),
              nn.PixelShuffle(2)
          )
      )
      temp //= 2
      self.decoder.append(
          nn.Sequential(
              *[NAFBlock(temp) for _ in range(n)]
          )
      )
    self.padder_size = 2 ** len(self.encoder)

  def _check_img_size(self, x):
    b, c, h, w = x.size()
    pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
    pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x

  def forward(self, x):
    B, C, H, W = x.shape
    img = self._check_img_size(x)
    x = self.head(x)
    encs = []
    for encoder, down in zip(self.encoder, self.down):
      x = encoder(x)
      encs.append(x)
      x = down(x)
    x = self.mid(x)
    for decoder, up, skip in zip(self.decoder, self.up, encs[::-1]):
      x = up(x)
      x = x + skip
      x = decoder(x)
    x = self.tail(x)
    x = x + img

    return x[:, :, :H, :W]
