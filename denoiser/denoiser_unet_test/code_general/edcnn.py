## EDCNN.py ##
## denoiser_unet_test/code_general/edcnn.py
## EDCNN : Edge Enhancement-based Densely Connected Network with Compound Loss for Low-Dose CT Denoising

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

class SobelConv2d(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int, kernel_size = 3, stride = 1, padding = 0, dilation = 1,
               groups = 1, bias = True, requires_grad = True):
    super(SobelConv2d, self).__init__()
    self.ch_in = in_channels
    self.ch_out = out_channels
    self.ksize = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups

    self.bias = bias if requires_grad else False

    if self.bias:
      self.bias = nn.Parameter(torch.zeros(size = (out_channels, ), dtype = torch.float32), requires_grad = True)
    else:
      self.bias = None
    
    self.sobel_weight = nn.Parameter(torch.zeros(
        size = (out_channels, int(in_channels/groups), kernel_size, kernel_size), requires_grad = False
    ))
    with torch.no_grad():
      kernel_mid = kernel_size//2
      for i in range(out_channels):
        if i % 4 == 0:
          self.sobel_weight[i, :, 0, :] = -1
          self.sobel_weight[i, :, 0, kernel_mid] = -2
          self.sobel_weight[i, :, -1, :] = 1
          self.sobel_weight[i, :, -1, kernel_mid] = 2

        elif i % 4 == 1:
          self.sobel_weight[i, :, :, 0] = -1
          self.sobel_weight[i, :, kernel_mid, 0] = -2
          self.sobel_weight[i, :, :, -1] = 1
          self.sobel_weight[i, :, kernel_mid, 0] = 2

        elif i % 4 == 2:
          self.sobel_weight[i, :, 0, 0] = -2
          for j in range(0, kernel_mid + 1):
            self.sobel_weight[i, :, kernel_mid - j, j] = -1
            self.sobel_weight[i, :, kernel_size - 1 - j, kernel_mid + j] = 1
          self.sobel_weight[i, :, -1, -1] = 2

        else:
          self.sobel_weight[i, :, -1, 0] = -2
          for j in range(0, kernel_mid + 1):
            self.sobel_weight[i, :, kernel_mid + j, j] = -1
            self.sobel_weight[i, :, j, kernel_mid + j] = 1
          self.sobel_weight[i, :, 0, -1] = 2
    if requires_grad:
      self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
    else:
      self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)
  def forward(self, x):
    if torch.cuda.is_available():
      self.sobel_factor = self.sobel_factor.cuda()
      if isinstance(self.bias, nn.Parameter):
        self.bias = self.bias.cuda()
    sobel_weight = self.sobel_weight * self.sobel_factor
    if torch.cuda.is_available():
      sobel_weight = sobel_weight.cuda()
    
    out = F.conv2d(x,sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return out
      


class EDCNN(nn.Module):
  def __init__(self, ch_in = 1, ch_out = 32, sobel_ch = 32):
    super(EDCNN, self).__init__()
    ## Edge Enhancement Module ##
    self.edge_enhance = SobelConv2d(ch_in, sobel_ch, kernel_size = 3, stride = 1, padding = 1, bias = True)
    
    self.block1 = nn.Sequential(
        nn.Conv2d(ch_in + sobel_ch, ch_out, kernel_size = 1, stride = 1, padding = 0), nn.LeakyReLU(),
        nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1), nn.LeakyReLU()
    )

    self.block2 = nn.Sequential(
        nn.Conv2d(ch_in + sobel_ch + ch_out, ch_out, kernel_size = 1, stride = 1, padding = 0), nn.LeakyReLU(),
        nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1), nn.LeakyReLU()
    )

    self.block3 = nn.Sequential(
        nn.Conv2d(ch_in + sobel_ch + ch_out, ch_out, kernel_size = 1, stride = 1, padding = 0), nn.LeakyReLU(),
        nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1), nn.LeakyReLU()
    )

    self.block4 = nn.Sequential(
        nn.Conv2d(ch_in + sobel_ch + ch_out, ch_out, kernel_size = 1, stride = 1, padding = 0), nn.LeakyReLU(),
        nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1), nn.LeakyReLU()
    )

    self.block5 = nn.Sequential(
        nn.Conv2d(ch_in + sobel_ch + ch_out, ch_out, kernel_size = 1, stride = 1, padding = 0), nn.LeakyReLU(),
        nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1), nn.LeakyReLU()
    )

    self.block6 = nn.Sequential(
        nn.Conv2d(ch_in + sobel_ch + ch_out, ch_out, kernel_size = 1, stride = 1, padding = 0), nn.LeakyReLU(),
        nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1), nn.LeakyReLU()
    )

    self.block7 = nn.Sequential(
        nn.Conv2d(ch_in + sobel_ch + ch_out, ch_out, kernel_size = 1, stride = 1, padding = 0), nn.LeakyReLU(),
        nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1), nn.LeakyReLU()
    )

    self.block8 = nn.Sequential(
        nn.Conv2d(ch_in + sobel_ch + ch_out, ch_out, kernel_size = 1, stride = 1, padding = 0),nn.LeakyReLU(),
        nn.Conv2d(ch_out, 1, kernel_size = 1, stride = 1, padding = 0)
    )

    self.tail = nn.LeakyReLU()
  
  def forward(self, x):
    out_0 = self.edge_enhance(x)
    out_0 = torch.cat((x, out_0), dim = 1)

    out_1 = self.block1(out_0)
    out_1 = torch.cat((out_0, out_1), dim = 1)

    out_2 = self.block2(out_1)
    out_2 = torch.cat((out_0, out_2), dim = 1)

    out_3 = self.block3(out_2)
    out_3 = torch.cat((out_0, out_3), dim = 1)

    out_4 = self.block4(out_3)
    out_4 = torch.cat((out_0, out_4), dim = 1)

    out_5 = self.block5(out_4)
    out_5 = torch.cat((out_0, out_5), dim = 1)

    out_6 = self.block6(out_5)
    out_6 = torch.cat((out_0, out_6), dim = 1)

    out_7 = self.block7(out_6)
    out_7 = torch.cat((out_0, out_7), dim = 1)

    out_8 = self.block8(out_7)

    out = self.tail(x + out_8)

    return out

if __name__ == "__main__":
  sample = torch.ones((2, 1, 512, 512))
  net = EDCNN()
  out = net(sample)
  print(out.shape)




    
