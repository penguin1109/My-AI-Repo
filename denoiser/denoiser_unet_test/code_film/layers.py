## denoiser_unet_test/code_film/layers.py
import torch
import torch.nn as nn

def conv(channel_in, channel_out, kernel_size, stride, padding):
  return nn.Sequential(
      nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding),
      nn.ReLU(inplace = True),
      nn.BatchNorm2d(channel_out)
  )
  
class FeatureExtractor(nn.Module):
  def __init__(self, ch_in, concat):
    super(FeatureExtractor,self).__init__()
    self.concat = concat
    if self.concat:
      self.ch_fin = ch_in*2*2
    else:
      self.ch_fin = ch_in*2
    self.model = nn.Sequential(
        conv(ch_in, channel_out = 128, kernel_size = 5, stride = 2, padding = 2),
        conv(128, 128, 3, 2, 1),
        conv(128, 128, 3, 2, 1),
        conv(128, 128, 3, 2, 1),
        conv(128, self.ch_fin, 3, 1, 1) ## skip-connection에 의해서 concat되어서 channel size가 2배가 되는 상황
    )
    self.gap = nn.AdaptiveAvgPool2d(1) ## alpha, beta
  def forward(self, x):
    return self.gap(self.model(x))

class FiLMLayer(nn.Module):
  def __init__(self, ch_in, concat = False):
    # [32, 64]
    super(FiLMLayer, self).__init__()
    self.concat = concat
    self.feature_extractor = FeatureExtractor(ch_in, self.concat)
  def forward(self, x, skip_x):
    batch_size, c, w, h = x.shape

    film_vector = self.feature_extractor(x)
    if self.concat:
      cat = torch.cat((x, skip_x), dim = 1)
      film_vector = film_vector.view(batch_size, c*2, 2, 1)
      for i in range(2*c):
        beta = film_vector[:, i, 0, :]
        gamma = film_vector[:, i, 1, :]
        beta = beta.view(cat.size(0), 1, 1)
        gamma = gamma.view(cat.size(0), 1, 1)
        cat[:, i, :, :] = cat[:, i, :, :] * gamma + beta
      return cat
    else:
      film_vector = film_vector.view(batch_size, c,2, 1)
      for i in range(c):
        beta = film_vector[:, i, 0, :]
        gamma = film_vector[:, i, 1, :]
        beta = beta.view(x.size(0), 1, 1)
        gamma = gamma.view(x.size(0), 1, 1)
        x[:, i, :, :] = x[:, i, :, :] * gamma + beta
      return x


   

    
    # x : input feature map (made by the convolution layer)

    
if __name__ == "__main__":
  sample = torch.zeros(size = (2, 1, 512, 512))
  fe = FeatureExtractor(1, concat = False)
  out = fe(sample)
  print(out.shape) ## (2, 128, 1, 1)
