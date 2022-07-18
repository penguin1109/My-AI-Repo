## denoiser_unet_test/code_general/tools.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.modules.loss import _Loss


class ResNetFeatExtractor(nn.Module):
  def __init__(self, blocks = [1,2, 3, 4], pretrained = True, progress = True):
    super(ResNetFeatExtractor, self).__init__()
    self.model = models.resnet50(pretrained, progress)
    del self.model.avgpool
    del self.model.fc
    self.blocks = blocks
  
  def forward(self, x):
    feats = []
    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)
    x = self.model.layer1(x)

    if 1 in self.blocks:
      feats.append(x)
    x = self.model.layer2(x)
    if 2 in self.blocks:
      feats.append(x)
    x = self.model.layer3(x)
    if 3 in self.blocks:
      feats.append(x)
    x = self.model.layer4(x)
    if 4 in self.blocks:
      feats.append(x)
    return feats
  
class CompoundLoss(_Loss):
  def __init__(self, blocks = [1,2, 3, 4], mse_weight = 1, resnet_weight = 0.01):
    super(CompoundLoss, self).__init__()
    self.mse_weight = mse_weight
    self.resnet_weight = resnet_weight
    self.blocks = blocks
    self.model = ResNetFeatExtractor(pretrained = True)
    if torch.cuda.is_available():
      self.model = self.model.cuda()
    self.model.eval()
    self.criterion = nn.MSELoss()
  
  def forward(self, x, y):
    running_loss = 0.0
    input_feats = self.model(torch.cat([x, x, x], dim = 1))
    target_feats = self.model(torch.cat([y, y, y], dim = 1))
    feats_num = len(self.blocks)

    for i in range(feats_num):
      running_loss += self.criterion(input_feats[i], target_feats[i])
    running_loss /= feats_num
    loss = self.mse_weight * self.criterion(x, y) + self.resnet_weight * running_loss
    return loss
