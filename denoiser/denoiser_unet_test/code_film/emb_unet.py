import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model

class FeatExtractor(nn.Module):
    def __init__(self, ch_in = 1):
        super(FeatExtractor, self).__init__()
        self.ch_in = ch_in
        ## Using Pretrained ResNext101 as the Feature Extractor
        self.net = model.resnext101_32x8d(pretrained = True)
    def forward(self, x):
        if x.size(1) == 1:
          x = torch.cat([x, x, x], dim = 1)
        return self.net(x)
        
class FiLM(nn.Module):
    def __init__(self, feat_in, feat_out):
        super(FiLM, self).__init__()
        self.fin = feat_in
        self.fout = feat_out
        self.lin = nn.Linear(feat_in, feat_out)
    def forward(self, img, x):
        b = x.size(0)
        vec = self.lin(x)
        vec = vec.view(b, self.fout//2, 2)
        gamma = torch.unsqueeze(torch.unsqueeze(vec[:, :, 0], dim = -1), dim = -1)
        beta = torch.unsqueeze(torch.unsqueeze(vec[:, :, 1], dim = -1), dim = -1)
        return (img * gamma) + beta

class SELayer(nn.Module):
    def __init__(self, ch_in, scale):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(ch_in, ch_in // scale)
        self.relu = nn.ReLU(inplace = True)
        self.lin2 = nn.Linear(ch_in // scale, ch_in)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        scale = self.pool(x)
        print(scale.shape)
        b, c = scale.size(0), scale.size(1)
        scale = scale.reshape((b, c))
        scale = self.sig(self.lin2(self.relu(self.lin1(scale))))
        scale = scale.reshape((b, c, 1, 1))
        return scale * x

class EncBlock(nn.Module):
    def __init__(self, ch_in, ch_out, downsample = True):
        super(EncBlock, self).__init__()
        self.ds = downsample
        self.downsample = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.block1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(ch_out, affine = True),
            nn.ReLU(inplace = True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ch_out,  ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(ch_out, affine = True),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        if self.ds:
            x = self.downsample(x)
        return self.block2(self.block1(x))

class DecBlock(nn.Module):
    def __init__(self, ch_in, ch_out, upsample = True):
        super(DecBlock, self).__init__()
        self.us = upsample
        self.upsample = nn.ConvTranspose2d(ch_in, ch_in, kernel_size = 2, stride = 2, dilation = 2, padding = 1, output_padding= 1)
        self.block1 = nn.Sequential(
            nn.Conv2d(ch_in * 2, ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(ch_out, affine = True),
            nn.ReLU(inplace = True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(ch_out, affine = True),
            nn.ReLU(inplace = True)
        )
    def forward(self, x1, skip):
        if self.us:
            x1 = self.upsample(x1)
        x = torch.cat([x1, skip], dim = 1)
        x = self.block2(self.block1(x))
        return x

class EmbUNet(nn.Module):
    def __init__(self, ch_in = 1, scale = 32):
        super(EmbUNet, self).__init__()
        self.feat_ext = FeatExtractor()
        self.film = FiLM(1000, scale * 2) ## film Layer for first UNet Encoder Block
        self.head = nn.Conv2d(ch_in, scale, kernel_size = 3, stride = 1, padding = 1)
        ## Encoder ##
        self.down1 = EncBlock(scale, scale, downsample = False)
        self.down2 = EncBlock(scale, scale * 2, downsample = True)
        self.se1 = SELayer(scale * 2, 16)
        self.down3 = EncBlock(scale * 2, scale * 4, downsample = True)
        self.se2 = SELayer(scale * 4, 16)

        self.bridge = EncBlock(scale * 4, scale * 4, downsample = True)

        self.up1 = DecBlock(scale * 4, scale * 2, upsample = True)
        self.up2 = DecBlock(scale * 2, scale, upsample = True)
        self.up3 = DecBlock(scale, scale // 2, upsample = True)

        self.tail = nn.Conv2d(scale // 2, ch_in, kernel_size = 1, stride = 1, padding = 0)
    def forward(self, x):
        head = self.head(x)
        feat = self.feat_ext(x)
        film = self.film(head, feat)
        
        d1 = self.down1(film)
        d2 = self.se1(self.down2(d1))
        d3 = self.se2(self.down3(d2))
        bridge = self.bridge(d3)

        u1 = self.up1(bridge, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        
        out = self.tail(u3)
        return out
