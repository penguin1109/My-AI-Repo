from layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
unet.py
TODO
(1) Implement a simple UNet (Most Basic) (O)
(2) Implement a UNet with FiLM Layers (Noise Conditional Training)
(3) Implement a UNet++ (With Super Vision)
(4) Implement a UNet with Attention
(5) Implement a UNet with the Residual Block 
"""

class encoding_block(nn.Module):
    def __init__(self,  ch_in, ch_out, ksize = 3, padding = 0, stride = 1, dilation = 1, bn = False):
        super(encoding_block, self).__init__()
        self.bn = bn
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(padding = (ksize-1)//2),
            nn.Conv2d(ch_in, ch_out, kernel_size = ksize, padding = padding, stride = stride, dilation = dilation),
            nn.PReLU())
        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(padding = (ksize-1)//2),
            nn.Conv2d(ch_out, ch_out, kernel_size = ksize, padding = padding, stride = stride, dilation = dilation),
            nn.PReLU(), )
        if self.bn:
            self.bn1 = nn.BatchNorm2d(ch_out)
            self.bn2 = nn.BatchNorm2d(ch_out)
    def forward(self, x):
        x1 = self.layer1(x)
        if self.bn:
            x1 = self.bn1(x1)
        x2 = self.layer2(x1)
        if self.bn:
            x2 = self.bn2(x2)
        return x2
        
        
class decoding_block(nn.Module):
    def __init__(self, ch_in, ch_out, upsampling = True):
        super(decoding_block, self).__init__()
        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode = 'bilinear', scale_factor = 2),
                nn.Conv2d(ch_in, ch_out, kernel_size = 1)
            )
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size = 2, stride = 2)
        self.conv = encoding_block(ch_in, ch_out,)
    def forward(self, in1, in2):
        out2 = self.up(in2)
        out1 = F.upsample(in1, out2.size()[2:], mode = 'bilinear')
        return self.conv(torch.cat([out1, out2], dim = 1))
        
class UNet(nn.Module):
    ## Simple UNet without the Residual Block
    def __init__(self, ch_in = 1):
        super(UNet, self).__init__()
        feat_n = 32
        feats = [feat_n * (2**i) for i in range(5)]
        ## Encoder ##
        self.conv1 = encoding_block(ch_in, feats[0], bn = True)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv2 = encoding_block(feats[0], feats[1], bn = True)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv3 = encoding_block(feats[1], feats[2], bn = True)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv4 = encoding_block(feats[2], feats[3], bn = True)
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2)
        
        ## Bridge ##
        self.bridge = encoding_block(feats[3], feats[4], bn = True)
        
        ## Decoder ##
        self.dec4 = decoding_block(feats[4], feats[3])
        self.dec3 = decoding_block(feats[3], feats[2])
        self.dec2 = decoding_block(feats[2], feats[1])
        self.dec1 = decoding_block(feats[1], feats[0])
        
        ## Tail ##
        self.tail = nn.Conv2d(feats[0], ch_in, kernel_size = 1)
    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.maxpool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.maxpool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.maxpool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.maxpool4(conv4)
        
        bridge = self.bridge(pool4)
        
        dec4 = self.dec4(conv4, bridge)
        dec3 = self.dec3(conv3, dec4)
        dec2 = self.dec2(conv2, dec3)
        dec1 = self.dec1(conv1, dec2)
        
        tail = F.upsample(self.tail(dec1), x.size()[2:], mode = 'bilinear')
        return tail
    
if __name__ == "__main__":
    sample = torch.ones((2, 1, 512, 512))
    net = UNet()
    output = net(sample)
    print(output.shape)
        