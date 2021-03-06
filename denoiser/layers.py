"""
contains the code for common layers to implement 
ASPP / SSA Module / ResBlock etc..

TODO
(1) SSA Module (O)
(2) ASPP Module (O)
(3) Squeeze & Excite Module

"""
import torch
import torch.nn as nn
import torch.functional as F

## Common Layers for all Networks ##
def conv1x1(ch_in, ch_out, stride):
    model = nn.Conv2d(ch_in, ch_out, kernel_size = (1,1), stride = stride, padding = 0)
    return model

def conv3x3(ch_in, ch_out, stride):
    model = nn.Conv2d(ch_in, ch_out, kernel_size = (3, 3), stride = stride, padding = 1)
    return model

## ASPP Layer
class ASPP(nn.Module):
    def __init__(self, ch_in, ch_out, rate = [6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = rate[0], dilation = rate[0]),
            nn.ReLU(inplace = True), nn.BatchNorm2d(ch_out),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = rate[1], dilation = rate[1]),
            nn.ReLU(inplace = True), nn.BatchNorm2d(ch_out),
        )
        self.aspp3 = nn.Sequential(
            nn.Covn2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = rate[2], dilation = rate[2]),
            nn.ReLU(inplace = True), nn.BatchNorm2d(ch_out)
        )
        self.last = nn.Conv2d(len(rate) * ch_out, ch_out, kernel_size = 1)
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        out = torch.cat([x1,x2,x3], dim = 1)

        return self.last(out)

## SE Block
class SEBlock(nn.Module):
    def __init__(self, ch, scale = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//scale, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(ch // scale, ch, bias = False),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) ## scaling the feature map using GAP (Average Pool)



class residual_block(nn.Module):
    ## Common Residual Block
    def __init__(self, ch_in, ch_mid, ch_out, downsample = False):
        super(residual_block, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.layer = nn.Sequential(
                conv1x1(ch_in, ch_mid, stride = 2),
                conv3x3(ch_mid, ch_mid, stride = 1),
                conv3x3(ch_mid, ch_out, stride = 1),
            )
            self.identity = conv1x1(ch_in, ch_out, stride = 2)
        else:
            self.layer = nn.Sequential(
                conv1x1(ch_in, ch_mid, stride = 1),
                conv3x3(ch_mid, ch_mid, stride = 1),
                conv3x3(ch_mid, ch_out, stride = 1),
            )
            self.identity = conv1x1(ch_in, ch_out, sride = 1)
    def forward(self, x):
        out = self.layer(x)
        if self.downsample == False:
            if out.size() is not x.size():
                x = self.identity(x)
        return out + x



class upsample(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride):
        super(upsample, self).__init__()
        self.layer = nn.ConvTranspose2d(ch_in, ch_out, kernel_size = kernel, stride = stride)
    def forward(self, x):
        return self.layer(x)

## Layers only for NBNet ##
class nbnet_conv_block(nn.Module):
    ## Convolution Block for NBNet
    def __init__(self, in_ch, out_ch, kernel_size = 3, padding = 1, stride = 1,  downsample = False):
        super(nbnet_conv_block, self).__init__()
        layers = [
                nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size,stride = stride, padding = padding,),
                nn.LeakyReLU(0.2, inplace = False),
                nn.Conv2d(in_channels = out_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding,),
                nn.LeakyReLU(0.2, inplace = False)
            ]
        self.skip = not downsample ## downsample = False -> skip = True
        if downsample:
            self.downsample = nn.Conv2d(in_channels = out_ch, out_channels = out_ch,
                kernel_size = 4, stride = 2, padding = 1)
        self.conv_block = nn.ModuleList(layers)
        self.identity = nn.Conv2d(in_channels = in_ch, out_channels = out_ch,
            kernel_size = 1, stride = 1, padding = 0)
    def forward(self, x):
        out = x
        for idx, block in enumerate(self.conv_block):
            out = block(out)
        # out = self.conv_block(x)
        x = self.identity(x)
        if self.skip == False: # downsample = True
            return self.downsample(out + x), out+x
        else: # downsample = False
            return out + x



class ssa(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out):
        super(ssa, self).__init__()
        self.out_ch = ch_out ## number of output channels
        self.conv = nbnet_conv_block(ch_in, ch_mid)
    def forward(self, x1, x2):
        b_, c_, w_, h_ = x1.shape
        cat = torch.cat((x1, x2), dim = 1)
        cat = self.conv(cat)
        B, C, w, h = cat.shape ## (w_, h_) == (w, h)
        V_t = cat.reshape(B, C, h*w) ## reshape to (b, k, w*h)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis = 2, keepdims = True))
        V = torch.transpose(V_t, dim0 = 2, dim1 = 1)
        
        a = V;b = torch.inverse(torch.matmul(V_t, V));c = V_t
        proj_mat = torch.matmul(torch.matmul(a,b), c)
        x1_ = torch.reshape(x1, shape = (B, c_, w_*h_))
        proj_feat = torch.matmul(proj_mat, torch.transpose(x1_, 2, 1))
        bridge = torch.transpose(proj_feat,1,2)
        bridge = torch.reshape(bridge,shape=(B, c_, w, h))
        
        return bridge ## output shape should be (b_, c_, w_, h_)

class skip_block(nn.Module):
    ## Blocks for the Skip Connection ## 
    def __init__(self, in_size, out_size, rep_n = 1):
        super(skip_block, self).__init__()
        self.blocks = []
        self.rep_n = rep_n
        self.blocks.append(nbnet_conv_block(in_size, 128))
        for i in range(self.rep_n-2):
            self.blocks.append(nbnet_conv_block(128, 128))
        self.blocks.append(nbnet_conv_block(128, out_size))
        self.blocks = nn.ModuleList(self.blocks)
        self.identity = nn.Conv2d(in_size, out_size, kernel_size = 1, bias = True)
        
    def forward(self, x):
        identity = self.identity(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return x + identity


class nbnet_upblock(nn.Module):
    ## Single Decoder Block ##
    def __init__(self, ch_in, ch_out, subnet_rep_num, subspace_dim = 16):
        super(nbnet_upblock, self).__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size = 2, stride = 2, bias = True)
        self.conv_block = nbnet_conv_block(ch_in, ch_out)
        self.num_subspace = subspace_dim
        self.ssa = ssa(ch_in, subspace_dim, ch_out)
        self.skip_m = skip_block(ch_out, ch_out, subnet_rep_num)
    
    def forward(self,x, bridge):
        # x : decoder input
        # bridge : encoder input (x2)
        up = self.up(x)
        bridge = self.skip_m(bridge)
        bridge = self.ssa(bridge, up)
        out = torch.cat((bridge, up), dim = 1)
        return self.conv_block(out)
        
### Layer Implementation for FiLM ###
  
        
if __name__ == "__main__":
    ssa = ssa(5, 15, 20)
    x1 = torch.ones((2, 5, 10, 10));x2 = torch.ones((2, 5, 10, 10))
    out = ssa(x1, x2)
    print(out.shape)
        
        
        
    
