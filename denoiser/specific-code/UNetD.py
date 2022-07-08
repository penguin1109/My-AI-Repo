import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

def conv1x1(ch_in, ch_out):
    model = nn.Conv2d(ch_in, ch_out, kernel_size = 1, stride=1, padding = 0, bias = True)
    return model

def conv3x3(ch_in, ch_out, bias=True):
    layer = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class UNetD(nn.Module):

    def __init__(self, in_chn=1, scale=32, depth=5, relu_slope=0.2, subspace_dim=16):
        super(UNetD, self).__init__()
        self.depth = depth
        self.down_path = []
        prev_channels = self.get_input_chn(in_chn)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*scale, downsample, relu_slope))
            prev_channels = (2**i) * scale

        self.up_path = []
        subnet_repeat_num = 1
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*scale, relu_slope, subnet_repeat_num, subspace_dim))
            prev_channels = (2**i)*scale
            subnet_repeat_num += 1

        self.last = conv3x3(prev_channels, in_chn, bias=True)
        self.up_path = nn.ModuleList(self.up_path)
        self.down_path = nn.ModuleList(self.down_path)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            if (i+1) < self.depth:
                x, x_up = down(x)
                blocks.append(x_up)
            else:
                x = down(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        pred = self.last(x)
        return pred

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print("weight")
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    print("bias")
                    nn.init.zeros_(m.bias)


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, subnet_repeat_num, subspace_dim=16):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)
        self.num_subspace = subspace_dim
        
        self.subnet = Subspace(in_size, self.num_subspace)
        self.skip_m = skip_blocks(out_size, out_size, subnet_repeat_num)

    def forward(self, x, bridge):
        up = self.up(x)
        bridge = self.skip_m(bridge)
        out = torch.cat([up, bridge], 1)
        if self.subnet:
            b_, c_, h_, w_ = bridge.shape
            sub = self.subnet(out)
            V_t = sub.reshape(b_, self.num_subspace, h_*w_)
            V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
            V = torch.transpose(V_t, 1,2)
            mat = torch.matmul(V_t, V)
            mat_inv = torch.inverse(mat)
            project_mat = torch.matmul(mat_inv, V_t)
            bridge_ = bridge.reshape(b_, c_, h_*w_)
            project_feature = torch.matmul(project_mat, torch.transpose(bridge_, 1,2))
            bridge = torch.matmul(V, project_feature)
            bridge = torch.transpose(bridge, 1, 2).reshape(b_, c_, h_, w_)
            out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = []
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.blocks  = nn.ModuleList(self.blocks)
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for idx, b in enumerate(self.blocks):
            x = b(x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = []
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        sc = self.shortcut(x)
        for idx, m in enumerate(self.blocks):
            x = m(x)
        return x + sc


class NBNet(nn.Module):
    def __init__(self, ch_in, scale = 32, depth = 5, relu_slope = 0.2, subspace_dim = 16):
        super(NBNet, self).__init__()
        self.depth = depth
        self.down_pth = []
        prev_channels = ch_in
        # [1(input) -> 32(1st) -> 64(2nd) -> 128(3rd) -> 256(4th)]
        for i in range(depth):
            downsample = True if (i+1) < depth else False ## 1, 2, 3, 4에 대해서만 down-sample
            self.down_pth.append(nbnet_conv_block(prev_channels, (2**i)*scale, downsample = downsample))
            prev_channels = (2**i)*scale
            
        self.up_pth = [];subnet_rep_num = 1;
        for i in reversed(range(depth-1)):
            self.up_pth.append(nbnet_upblock(prev_channels, (2**i)*scale,subnet_rep_num, subspace_dim))
            prev_channels = (2**i)*scale
            subnet_rep_num += 1
        self.tail = conv3x3(prev_channels, ch_in,)
        self.down_pth = nn.ModuleList(self.down_pth)
        self.up_pth = nn.ModuleList(self.up_pth)
        
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_pth):
            if (i + 1) < self.depth: # downsample O
                x, x_up = down(x) # (encoder input, skip connection input)
                blocks.append(x_up)
            else: # downsample X
                x = down(x)  
        # x = [B, 512, W, H]
        for i, up in enumerate(self.up_pth):
            x = up(x, blocks[-i-1])

        output = self.tail(x)
        return output

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xaviar_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
                    

if __name__ == "__main__":
    sample = torch.zeros((2, 1, 512, 512))
    net1= UNetD(1)
    net2 = NBNet(1)
    # print(net1)
    print(net2)
    #out = net(sample)
    #print(out.shape)
