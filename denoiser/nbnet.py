import torch
import torch.nn as nn
from layers import *

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
        self.tail = conv3x3(prev_channels, ch_in, stride = 1)
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
    sample = torch.ones((5, 1, 128, 128))
    net = NBNet(1)
    output = net(sample)
    print(output.shape)
            
        
        
        