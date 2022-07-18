## ECNDNet.py ##
## ECNDNet
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation = 1):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = dilation, dilation = dilation),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.model(x)
    
class ECNDNet(nn.Module):
    def __init__(self, ch_in = 1, ch_out = 64):
        super(ECNDNet, self).__init__()
        self.layers = []
        self.head = nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1)
        
        for i in range(1, 16):
            if i in [2, 5, 9, 12]:
                self.layers.append(ConvBlock(ch_out, ch_out, dilation = 2))
            else:
                self.layers.append(ConvBlock(ch_out, ch_out, dilation = 1))
        self.layers = nn.Sequential(*self.layers)
        self.tail = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(ch_out, 1, kernel_size = 3, stride = 1, padding = 1)
        )
    def forward(self, x):
        head = self.head(x)
        head = self.layers(head)
        tail = self.tail(head)
        return (tail + x)
        
