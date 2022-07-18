## FFDNet.py ##
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, batch_norm = False, relu = True):
        super(ConvBlock, self).__init__()
        if batch_norm:
            self.norm = nn.BatchNorm2d(ch_out)
        else:
            self.norm = nn.LayerNorm(normalized_shape = [ch_out, 512, 512])
        if relu:
            self.activation = nn.ReLU(inplace = True)
        else:
            self.activation = nn.Tanh()
            
        self.model = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            self.norm, self.activation
        )
    def forward(self, x):
        return self.model(x)
    
class FFDNet(nn.Module):
    def __init__(self, ch_in, feat = 64, ch_out = 4):
        super(FFDNet, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(ch_in, feat, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(feat)
        )
        self.middle = []
        for i in range(15):
            self.middle.append(ConvBlock(feat, feat, True, True))
        self.middle = nn.Sequential(*self.middle)
        
        self.tail = nn.Conv2d(feat, ch_out, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        head = self.head(x)
        middle = self.middle(head)
        tail = self.tail(middle)
        out = torch.sum(tail, dim = 1, keepdim = True) ## Channel기준으로 ch_out개수만큼의 feature map을 모두 더해준다.
        return out

    
    
        
        
