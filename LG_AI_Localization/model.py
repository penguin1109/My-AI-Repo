import torch, timm
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from modules import *

class BaseLine(nn.Module):
    ## 기본적으로 baseline으로 사용하는, 제일 간단한 network 구조인 backbone cnn + fc layer이다.
    def __init__(self,num_cls = 1, *args):
        # num_cls는 regression이면 1이다.
        self.backbone = timm.create_model(args.bbone, pretrained = True, num_classes = 0) ## num_class = 0으로 설정해야 출력이 pooling된 이후의, classifier이전의 층이다.
        self.out_size = self.backbone.num_features
        self.fc = nn.Linear(self.out_size, num_cls)
        self.relu = nn.ReLU(inplace = True)

    def forward(self,x):
        feats = self.backbone(x)
        return self.fc(self.relu(x))

class RAN(nn.Module):
    """
    RAN (Residual Attention Network)
    """
    def __init__(self, *args):
        super(RAN, self).__init__()
        self.args = args
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.layer2 = ResBlock(64, 64, 256, False)
        self.layer3 = ResBlock(128, 128, 512, True)
        self.layer4 = ResBlock(256, 256, 1024, True)
        self.layer5 = nn.Sequential(
            ResBlock(512, 512, 2048, False),
            ResBlock(512, 512, 2048, False),
            ResBlock(512, 512, 2048, True),
        )
        self.fc = nn.Linear(2048, 1)
        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)

        for m in self.module():
            if isinstance(m, nn.Linear):
                nn.init_xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Conv2d):
                nn.init_xavier_uniform_(m.weight.data)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc(self.acgpool(self.layer5(out)))
        return out

class Regressor(nn.Sequential):
    def __init__(self, in_ch, out_ch, d_rate):
        super(Regressor, self).__init__()
        self.covnA = nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1)
        self.leakyreluA = nn.ReLU()
        self.convB = nn.Conv2d(out_ch, out_ch, kernel_size = 1, stride = 1)
        self.leakyreluB = nn.ReLU()
        self.dropout = nn.Dropout(p = d_rate)
        self.convC = nn.Conv2d(out_ch, 1, kernel_size = 1, stride = 1)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.leakyreluA(self.convA(x))
        x = self.leakyreluB(self.convB(x))
        x = self.activation(self.convC(self.dropout(x)))
        return x

class Global_Regressor(nn.Module):
    def __init__(self,*args):
        super(Global_Regressor, self).__init__()
        self.args = args
        self.encoder = ptcv_get_model("bn_vgg_16", pretrained = True)
        self.avg_pool = nn.AvgPool2d(kernel_size = 7)
        self.regressor = Regressor(in_ch = 1536, out_ch = 512)

    def forward_siamese(self, x):
        x = self.encoder.features.stage1(x)
        x = self.encoder.features.stage2(x)
        x = self.encoder.features.stage3(x)
        x = self.encoder.features.stage4(x)
        x = self.encoder.features.stage5(x)
        x = self.avg_pool(self.encoder.features.stage6(x))
        return x

    def forward(self, phase, **kwargs):
        if phase == 'train':
            x_1_1, x_1_2, x_2 = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2']
            x_1_1 = self.forward_siamese(x_1_1)
            x_1_2 = self.forward_siamese(x_1_2)
            x_2 = self.forward_siamese(x_2)
            x = torch.cat([x_1_1, x_1_2, x_2], dim = 1)
            output = self.regressor(x)
            return output
        elif phase == 'test': ## test mode
            x_1_1, x_1_2, x_2 = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2']
            x = torch.cat([x_1_1, x_1_2, x_2], dim = 1)
            output = self.regressor(x)
            return output
        elif phase == 'extract':
            return self.forward_siamese(kwargs['x'])

class Local_Regressor(nn.Module):
    def __init__(self, *args):
        super(Local_Regressor, self).__init__()
        self.reg_num = args.reg_num
        self.encoder = nn.ModuleList([
            ptcv_get_model("bn_vgg_16", pretrained = True) for _ in range(self.reg_num)
        ])
        self.avg_pool = nn.AvgPool2d(kernel_size = 7)
        ## 최종적으로 나이를 예측하는데 사용되는 regressor
        self.regressor = nn.ModuleList([
            Regressor(1536, 512) for _ in range(self.reg_num)
        ])
    def forward_siamese(self, x, idx):
        x = self.encoder[idx].features(x)
        x = self.avg_pool(x)
        return x
    def forward(self, phase, **kwargs):
        if phase == 'train':
            x_1, x_2, x_test, idx = kwargs['x_1'], kwargs['x_2'], kwargs['x_test'], kwargs['idx']
            x_1, x_2, x_test = self.forward_siamese(x_1, idx), self.forward_siamese(x_2, idx), self.forward_siamese(x_test, idx)
            x_cat = torch.cat([x_1, x_2, x_test], dim = 1)
            outs = self.regressor[idx](x_cat)
            return outs.squeeze()
        elif phase == 'test':
            x_1_1, x_1_2, x_2, idx = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2'], kwargs['idx']
            x = torch.cat([x_1_1, x_1_2, x_2], dim = 1)
            return self.regressor[idx](x)
        elif phase == 'extract':
            x, idx = kwargs['x'], kwargs['idx']
            x = self.forward_siamese(x, idx)
            return x