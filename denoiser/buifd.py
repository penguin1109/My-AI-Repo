import torch
import torch.nn as nn

# Fusion Net는 말 SNR (Signal To Noise Ratio)를 학습한다.

class DNCNN(nn.Module):
  # A denoising Network only based on the CNN Layer with kernel size = 3 and No Padding
  # Predicts the Noise Values in the Noisy Input Image
  # Subtracts the Noise values from the Noisy Input
    def __init__(self, channels, layer_n = 17):
        super(DNCNN, self).__init__()
        ksize = 3;padding = 1;features = 64
        layers = []
    ## First Convolution Layer ##
        layers.append(nn.Conv2d(in_channels = channels, out_channels = features, kernel_size = ksize, padding = padding, bias = False))
        layers.append(nn.ReLU(inplace = True))
    ## Middle Convolution Layer ##
        for _ in range(layer_n - 2):
            layers.append(nn.Conv2d(in_channels = features, out_channels = features, kernel_size = ksize, padding = padding, bias = False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace = True))
    ## Final Convolution Layer ##
        layers.append(nn.Conv2d(in_channels = features, out_channels = channels, kernel_size = ksize, padding = padding, bias = False))
    
        self.dncnn = nn.ModuleList(layers)
    
    def forward(self, x):
        for idx, layer in enumerate(self.dncnn):
          x = layer(x)
        return x
  
class DnCNN_RL(nn.Module):
    def __init__(self, ch_n, layer_n = 17):
        super(DnCNN_RL, self).__init__()
        self.dncnn = DNCNN(channels = ch_n, layer_n = layer_n)
    def forward(self, x):
        return self.dncnn(x)
 

class FinalFusion(nn.Module):
    def __init__(self, ch):
        super(FinalFusion, self).__init__()
        ksize = 3;padding = 1;features = 64;dilation = 1
        layers = []
    ## Five Different Inputs are Sent to the Final Fusion Network
    # noise-level, prior image, noisy input, 
    # pixel-wise multiplication(prior image, noise-level)
    # pixel-wise multiplication(noisy input, noise-level)
        layers.append(nn.Conv2d(
        in_channels = 5 * ch, out_channels = features, kernel_size = ksize, padding = padding, dilation = dilation, bias = False
        ))
        layers.append(nn.Conv2d(
        in_channels = features, out_channels = features, kernel_size = ksize, padding = padding, dilation = dilation, bias = False
        ))
        layers.append(nn.Conv2d(
        in_channels =features, out_channels = features, kernel_size = ksize, padding = padding, dilation = dilation, bias = False
        ))
        self.fusion_layers = nn.ModuleList(layers)
    def forward(self, x,y,z):
        noisy_input = y
        prior = x
        noise_level = z
        channel_0 = noisy_input;channel_1 = prior;channel_2 = noise_level
        channel_3 = noisy_input * (1-noise_level)
        channel_4 = prior * noise_level
        out = torch.cat((channel_0, channel_1, channel_2, channel_3, channel_4), dim = 1)
        for idx, layer in enumerate(self.fusion_layers):
            out = layer(out)
        return out
    
    
    
class NoiseCNN(nn.Module):
    ## Noise Level CNN
    def __init__(self, channels, layer_n = 5):
        super(NoiseCNN, self).__init__()
        ksize = 5;padding = 2;features = 64
        layers =[]
        ## First Convolution Layer ##
        layers.append(nn.Conv2d(in_channels = channels, out_channels = features, kernel_size = ksize, padding = padding, bias = False))
        layers.append(nn.ReLU(inplace = True))
        ## Middle Convolution Layer ##
        for _ in range(layer_n):
            layers.append(nn.Conv2d(in_channels = features, out_channels = features, kernel_size = ksize, padding = padding, bias = False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace = True))
        ## Final Convolution Layer ##
        layers.append(nn.Conv2d(in_channels = features, out_channels = channels, kernel_size = ksize, padding = padding, bias = False))
    
        self.noisecnn = nn.ModuleList(layers)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        for idx, layer in enumerate(self.noisecnn):
            x = layer(x)
        noise_level = x
        noise_level = self.sigmoid(noise_level)
        return noise_level
  
class BUIFD(nn.Module):
    def __init__(self, ch=1, n_layer = 17):
        super(BUIFD, self).__init__()
        self.dncnn = DNCNN(ch, n_layer)
        self.noisecnn = NoiseCNN(ch)
        self.finalfusion = FinalFusion(ch)
    def forward(self, x):
        noisy = x
        noise = self.dncnn(x)
        prior = noisy - noise

        noise_level = self.noisecnn(x)

        denoised = self.finalfusion(prior, noisy, noise_level)
        return denoised
