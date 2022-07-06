import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math

def gaussian(ksize,sigma):
    """fnc for creating the gaussian distribution for a given matrix with (ksize x ksize)"""
    gauss = torch.Tensor([math.exp(-(x-ksize//2)**2) / float(2*sigma**2)])
    return gauss / gauss.sum() # Normalize to range [0-1]


def create_window(ksize, channel):
    1D_window = gaussian(ksize, 1.5).unsqueeze(1)
    2D_window = 1D_window.mm(1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(2D_window.expand(channel, 1, ksize, ksize))
    return window

def _ssim(img1, img2, window, ksize, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = ksize//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = ksize//2, groups = channel)

    mu1_sq = mu1.pow(2) # mu1 ^ 2
    mu2_sq = mu2.pow(2) # mu2 ^ 2
    mu1_mu2 = mu1 * mu2 # mu1 ^ 2 * mu2 ^ 2

    sigma1_sq = F.conv2d(img1 * img1, window, padding = ksize // 2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding = ksize//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding =ksize//2, groups = channel) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    c3= c2/2

    ssim_map = ((2*mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) *(sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, ksize = 11, size_average = True):
    B, C, W, H = img1.size()
    window = create_window(ksize, C)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, ksize, channel, size_average)

