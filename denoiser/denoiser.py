import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio

"""
denoiser-unet-test > code > denoiser.py
contains the pipeline code for training the denoiser and evalutaing the denoiser
"""

class DenoiserBasePipeline:
    def __init__(self, model, device, criterion, optimizer, scheduler = None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _forward(self, data):
        output = self.model(data.to(device)) ## (B, 1, 512, 512)
        return output

    def train_step(self, data):
        x, y = data[0], data[1]
        prediction = self._forward(x)
        loss = self.criterion(prediction, y) ## change the criterion to get (prediction, ground truth) 
        return prediction, loss
    
    def validate(self, data_loader):
        validate_loop = tqdm(data_loader) # iterable data loader
        running_loss = 0.0
        for idx, data in enumerate(validate_loop):
            x, y = data[0], data[1]
            prediction = self._forward(x)
            running_loss += self.criterion(prediction, y.to(self.device))
        return running_loss / len(validate_loop)
    
    def test(self, data_loader):
        psnr = 0
        test_loop = tqdm(data_loader)
        for idx, data in enumerate(test_loop):
            x, y = data[0], data[1]
            prediction = self._forward(x) ## _forward func에서 알아서 device 할당
            # to calculate the PSNR using the skimage library, the img_in and img_out must be ndarray
            prediction = np.clip(prediction.cpu().detach().numpy()[0, 0], 0, 1).astype(np.float32)
            y = y.numpy() # device할당은 안되어 있음
            psnr += peak_signal_noise_ratio(prediction, y)
        return psnr / len(test_loop)


class DenoiserTrainer():
    def __init__(self, ):
        pass

class DenoiserEncDec():
    """Instance of the Denoiser on an Encoder-Decoder based Structure"""
    def __init__(self, model_pt, in_ch, device):
        self.model = torch.jit.load(model_pt) ## load the pretrained denoiser
        self.in_ch = in_ch
        self.device = device
        self.model.to(self.device)

    @propery
    def model(self):
        return self._model
    
    def _load_model(self, model_pth, n_channels):
        self._model = torch.jit.load(model_pth)
    def run(self, img):
        return
    
    def infer(self, imgs):
        return
        
