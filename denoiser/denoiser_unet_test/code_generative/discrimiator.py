## denoiser_unest_test/generative_model/discriminator.py

class Discriminator(nn.Module):
  def __init__(self, ch_in = 1):
    super(Discriminator, self).__init__()
    def discriminator_block(ch_in, ch_out, normalization = True):
      layers = [nn.Conv2d(ch_in, ch_out, kernel_size = 5, stride = 2, padding = 2)]
      if normalization:
        layers.append(nn.InstanceNorm2d(ch_out))
      layers.append(nn.LeakyReLU(0.2, inplace = True))
      return layers
  
    self.model = nn.Sequential(
        *discriminator_block(ch_in * 2, 64, normalization = False),
        *discriminator_block(64, 128),
        *discriminator_block(128, 256),
        *discriminator_block(256, 512),
        nn.ZeroPadd((1, 0, 1, 0)),
        nn.Conv2d(512, 1, kernel_size = 5, stride = 1, paddng = 2, bias = False),
        nn.Tanh(inplace = True)
    )

  def forward(self, x):
    return self.model(x)
