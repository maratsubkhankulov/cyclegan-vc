import torch

class Downsample(torch.nn.Module):
   
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
      super(Downsample, self).__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.conv = torch.nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding)
      self.norm = torch.nn.InstanceNorm1d(num_features=out_channels, affine=True)
      self.glu = torch.nn.GLU(dim=1)
    
    def forward(self, x):
      """ [B, D, T] -> [B, D, T]"""
      x = self.conv(x)
      x = self.norm(x)
      x = self.glu(x)
      return x
  
class ResidualBlock(torch.nn.Module):
  def __init__(self, in_channels1, out_channels1, in_channels2, out_channels2, kernel_size, stride):
    super(ResidualBlock, self).__init__()
    self.conv1 = torch.nn.Conv1d(in_channels=in_channels1,
                                out_channels=out_channels1,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding='same')
    self.norm1 = torch.nn.InstanceNorm1d(num_features=out_channels1, affine=True)
    self.glu = torch.nn.GLU(dim=1)
    self.conv2 = torch.nn.Conv1d(in_channels=in_channels2,
                                out_channels=out_channels2,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding='same')
    self.norm2 = torch.nn.InstanceNorm1d(num_features=out_channels2, affine=True)
  
  def forward(self, x):
    """ [B, D, T] -> [B, D, T]"""
    residual = x.clone()
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.glu(x)
    x = self.conv2(x)
    x = self.norm2(x)
    return x + residual

# from https://github.com/serkansulun/pytorch-pixelshuffle1d
class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        """ [B, D, T] -> [B, D/upscale_factor, T*upscale_factor]"""
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class Upsample(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0)):
      super(Upsample, self).__init__()
      self.conv = torch.nn.Conv1d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding)
      self.pixel_shuffle = PixelShuffle1D(upscale_factor=2)
      self.norm = torch.nn.InstanceNorm1d(num_features=out_channels//2, affine=True)
      self.glu = torch.nn.GLU(dim=1)
    
    def forward(self, x):
      x = self.conv(x)
      x = self.pixel_shuffle(x)
      x = self.norm(x)
      x = self.glu(x)
      return x

class Generator(torch.nn.Module):
    
    def __init__(self):
      super(Generator, self).__init__()
      self.conv1 = torch.nn.Conv1d(in_channels=24, out_channels=128, kernel_size=5, stride=1, padding=2)
      self.glu = torch.nn.GLU(dim=1)
      self.downsample_twice = torch.nn.Sequential(
        Downsample(in_channels=64, out_channels=256, kernel_size=5, stride=2, padding=2),
        Downsample(in_channels=128, out_channels=512*2, kernel_size=5, stride=2, padding=2),
      )
      self.residual_blocks = torch.nn.Sequential(
        *[ResidualBlock(in_channels1=512, out_channels1=1024,
                        in_channels2=512, out_channels2=512,
                        kernel_size=3, stride=1) for _ in range(6)]
      )
      self.upsample_twice = torch.nn.Sequential(
         Upsample(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=2),
         Upsample(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2),
      )
      self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=24, kernel_size=15, stride=1, padding='same')

    def forward(self, x):
      """ [B, D, T] -> [B, D, T]
      B - batch size
      D - number of mel-cepstral coefficients
      T - number of time steps - 128 for training
      """
      x = self.conv1(x)
      x = self.glu(x)
      x = self.downsample_twice(x)
      x = self.residual_blocks(x)
      x = self.upsample_twice(x)
      x = self.conv2(x)
      return x

class Downsample2d(torch.nn.Module):
   
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=[0, 0]):
      super(Downsample2d, self).__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding)
      self.norm = torch.nn.InstanceNorm2d(num_features=out_channels, affine=True)
      self.glu = torch.nn.GLU(dim=1)
    
    def forward(self, x):
      """ [B, D, T] -> [B, D, T]"""
      x = self.conv(x)
      x = self.norm(x)
      x = self.glu(x)
      return x

class Discriminator(torch.nn.Module):
   
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=(1,1))
    self.glu = torch.nn.GLU(dim=1)
    
    self.d1 = Downsample2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
    self.d2 = Downsample2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
    self.d3 = Downsample2d(in_channels=256, out_channels=1024, kernel_size=(6, 3), stride=(1, 2), padding=(0,0))
    
    self.fc = torch.nn.Linear(in_features=3584, out_features=1)
    self.sigmoid = torch.nn.Sigmoid()     

  def forward(self, x):
    x = self.conv1(x)
    x = self.glu(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = x.flatten(start_dim=1)
    x = self.fc(x)
    x = self.sigmoid(x)
    return x

class CycleGAN(torch.nn.Module):
      
      def __init__(self):
        super(CycleGAN, self).__init__()
        self.Gx_y = Generator()
        self.Gy_x = Generator()
        self.Dy = Discriminator()
        self.Dx = Discriminator()