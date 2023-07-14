import unittest
import torch

from model import CycleGAN, Discriminator, Downsample, Generator, ResidualBlock, Upsample

class ModelTest(unittest.TestCase):
  def test_residual_block(self):
    residual = ResidualBlock(in_channels1=1024, out_channels1=1024, 
                in_channels2=512, out_channels2=1024,
                kernel_size=3, stride=1)

    residual.forward(torch.randn(1, 1024, 1024)).size()

  def test_downsample_block(self):
    downsample = Downsample(in_channels=24, out_channels=256, kernel_size=5, stride=2)
    print(downsample.forward(torch.randn(1, 24, 1024)).size())

  def test_upsample_block(self):
    upsample1 = Upsample(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=2)
    upsample2 = Upsample(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
    x = torch.randn(1, 512, 1024)
    x = upsample1.forward(x)
    x = upsample2.forward(x)

  def test_generator(self):
    generator = Generator()
    print(generator.forward(torch.randn(1, 24, 128)).size())

  def test_discriminator(self):
    discriminator = Discriminator()
    print(discriminator.forward(torch.randn(1, 1, 24, 128)).size())

  def test_cyclegan(self):
    cycleGAN = CycleGAN()

    # Forward-inverse mapping
    x = torch.randn(1, 24, 128)
    y_hat = cycleGAN.Gx_y.forward(x)
    print(f'y_hat: {y_hat.size()}')

    x_hat = cycleGAN.Gy_x.forward(y_hat)

    print(f'x_hat: {x_hat.size()}')
    x_hat = x_hat.unsqueeze(0)
    print(f'x_hat: {x_hat.size()}')

    print(f'Dy: {cycleGAN.Dy.forward(x_hat).size()}')

    # Inverse-forward mapping
    y = torch.randn(1, 24, 128)
    x_hat = cycleGAN.Gy_x.forward(y)
    y_hat = cycleGAN.Gx_y.forward(x_hat)

    x_hat = x_hat.unsqueeze(0)

    cycleGAN.Dx.forward(x_hat).size()