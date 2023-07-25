import argparse
import time
import torch
import numpy as np
import os
import glob
import pyworld as pw
import soundfile as sf

from dataloader import WorldDataset
from torch.utils.data import DataLoader, random_split
from model import CycleGAN
from datetime import datetime


def save_ckpt(cycleGAN, ckpt_dir):
  current_time = datetime.now()
  timestamp = current_time.strftime('%b%d_%H-%M-%S')
  path = ckpt_dir + timestamp + ".pt"

  # Save state dict of cycleGAN
  torch.save(cycleGAN.state_dict(), path)

def load_ckpt(cycleGAN, ckpt_dir):
  files = glob.glob(os.path.join(ckpt_dir, '*.pt'))
  
  files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
  if len(files) == 0:
    raise Exception('No checkpoint found')

  latest_ckpt = files[0]
  cycleGAN.load_state_dict(torch.load(latest_ckpt))
  cycleGAN.eval()

def synthesize_mcep(f0, sp, ap, mcep, fs, frame_period):
  sp = pw.decode_spectral_envelope(mcep.numpy(), fs, fft_size=1024)
  y = pw.synthesize(f0, sp, ap, fs, frame_period)
  return y

def log_output_for_eval(source_features, target_features, xy_mcep, yx_mcep, path):
  """ Saves wavefiles for evaluation:
      - source waveform (x)
      - target waveform (y)
      - synthesized waveform for Gx_y(source_features.mcep)
      - synthesized waveform for Gy_x(target_features.mcep)
      """

  fs = 16000
  frame_period = 5.0

  # Save source and target waveforms
  source_wav = source_features[5][0].numpy()
  target_wav = target_features[5][0].numpy()

  source_name = source_features[4][0]
  target_name = target_features[4][0]

  sf.write(path + '/' + source_name + '_source.wav', source_wav, fs)
  sf.write(path + '/' + target_name + '_target.wav', target_wav, fs)
  
  def synthesize_and_save(features, name, path, mcep):
    f0 = features[0][0].numpy()
    sp = features[2][0].numpy()
    ap = features[3][0].numpy()

    mcep = mcep[-1].squeeze(0).transpose(-2, -1).to(dtype=torch.float64).contiguous().detach()
    mcep = mcep.cpu()

    # pad mcep to min number of frames of f0
    if mcep.shape[0] < f0.shape[0]:
      mcep = torch.cat((mcep, torch.zeros((f0.shape[0] - mcep.shape[0], mcep.shape[1]))), dim=0)
    # else clip mcep to match f0
    else:
      mcep = mcep[:f0.shape[0], :]

    wav = synthesize_mcep(f0, sp, ap, mcep, fs, frame_period)
    sf.write(path + '/' + name + '.wav', wav, fs)
  
  synthesize_and_save(source_features, source_name + '_to_' + target_name, path, xy_mcep)
  synthesize_and_save(source_features, source_name + '_orig', path, source_features[6].transpose(1, 2))

  synthesize_and_save(target_features, target_name + '_to_' + source_name, path, yx_mcep)
  synthesize_and_save(target_features, target_name + '_orig', path, target_features[6].transpose(1, 2))

def train_cyclegan(
  source_speaker,
  target_speaker,
  train_data_dir,
  eval_data_dir,
  checkpoint_dir,
  resume_from_checkpoint,
  eval_output_dir,
):
  # Loss parameters
  cyc_tradeoff_parameter=10
  identity_tradeoff_parameter=5

  def training_iteration(x,
                        y,
                        real_labels,
                        fake_labels,
                        optimizer_d,
                        optimizer_g,
                        criterion,
                        Gx_y,
                        Gy_x,
                        Dx,
                        Dy):

    # ======================================================== #
    #                      DISCRIMINATOR LOSS                  #
    # ======================================================== #

    def adversarial_loss(G, D, source, target):
      # Discriminator should learn to classify real and fake features
      # correctly. Thus we give it a set for real features with real labels
      # and a set of fake features with fake labels.
      real_outputs = D.forward(target.unsqueeze(0))
      d_loss_real = criterion(real_outputs, real_labels)
      real_score = real_outputs

      # Generate fake features
      # from IPython.core.debugger import Tracer; Tracer()() 
      z = source
      fake_features = G.forward(z)
      fake_outputs = D.forward(fake_features.unsqueeze(0))
      d_loss_fake = criterion(fake_outputs, fake_labels)
      fake_score = fake_outputs

      d_loss = d_loss_real + d_loss_fake
      return d_loss, real_score, fake_score

    dy_loss, real_score, fake_score = adversarial_loss(G=Gx_y, D=Dy, source=x, target=y)
    dx_loss, real_score, fake_score = adversarial_loss(G=Gy_x, D=Dx, source=y, target=x)
    d_loss = dx_loss + dy_loss

    # ======================================================== #
    #                      GENERATOR LOSS                      #
    # ======================================================== #

    def cycle_consistency_loss(Gx_y, Gy_x, x, y):
      forward_loss = torch.norm(Gy_x(Gx_y(x)) - x)
      inverse_forward_loss = torch.norm(Gx_y(Gy_x(y)) - y)
      return forward_loss + inverse_forward_loss
    
    def identity_loss(Gx_y, Gy_x, x, y):
      return torch.norm(Gx_y(y) - y) + torch.norm(Gy_x.forward(x) - x)
    
    g_loss = cycle_consistency_loss(Gx_y, Gy_x, x, y) * cyc_tradeoff_parameter + \
      identity_loss(Gx_y, Gy_x, x, y) * identity_tradeoff_parameter

    full_loss = d_loss + g_loss

    # ======================================================== #
    #                     TRAIN GENERATOR                      #
    # ======================================================== #
    optimizer_g.zero_grad()
    optimizer_g.zero_grad()
    g_loss.backward()
    optimizer_g.step()

    # ======================================================== #
    #                     TRAIN DISCRIMINATOR                  #
    # ======================================================== #
    optimizer_d.zero_grad()
    optimizer_d.zero_grad()
    d_loss.backward()
    optimizer_d.step()

    # Return all losses
    return full_loss

  device = 'cuda:0'

  batch_size=1
  cycleGAN = CycleGAN()
  if resume_from_checkpoint:
    load_ckpt(cycleGAN, checkpoint_dir)
  cycleGAN = cycleGAN.to(device)

  # Create labels which are later used as input for the BCU loss
  real_labels = torch.ones(batch_size, 1, device=device)
  fake_labels = torch.zeros(batch_size, 1, device=device)

  optimizer_d = torch.optim.Adam(list(cycleGAN.Dx.parameters()) + list(cycleGAN.Dy.parameters()), lr=0.0001)
  optimizer_g = torch.optim.Adam(list(cycleGAN.Gx_y.parameters()) + list(cycleGAN.Gy_x.parameters()), lr=0.0002)

  criterion = torch.nn.BCELoss()
  
  eval_dataset = WorldDataset(source_speaker, target_speaker, eval_data_dir, batch_size=1, train=False, sr=16000)
  eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1)

  dataset = WorldDataset(source_speaker, target_speaker, train_data_dir, batch_size=1, train=True, sr=16000)
  train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
  
  train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
  test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

  checkpoint_every = 500
  stat_every = 100
  log_every = 162*3

  def train(epochs):
    start = time.time()
    iteration = 0
    for i in range(epochs):
      print(f'Epoch {i}')
      for batch in train_dataloader:
        print(f'iteration {iteration}')
        
        # convert to float32 because model is defined this way
        source_features = batch[0]
        target_features = batch[1]
        x = source_features[-1].to(dtype=torch.float32).transpose(1, 2)
        y = target_features[-1].to(dtype=torch.float32).transpose(1, 2)
        x = x.to(device)
        y = y.to(device)

        loss = training_iteration(
                            x=x,
                            y=y,
                            real_labels=real_labels,
                            fake_labels=fake_labels,
                            optimizer_d=optimizer_d,
                            optimizer_g=optimizer_g,
                            criterion=criterion,
                            Gx_y=cycleGAN.Gx_y,
                            Gy_x=cycleGAN.Gy_x,
                            Dx=cycleGAN.Dx,
                            Dy=cycleGAN.Dy)

        iteration += 1

        if iteration % log_every == 1:
          it = iter(eval_dataloader)
          for i in range(10):
            source_features, target_features = next(it)

            cycleGAN.eval()
            # Source to target
            source_mcep = source_features[6].to(dtype=torch.float32).transpose(1, 2)
            source_mcep = source_mcep.to(device)
            gxy_mcep = cycleGAN.Gx_y(source_mcep)

            # Target to source
            target_mcep = target_features[6].to(dtype=torch.float32).transpose(1, 2)
            target_mcep = target_mcep.to(device)
            gyx_mcep = cycleGAN.Gy_x(target_mcep)
            cycleGAN.train()

            log_output_for_eval(source_features, target_features, gxy_mcep, gyx_mcep, eval_output_dir)

        if iteration % stat_every == 0:
          t_batch = next(iter(test_dataloader))
          source_features = t_batch[0][-1].to(dtype=torch.float32).transpose(1, 2)
          target_features = t_batch[1][-1].to(dtype=torch.float32).transpose(1, 2)
          source_features = source_features.to(device)
          target_features = target_features.to(device)
          cycleGAN.eval()
          test_loss = training_iteration(
                            x=source_features,
                            y=target_features,
                            real_labels=real_labels,
                            fake_labels=fake_labels,
                            optimizer_d=optimizer_g,
                            optimizer_g=optimizer_g,
                            criterion=criterion,
                            Gx_y=cycleGAN.Gx_y,
                            Gy_x=cycleGAN.Gy_x,
                            Dx=cycleGAN.Dx,
                            Dy=cycleGAN.Dy)
          cycleGAN.train()


          elapsed = time.time() - start
          start = time.time()
          print(f'Iterations per second: {stat_every / elapsed}')
          print(f'loss: {loss.item()}')
          print(f'Test loss: {test_loss.item()}')
        if iteration % checkpoint_every == 0:
          save_ckpt(cycleGAN, checkpoint_dir)

  train(epochs = 100)

if __name__ == '__main__':
  args_parser = argparse.ArgumentParser()
  args_parser.add_argument('--source_speaker', default='SF1', help='Source speaker ID')
  args_parser.add_argument('--target_speaker', default='TF2', help='Target speaker ID')
  args_parser.add_argument('--train_data_dir', default='./data/vcc2016_training/', help='Path to training data')
  args_parser.add_argument('--eval_data_dir', default='./data/evaluation_all/', help='Path to evaluation data')
  args_parser.add_argument('--checkpoint_dir', default='checkpoints', help='Path to checkpoint directory')
  args_parser.add_argument('--resume_from_checkpoint', default=False, help='Resume training from latest checkpoint')
  args_parser.add_argument('--eval_output_dir', default='eval_output', help='Path to evaluation output directory')

  args = args_parser.parse_args()

  train_cyclegan(
    source_speaker=args.source_speaker,
    target_speaker=args.target_speaker,
    train_data_dir=args.train_data_dir,
    eval_data_dir=args.eval_data_dir,
    checkpoint_dir=args.checkpoint_dir,
    resume_from_checkpoint=args.resume_from_checkpoint,
    eval_output_dir=args.eval_output_dir,
  )