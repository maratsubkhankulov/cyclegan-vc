
import argparse
import numpy as np

import torch

from dataloader import WorldDataset


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data')
  parser.add_argument('--source_speaker', type=str, default='SF1')
  parser.add_argument('--target_speaker', type=str, default='TM1')
  return parser.parse_args()

def get_f0_statistics(f0s):
  log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
  log_f0s_mean = log_f0s_concatenated.mean()
  log_f0s_std = log_f0s_concatenated.std()
  return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, source_mean, source_std, target_mean, target_std):
  """" Converts pitch from source speaker to target speaker """
  return torch.exp((torch.log(f0) - source_mean) / source_std * target_std + target_mean)

def generate_f0_statistics(args):
  source_f0s = []
  target_f0s = []
  for i, (source, target) in enumerate(dataloader):
    source_f0s.append(source[0].numpy().flatten())
    target_f0s.append(target[0].numpy().flatten())
  
  source_logf0_mean, soure_logf0_std = get_f0_statistics(source_f0s)
  target_logf0_mean, target_logf0_std = get_f0_statistics(target_f0s)

  print('{} log f0 mean: {}'.format(args.source_speaker, source_logf0_mean))
  print('{} log f0 std: {}'.format(args.source_speaker, soure_logf0_std))
  print('{} log f0 mean: {}'.format(args.target_speaker, target_logf0_mean))
  print('{} log f0 std: {}'.format(args.target_speaker, target_logf0_std))


if __name__ == '__main__':
  
  args = get_args()

  dataset = WorldDataset(source_speaker=args.source_speaker, target_speaker=args.target_speaker, train=False, path=args.data_dir)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

  generate_f0_statistics(args)