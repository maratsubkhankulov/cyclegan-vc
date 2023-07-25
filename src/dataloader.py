import os

import random
import torch
import pyworld as pw
import numpy as np
import librosa
from collections import defaultdict

# World dataset that loads waveform tensors from a given path
# The directory structure is as follows:
# data/vcc2016_training/
#  /SF1/
#    /10001.wav
#    /10002.wav
#    ...
#  /SF2/
#    /20001.wav
#    /20002.wav
#    ...
# The dataloader is iterable and returns batches of tensors given to init function
# The tensors are of shape (batch_size, 2, n_samples) representing a source and target
# pair. 
class WorldDataset(torch.utils.data.Dataset):
  """
  WorldDataset is a torch.utils.data.Dataset that loads waveforms and extracts features
  using World vocoder.
  """
    
  def __init__(self, source_speaker, target_speaker, path, batch_size=1, sr=16_000, train=True, device='cpu:0'):
    self.source_speaker = source_speaker
    self.target_speaker = target_speaker
    self.path = path
    self.batch_size = batch_size
    self.sr = sr
    self.train = train
    self.device = device
    self.n_frames = 128
    self.speaker_ids = [] # speaker 
    self.speaker_file_dict = defaultdict(list) # speaker -> [file1, file2, ...]
    self.feature_cache = {}
    self.load_files()

  def load_files(self):
    for speaker in os.listdir(self.path):
      speaker_path = os.path.join(self.path, speaker)
      self.speaker_ids.append(speaker)
      if os.path.isdir(speaker_path):
        for file in sorted(os.listdir(speaker_path)):
          file_path = os.path.join(speaker_path, file)
          if os.path.isfile(file_path):
            self.speaker_file_dict[speaker].append(file_path)
    
    if self.train:
      for files in self.speaker_file_dict.values():
        random.shuffle(files)
  
  def extract_features(self, file_path):
    wav, _ = librosa.load(file_path, sr=self.sr, mono=True)
    wav = wav.astype(np.float64)

    f0, time_axis = pw.harvest(wav, self.sr, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
    sp = pw.cheaptrick(wav, f0, time_axis, self.sr)
    ap = pw.d4c(wav, f0, time_axis, self.sr)
    mcep = pw.code_spectral_envelope(sp, self.sr, 24)
    
    if self.train:
      mcep = self.sample_mcep_segment(mcep, self.n_frames)

    # Example name = {example_id}_{speaker_id}
    speaker_id = file_path.split('/')[-2]
    example_id = file_path.split('/')[-1].replace('.wav', '')
    example_name = example_id + '_' + speaker_id
    
    f0 = torch.tensor(f0, device=self.device)
    time_axis = torch.tensor(time_axis, device=self.device)
    sp = torch.tensor(sp, device=self.device)
    ap = torch.tensor(ap, device=self.device)
    mcep = torch.tensor(mcep, device=self.device)
    return f0, time_axis, sp, ap, example_name, wav, mcep

  def extract_features_memoized(self, file_path):
    if file_path in self.feature_cache:
      return self.feature_cache[file_path]
    else:
      features = self.extract_features(file_path)
      self.feature_cache[file_path] = features
      return features 

  def sample_mcep_segment(self, mcep, n_frames):
    """
    Randomly sample a 128 frame segment
    """
    start = random.randint(0, mcep.shape[0] - n_frames)
    end = start + self.n_frames
    return mcep[start:end]
  
  def __len__(self):
    return len(self.speaker_file_dict['SF1']) // self.batch_size
  
  def __getitem__(self, idx):
    source = self.speaker_file_dict[self.source_speaker][idx]
    target = self.speaker_file_dict[self.target_speaker][idx]

    source_features = self.extract_features_memoized(source)
    target_features = self.extract_features_memoized(target)

    return (source_features, target_features)