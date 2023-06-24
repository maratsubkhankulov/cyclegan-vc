from collections import defaultdict
import os
import torchaudio
import torch
import torch.nn.functional as F

# Audio dataset that loads waveform tensors from a given path
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
class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, batch_size=1, sr=16_000):
      self.path = path
      self.batch_size = batch_size
      self.sr = sr
      self.speaker_ids = [] # speaker 
      self.speaker_file_dict = defaultdict(list) # speaker name to list of file paths
      self.sources = ['SF1', 'SF2']
      self.targets = ['TF2', 'TM2']
      self.load_files()

    def load_files(self):
      # offset into self.files
      offset = 0
      for speaker in os.listdir(self.path):
        speaker_path = os.path.join(self.path, speaker)
        self.speaker_ids.append(speaker)
        if os.path.isdir(speaker_path):
          for file in os.listdir(speaker_path):
            file_path = os.path.join(speaker_path, file)
            if os.path.isfile(file_path):
              self.speaker_file_dict[speaker].append(file_path)
    
    def __len__(self):
      return len(self.speaker_file_dict['SF1']) // self.batch_size
    
    def __getitem__(self, idx):
      source = self.speaker_file_dict['SF1'][idx]
      target = self.speaker_file_dict['TF2'][idx]

      source_wav, sample_rate = torchaudio.load(source)
      source_wav = torchaudio.transforms.Resample(sample_rate, self.sr)(source_wav)

      target_wav, sample_rate = torchaudio.load(target)
      target_wav = torchaudio.transforms.Resample(sample_rate, self.sr)(target_wav)

      pad_len = max(source_wav.shape[1], target_wav.shape[1])
      source_wav = F.pad(source_wav, (0, pad_len - source_wav.shape[1]))
      target_wav = F.pad(target_wav, (0, pad_len - target_wav.shape[1]))
      return torch.cat([source_wav, target_wav])
