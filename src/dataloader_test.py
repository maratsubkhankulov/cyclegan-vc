import unittest
from dataloader import AudioDataset

import torchaudio
from torch.utils.data import DataLoader

class AudioDatasetTest(unittest.TestCase):

  def test_loading(self):
    # load file at data/vcc2016_training/SF1/100001.wav
    waveform, sample_rate = torchaudio.load('data/vcc2016_training/SF1/100001.wav')
    print(waveform.size())
  
  def test_audio_dataset(self):
    dataset = AudioDataset('data/vcc2016_training', batch_size=1, sr=22050)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    iterator = iter(train_dataloader)
    for batch in iterator:
      print(batch.shape)
