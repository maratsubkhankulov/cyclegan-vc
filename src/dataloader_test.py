import unittest
from dataloader import AudioDataset

import torchaudio
from torch.utils.data import DataLoader

class AudioDatasetTest(unittest.TestCase):

  def test_audio_dataset(self):
    dataset = AudioDataset('data/vcc2016_training', batch_size=1, sr=16000)
    batch = dataset[0]
    # assert that batch[0] and batch[1] are tuples of 5 tensors
    self.assertEqual(len(batch[0]), 5)
    self.assertEqual(len(batch[1]), 5)

    # assert that batch[_][4] is of shape (1, 24, 128)
    self.assertEqual(batch[0][4].shape, (1, 24, 128))
    self.assertEqual(batch[1][4].shape, (1, 24, 128))
