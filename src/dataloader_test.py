import unittest
from dataloader import WorldDataset

class WorldDatasetTest(unittest.TestCase):

  def test_audio_dataset(self):
    train_dataset = WorldDataset('data/vcc2016_training', batch_size=1, train=True, sr=16000)
    test_dataset = WorldDataset('data/vcc2016_training', batch_size=1, train=False, sr=16000)

    batch = train_dataset[0]
    
    self.assertEqual(len(batch[0]), 5)
    self.assertEqual(len(batch[1]), 5)

    self.assertEqual(batch[0][4].shape, (128, 24))
    self.assertEqual(batch[1][4].shape, (128, 24))

    batch = test_dataset[0]

    self.assertEqual(len(batch[0]), 5)
    self.assertEqual(len(batch[1]), 5)

    self.assertNotEqual(batch[0][4].shape, (24, 128))
    self.assertNotEqual(batch[1][4].shape, (24, 128))

