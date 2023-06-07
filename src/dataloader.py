import torchaudio
import torch

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
# The tensors are of shape (batch_size, 1, n_samples)
# The dataloader also supports custom sampling rates
class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, batch_size=1, sr=22050):
        self.path = path
        self.batch_size = batch_size
        self.sr = sr
        self.files = []
        self.speaker_ids = []
        self.speaker_dict = {}
        self.speaker_count = 0
        self.load_files()
    
    def load_files(self):
        for speaker in os.listdir(self.path):
            speaker_path = os.path.join(self.path, speaker)
            if os.path.isdir(speaker_path):
                for file in os.listdir(speaker_path):
                    file_path = os.path.join(speaker_path, file)
                    if os.path.isfile(file_path):
                        self.files.append(file_path)
                        if speaker not in self.speaker_dict:
                            self.speaker_dict[speaker] = self.speaker_count
                            self.speaker_count += 1
                        self.speaker_ids.append(self.speaker_dict[speaker])