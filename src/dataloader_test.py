import torchaudio
import torch

# load file at data/vcc2016_training/SF1/100001.wav
waveform, sample_rate = torchaudio.load('data/vcc2016_training/SF1/100001.wav')
print(waveform.size())