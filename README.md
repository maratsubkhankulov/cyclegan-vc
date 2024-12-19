# CycleGAN-VC

Implementation of [Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1711.11293.pdf) (Kaneko & Kameoka, 2017).

## Overview

This project implements voice conversion between speakers using CycleGAN without requiring parallel data. Generated samples are available [here](https://docs.google.com/presentation/d/1fJlDwIBgM_ANQZxCtv_c3xnUkahVrS85HfuMIoaFVeg/edit?usp=sharing).

## Implementation Notes

The model preserves the original speaker's prosody due to the convolution receptive field size, while log(f0) conversion handles pitch shifting effectively. Implementation choices were informed by [this reference repository](https://github.com/pritishyuvraj/Voice-Conversion-GAN).

## Setup

1. Download [training data](https://datashare.ed.ac.uk/download/DS_10283_2211.zip)
2. Extract vcc2016_training and evaluation_all
3. Set up Python environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Preprocessing

Generate f0 statistics:
```bash
python3 src/preprocess.py --data_dir ./data/vcc2016_training/ \
                         --source_speaker SF1 \
                         --target_speaker TM3
```

### Training

```bash
python3 src/train.py --resume_from_checkpoint False \
                     --checkpoint_dir SF1_TM3_checkpoints \
                     --source_speaker SF1 \
                     --target_speaker TM3 \
                     --source_logf0_mean 5.38879422525781 \
                     --source_logf0_std 0.2398814107162179 \
                     --target_logf0_mean 4.858265991213904 \
                     --target_logf0_std 0.23171982666578547
```

### Additional Parameters

Both scripts support additional parameters - use `--help` for full options:
- `preprocess.py`: Configure data directory and speakers
- `train.py`: Set data directories, checkpoints, and f0 statistics
