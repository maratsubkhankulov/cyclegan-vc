# CycleGAN VC

This code reproduces the CycleGAN-VC paper: [PARALLEL-DATA-FREE VOICE CONVERSION USING CYCLE-CONSISTENT ADVERSARIAL NETWORKS, T. Kaneko, H. Kameoka 2017](https://arxiv.org/pdf/1711.11293.pdf)

I used [this repository](https://github.com/pritishyuvraj/Voice-Conversion-GAN) as a reference when stuck for ideas about how to interpret the paper.

## Generated samples

## Installation

1. Download training data
2. Install code dependencies
3. Generate f0 stats
4. Run training

## Usage

If all goes well, you should see something like this
```
python3 src/train.py --resume_from_checkpoint False --checkpoint_dir SF1_TM3_checkpoints --source_speaker SF1 --target_speaker TM3 --source_logf0_mean 5.38879422525781 --source_logf0_std 0.2398814107162179 --target_logf0_mean  4.858265991213904 --target_logf0_std 0.23171982666578547
Epoch 0
Iteration: 10, it/s: 0.49, d_loss: 0.0083846, g_loss: 613.42, test_d_loss: 0.0028683, test_g_loss: 515.88
Iteration: 20, it/s: 0.75, d_loss: 0.0016719, g_loss: 602.03, test_d_loss: 0.0137144, test_g_loss: 644.71
Iteration: 30, it/s: 0.69, d_loss: 0.0010094, g_loss: 620.05, test_d_loss: 0.0019095, test_g_loss: 566.79
...
```

### f0 stats

```
SF1 log f0 mean: 5.388794225257816
SF1 log f0 std: 0.2398814107162179
TM3 log f0 mean: 4.858265991213904
TM3 log f0 std: 0.23171982666578547
```

## TODO:
- [x] Just try to implement the architecture of CycleGAN Generator and Discriminator
- [x] Write model with Gx->y, Gy->x and Dy submodules
- [x] Reproduce inverse-forward and forward-inverse passes
- [x] Write the shell of training loop on random data
- [x] Create variables for outputs and implement loss function
- [x] Add backprop step
- [x] Zero gradients
- [x] Add optimizer
- [x] Log losses every few iterations
- [x] Verify implementation against reference
  - reference has 2 discriminators, 2 generators
  - difference in use of combined losses for both pairs
  - difference in use of pairs of images for each batch
    - Looks similar to how loss sample distributions are defined in the paper
    - Can proceed with my code, then debug and update understanding
- [x] Load data and extract features using WORLD vocoder
- [x] Load all necessary data in AudioDataset
  - corresponds to 12 mins cited from the paper
- [x] Test out feature extraction code here: https://github.com/pritishyuvraj/Voice-Conversion-GAN/blob/master/preprocess.py
- [x] How to obtain MCEPs?
- [?] The input to generator has 23 channels - does this correspond to the 24 MCEPs per time step?
- [x]  - How to pack the other data into the input?
  - You don't - the generators map from MCEP to MCEP
- [x] Change generator architecture to use 1d convolution
  - [x] Fix dataset implementation to produce [B, T, D] tensors
- [x] Change loss function to use source and target
- [x] Run training over supplied features
- [x] Vocode a full-length audio sample
- [x] Pad features
  - if train=True, WorldDataset should return truncated segments
  - else: WorldDataset should return full segments that can be synthesized into waveforms
  - [x] there should be a test
- [x] Adjust loss coefficients to those used in the paper
- [x] Extract model from notebook and add a test
- [x] Extract train.py from notebook
- [x] Add checkpointing
- [x] Enable GPU training
- [x] Add test/validation dataset and graph how that varies with training loss
- [x] Eval program to generate side-by-side samples given a training module.
  - [x] Added feature id, full wav to dataset items
  - [x] Added synthesize_mcep() and save_output_for_eval()
  - [x] Implement parallel eval generation by using parallel eval dataset
- [x] Use two optimizers - for generators and discriminators
- [x] Produce eval samples for SF1->TF2, SF1->TM3, SM1->TF2, SM1->TM3
- [x] Convert F0 using logarithm Gaussian normalized transformation
- [] Update loss function
  - [x] Decay learning rate for 20k iterations after the first 20k iterations
  - [] Swap BCE loss for least squares loss
  - [x] Stop using identity loss after 1k iterations
- [] Cleanup
  - [] Add output samples to README
  - [] Add diagram
- [] complain to Konstantinos