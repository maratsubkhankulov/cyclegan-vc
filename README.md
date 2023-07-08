# CycleGAN VC

This repository reproduces https://arxiv.org/pdf/1711.11293.pdf.

# TODO:
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
- [] Run training over supplied features
- [] If loss doesn't improve, debug


TODOS:
- [x] Load all necessary data in AudioDataset
  - corresponds to 12 mins cited from the paper
- [x] Test out feature extraction code here: https://github.com/pritishyuvraj/Voice-Conversion-GAN/blob/master/preprocess.py
- [x] How to obtain MCEPs?
- [?] The input to generator has 23 channels - does this correspond to the 24 MCEPs per time step?
- []  - How to pack the other data into the input?
  - You don't - the generators map from MCEP to MCEP