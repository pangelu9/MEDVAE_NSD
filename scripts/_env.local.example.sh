#!/bin/bash
# Template for local, machine-specific overrides. Copy to scripts/_env.local.sh
# (which is gitignored) and edit. _env.sh sources it automatically if present, so
# you can keep site paths and real data filenames out of version control.
export CONDA_ROOT=/path/to/miniconda3                  # conda install with the `medvae` env
export CCN_DATA_ROOT=/path/to/data                     # dir containing nsd/ and nsd_classification/
export CCN_ANN_FEATURES=ann_features.npy               # your real ANN-features filename
# export CONDA_ENV=medvae                              # if your env has a different name
# export CCN_RESULTS_DIR=/path/to/results              # where checkpoints/eval outputs go
