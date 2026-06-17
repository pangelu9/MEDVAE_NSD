"""Template for local data-name overrides.

The code refers to data files by generic names (see ccn_config.py). If your
actual files are named differently, copy this file to ``ccn_config_local.py``
(which is gitignored) and edit the values to match your layout. ccn_config.py
imports ``*`` from ccn_config_local.py automatically when it exists.

Only override the names you need — anything you leave out keeps the generic
default from ccn_config.py. ``{sid}`` is the zero-padded subject id (01..08),
``{n_dims}`` the latent dimensionality. Paths are relative to their data dir.
"""

# --- the three files the documented pipeline needs ---
FMRI_FILE_TEMPLATE = "fmri_subject{sid}.npz"      # per-subject fMRI .npz in $CCN_DATA_ROOT/nsd/data/
ANN_FEATURES_FILE  = "ann_features.npy"           # ANN (image-model) features, same dir
LABELS_FILE        = "category_labels.npy"         # multi-hot category labels

# --- optional: only needed for the reconstruction / cross-trial analyses ---
# NOISE_CEILING_TEMPLATE   = "subj{sid}_noiseceiling.npy"
# FMRI_TRIAL_FILE_TEMPLATE = "fmri_subject{sid}_trials.npz"
# CROSS_TRIAL_SUBPATH      = "vae/cross_trial_data"

# --- optional: trained-checkpoint fallback name (or just pass --vae_checkpoint) ---
# VAE_CHECKPOINT_TEMPLATE  = "medvae_{n_dims}d.pt"
