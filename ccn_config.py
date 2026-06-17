"""Central configuration for the MED-VAE cross-subject alignment codebase.

Every filesystem location used by the code is resolved here, so the rest of the
repository contains no hard-coded absolute paths.

Data is NOT shipped with this repository. Point the code at wherever the NSD
fMRI responses and the ANN activation files live by setting the environment
variable ``CCN_DATA_ROOT`` (or editing ``DATA_ROOT`` below). ``DATA_ROOT`` must
be a directory containing the ``nsd/`` and ``nsd_classification/`` trees.

    export CCN_DATA_ROOT=/path/to/code_may

Optional overrides:
    CCN_RESULTS_DIR   where trained MED-VAE checkpoints (.pt) are written/read
                      (default: <repo>/medvae/results)
"""
import os
import sys

# --------------------------------------------------------------------------
# Repository layout (code)
# --------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MEDVAE_DIR = os.path.join(REPO_ROOT, "medvae")
BASELINES_DIR = os.path.join(REPO_ROOT, "baselines")
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")

# --------------------------------------------------------------------------
# Data location (NOT part of this repo)
# --------------------------------------------------------------------------
# Directory containing the original data trees: nsd/ and nsd_classification/.
# Set CCN_DATA_ROOT (see module docstring); the placeholder default fails loudly
# with a clear path if it is left unset.
DATA_ROOT = os.environ.get("CCN_DATA_ROOT", "/path/to/data")

NSD_DIR = os.path.join(DATA_ROOT, "nsd")                 # contains data/, labels, ...
NSD_DATA_DIR = os.path.join(NSD_DIR, "data")             # fMRI npz + ANN activations
NSD_CLASSIFICATION_DIR = os.path.join(DATA_ROOT, "nsd_classification")
NSD_CLASSIFICATION_DATA_DIR = os.path.join(NSD_CLASSIFICATION_DIR, "data")
MINDEYE_DATA_DIR = os.path.join(NSD_CLASSIFICATION_DIR, "data_mindeye2")

# Per-subject NSD noise-ceiling (ncsnr) data, used by the reconstruction eval.
NOISE_CEILING_DIR = NSD_CLASSIFICATION_DATA_DIR

# Default location of trained MED-VAE checkpoints (.pt).
RESULTS_DIR = os.environ.get("CCN_RESULTS_DIR", os.path.join(MEDVAE_DIR, "results"))

# --------------------------------------------------------------------------
# Data file names (generic, public-facing defaults)
# --------------------------------------------------------------------------
# This repository ships NO data, so the names below are deliberately generic
# placeholders. To run on your own data without committing its (possibly
# idiosyncratic) filenames, either set the matching CCN_* environment variable,
# or create ``ccn_config_local.py`` next to this file (it is gitignored) that
# reassigns any of the names below. All full paths are assembled *after* that
# local import, so a single override propagates everywhere. ``{sid}`` is the
# zero-padded subject id (01..08); ``{n_dims}`` the latent dimensionality.
#
# A worked example mapping these to real files is in ccn_config_local.example.py.

# fMRI responses (per-subject .npz in NSD_DATA_DIR)
FMRI_FILE_TEMPLATE        = os.environ.get("CCN_FMRI_TEMPLATE",         "fmri_subject{sid}.npz")
FMRI_TRIAL_FILE_TEMPLATE  = os.environ.get("CCN_FMRI_TRIAL_TEMPLATE",   "fmri_subject{sid}_trials.npz")
FMRI_MINDEYE_TEMPLATE     = os.environ.get("CCN_FMRI_MINDEYE_TEMPLATE", os.path.join("mindeye", "fmri_subject{sid}.npz"))
NOISE_CEILING_TEMPLATE    = os.environ.get("CCN_NOISE_CEILING_TEMPLATE", "subj{sid}_noiseceiling.npy")
CROSS_TRIAL_SUBPATH       = os.environ.get("CCN_CROSS_TRIAL_SUBPATH",   os.path.join("vae", "cross_trial_data"))

# Image labels
LABELS_FILE               = os.environ.get("CCN_LABELS_FILE",           "category_labels.npy")
LABELS_ALL_FILE           = os.environ.get("CCN_LABELS_ALL_FILE",       "labels_all.npy")
LABELS_ORIGINAL_FILE      = os.environ.get("CCN_LABELS_ORIGINAL_FILE",  "labels_original.npy")
LABELS_MINDEYE_FILE       = os.environ.get("CCN_LABELS_MINDEYE_FILE",   os.path.join("mindeye", "category_labels.npy"))
LABELS_MINDEYE_CLIP_FILE  = os.environ.get("CCN_LABELS_MINDEYE_CLIP_FILE", os.path.join("mindeye_clip", "category_labels.npy"))
LABELS_PER_SUBJECT_TEMPLATE = os.environ.get(
    "CCN_LABELS_PER_SUBJECT_TEMPLATE", os.path.join("labels_subject_{sid}", "category_labels.npy"))

# ANN activation scaffolds (one row per image presentation)
ANN_FEATURES_FILE         = os.environ.get("CCN_ANN_FEATURES",          "ann_features.npy")
ACTIVATIONS_ALL_FILE      = os.environ.get("CCN_ACTIVATIONS_ALL_FILE",  "activations_all.npy")
ACTIVATIONS_ORIGINAL_FILE = os.environ.get("CCN_ACTIVATIONS_ORIGINAL_FILE", "activations_original.npy")

# Alternative ANN scaffolds (experiments outside the documented release path).
# Each value is a path relative to its base dir; generic placeholders here.
ANN_INFER_SUBDIR          = os.environ.get("CCN_ANN_INFER_SUBDIR",      "ann_inference_results")
ANN_SUBPATH_MINDEYE       = os.environ.get("CCN_ANN_MINDEYE",      os.path.join("mindeye", "ann_features.npy"))
ANN_SUBPATH_UNT_51060     = os.environ.get("CCN_ANN_UNT_51060",    os.path.join("untrained", "ann_features.npy"))
ANN_SUBPATH_UNT_8000      = os.environ.get("CCN_ANN_UNT_8000",     "ann_features_untrained.npy")
ANN_SUBPATH_RAND_8000     = os.environ.get("CCN_ANN_RAND_8000",    os.path.join("random", "ann_features.npy"))
ANN_SUBPATH_AVGPOOL       = os.environ.get("CCN_ANN_AVGPOOL",      "ann_features_avgpool2048.npy")
ANN_SUBPATH_CLIP_VITL14   = os.environ.get("CCN_ANN_CLIP_VITL14",  os.path.join("clip_vitl14", "ann_features.npy"))
ANN_SUBPATH_DINO_RN50     = os.environ.get("CCN_ANN_DINO_RN50",    os.path.join("dino_rn50", "ann_features.npy"))
ANN_SUBPATH_DINO_VITB16   = os.environ.get("CCN_ANN_DINO_VITB16",  os.path.join("dino_vitb16", "ann_features.npy"))
ANN_SUBPATH_CLIP_RN50     = os.environ.get("CCN_ANN_CLIP_RN50",    os.path.join("clip_rn50", "ann_features.npy"))
ANN_SUBPATH_MNIST_RN50    = os.environ.get("CCN_ANN_MNIST_RN50",   "mnist_features_rn50_2048.npy")
ANN_SUBPATH_MNIST_784     = os.environ.get("CCN_ANN_MNIST_784",    "mnist_features_784.npy")

# Fallback trained-checkpoint filename (used by the evaluator when --vae_checkpoint
# is omitted).
VAE_CHECKPOINT_TEMPLATE   = os.environ.get("CCN_VAE_CHECKPOINT_TEMPLATE", "medvae_{n_dims}d.pt")

# --------------------------------------------------------------------------
# Local, uncommitted overrides (gitignored): map the generic names above to your
# real data filenames. Copy ccn_config_local.example.py -> ccn_config_local.py.
# --------------------------------------------------------------------------
try:
    from ccn_config_local import *  # noqa: F401,F403
except ImportError:
    pass

# --------------------------------------------------------------------------
# Assemble full paths from the (possibly-overridden) names above
# --------------------------------------------------------------------------
# Multi-hot COCO super-category labels (70502 x 12).
LABELS_PATH = os.path.join(NSD_DATA_DIR, LABELS_FILE)

# Preprocessed trial-separated data for the cross-trial reconstruction (Fig 2C).
CROSS_TRIAL_DATA_DIR = os.path.join(DATA_ROOT, CROSS_TRIAL_SUBPATH)

# Logical name -> absolute path. The eval scripts accept --ann_activations to
# select one explicitly; if omitted they fall back to checkpoint-name matching.
_INFER_DIR = os.path.join(NSD_CLASSIFICATION_DIR, "code", ANN_INFER_SUBDIR)
ANN_ACTIVATIONS = {
    # Main paper scaffold: ImageNet-pretrained ResNet-50 (streams ROIs)
    "rn50_streams":         os.path.join(NSD_DATA_DIR, ANN_FEATURES_FILE),
    # ResNet-50 with MindEye2 ROIs — image-decoding setup
    "rn50_mindeye":         os.path.join(NSD_DATA_DIR, ANN_SUBPATH_MINDEYE),
    # Untrained (random-init) ResNet-50 controls
    "rn50_untrained_51060": os.path.join(NSD_DATA_DIR, ANN_SUBPATH_UNT_51060),
    "rn50_untrained_8000":  os.path.join(NSD_DATA_DIR, ANN_SUBPATH_UNT_8000),
    "rn50_random_8000":     os.path.join(_INFER_DIR, ANN_SUBPATH_RAND_8000),
    # avgpool-only RN50 (2048-d)
    "rn50_avgpool_2048":    os.path.join(NSD_DATA_DIR, ANN_SUBPATH_AVGPOOL),
    # Out of scope for the core release; kept for the eval auto-detect fallback.
    "clip_vitl14":          os.path.join(NSD_DATA_DIR, ANN_SUBPATH_CLIP_VITL14),
    "dino_rn50":            os.path.join(_INFER_DIR, ANN_SUBPATH_DINO_RN50),
    "dino_vitb16":          os.path.join(_INFER_DIR, ANN_SUBPATH_DINO_VITB16),
    "clip_rn50":            os.path.join(_INFER_DIR, ANN_SUBPATH_CLIP_RN50),
    "mnist_rn50_2048":      os.path.join(DATA_ROOT, "ccn_final", "MNIST_experiment", ANN_SUBPATH_MNIST_RN50),
    "mnist_784":            os.path.join(DATA_ROOT, "ccn_final", "MNIST_experiment", ANN_SUBPATH_MNIST_784),
}

# MindAligner / MindEye2 decoder location. Only used by the (excluded from this
# release) image-decoding path for the 'mindeye' dataset; install separately.
MINDALIGNER_DIR = os.environ.get(
    "CCN_MINDALIGNER_DIR",
    os.path.join(DATA_ROOT, "mindaligner", "MindAligner"))


def setup_paths():
    """Put the package directories on sys.path so cross-package bare imports
    resolve (evaluation/ and baselines/ import medvae modules by bare name,
    exactly as the original code did via sys.path.insert)."""
    for d in (MEDVAE_DIR, BASELINES_DIR, EVALUATION_DIR, REPO_ROOT):
        if d and d not in sys.path:
            sys.path.insert(0, d)
