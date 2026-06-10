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
DATA_ROOT = os.environ.get("CCN_DATA_ROOT", "/well/costa/users/odx145/code_may")

NSD_DIR = os.path.join(DATA_ROOT, "nsd")                 # contains data/, labels, ...
NSD_DATA_DIR = os.path.join(NSD_DIR, "data")             # fMRI npz + ANN activations
NSD_CLASSIFICATION_DIR = os.path.join(DATA_ROOT, "nsd_classification")
NSD_CLASSIFICATION_DATA_DIR = os.path.join(NSD_CLASSIFICATION_DIR, "data")
MINDEYE_DATA_DIR = os.path.join(NSD_CLASSIFICATION_DIR, "data_mindeye2")

# Per-subject NSD noise-ceiling (ncsnr) data, used by the reconstruction eval.
NOISE_CEILING_DIR = NSD_CLASSIFICATION_DATA_DIR

# Preprocessed trial-separated data for the cross-trial reconstruction analysis
# (Fig 2C cross-trial). This ~8 GB tree was produced by the trial-conversion
# step and ships alongside the original data, not inside this repo.
CROSS_TRIAL_DATA_DIR = os.path.join(
    DATA_ROOT, "vae", "cross_trial_data_shared1000",
    "trial_separated_shared1000_unified", "trial_separated_preprocessed")

# Default location of trained MED-VAE checkpoints (.pt).
RESULTS_DIR = os.environ.get("CCN_RESULTS_DIR", os.path.join(MEDVAE_DIR, "results"))

# --------------------------------------------------------------------------
# fMRI responses / image labels
# --------------------------------------------------------------------------
# Per-subject fMRI npz files live in NSD_DATA_DIR as:
#     fmri_subject{NN}_streams_overl_NEW.npz       (NN = 01..08)
# Multi-hot COCO super-category labels (70502 x 12):
LABELS_PATH = os.path.join(NSD_DATA_DIR, "labels_all_aligned.npy")

# --------------------------------------------------------------------------
# ANN activation scaffolds (one row per image presentation)
# --------------------------------------------------------------------------
# Logical name -> absolute path. The eval scripts accept --ann_activations to
# select one of these explicitly; if omitted they fall back to auto-detecting
# from the checkpoint name (legacy behaviour, preserved verbatim).
ANN_ACTIVATIONS = {
    # Main paper scaffold: ImageNet-pretrained ResNet-50, 51060-d (streams ROIs)
    "rn50_streams":         os.path.join(NSD_DATA_DIR, "aligned_all_activations_fair_resnet50_hendrycks.npy"),
    # ResNet-50 used with MindEye2 ROIs (51048-d) — image-decoding setup
    "rn50_mindeye":         os.path.join(NSD_DATA_DIR, "final_datasets_mindeye2", "averaged", "activations_all.npy"),
    # Untrained (random-init) ResNet-50 controls
    "rn50_untrained_51060": os.path.join(NSD_DATA_DIR, "final_datasets_mindeye2_untrained_resnet", "activations_all.npy"),
    "rn50_untrained_8000":  os.path.join(NSD_DATA_DIR, "aligned_all_activations_untrained_resnet50.npy"),
    "rn50_random_8000":     os.path.join(NSD_CLASSIFICATION_DIR, "code",
                                         "inference_results_subject_01_02_03_04_05_06_07_08_all_MULTI",
                                         "resnet50_random_aligned", "activations_all.npy"),
    # avgpool-only RN50 (2048-d)
    "rn50_avgpool_2048":    os.path.join(NSD_DATA_DIR, "aligned_rn50_sup_avgpool_2048.npy"),
    # Kept so the eval auto-detect logic is unchanged (out of scope for the
    # core release, but referenced by the string-matching fallback):
    "clip_vitl14":          os.path.join(NSD_DATA_DIR, "final_datasets_mindeye2", "averaged_CLIP", "activations_CLIP_averaged.npy"),
    "dino_rn50":            os.path.join(NSD_CLASSIFICATION_DIR, "code",
                                         "inference_results_subject_01_02_03_04_05_06_07_08_all_MULTI",
                                         "dino_rn50_aligned", "all_activations_dino_rn50_fair_Conv+ReLU_rate8_aligned.npy"),
    "dino_vitb16":          os.path.join(NSD_CLASSIFICATION_DIR, "code",
                                         "inference_results_subject_01_02_03_04_05_06_07_08_all_MULTI",
                                         "dino_vitb16_aligned", "all_activations_dino_vitb16_subset.npy"),
    "clip_rn50":            os.path.join(NSD_CLASSIFICATION_DIR, "code",
                                         "inference_results_subject_01_02_03_04_05_06_07_08_all_MULTI",
                                         "clip_rn50_aligned", "activations_all.npy"),
    # MNIST sanity-check scaffolds (out of scope for the core release; kept so
    # the eval auto-detect logic is byte-identical to the original):
    "mnist_rn50_2048":      os.path.join(DATA_ROOT, "ccn_final", "MNIST_experiment", "mimic_mnist_rn50_2048.npy"),
    "mnist_784":            os.path.join(DATA_ROOT, "ccn_final", "MNIST_experiment", "mimic_mnist_activations.npy"),
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
