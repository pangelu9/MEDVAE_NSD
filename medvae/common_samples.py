"""Common-sample selection used on the real --train path (find_common_samples is called
unconditionally in main.py). The exploratory cross-subject alignment suite
that used to live here was removed for the published repo."""

import numpy as np
from load_data import load_activations, load_labels, load_fmri_data
import gc

# --- MEDVAE: resolve data locations via the central config ------------
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from ccn_config import NSD_DATA_DIR


def find_common_samples(args):
    subject_prefix: str = "fmri_subject"
    data_dir=NSD_DATA_DIR
    subject_ids = ["01", "02", "03", "04", "05", "06", "07", "08"]
    import os

    if args.dataset == "streams":
        file_paths = [os.path.join(data_dir,f"fmri_subject{subject_id}_streams_overl_NEW.npz") for subject_id in subject_ids]
        filename_labels = "labels_all_aligned.npy"
    elif args.dataset == "mindeye":
        file_paths = [os.path.join(data_dir,f"final_datasets_mindeye2/averaged/fmri_subject{subject_id}_averaged.npz") for subject_id in subject_ids]
        
        filename_labels = "data/final_datasets_mindeye2/averaged/labels_all.npy"
    method_name="vae_shared872"

    
    nn_activations = load_activations(args, filename=args.filename)
    print("Loading fMRI data...")
    fmri_data_list = load_fmri_data(file_paths, args)

    
    labels = load_labels(args, filename=filename_labels)

    fmri_common, nn_common, labels_common, idx_common = extract_common_samples(fmri_data_list, nn_activations, labels)

    del fmri_data_list, nn_activations, labels
    gc.collect()

    return fmri_common, nn_common, labels_common, idx_common


def common_samples_mask(fmri_data_list):
    """
    Returns the indices of time-points that are valid (non-NaN) for **every** subject.
    """
    mask = ~np.isnan(fmri_data_list[0]).any(axis=1)
    for arr in fmri_data_list[1:]:
        mask &= ~np.isnan(arr).any(axis=1)
    return np.where(mask)[0]


def extract_common_samples(fmri_data_list, nn_activations, labels):
    """
    Slice all modalities to keep only the common valid time-points.

    Returns
    -------
    fmri_common  : list[np.ndarray]   # one array per subject
    nn_common    : np.ndarray
    labels_common: np.ndarray
    idx_common   : np.ndarray         # the indices (same that extract_latents_and_all_recons would return)
    """
    idx_common = common_samples_mask(fmri_data_list)

    fmri_common = [subj[idx_common] for subj in fmri_data_list]

    # handle (1, n) or (n,) label shapes
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels_common = labels[:, idx_common]
    else:
        labels_common = labels[idx_common]

    nn_common = nn_activations[idx_common]

    print("Number of common samples:", len(idx_common))

    return fmri_common, nn_common, labels_common, idx_common
