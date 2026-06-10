"""
Universal evaluation for cross-subject alignment methods: MED-VAE, SRM, Procrustes.

Produces a single consolidated pkl per method with all metrics (alignment,
cross-subject fMRI prediction, reconstruction, retrieval, decoding, silhouette,
cross-trial) consumed by the visualisation scripts.

Usage:
  # VAE
  python evaluate_methods.py --method vae --vae_checkpoint path/to/model.pt \\
      --n_dims 32 --mode not_all8

  # SRM / Procrustes (pass the saved model pkl directly)
  python evaluate_methods.py --method srm --alignment_model_path path/to/srm_model.pkl \\
      --n_dims 32 --mode not_all8
  python evaluate_methods.py --method procrustes --alignment_model_path path/to/proc_model.pkl \\
      --n_dims 32 --mode not_all8
"""

import argparse
import os
import sys
import pickle
import csv
import numpy as np
import h5py
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

# --- MEDVAE: resolve data + package locations via the central config ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import ccn_config
from ccn_config import (MEDVAE_DIR, EVALUATION_DIR, NSD_DIR,
                        NOISE_CEILING_DIR, CROSS_TRIAL_DATA_DIR, RESULTS_DIR,
                        LABELS_PATH, ANN_ACTIVATIONS)
for _p in (EVALUATION_DIR, MEDVAE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from metrics import (
        test_multilabel_decoding_balanced,
        calculate_silhouette_scores
    )
    HAVE_DECODING_SILHOUETTE = True
except ImportError as e:
    print(f"Warning: Could not import decoding/silhouette functions: {e}")
    HAVE_DECODING_SILHOUETTE = False

from evaluation_pipeline import compute_alignment_metrics, compute_cross_subject_retrieval

ANN_ACTIVATIONS_PATH_MINDEYE = ANN_ACTIVATIONS["rn50_mindeye"]
ANN_ACTIVATIONS_PATH_CLIP = ANN_ACTIVATIONS["clip_vitl14"]
ANN_ACTIVATIONS_PATH_STREAMS = ANN_ACTIVATIONS["rn50_streams"]
ANN_ACTIVATIONS_PATH_DINO_RN50 = ANN_ACTIVATIONS["dino_rn50"]
ANN_ACTIVATIONS_PATH_DINO_VITB16 = ANN_ACTIVATIONS["dino_vitb16"]
ANN_ACTIVATIONS_PATH_CLIPRN50 = ANN_ACTIVATIONS["clip_rn50"]
ANN_ACTIVATIONS_PATH = ANN_ACTIVATIONS_PATH_MINDEYE

# ==============================================================================
# Configuration
# ==============================================================================

SUBJ_DIMS_MINDEYE = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
SUBJ_DIMS_STREAMS = {1: 20732, 2: 20735, 3: 20736, 4: 20733, 5: 20733, 6: 20734, 7: 20726, 8: 20733}

SUBJ_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
IDX_TO_SUBJ = {v: k for k, v in SUBJ_TO_IDX.items()}

NC_DATA_DIR = NOISE_CEILING_DIR


# ==============================================================================
# Method Handlers
# ==============================================================================

class VAEHandler:
    """Wraps a MED-VAE model for evaluation."""

    def __init__(self, model, config, device, batch_size=256):
        self.model = model
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.has_ann_metrics = True

    def encode_to_latent(self, fmri, subj_idx):
        import torch
        latents = []
        for start in range(0, len(fmri), self.batch_size):
            end = min(start + self.batch_size, len(fmri))
            batch = torch.tensor(fmri[start:end], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                mu, _ = self.model.encoders[subj_idx](batch)
            latents.append(mu.cpu().numpy())
        return np.vstack(latents)

    def predict_fmri(self, src_fmri, src_idx, tgt_idx):
        import torch
        pred_list = []
        for start in range(0, len(src_fmri), self.batch_size):
            end = min(start + self.batch_size, len(src_fmri))
            batch = torch.tensor(src_fmri[start:end], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                mu, _ = self.model.encoders[src_idx](batch)
                pred = self.model.fmri_decoders[tgt_idx](mu)
            pred_list.append(pred.cpu().numpy())
        return np.vstack(pred_list)

    def reconstruct_fmri(self, fmri, subj_idx):
        return self.predict_fmri(fmri, subj_idx, subj_idx)


class AlignmentTransformer:
    """Wrapper for SRM/Procrustes alignment models."""

    def __init__(self, model_dict, method):
        self.W_list = model_dict['W_list']
        self.pca_models = model_dict['pca_models']
        self.training_means = model_dict['training_means']
        self.n_components = model_dict['n_components']
        self.method = method

    def to_shared_space(self, fmri, subj_idx):
        pca_features = self.pca_models[subj_idx].transform(fmri)
        centered = pca_features - self.training_means[subj_idx]
        shared = centered @ self.W_list[subj_idx]
        return shared

    def transform(self, source_fmri, source_idx, target_idx):
        shared = self.to_shared_space(source_fmri, source_idx)
        target_centered = shared @ self.W_list[target_idx].T
        target_pca = target_centered + self.training_means[target_idx]
        predicted_target = self.pca_models[target_idx].inverse_transform(target_pca)
        return predicted_target


class BaselineHandler:
    """Wraps an AlignmentTransformer (SRM/Procrustes) for evaluation."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.has_ann_metrics = False

    def encode_to_latent(self, fmri, subj_idx):
        return self.transformer.to_shared_space(fmri, subj_idx)

    def predict_fmri(self, src_fmri, src_idx, tgt_idx):
        return self.transformer.transform(src_fmri, src_idx, tgt_idx)

    def reconstruct_fmri(self, fmri, subj_idx):
        return self.transformer.transform(fmri, subj_idx, subj_idx)


# ==============================================================================
# Model Loading
# ==============================================================================

def load_vae_model(checkpoint_path, device):
    import torch
    vae_dir = MEDVAE_DIR
    if vae_dir not in sys.path:
        sys.path.insert(0, vae_dir)
    from model import HybridMultiEncoderVAE

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    print(f"Loading VAE from {checkpoint_path}")
    print(f"  Latent dim: {config.get('latent_dim', 512)}")

    input_dims = config.get('input_dim', [15724, 14278, 15226, 13153, 13039, 17907, 12682, 14386, 51048])
    output_dims = config.get('output_dim', [15724, 14278, 15226, 13153, 13039, 17907, 12682, 14386])
    latent_dim = config.get('latent_dim', 512)
    hidden_dim = config.get('hidden_dim', 1024)
    nn_output_dim = config.get('nn_output_dim', 51048)
    dropout_rate = config.get('dropout_rate', 0.3)

    model = HybridMultiEncoderVAE(
        input_dims=input_dims,
        output_dims=output_dims,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        nn_output_dim=nn_output_dim,
        dropout_rate=dropout_rate,
        only_nn_encoder=False,
        only_fmri_encoders=False,
        use_nn_decoder=True,
        use_fmri_decoders=True
    )

    if checkpoint.get('finetune', False):
        base_model_name = checkpoint['base_model']
        base_path = os.path.join(os.path.dirname(checkpoint_path), '..', base_model_name)
        if not os.path.exists(base_path):
            base_path = os.path.join(os.path.dirname(checkpoint_path), base_model_name)
        if not os.path.exists(base_path):
            base_path = os.path.join(vae_dir, 'results', base_model_name)
        print(f"  Finetuned checkpoint. Loading base model: {base_path}")
        base_checkpoint = torch.load(base_path, map_location=device, weights_only=False)
        model.load_state_dict(base_checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  Number of encoders: {len(model.encoders)}")
    return model, config


def load_alignment_model(method, alignment_model_path, n_components):
    """Load SRM/Procrustes alignment model from a .pkl file or directory."""
    if os.path.isfile(alignment_model_path):
        model_path = alignment_model_path
    else:
        model_dir = alignment_model_path
        possible_paths = [
            os.path.join(model_dir, f"{method}_shared872_{n_components}d_model.pkl"),
            os.path.join(model_dir, f"{method}_shared872_{n_components}d.pkl"),
            os.path.join(model_dir, f"detsrm_shared872_{n_components}d_model.pkl"),
            os.path.join(model_dir, f"detsrm_shared872_{n_components}d.pkl"),
            os.path.join(model_dir, f"srm_shared872_{n_components}d_model.pkl"),
            os.path.join(model_dir, f"procrustes_shared872_{n_components}d_model.pkl"),
            os.path.join(model_dir, f"procrustes_a_shared872_{n_components}d_model.pkl"),
        ]
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        if model_path is None:
            raise FileNotFoundError(f"Alignment model not found. Tried: {possible_paths}")

    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    print(f"Loaded {method.upper()} alignment model from {model_path}")
    print(f"  Subjects: {model_dict['n_subjects']}")
    print(f"  Components: {model_dict['n_components']}")
    print(f"  PCA features: {model_dict['n_pca_features']}")

    return model_dict


# ==============================================================================
# Noise Ceilings
# ==============================================================================

def load_noise_ceilings(subjects, nc_data_dir=NC_DATA_DIR):
    nc_dict = {}
    for subj in subjects:
        subj_str = f"{subj:02d}"
        nc_path = os.path.join(nc_data_dir, f"subj{subj_str}", "ncsnr_processed",
                               f"subj{subj_str}_noiseceiling_NC.npy")
        if not os.path.exists(nc_path):
            print(f"  Warning: NC file not found for subject {subj}: {nc_path}")
            return None
        nc = np.load(nc_path)
        nc_dict[subj] = nc
        print(f"  Subject {subj}: loaded NC array ({nc.shape[0]} voxels, mean NC={np.nanmean(nc):.4f})")
    return nc_dict


# ==============================================================================
# Data Loading
# ==============================================================================

def get_common_valid_test_indices_streams(data_path, subjects, test_size=0.1):
    from sklearn.model_selection import train_test_split

    valid_masks = []
    for subj in subjects:
        subj_str = f"{subj:02d}"
        npz_path = os.path.join(data_path, "data", f"fmri_subject{subj_str}_streams_overl_NEW.npz")
        fmri_data = np.load(npz_path)['fmri_data']
        valid_masks.append(~np.isnan(fmri_data).any(axis=1))

    all_valid = np.all(valid_masks, axis=0)
    valid_indices = np.where(all_valid)[0]
    print(f"  Rows valid for ALL subjects {subjects}: {len(valid_indices)}")

    if test_size >= 1.0:
        print(f"  Using ALL {len(valid_indices)} valid indices (test_size=1.0)")
        return valid_indices

    _, test_idx = train_test_split(valid_indices, test_size=test_size, random_state=42)
    print(f"  Common test indices: {len(test_idx)}")
    return test_idx


def load_subject_test_data_streams(data_path, subj, test_size=0.1, common_test_indices=None):
    from sklearn.model_selection import train_test_split

    subj_str = f"{subj:02d}"
    npz_path = os.path.join(data_path, "data", f"fmri_subject{subj_str}_streams_overl_NEW.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Streams data not found at {npz_path}")

    npz_data = np.load(npz_path)
    fmri_data = npz_data['fmri_data']

    if 'image_indices' in npz_data:
        all_img_indices = npz_data['image_indices']
    elif 'labels' in npz_data:
        all_img_indices = npz_data['labels']
        if len(all_img_indices.shape) > 1:
            all_img_indices = np.arange(len(fmri_data))
    else:
        all_img_indices = np.arange(len(fmri_data))

    print(f"  Loaded subj{subj_str} from streams: {fmri_data.shape[-1]} voxels, {len(fmri_data)} samples")
    n_samples = len(fmri_data)

    if common_test_indices is not None:
        test_indices = common_test_indices
    else:
        indices = np.arange(n_samples)
        if test_size >= 1.0:
            test_indices = indices
        else:
            _, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
        test_fmri_temp = fmri_data[test_indices]
        valid_mask = ~np.isnan(test_fmri_temp).any(axis=1)
        test_indices = test_indices[valid_mask]

    test_fmri = fmri_data[test_indices]
    test_img_idx = all_img_indices[test_indices] if len(all_img_indices) == n_samples else test_indices

    print(f"  Test samples: {len(test_fmri)}, Unique images: {len(np.unique(test_img_idx))}")
    return test_fmri.astype(np.float32), test_img_idx


def load_subject_test_data(data_path, subj, new_test=True, dataset='streams',
                           test_size=0.1, common_test_indices=None):
    return load_subject_test_data_streams(data_path, subj, test_size, common_test_indices)


def get_shared_images(data_path, target_subjects=[1, 2, 5, 7], exclude_subjects=[3, 4, 6, 8],
                      new_test=True, mode='not_all8', dataset='streams', test_size=0.1):
    print("\n" + "="*60)
    print(f"FINDING SHARED IMAGES (mode={mode}, dataset={dataset})")
    print("="*60)

    target_images = {}
    target_data = {}

    common_test_indices = None
    if dataset == 'streams':
        common_test_indices = get_common_valid_test_indices_streams(data_path, target_subjects, test_size)

    for subj in target_subjects:
        voxels, img_indices = load_subject_test_data(
            data_path, subj, new_test, dataset=dataset, test_size=test_size,
            common_test_indices=common_test_indices
        )
        target_images[subj] = set(np.unique(img_indices))
        target_data[subj] = {'voxels': voxels, 'img_indices': img_indices}
        print(f"  Subject {subj}: {len(target_images[subj])} unique test images")

    common_target = target_images[target_subjects[0]]
    for subj in target_subjects[1:]:
        common_target = common_target.intersection(target_images[subj])
    print(f"\nCommon among subjects {target_subjects}: {len(common_target)} images")

    if mode == 'all_common':
        return common_target, target_data

    exclude_images_per_subj = {}
    for subj in exclude_subjects:
        try:
            _, img_indices = load_subject_test_data(data_path, subj, new_test, dataset=dataset, test_size=test_size)
            exclude_images_per_subj[subj] = set(np.unique(img_indices))
            print(f"  Subject {subj}: {len(exclude_images_per_subj[subj])} unique test images")
        except Exception as e:
            print(f"  Subject {subj}: Could not load - {e}")
            exclude_images_per_subj[subj] = set()

    if mode == 'exclusive':
        exclude_union = set()
        for subj_images in exclude_images_per_subj.values():
            exclude_union = exclude_union.union(subj_images)
        selected_images = common_target - exclude_union
        print(f"\nEXCLUSIVE IMAGES (not in ANY of {exclude_subjects}): {len(selected_images)}")
    elif mode == 'not_all8':
        if exclude_images_per_subj:
            common_exclude = exclude_images_per_subj[exclude_subjects[0]]
            for subj in exclude_subjects[1:]:
                common_exclude = common_exclude.intersection(exclude_images_per_subj[subj])
        else:
            common_exclude = set()
        selected_images = common_target - common_exclude
        print(f"\nSHARED NOT ALL 8 (in {target_subjects} but not ALL of {exclude_subjects}): {len(selected_images)}")
    elif mode == 'shared_only':
        if exclude_images_per_subj:
            common_exclude = exclude_images_per_subj[exclude_subjects[0]]
            for subj in exclude_subjects[1:]:
                common_exclude = common_exclude.intersection(exclude_images_per_subj[subj])
        else:
            common_exclude = set()
        selected_images = common_target.intersection(common_exclude)
        print(f"\nSHARED ONLY (in ALL 8 subjects): {len(selected_images)}")

    return selected_images, target_data


def extract_common_fmri(target_data, selected_images, subjects):
    common_fmri = {}
    sorted_images = sorted(selected_images)

    for subj in subjects:
        voxels = target_data[subj]['voxels']
        img_indices = target_data[subj]['img_indices']

        img_to_voxels = defaultdict(list)
        for i, img_idx in enumerate(img_indices):
            if img_idx in selected_images:
                img_to_voxels[img_idx].append(voxels[i])

        fmri_list = []
        for img_idx in sorted_images:
            trials = img_to_voxels[img_idx]
            if len(trials) > 0:
                fmri_list.append(np.mean(trials, axis=0))

        common_fmri[subj] = np.array(fmri_list)
        print(f"  Subject {subj}: {common_fmri[subj].shape[0]} images, {common_fmri[subj].shape[1]} voxels")

    return common_fmri, sorted_images


def load_vae_test_split(data_path, subjects, remove_all_overlaps=False, filter_no_fmri=False):
    from sklearn.model_selection import train_test_split

    print("\n" + "="*60)
    print("REPLICATING VAE TRAINING TEST SPLIT")
    print("="*60)

    subject_ids = [f"{s:02d}" for s in sorted(subjects)]

    brain_data_list = []
    for subj_str in subject_ids:
        fpath = os.path.join(data_path, "data", f"fmri_subject{subj_str}_streams_overl_NEW.npz")
        fmri = np.load(fpath, mmap_mode='r')['fmri_data']
        brain_data_list.append(fmri)
        print(f"  Loaded subject {subj_str}: {fmri.shape}")

    nn_path = os.path.join(data_path, "data", "final_datasets_mindeye2", "averaged", "activations_all.npy")
    nn_activations = np.load(nn_path, mmap_mode='r')
    n_samples_orig = nn_activations.shape[0]

    if remove_all_overlaps:
        print("  Applying remove_all_overlaps...")
        n_valid_per_row = np.zeros(n_samples_orig, dtype=int)
        for bd in brain_data_list:
            valid = ~np.isnan(bd[:n_samples_orig]).any(axis=1)
            n_valid_per_row += valid.astype(int)
        overlap_rows = n_valid_per_row > 1
        for i in range(len(brain_data_list)):
            bd = brain_data_list[i].copy() if not brain_data_list[i].flags.writeable else brain_data_list[i]
            if not bd.flags.writeable:
                bd = bd.copy()
            bd[overlap_rows] = np.nan
            brain_data_list[i] = bd

    if filter_no_fmri:
        print("  Applying filter_no_fmri...")
        has_fmri = np.zeros(n_samples_orig, dtype=bool)
        for bd in brain_data_list:
            has_fmri |= ~np.isnan(bd[:n_samples_orig]).all(axis=1)
        valid_indices = np.where(has_fmri)[0]
        brain_data_list = [bd[valid_indices] for bd in brain_data_list]
        n_samples = len(valid_indices)
    else:
        n_samples = n_samples_orig

    indices = np.arange(n_samples)
    _, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    print(f"  Train: {n_samples - len(test_indices)}, Test: {len(test_indices)}")

    per_subject_fmri = {}
    for idx, subj in enumerate(sorted(subjects)):
        test_fmri = brain_data_list[idx][test_indices]
        valid_mask = ~np.isnan(test_fmri).any(axis=1)
        per_subject_fmri[subj] = test_fmri[valid_mask]
        print(f"  Subject {subj}: {valid_mask.sum()}/{len(test_indices)} valid test images → {per_subject_fmri[subj].shape}")

    return per_subject_fmri


# ==============================================================================
# Cross-Trial Data Loading
# ==============================================================================

def load_cross_trial_data_streams(hero_subjects, mode='not_all8',
                                   cross_trial_dir=CROSS_TRIAL_DATA_DIR):
    if not os.path.isdir(cross_trial_dir):
        print(f"  Cross-trial data directory not found: {cross_trial_dir}")
        return None

    all_subjects = list(range(1, 9))
    exclude_subjects = [s for s in all_subjects if s not in hero_subjects]

    image_to_valid_subjects = defaultdict(set)
    subject_data = {}

    for subj in all_subjects:
        fpath = os.path.join(cross_trial_dir, f"fmri_subject{subj:02d}_trials_preprocessed.npz")
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found")
            continue
        npz = np.load(fpath)
        fmri_data = npz['fmri_data']
        image_ids = npz['image_ids']

        valid_rows = ~np.all(np.isnan(fmri_data), axis=1)
        valid_image_ids = np.unique(image_ids[valid_rows])
        for img_id in valid_image_ids:
            image_to_valid_subjects[img_id].add(subj)

        n_valid = int(np.sum(valid_rows))
        print(f"  Subject {subj}: {n_valid}/{len(fmri_data)} valid rows, "
              f"{len(valid_image_ids)} images with data")

        if subj in hero_subjects:
            subject_data[subj] = {'fmri_data': fmri_data, 'image_ids': image_ids}

    hero_set = set(hero_subjects)
    all_set = set(all_subjects)
    exclude_set = set(exclude_subjects)

    if mode == 'not_all8':
        selected = {img for img, subjs in image_to_valid_subjects.items()
                    if hero_set.issubset(subjs) and subjs != all_set}
    elif mode == 'exclusive':
        selected = {img for img, subjs in image_to_valid_subjects.items()
                    if hero_set.issubset(subjs) and not subjs.intersection(exclude_set)}
    else:
        selected = {img for img, subjs in image_to_valid_subjects.items()
                    if hero_set.issubset(subjs)}

    print(f"  Cross-trial images selected ({mode}): {len(selected)}")

    cross_trial_data = {}
    for subj in hero_subjects:
        if subj not in subject_data:
            continue
        fmri_data = subject_data[subj]['fmri_data']
        image_ids = subject_data[subj]['image_ids']
        grouped = defaultdict(list)
        for i, img_id in enumerate(image_ids):
            if img_id in selected and not np.all(np.isnan(fmri_data[i])):
                grouped[img_id].append(fmri_data[i])
        grouped = {k: v for k, v in grouped.items() if len(v) >= 2}
        n_trials = sum(len(v) for v in grouped.values())
        cross_trial_data[subj] = {
            'grouped_trials': grouped, 'n_images': len(grouped), 'n_total_trials': n_trials
        }
        print(f"  Subject {subj}: {len(grouped)} multi-trial images, {n_trials} trials")

    return cross_trial_data if cross_trial_data else None


# ==============================================================================
# Shared Metric Computation
# ==============================================================================

def compute_cross_subject_fmri_prediction(handler, common_fmri, subjects,
                                           novel_subject=None, nc_dict=None, nc_threshold=0.1):
    print(f"\n{'='*60}")
    print(f"CROSS-SUBJECT fMRI PREDICTION")
    print(f"{'='*60}")

    n_subj = len(subjects)
    pred_matrix = np.zeros((n_subj, n_subj))
    pred_matrix_img = np.zeros((n_subj, n_subj))
    nc_norm_matrix = np.full((n_subj, n_subj), np.nan)

    for i, src_subj in enumerate(subjects):
        src_idx = SUBJ_TO_IDX[src_subj]
        src_fmri = common_fmri[src_subj]
        n_samples = src_fmri.shape[0]

        for j, tgt_subj in enumerate(subjects):
            tgt_idx = SUBJ_TO_IDX[tgt_subj]
            tgt_fmri = common_fmri[tgt_subj]

            pred_fmri = handler.predict_fmri(src_fmri, src_idx, tgt_idx)

            n_voxels = tgt_fmri.shape[1]
            corr_vals = []
            per_voxel_r = np.full(n_voxels, np.nan)
            for v in range(n_voxels):
                if np.std(tgt_fmri[:, v]) > 0 and np.std(pred_fmri[:, v]) > 0:
                    r, _ = pearsonr(tgt_fmri[:, v], pred_fmri[:, v])
                    if not np.isnan(r):
                        corr_vals.append(r)
                        per_voxel_r[v] = r
            pred_matrix[i, j] = np.mean(corr_vals) if corr_vals else 0

            img_corrs = []
            for s in range(n_samples):
                if np.std(tgt_fmri[s]) > 0 and np.std(pred_fmri[s]) > 0:
                    r, _ = pearsonr(tgt_fmri[s], pred_fmri[s])
                    if not np.isnan(r):
                        img_corrs.append(r)
            pred_matrix_img[i, j] = np.mean(img_corrs) if img_corrs else 0

            nc_str = ""
            if nc_dict is not None and tgt_subj in nc_dict:
                nc = nc_dict[tgt_subj]
                if nc.shape[0] == n_voxels:
                    r_clamped = np.maximum(per_voxel_r, 0.0)
                    r_squared = r_clamped ** 2
                    valid = ~np.isnan(per_voxel_r) & (nc > nc_threshold)
                    if np.sum(valid) > 0:
                        nc_norm = np.mean(r_squared[valid] / nc[valid]) * 100
                        nc_norm_matrix[i, j] = nc_norm
                        nc_str = f", NC_norm={nc_norm:.2f}%"

            print(f"  S{src_subj}→S{tgt_subj}: r_voxel = {pred_matrix[i, j]:.4f} ({len(corr_vals)} voxels), "
                  f"r_image = {pred_matrix_img[i, j]:.4f} ({len(img_corrs)} images){nc_str}")

    diag_mean = np.diag(pred_matrix).mean()
    diag_mean_img = np.diag(pred_matrix_img).mean()
    if novel_subject is not None and novel_subject in subjects:
        novel_idx = subjects.index(novel_subject)
        novel_pairs = [(i, j) for i in range(n_subj) for j in range(i+1, n_subj)
                       if i == novel_idx or j == novel_idx]
        offdiag_mean = np.mean([pred_matrix[i, j] for i, j in novel_pairs])
        offdiag_mean_img = np.mean([pred_matrix_img[i, j] for i, j in novel_pairs])
    else:
        mask = ~np.eye(n_subj, dtype=bool)
        offdiag_mean = pred_matrix[mask].mean()
        offdiag_mean_img = pred_matrix_img[mask].mean()

    print(f"\nSummary (per-voxel):")
    print(f"  Diagonal (self-recon) mean: {diag_mean:.4f}")
    print(f"  Off-diagonal (cross-pred) mean: {offdiag_mean:.4f}")

    result = {
        'fmri_prediction_matrix': pred_matrix,
        'fmri_pred_diagonal_mean': diag_mean,
        'fmri_pred_offdiag_mean': offdiag_mean,
        'fmri_prediction_matrix_image': pred_matrix_img,
        'fmri_pred_diagonal_mean_image': diag_mean_img,
        'fmri_pred_offdiag_mean_image': offdiag_mean_img,
    }

    if not np.all(np.isnan(nc_norm_matrix)):
        mask = ~np.eye(n_subj, dtype=bool)
        nc_diag_mean = np.nanmean(np.diag(nc_norm_matrix))
        nc_offdiag_mean = np.nanmean(nc_norm_matrix[mask])
        result['fmri_pred_nc_norm_matrix'] = nc_norm_matrix
        result['fmri_pred_nc_norm_diagonal_mean'] = nc_diag_mean
        result['fmri_pred_nc_norm_offdiag_mean'] = nc_offdiag_mean
        print(f"\n  Diagonal NC-norm mean: {nc_diag_mean:.2f}%")
        print(f"  Off-diagonal NC-norm mean: {nc_offdiag_mean:.2f}%")

    return result


def compute_reconstruction_quality(handler, common_fmri, subjects,
                                   nc_dict=None, nc_threshold=0.1):
    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION QUALITY (self-reconstruction)")
    print(f"{'='*60}")

    voxel_corrs, r2_scores, global_corrs = [], [], []
    nc_norm_scores = []
    per_subject = {}

    for subj in subjects:
        subj_idx = SUBJ_TO_IDX[subj]
        orig_fmri = common_fmri[subj]

        pred_fmri = handler.reconstruct_fmri(orig_fmri, subj_idx)

        n_voxels = orig_fmri.shape[1]
        voxel_corr_list = []
        per_voxel_r = np.full(n_voxels, np.nan)
        for v in range(n_voxels):
            if np.std(orig_fmri[:, v]) > 0 and np.std(pred_fmri[:, v]) > 0:
                r, _ = pearsonr(orig_fmri[:, v], pred_fmri[:, v])
                if not np.isnan(r):
                    voxel_corr_list.append(r)
                    per_voxel_r[v] = r
        mean_voxel_corr = np.mean(voxel_corr_list) if voxel_corr_list else 0.0

        ss_res = np.sum((orig_fmri - pred_fmri) ** 2)
        ss_tot = np.sum((orig_fmri - orig_fmri.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        global_corr, _ = pearsonr(orig_fmri.ravel(), pred_fmri.ravel())

        voxel_corrs.append(mean_voxel_corr)
        r2_scores.append(r2)
        global_corrs.append(global_corr)

        subj_result = {
            'voxel_correlation': mean_voxel_corr,
            'r2': r2,
            'global_correlation': global_corr
        }

        if nc_dict is not None and subj in nc_dict:
            nc = nc_dict[subj]
            if nc.shape[0] == n_voxels:
                r_clamped = np.maximum(per_voxel_r, 0.0)
                r_squared = r_clamped ** 2
                valid = ~np.isnan(per_voxel_r) & (nc > nc_threshold)
                if np.sum(valid) > 0:
                    nc_norm = np.mean(r_squared[valid] / nc[valid]) * 100
                    nc_norm_scores.append(nc_norm)
                    subj_result['nc_norm_voxel_corr'] = nc_norm
                    subj_result['nc_norm_n_voxels'] = int(np.sum(valid))

        per_subject[subj] = subj_result
        nc_str = f", NC_norm={subj_result.get('nc_norm_voxel_corr', 0):.2f}%" if 'nc_norm_voxel_corr' in subj_result else ""
        print(f"  Subject {subj}: voxel_corr={mean_voxel_corr:.4f}, R²={r2:.4f}, global_corr={global_corr:.4f}{nc_str}")

    avg_voxel_corr = np.mean(voxel_corrs)
    avg_r2 = np.mean(r2_scores)
    avg_global_corr = np.mean(global_corrs)

    print(f"\n  Mean voxel correlation: {avg_voxel_corr:.4f}")
    print(f"  Mean R²: {avg_r2:.4f}")
    print(f"  Mean global correlation: {avg_global_corr:.4f}")

    result = {
        'recon_per_subject': per_subject,
        'recon_avg_voxel_corr': avg_voxel_corr,
        'recon_avg_r2': avg_r2,
        'recon_avg_global_corr': avg_global_corr
    }

    if nc_norm_scores:
        avg_nc_norm = np.mean(nc_norm_scores)
        result['recon_avg_nc_norm_voxel_corr'] = avg_nc_norm
        print(f"  Mean NC-normalised voxel corr: {avg_nc_norm:.2f}%")

    return result


def compute_cross_trial_correlations(handler, cross_trial_data, subjects):
    print(f"\n{'='*60}")
    print(f"CROSS-TRIAL CORRELATION")
    print(f"{'='*60}")

    per_subject = {}
    agg_sample, agg_direct, agg_voxel, agg_dvoxel, agg_ratio = [], [], [], [], []

    for subj in subjects:
        if subj not in cross_trial_data:
            print(f"  Subject {subj}: no trial data, skipping")
            continue

        subj_idx = SUBJ_TO_IDX[subj]
        grouped_trials = cross_trial_data[subj]['grouped_trials']

        all_fmri, all_img_ids = [], []
        for img_id, trials_list in grouped_trials.items():
            for trial in trials_list:
                all_fmri.append(trial)
                all_img_ids.append(img_id)

        all_fmri = np.array(all_fmri)
        all_img_ids = np.array(all_img_ids)

        all_recon = handler.reconstruct_fmri(all_fmri, subj_idx)

        gt_i_list, recon_j_list, gt_j_list = [], [], []
        for img_id in grouped_trials:
            idxs = np.where(all_img_ids == img_id)[0]
            for i in idxs:
                for j in idxs:
                    if i != j:
                        gt_i_list.append(all_fmri[i])
                        recon_j_list.append(all_recon[j])
                        gt_j_list.append(all_fmri[j])

        all_gt_i = np.array(gt_i_list)
        all_recon_j = np.array(recon_j_list)
        all_gt_j = np.array(gt_j_list)
        n_pairs = len(all_gt_i)

        sample_corrs_recon, sample_corrs_direct = [], []
        for p in range(n_pairs):
            recon_ok = np.var(all_gt_i[p]) > 1e-8 and np.var(all_recon_j[p]) > 1e-8
            direct_ok = np.var(all_gt_i[p]) > 1e-8 and np.var(all_gt_j[p]) > 1e-8
            if recon_ok:
                r = np.corrcoef(all_gt_i[p], all_recon_j[p])[0, 1]
                if not np.isnan(r):
                    sample_corrs_recon.append(r)
            if direct_ok:
                r = np.corrcoef(all_gt_i[p], all_gt_j[p])[0, 1]
                if not np.isnan(r):
                    sample_corrs_direct.append(r)

        mean_sample = np.mean(sample_corrs_recon) if sample_corrs_recon else 0.0
        mean_direct = np.mean(sample_corrs_direct) if sample_corrs_direct else 0.0

        n_voxels = all_gt_i.shape[1]
        voxel_recon, voxel_direct = [], []
        for v in range(n_voxels):
            gt_v = all_gt_i[:, v]
            rec_v = all_recon_j[:, v]
            gt_j_v = all_gt_j[:, v]
            if np.std(gt_v) > 0 and np.std(rec_v) > 0:
                r, _ = pearsonr(gt_v, rec_v)
                if not np.isnan(r):
                    voxel_recon.append(r)
            if np.std(gt_v) > 0 and np.std(gt_j_v) > 0:
                r, _ = pearsonr(gt_v, gt_j_v)
                if not np.isnan(r):
                    voxel_direct.append(r)

        mean_voxel = np.mean(voxel_recon) if voxel_recon else 0.0
        mean_dvoxel = np.mean(voxel_direct) if voxel_direct else 0.0
        ratio = mean_sample / mean_direct if mean_direct > 0 else 0.0

        per_subject[subj] = {
            'mean_sample_corr': mean_sample,
            'mean_direct_corr': mean_direct,
            'mean_voxel_corr': mean_voxel,
            'mean_direct_voxel_corr': mean_dvoxel,
            'performance_ratio': ratio,
            'n_images': cross_trial_data[subj]['n_images'],
            'n_pairs': n_pairs
        }
        print(f"  Subject {subj}: sample_corr={mean_sample:.4f}, ceiling={mean_direct:.4f}, "
              f"ratio={ratio:.2%}, voxel_corr={mean_voxel:.4f} ({cross_trial_data[subj]['n_images']} images)")

        agg_sample.append(mean_sample)
        agg_direct.append(mean_direct)
        agg_voxel.append(mean_voxel)
        agg_dvoxel.append(mean_dvoxel)
        agg_ratio.append(ratio)

    if not agg_sample:
        return None

    result = {
        'cross_trial_per_subject': per_subject,
        'cross_trial_avg_sample_corr': np.mean(agg_sample),
        'cross_trial_avg_direct_corr': np.mean(agg_direct),
        'cross_trial_avg_voxel_corr': np.mean(agg_voxel),
        'cross_trial_avg_direct_voxel_corr': np.mean(agg_dvoxel),
        'cross_trial_avg_performance_ratio': np.mean(agg_ratio)
    }
    print(f"\n  Mean sample corr: {result['cross_trial_avg_sample_corr']:.4f}")
    print(f"  Mean ceiling:     {result['cross_trial_avg_direct_corr']:.4f}")
    print(f"  Mean voxel corr:  {result['cross_trial_avg_voxel_corr']:.4f}")
    print(f"  Mean ratio:       {result['cross_trial_avg_performance_ratio']:.2%}")
    return result


# ==============================================================================
# ANN Metrics (VAE-only)
# ==============================================================================

def compute_ann_to_fmri_prediction(model, common_fmri, sorted_images, subjects, device,
                                    ann_activations_path=ANN_ACTIVATIONS_PATH,
                                    batch_size=256, nc_dict=None, nc_threshold=0.1,
                                    fmri_latent_list=None, nn_output_dim=None):
    import torch
    print(f"\n{'='*60}")
    print(f"ANN → fMRI PREDICTION")
    print(f"{'='*60}")

    nn_enc_idx = model.nn_encoder_idx
    if nn_enc_idx < 0:
        print("  Skipping: model has no ANN encoder")
        return {}

    print(f"  Loading ANN activations from {ann_activations_path}")
    all_activations = np.load(ann_activations_path)
    if nn_output_dim is not None and all_activations.shape[1] > nn_output_dim:
        all_activations = all_activations[:, :nn_output_dim]
    all_activations = (all_activations - all_activations.mean(axis=0)) / (all_activations.std(axis=0) + 1e-8)
    all_activations = all_activations.astype(np.float32)
    ann_test = all_activations[sorted_images]
    print(f"  ANN test shape: {ann_test.shape}")
    del all_activations

    n_samples = ann_test.shape[0]
    latents = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = torch.tensor(ann_test[start:end], dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, _ = model.encoders[nn_enc_idx](batch)
        latents.append(mu.cpu().numpy())
    latent_z = np.vstack(latents)

    ann_fmri_comp_corr = None
    ann_fmri_rsa_eucl = None
    ann_fmri_rsa_pear = None
    if fmri_latent_list is not None and len(fmri_latent_list) > 0:
        n_components = latent_z.shape[1]
        n_subj = len(fmri_latent_list)
        per_subj_comp = np.zeros((n_subj, n_components))
        for s_idx in range(n_subj):
            fmri_lat = fmri_latent_list[s_idx]
            for d in range(n_components):
                r, _ = pearsonr(latent_z[:, d], fmri_lat[:, d])
                per_subj_comp[s_idx, d] = r if not np.isnan(r) else 0.0
        ann_fmri_comp_corr = per_subj_comp.mean(axis=0)
        print(f"  ANN-fMRI per-component correlation mean: {ann_fmri_comp_corr.mean():.4f}")

        ann_rdm_eucl = pdist(latent_z, 'euclidean')
        ann_rdm_pear = pdist(latent_z, 'correlation')
        rsa_eucl_per_subj, rsa_pear_per_subj = [], []
        for s_idx in range(n_subj):
            fmri_rdm_eucl = pdist(fmri_latent_list[s_idx], 'euclidean')
            fmri_rdm_pear = pdist(fmri_latent_list[s_idx], 'correlation')
            r_eucl, _ = pearsonr(ann_rdm_eucl, fmri_rdm_eucl)
            r_pear, _ = pearsonr(ann_rdm_pear, fmri_rdm_pear)
            rsa_eucl_per_subj.append(r_eucl if not np.isnan(r_eucl) else 0.0)
            rsa_pear_per_subj.append(r_pear if not np.isnan(r_pear) else 0.0)
        ann_fmri_rsa_eucl = float(np.mean(rsa_eucl_per_subj))
        ann_fmri_rsa_pear = float(np.mean(rsa_pear_per_subj))

    voxel_corrs, r2_scores, global_corrs = [], [], []
    nc_norm_scores = []
    per_subject = {}

    for subj in subjects:
        subj_idx = SUBJ_TO_IDX[subj]
        orig_fmri = common_fmri[subj]
        pred_fmri_list = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            z_batch = torch.tensor(latent_z[start:end], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = model.fmri_decoders[subj_idx](z_batch)
            pred_fmri_list.append(pred.cpu().numpy())
        pred_fmri = np.vstack(pred_fmri_list)

        n_voxels = orig_fmri.shape[1]
        voxel_corr_list = []
        per_voxel_r = np.full(n_voxels, np.nan)
        for v in range(n_voxels):
            if np.std(orig_fmri[:, v]) > 0 and np.std(pred_fmri[:, v]) > 0:
                r, _ = pearsonr(orig_fmri[:, v], pred_fmri[:, v])
                if not np.isnan(r):
                    voxel_corr_list.append(r)
                    per_voxel_r[v] = r
        mean_voxel_corr = np.mean(voxel_corr_list) if voxel_corr_list else 0.0

        ss_res = np.sum((orig_fmri - pred_fmri) ** 2)
        ss_tot = np.sum((orig_fmri - orig_fmri.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        global_corr, _ = pearsonr(orig_fmri.ravel(), pred_fmri.ravel())

        voxel_corrs.append(mean_voxel_corr)
        r2_scores.append(r2)
        global_corrs.append(global_corr)

        subj_result = {'voxel_correlation': mean_voxel_corr, 'r2': r2, 'global_correlation': global_corr}
        if nc_dict is not None and subj in nc_dict:
            nc = nc_dict[subj]
            if nc.shape[0] == n_voxels:
                r_clamped = np.maximum(per_voxel_r, 0.0)
                r_squared = r_clamped ** 2
                valid = ~np.isnan(per_voxel_r) & (nc > nc_threshold)
                if np.sum(valid) > 0:
                    nc_norm = np.mean(r_squared[valid] / nc[valid]) * 100
                    nc_norm_scores.append(nc_norm)
                    subj_result['nc_norm_voxel_corr'] = nc_norm
        per_subject[subj] = subj_result
        print(f"  Subject {subj}: voxel_corr={mean_voxel_corr:.4f}, R²={r2:.4f}")

    result = {
        'ann2fmri_per_subject': per_subject,
        'ann2fmri_avg_voxel_corr': np.mean(voxel_corrs),
        'ann2fmri_avg_r2': np.mean(r2_scores),
        'ann2fmri_avg_global_corr': np.mean(global_corrs)
    }
    if nc_norm_scores:
        result['ann2fmri_avg_nc_norm_voxel_corr'] = np.mean(nc_norm_scores)
    if ann_fmri_comp_corr is not None:
        result['ann_fmri_per_component_corr'] = ann_fmri_comp_corr
        result['ann_fmri_avg_comp_corr'] = float(ann_fmri_comp_corr.mean())
    if ann_fmri_rsa_eucl is not None:
        result['ann_fmri_rsa_euclidean'] = ann_fmri_rsa_eucl
        result['ann_fmri_rsa_pearson'] = ann_fmri_rsa_pear
    return result


def compute_ann_to_ann_reconstruction(model, sorted_images, device,
                                       ann_activations_path=ANN_ACTIVATIONS_PATH,
                                       batch_size=256, nn_output_dim=None):
    import torch
    print(f"\n{'='*60}")
    print(f"ANN → ANN RECONSTRUCTION")
    print(f"{'='*60}")

    nn_enc_idx = model.nn_encoder_idx
    if nn_enc_idx < 0 or model.nn_decoder is None:
        print("  Skipping: model has no ANN encoder/decoder")
        return {}

    all_activations = np.load(ann_activations_path)
    if nn_output_dim is not None and all_activations.shape[1] > nn_output_dim:
        all_activations = all_activations[:, :nn_output_dim]
    all_activations = (all_activations - all_activations.mean(axis=0)) / (all_activations.std(axis=0) + 1e-8)
    all_activations = all_activations.astype(np.float32)
    ann_test = all_activations[sorted_images]
    del all_activations

    n_samples, n_features = ann_test.shape
    pred_ann_list = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = torch.tensor(ann_test[start:end], dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, _ = model.encoders[nn_enc_idx](batch)
            pred = model.nn_decoder(mu)
        pred_ann_list.append(pred.cpu().numpy())
    pred_ann = np.vstack(pred_ann_list)

    feature_corrs = np.full(n_features, np.nan)
    for f in range(n_features):
        if np.std(ann_test[:, f]) > 0 and np.std(pred_ann[:, f]) > 0:
            r, _ = pearsonr(ann_test[:, f], pred_ann[:, f])
            if not np.isnan(r):
                feature_corrs[f] = r
    valid = ~np.isnan(feature_corrs)
    mean_feature_corr = np.mean(feature_corrs[valid]) if valid.any() else 0.0

    image_corrs = []
    for i in range(n_samples):
        if np.std(ann_test[i]) > 0 and np.std(pred_ann[i]) > 0:
            r, _ = pearsonr(ann_test[i], pred_ann[i])
            if not np.isnan(r):
                image_corrs.append(r)
    mean_image_corr = np.mean(image_corrs) if image_corrs else 0.0

    ss_res = np.sum((ann_test - pred_ann) ** 2)
    ss_tot = np.sum((ann_test - ann_test.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    global_corr, _ = pearsonr(ann_test.ravel(), pred_ann.ravel())

    print(f"  Per-feature correlation: {mean_feature_corr:.4f}")
    print(f"  Per-image correlation: {mean_image_corr:.4f}")
    print(f"  R²: {r2:.4f}")

    return {
        'ann2ann_mean_feature_corr': mean_feature_corr,
        'ann2ann_mean_image_corr': mean_image_corr,
        'ann2ann_r2': r2,
        'ann2ann_global_corr': global_corr,
        'ann2ann_n_features': n_features,
        'ann2ann_n_images': n_samples,
    }


def compute_fmri_to_ann_reconstruction(model, common_fmri, sorted_images, subjects, device,
                                        ann_activations_path=ANN_ACTIVATIONS_PATH,
                                        batch_size=256, nn_output_dim=None):
    import torch
    print(f"\n{'='*60}")
    print(f"fMRI → ANN RECONSTRUCTION")
    print(f"{'='*60}")

    if model.nn_decoder is None:
        print("  Skipping: model has no NN decoder")
        return {}

    all_activations = np.load(ann_activations_path)
    if nn_output_dim is not None and all_activations.shape[1] > nn_output_dim:
        all_activations = all_activations[:, :nn_output_dim]
    all_activations = (all_activations - all_activations.mean(axis=0)) / (all_activations.std(axis=0) + 1e-8)
    all_activations = all_activations.astype(np.float32)
    ann_gt = all_activations[sorted_images]
    n_samples, n_features = ann_gt.shape
    del all_activations

    per_subject = {}
    all_feature_corrs, all_image_corrs = [], []

    for subj in subjects:
        subj_idx = SUBJ_TO_IDX[subj]
        fmri = common_fmri[subj]
        latents = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = torch.tensor(fmri[start:end], dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, _ = model.encoders[subj_idx](batch)
            latents.append(mu.cpu().numpy())
        latent_z = np.vstack(latents)

        pred_ann_list = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            z_batch = torch.tensor(latent_z[start:end], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = model.nn_decoder(z_batch)
            pred_ann_list.append(pred.cpu().numpy())
        pred_ann = np.vstack(pred_ann_list)

        feature_corrs = np.full(n_features, np.nan)
        for f in range(n_features):
            if np.std(ann_gt[:, f]) > 0 and np.std(pred_ann[:, f]) > 0:
                r, _ = pearsonr(ann_gt[:, f], pred_ann[:, f])
                if not np.isnan(r):
                    feature_corrs[f] = r
        valid = ~np.isnan(feature_corrs)
        mean_feature_corr = np.mean(feature_corrs[valid]) if valid.any() else 0.0

        image_corrs = []
        for i in range(n_samples):
            if np.std(ann_gt[i]) > 0 and np.std(pred_ann[i]) > 0:
                r, _ = pearsonr(ann_gt[i], pred_ann[i])
                if not np.isnan(r):
                    image_corrs.append(r)
        mean_image_corr = np.mean(image_corrs) if image_corrs else 0.0

        ss_res = np.sum((ann_gt - pred_ann) ** 2)
        ss_tot = np.sum((ann_gt - ann_gt.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        per_subject[subj] = {
            'feature_correlation': mean_feature_corr,
            'image_correlation': mean_image_corr,
            'r2': r2,
        }
        print(f"  Subject {subj}: per-feature={mean_feature_corr:.4f}, per-image={mean_image_corr:.4f}, R²={r2:.4f}")
        all_feature_corrs.append(mean_feature_corr)
        all_image_corrs.append(mean_image_corr)

    return {
        'fmri2ann_per_subject': per_subject,
        'fmri2ann_avg_feature_corr': np.mean(all_feature_corrs),
        'fmri2ann_avg_image_corr': np.mean(all_image_corrs),
    }


# ==============================================================================
# CSV Export
# ==============================================================================

def save_results_to_csv(results, method, n_dims, mode, dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{method}_{n_dims}d_{mode}_{dataset}"

    pred_csv = os.path.join(output_dir, f"cross_subj_prediction_{base_name}.csv")
    with open(pred_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'n_dims', 'source', 'target', 'r_voxel', 'is_diagonal'])
        pred_matrix = results['fmri_prediction_matrix']
        subjects = results['subjects']
        for i, src in enumerate(subjects):
            for j, tgt in enumerate(subjects):
                writer.writerow([method, n_dims, f"S{src}", f"S{tgt}",
                                 f"{pred_matrix[i, j]:.6f}", "yes" if i == j else "no"])
    print(f"  Saved: {pred_csv}")

    align_csv = os.path.join(output_dir, f"alignment_metrics_{base_name}.csv")
    with open(align_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'n_dims', 'metric', 'value'])
        for key in ['avg_comp_corr', 'avg_rsa_pearson', 'avg_rsa_euclidean',
                     'fmri_pred_diagonal_mean', 'fmri_pred_offdiag_mean',
                     'recon_avg_voxel_corr', 'recon_avg_r2', 'recon_avg_global_corr',
                     'recon_avg_nc_norm_voxel_corr']:
            if results.get(key) is not None:
                writer.writerow([method, n_dims, key, f"{results[key]:.6f}"])
        # comp_corr matrix diag/offdiag
        for key in ['comp_corr_diagonal_mean', 'comp_corr_offdiag_mean',
                     'pred_diagonal_mean', 'pred_offdiag_mean']:
            if results.get(key) is not None:
                writer.writerow([method, n_dims, key, f"{results[key]:.6f}"])
        if results.get('decoding'):
            writer.writerow([method, n_dims, 'decoding_balanced_accuracy',
                             f"{results['decoding']['balanced_accuracy']:.6f}"])
        if results.get('silhouette'):
            writer.writerow([method, n_dims, 'silhouette_combined',
                             f"{results['silhouette']['silhouette_combined']:.6f}"])
        if results.get('cross_trial'):
            ct = results['cross_trial']
            for k in ['cross_trial_avg_sample_corr', 'cross_trial_avg_direct_corr',
                       'cross_trial_avg_voxel_corr', 'cross_trial_avg_performance_ratio']:
                writer.writerow([method, n_dims, k, f"{ct[k]:.6f}"])
    print(f"  Saved: {align_csv}")

    retrieval_csv = os.path.join(output_dir, f"retrieval_{base_name}.csv")
    with open(retrieval_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'n_dims', 'metric', 'top_k', 'accuracy'])
        if results.get('retrieval'):
            for metric_name in ['euclidean', 'cosine']:
                if metric_name in results['retrieval']:
                    for k, acc in results['retrieval'][metric_name]['mean'].items():
                        writer.writerow([method, n_dims, metric_name, k, f"{acc:.4f}"])
    print(f"  Saved: {retrieval_csv}")


# ==============================================================================
# CLI
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Universal evaluation for cross-subject alignment methods")

    parser.add_argument("--method", type=str, required=True,
                        choices=['vae', 'srm', 'procrustes'],
                        help="Alignment method")
    parser.add_argument("--dataset", type=str, default="streams",
                        choices=['streams'])
    parser.add_argument("--n_dims", type=int, default=32,
                        help="Latent / shared space dimensionality")

    # VAE-specific
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to VAE checkpoint (required for --method vae)")
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="(Deprecated alias for --n_dims for VAE backward compat)")

    # Baseline-specific
    parser.add_argument("--alignment_model_path", type=str, default=None,
                        help="Path to alignment model .pkl file or directory "
                             "(required for --method srm/procrustes)")
    parser.add_argument("--n_components", type=int, default=None,
                        help="(Deprecated alias for --n_dims for baseline backward compat)")

    # Data
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    parser.add_argument("--mode", type=str, default="not_all8",
                        choices=['exclusive', 'not_all8', 'all_common', 'shared_only', 'vae_test'])
    parser.add_argument("--new_test", action="store_true", default=True)
    parser.add_argument('--test_size', type=float, default=0.1)

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: EVALUATION_DIR/results_vae or results)")

    # VAE ANN activations override
    parser.add_argument('--ann_activations', type=str, default=None)
    parser.add_argument('--novel_subject', type=int, default=None)

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--silhouette_reduce', type=lambda s: str(s).lower() not in ('false', '0', 'no'),
                        default=True)

    args = parser.parse_args()

    # Resolve deprecated aliases
    if args.latent_dim is not None:
        args.n_dims = args.latent_dim
    if args.n_components is not None:
        args.n_dims = args.n_components

    # Auto-detect data path
    if args.data_path is None:
        args.data_path = NSD_DIR + os.sep

    # Auto-detect output dir
    if args.output_dir is None:
        if args.method == 'vae':
            args.output_dir = os.path.join(EVALUATION_DIR, "results_vae")
        else:
            args.output_dir = os.path.join(EVALUATION_DIR, "results")

    # Auto-detect VAE checkpoint
    if args.method == 'vae' and args.vae_checkpoint is None:
        args.vae_checkpoint = os.path.join(
            RESULTS_DIR,
            f"medvae_fmri_nn2nn_fmri_resnet50_hendrycks_fair_nnweight5_{args.n_dims}_b1.0.pt")

    # Baseline model path (SRM/Procrustes) must be supplied explicitly — point to the
    # fitted alignment model .pkl produced by baselines/fit_baselines.py (or fit_*_random.py).
    if args.method in ('srm', 'procrustes') and args.alignment_model_path is None:
        raise SystemExit(
            f"--alignment_model_path is required for --method {args.method}: pass the path to "
            "the fitted SRM/Procrustes model .pkl (see baselines/).")

    return args


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = get_args()

    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)

    print("="*70)
    print("CROSS-SUBJECT ALIGNMENT EVALUATION")
    print("="*70)
    print(f"Method: {args.method.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Dims: {args.n_dims}")
    print(f"Mode: {args.mode}")
    print(f"Subjects: {args.subjects}")
    print(f"Seed: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model and create handler ---
    if args.method == 'vae':
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        print("\n" + "="*60)
        print("LOADING VAE MODEL")
        print("="*60)
        model, config = load_vae_model(args.vae_checkpoint, device)
        handler = VAEHandler(model, config, device)
    else:
        device = None
        model = None
        config = None
        print("\n" + "="*60)
        print(f"LOADING {args.method.upper()} ALIGNMENT MODEL")
        print("="*60)
        model_dict = load_alignment_model(args.method, args.alignment_model_path, args.n_dims)
        transformer = AlignmentTransformer(model_dict, args.method)
        handler = BaselineHandler(transformer)

    # --- Load test data ---
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)

    if args.mode == 'vae_test':
        remove_overlaps = False
        filter_fmri = False
        if config is not None:
            remove_overlaps = config.get('remove_all_overlaps', False)
            filter_fmri = config.get('filter_no_fmri', False)
        if 'rvoverl_all' in (args.vae_checkpoint or ''):
            remove_overlaps = True
        if 'filtfmri' in (args.vae_checkpoint or '') or 'filter_no_fmri' in (args.vae_checkpoint or ''):
            filter_fmri = True
        common_fmri = load_vae_test_split(
            args.data_path, args.subjects,
            remove_all_overlaps=remove_overlaps, filter_no_fmri=filter_fmri
        )
        sorted_images = list(range(len(next(iter(common_fmri.values())))))
        target_data = None
    else:
        selected_images, target_data = get_shared_images(
            args.data_path, target_subjects=args.subjects,
            new_test=args.new_test, mode=args.mode,
            dataset=args.dataset, test_size=args.test_size
        )
        print("\n" + "="*60)
        print("EXTRACTING COMMON fMRI DATA")
        print("="*60)
        common_fmri, sorted_images = extract_common_fmri(target_data, selected_images, args.subjects)

    # --- Load noise ceilings ---
    print("\n" + "="*60)
    print("LOADING NOISE CEILINGS")
    print("="*60)
    nc_dict = load_noise_ceilings(args.subjects)
    if nc_dict is None:
        print("  NC normalisation will be skipped (missing files)")

    # --- Evaluate ---
    if args.mode == 'vae_test':
        print("\n[vae_test mode] Per-subject metrics only (no pairwise)")
        recon_metrics = compute_reconstruction_quality(handler, common_fmri, args.subjects, nc_dict=nc_dict)
        alignment_metrics = None
        fmri_pred_metrics = None
        retrieval_results = None
        latent_space_list = None
    else:
        # Encode to latent / shared space
        print("\n" + "="*60)
        print("ENCODING TO LATENT/SHARED SPACE")
        print("="*60)
        latent_space_list = []
        for subj in args.subjects:
            subj_idx = SUBJ_TO_IDX[subj]
            fmri = common_fmri[subj]
            latent = handler.encode_to_latent(fmri, subj_idx)
            latent_space_list.append(latent)
            print(f"  Subject {subj}: {fmri.shape} → {latent.shape}")

        alignment_metrics = compute_alignment_metrics(
            latent_space_list, args.subjects, novel_subject=args.novel_subject)

        fmri_pred_metrics = compute_cross_subject_fmri_prediction(
            handler, common_fmri, args.subjects,
            novel_subject=args.novel_subject, nc_dict=nc_dict)

        retrieval_results = compute_cross_subject_retrieval(
            latent_space_list, args.subjects, novel_subject=args.novel_subject)

        recon_metrics = compute_reconstruction_quality(handler, common_fmri, args.subjects, nc_dict=nc_dict)

    # --- ANN metrics (VAE-only) ---
    ann2fmri_metrics = None
    ann2ann_metrics = None
    fmri2ann_metrics = None

    if args.mode != 'vae_test' and handler.has_ann_metrics:
        nn_input_dim = config.get('input_dim', [])[-1] if config.get('input_dim') else 0
        vae_name = os.path.basename(args.vae_checkpoint).lower()
        if args.ann_activations is not None:
            ann_act_path = ANN_ACTIVATIONS.get(args.ann_activations, args.ann_activations)
        elif 'vitb16' in vae_name or 'vit' in vae_name:
            ann_act_path = ANN_ACTIVATIONS_PATH_DINO_VITB16
        elif 'dino' in vae_name:
            ann_act_path = ANN_ACTIVATIONS_PATH_DINO_RN50
        elif 'avgpool' in vae_name and nn_input_dim == 2048:
            ann_act_path = ANN_ACTIVATIONS["rn50_avgpool_2048"]
        elif 'mnist' in vae_name and 'rn50' in vae_name and nn_input_dim == 2048:
            ann_act_path = ANN_ACTIVATIONS["mnist_rn50_2048"]
        elif 'mnist' in vae_name and nn_input_dim == 784:
            ann_act_path = ANN_ACTIVATIONS["mnist_784"]
        elif nn_input_dim == 51060 and 'untrained' in vae_name:
            ann_act_path = ANN_ACTIVATIONS["rn50_untrained_51060"]
        elif nn_input_dim == 51060:
            ann_act_path = ANN_ACTIVATIONS_PATH_STREAMS
        elif nn_input_dim == 26368:
            ann_act_path = ANN_ACTIVATIONS_PATH_CLIP
        elif nn_input_dim == 49000 or nn_input_dim == 74524:
            ann_act_path = ANN_ACTIVATIONS_PATH_CLIPRN50
        elif 'rn50random' in vae_name and nn_input_dim == 8000:
            ann_act_path = ANN_ACTIVATIONS["rn50_random_8000"]
        elif 'untrained' in vae_name and nn_input_dim == 8000:
            ann_act_path = ANN_ACTIVATIONS["rn50_untrained_8000"]
        else:
            ann_act_path = ANN_ACTIVATIONS_PATH_MINDEYE
        print(f"  ANN activations: {ann_act_path} (nn_input_dim={nn_input_dim})")

        ann2fmri_metrics = compute_ann_to_fmri_prediction(
            model, common_fmri, sorted_images, args.subjects, device,
            ann_activations_path=ann_act_path, nc_dict=nc_dict,
            fmri_latent_list=latent_space_list, nn_output_dim=nn_input_dim
        )
        ann2ann_metrics = compute_ann_to_ann_reconstruction(
            model, sorted_images, device,
            ann_activations_path=ann_act_path, nn_output_dim=nn_input_dim
        )
        fmri2ann_metrics = compute_fmri_to_ann_reconstruction(
            model, common_fmri, sorted_images, args.subjects, device,
            ann_activations_path=ann_act_path, nn_output_dim=nn_input_dim
        )

    # --- Cross-trial correlation ---
    cross_trial_metrics = None
    if args.mode != 'vae_test':
        print("\n" + "="*60)
        print("LOADING CROSS-TRIAL DATA")
        print("="*60)
        cross_trial_data = load_cross_trial_data_streams(args.subjects, mode=args.mode)
        if cross_trial_data is not None:
            cross_trial_metrics = compute_cross_trial_correlations(handler, cross_trial_data, args.subjects)
        else:
            print("  Skipping cross-trial correlation (no data found)")

    # --- Decoding & silhouette ---
    decoding_results = None
    silhouette_results = None
    if args.mode == 'vae_test':
        print("\n  [vae_test mode] Skipping decoding and silhouette")
    elif HAVE_DECODING_SILHOUETTE and latent_space_list is not None:
        print("\n" + "="*60)
        print("LOADING LABELS")
        print("="*60)
        all_labels = np.load(LABELS_PATH)
        test_labels = all_labels[sorted_images]
        print(f"  Labels shape: {test_labels.shape}")

        method_tag = f"{args.method}_{args.n_dims}d_{args.mode}_{args.dataset}"
        decoding_results = test_multilabel_decoding_balanced(
            latent_space_list, test_labels, method_name=method_tag
        )

        print("\n" + "="*60)
        print("SILHOUETTE SCORES")
        print("="*60)
        # Always dump the per-subject latents + labels behind the silhouette, so the
        # combined-silhouette bootstrap (visualisation/figure2_stats.py) is fully
        # reproducible from the released eval outputs. Set DUMP_SILH_LATENTS to override
        # the destination; otherwise it lands next to the results pkl.
        import pickle as _pk
        _silh_lat_path = os.environ.get('DUMP_SILH_LATENTS') or os.path.join(
            args.output_dir,
            f"silh_latents_{args.method}_{args.n_dims}d_{args.mode}_{args.dataset}.pkl")
        _pk.dump({'latents': latent_space_list, 'labels': test_labels},
                 open(_silh_lat_path, 'wb'))
        print(f"  Dumped silhouette latents -> {_silh_lat_path}")
        try:
            silhouette_combined, silhouette_per_subject = calculate_silhouette_scores(
                latent_space_list, test_labels, reduce=args.silhouette_reduce
            )
            silhouette_results = {
                'silhouette_combined': silhouette_combined,
                'silhouette_per_subject': silhouette_per_subject
            }
            print(f"  Combined silhouette: {silhouette_combined:.4f}")
            print(f"  Per-subject: {[f'{s:.4f}' for s in silhouette_per_subject]}")
        except Exception as e:
            print(f"  Warning: Silhouette computation failed: {e}")
    else:
        print("\nSkipping decoding and silhouette (import failed or no latent space)")

    # --- Combine results ---
    results = {
        'method': args.method,
        'dataset': args.dataset,
        'n_dims': args.n_dims,
        'n_components': args.n_dims,
        'latent_dim': args.n_dims,
        'mode': args.mode,
        'subjects': args.subjects,
        'n_images': len(sorted_images) if sorted_images is not None else sum(v.shape[0] for v in common_fmri.values()) // len(common_fmri),
        **(alignment_metrics or {}),
        **(fmri_pred_metrics or {}),
        **recon_metrics,
        **(ann2fmri_metrics or {}),
        **(ann2ann_metrics or {}),
        **(fmri2ann_metrics or {}),
        'retrieval': retrieval_results,
        'decoding': decoding_results,
        'silhouette': silhouette_results,
        'cross_trial': cross_trial_metrics
    }

    # Backward compatibility: add old key aliases
    if 'component_corr_matrix' in results:
        results['prediction_matrix'] = results['component_corr_matrix']
    if 'comp_corr_diagonal_mean' in results:
        results['pred_diagonal_mean'] = results['comp_corr_diagonal_mean']
    if 'comp_corr_offdiag_mean' in results:
        results['pred_offdiag_mean'] = results['comp_corr_offdiag_mean']

    if args.method == 'vae':
        results['vae_checkpoint'] = args.vae_checkpoint
        results['vae_name'] = os.path.basename(args.vae_checkpoint).replace('.pt', '')

    # --- Save ---
    output_file = os.path.join(
        args.output_dir,
        f"alignment_eval_{args.method}_{args.n_dims}d_{args.mode}_{args.dataset}.pkl"
    )
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to: {output_file}")

    csv_output_dir = os.path.join(args.output_dir, "results")
    save_results_to_csv(results, args.method, args.n_dims, args.mode, args.dataset, csv_output_dir)

    # --- Summary ---
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Method: {args.method.upper()}-{args.n_dims}D")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Images: {results['n_images']}")

    if alignment_metrics:
        print(f"\nAlignment Metrics:")
        print(f"  Component Correlation: {results['avg_comp_corr']:.4f}")
        print(f"  RSA (Pearson): {results['avg_rsa_pearson']:.4f}")
        print(f"  RSA (Euclidean): {results['avg_rsa_euclidean']:.4f}")

    if fmri_pred_metrics:
        print(f"\nCross-Subject fMRI Prediction:")
        print(f"  Diagonal: {results['fmri_pred_diagonal_mean']:.4f}")
        print(f"  Off-diagonal: {results['fmri_pred_offdiag_mean']:.4f}")
        if results.get('fmri_pred_nc_norm_diagonal_mean') is not None:
            print(f"  NC-norm diagonal: {results['fmri_pred_nc_norm_diagonal_mean']:.2f}%")
            print(f"  NC-norm off-diag: {results['fmri_pred_nc_norm_offdiag_mean']:.2f}%")

    if retrieval_results:
        print(f"\nRetrieval (Euclidean):")
        for k in [1, 5, 10]:
            if k in retrieval_results['euclidean']['mean']:
                print(f"  Top-{k}: {retrieval_results['euclidean']['mean'][k]:.2f}%")

    print(f"\nReconstruction:")
    print(f"  Voxel correlation: {results['recon_avg_voxel_corr']:.4f}")
    print(f"  R²: {results['recon_avg_r2']:.4f}")

    if results.get('cross_trial'):
        ct = results['cross_trial']
        print(f"\nCross-Trial:")
        print(f"  Sample corr: {ct['cross_trial_avg_sample_corr']:.4f}")
        print(f"  Ceiling: {ct['cross_trial_avg_direct_corr']:.4f}")
        print(f"  Ratio: {ct['cross_trial_avg_performance_ratio']:.2%}")

    if results.get('decoding'):
        print(f"\nDecoding: balanced_acc={results['decoding']['balanced_accuracy']:.4f}")
    if results.get('silhouette'):
        print(f"Silhouette: {results['silhouette']['silhouette_combined']:.4f}")


if __name__ == "__main__":
    main()
