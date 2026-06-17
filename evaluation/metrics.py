import numpy as np
import torch
import os 
import sys 
from sklearn.decomposition import PCA
try:
    from umap import UMAP
except ImportError:
    UMAP = None
import pathlib, pickle
import time
from typing import List, Optional, Dict, Any

from tqdm import tqdm

from scipy.spatial.distance import pdist

# --- MEDVAE: put all package dirs (medvae/, baselines/, evaluation/) on
# sys.path so cross-package bare imports resolve regardless of caller ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from ccn_config import setup_paths
setup_paths()

from sklearn.metrics import balanced_accuracy_score, hamming_loss
from scipy.stats import spearmanr, pearsonr
from savemetrics import save_alignment_metrics

os.environ['OMP_NUM_THREADS'] = '1'     # force single-thread
os.environ['PYTHONHASHSEED'] = '0'

def test_multilabel_decoding_balanced(aligned_data, labels, method_name,
                                      subset_tag=None,
                                      max_imbalance=None, verbose=True,
                                      output_dir="results/other"): 
    """
    Multi-label LOSO decoding with imbalance handling & within/cross scores.
    
    Parameters
    ----------
    aligned_data : list of (n_samples, n_components) arrays
        Aligned brain data for each subject
    labels : (n_samples, n_labels) array or list of such arrays
        Multi-label targets (binary)
    method_name : str
        Method name for printing
    max_imbalance : float or None
        Maximum allowed class imbalance ratio (e.g., 10.0 means max 10:1).
        If None, keeps all labels regardless of imbalance.
    verbose : bool
        Print per-fold details
        
    Returns
    -------
    dict with keys:
        - exact_match_accuracy : float (cross-subject)
        - balanced_accuracy : float (cross-subject, per-label averaged)
        - within_exact, within_balanced : training set scores
        - per_subject_exact, per_subject_balanced : per-fold results
        - n_valid_labels_per_fold : how many labels survived filtering
    """
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    import numpy as np
    
    # Handle default
    if max_imbalance is None:
        max_imbalance = float('inf')
    
    # Replicate labels if needed
    n_subjects = len(aligned_data)
    if isinstance(labels, np.ndarray):
        labels = [labels for _ in range(n_subjects)]
    else:
        print("Labels is already a list of np arrays, one for each subject")
    
    # Metrics helper - FIXED VERSION
    def _metrics(y_true, y_pred):
        """Compute exact match and per-label averaged balanced accuracy."""
        exact = np.mean(np.all(y_pred == y_true, axis=1))
        
        # Per-label balanced accuracy (better than global ravel)
        bal_per_label = []
        for lab in range(y_true.shape[1]):
            try:
                bal_lab = balanced_accuracy_score(y_true[:, lab], y_pred[:, lab])
                bal_per_label.append(bal_lab)
            except ValueError:  # Handle constant predictions
                bal_per_label.append(0.5)  # Chance level
        
        bal_avg = np.mean(bal_per_label)
        hm = hamming_loss(y_true, y_pred)
        return exact, bal_avg, hm 
    
    # Storage
    exact_cross, exact_within = [], []
    bal_cross, bal_within = [], []
    hamming_within, hamming_cross = [], []
    n_valid_labels = []
    
    print(f"\n{'='*70}")
    print(f"MULTI-LABEL DECODING (balanced, LOSO): {method_name}")
    print(f"{'='*70}")
    
    # LOSO loop
    for test_subj in range(n_subjects):
        train_idx = [i for i in range(n_subjects) if i != test_subj]
        
        # Stack data
        X_train = np.vstack([aligned_data[i] for i in train_idx])
        y_train = np.vstack([labels[i] for i in train_idx])
        X_test = aligned_data[test_subj]
        y_test = labels[test_subj]
        
        # Filter labels by imbalance (on TRAIN set)
        n_labels = y_train.shape[1]
        keep_lab = []
        
        for lab in range(n_labels):
            cls, cnt = np.unique(y_train[:, lab], return_counts=True)
            if len(cls) < 2:  # Constant label
                continue
            ratio = cnt.max() / cnt.min()
            if ratio <= max_imbalance:
                keep_lab.append(lab)
        
        n_valid_labels.append(len(keep_lab))
        
        # Handle no valid labels
        if not keep_lab:
            if verbose:
                print(f"  Fold {test_subj+1}: No valid labels (all too imbalanced)")
            exact_cross.append(np.nan)
            exact_within.append(np.nan)
            bal_cross.append(np.nan)
            bal_within.append(np.nan)
            continue
        
        # Subset to valid labels
        y_tr = y_train[:, keep_lab]
        y_te = y_test[:, keep_lab]
        
        # Train classifier with balanced class weights
        clf = MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=42,
                             class_weight='balanced', solver='lbfgs')
        )
        clf.fit(X_train, y_tr)
        
        # Predictions
        pred_tr = clf.predict(X_train)
        pred_te = clf.predict(X_test)
        
        # Evaluate
        ex_w, bal_w, hm_w = _metrics(y_tr, pred_tr) 
        ex_c, bal_c, hm_c = _metrics(y_te, pred_te)
        
        exact_within.append(ex_w)
        exact_cross.append(ex_c)
        bal_within.append(bal_w)
        bal_cross.append(bal_c)
        hamming_within.append(hm_w)
        hamming_cross.append(hm_c)
        
        if verbose:
            print(f"  Fold {test_subj+1}: valid_labels={len(keep_lab)}, "
                  f"cross_exact={ex_c:.3f}, cross_bal={bal_c:.3f}")
    
    # Aggregate (using nanmean to handle folds with no valid labels)
    m_ex_w, s_ex_w = np.nanmean(exact_within), np.nanstd(exact_within)
    m_ex_c, s_ex_c = np.nanmean(exact_cross), np.nanstd(exact_cross)
    m_bal_w, s_bal_w = np.nanmean(bal_within), np.nanstd(bal_within)
    m_bal_c, s_bal_c = np.nanmean(bal_cross), np.nanstd(bal_cross)
    m_hm_w, s_hm_w = np.nanmean(hamming_within), np.nanstd(hamming_within)
    m_hm_c, s_hm_c = np.nanmean(hamming_cross),  np.nanstd(hamming_cross)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Within-subject  exact-match:  {m_ex_w:.4f} ± {s_ex_w:.4f}")
    print(f"  Cross-subject   exact-match:  {m_ex_c:.4f} ± {s_ex_c:.4f}")
    print(f"  Within-subject  balanced-acc: {m_bal_w:.4f} ± {s_bal_w:.4f}")
    print(f"  Cross-subject   balanced-acc: {m_bal_c:.4f} ± {s_bal_c:.4f}")
    print(f"  Within-subject  Hamming-loss: {m_hm_w:.4f} ± {s_hm_w:.4f}")
    print(f"  Cross-subject   Hamming-loss: {m_hm_c:.4f} ± {s_hm_c:.4f}") 
    print(f"  Avg valid labels per fold:    {np.mean(n_valid_labels):.1f} / {n_labels}")
    
    # Chance level
    print(f"  Chance (balanced-acc):        0.5000")
    print(f"  Chance (exact-match):         {0.5**n_labels:.6f}")
    print(f"{'='*70}")

    save_within_metrics(exact_within, bal_within, tag=method_name, out_dir=output_dir,subset_tag=subset_tag)

    
    return {
        'exact_match_accuracy': m_ex_c,
        'balanced_accuracy': m_bal_c, 
        'exact_match_std': s_ex_c,
        'balanced_accuracy_std': s_bal_c,
        'per_subject_exact': exact_cross,
        'per_subject_balanced': bal_cross,
        'within_exact': exact_within,
        'within_balanced': bal_within,
        'n_valid_labels_per_fold': n_valid_labels,
        # For compatibility with old code
        'f1_micro': m_bal_c,  # Use balanced acc as proxy
        'hamming_loss_cross': m_hm_c,        
        'hamming_loss_within': m_hm_w,  
        'per_subject_hamming_cross': hamming_cross,
        'per_subject_hamming_within': hamming_within
    }


def test_multilabel_decoding_single_subject_generalization(
        aligned_data, labels, method_name,
        max_imbalance=None, verbose=True,
        output_dir=None
    ):
    """
    Multi-label decoding: train on a SINGLE subject, test on ALL OTHERS.
    Repeat for every subject and average the cross-subject scores.
    
    Parameters & return dict are identical to the original LOSO function.
    """
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, hamming_loss
    import numpy as np

    # ---------- setup ----------
    if max_imbalance is None:
        max_imbalance = float('inf')

    n_subjects = len(aligned_data)
    if isinstance(labels, np.ndarray):          # replicate if single array
        labels = [labels for _ in range(n_subjects)]

    # helper metrics ---------------------------------------------------------
    def _metrics(y_true, y_pred):
        exact = np.mean(np.all(y_pred == y_true, axis=1))
        bal_per_label = []
        for lab in range(y_true.shape[1]):
            try:
                bal_per_label.append(
                    balanced_accuracy_score(y_true[:, lab], y_pred[:, lab]))
            except ValueError:                  # constant prediction
                bal_per_label.append(0.5)
        bal_avg = np.mean(bal_per_label)
        hm = hamming_loss(y_true, y_pred)
        return exact, bal_avg, hm

    # storage ----------------------------------------------------------------
    exact_cross, exact_within = [], []
    bal_cross, bal_within     = [], []
    hamming_cross, hamming_within = [], []
    n_valid_labels = []

    print(f"\n{'='*70}")
    print(f"MULTI-LABEL DECODING (single-subject → others): {method_name}")
    print(f"{'='*70}")

    # ===== main loop: each subject acts as the training set once ============
    for train_subj in range(n_subjects):
        # ---- define train / test split ----
        X_train = aligned_data[train_subj]
        y_train = labels[train_subj]

        X_test = np.vstack([aligned_data[i] for i in range(n_subjects)
                            if i != train_subj])
        y_test = np.vstack([labels[i] for i in range(n_subjects)
                            if i != train_subj])

        # ---- label filtering by imbalance (on the SINGLE training subject) ----
        n_labels = y_train.shape[1]
        keep_lab = []
        for lab in range(n_labels):
            cls, cnt = np.unique(y_train[:, lab], return_counts=True)
            if len(cls) < 2:
                continue
            ratio = cnt.max() / cnt.min()
            if ratio <= max_imbalance:
                keep_lab.append(lab)

        n_valid_labels.append(len(keep_lab))

        if not keep_lab:                # no label survived
            if verbose:
                print(f"  Train-subj {train_subj+1}: No valid labels")
            for container in (exact_cross, exact_within,
                              bal_cross, bal_within,
                              hamming_cross, hamming_within):
                container.append(np.nan)
            continue

        y_tr = y_train[:, keep_lab]
        y_te = y_test[:, keep_lab]

        # ---- train ----
        clf = MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=42,
                             class_weight='balanced', solver='lbfgs'))
        clf.fit(X_train, y_tr)

        # ---- evaluate ----
        pred_tr = clf.predict(X_train)
        pred_te = clf.predict(X_test)

        ex_w, bal_w, hm_w = _metrics(y_tr, pred_tr)   # training re-substitution
        ex_c, bal_c, hm_c = _metrics(y_te, pred_te)   # generalisation to others

        exact_within.append(ex_w);  exact_cross.append(ex_c)
        bal_within.append(bal_w);   bal_cross.append(bal_c)
        hamming_within.append(hm_w); hamming_cross.append(hm_c)

        if verbose:
            print(f"  Train-subj {train_subj+1}: valid_labels={len(keep_lab)}, "
                  f"cross_exact={ex_c:.3f}, cross_bal={bal_c:.3f}")

    # ---------- aggregate ----------
    def agg(x):
        return np.nanmean(x), np.nanstd(x)

    m_ex_w, s_ex_w = agg(exact_within);  m_ex_c, s_ex_c = agg(exact_cross)
    m_bal_w, s_bal_w = agg(bal_within);  m_bal_c, s_bal_c = agg(bal_cross)
    m_hm_w, s_hm_w = agg(hamming_within); m_hm_c, s_hm_c = agg(hamming_cross)

    print(f"\n{'='*70}")
    print(f"SUMMARY (single-subject → others)")
    print(f"{'='*70}")
    print(f"  Within-subject (re-sub)  exact-match:  {m_ex_w:.4f} ± {s_ex_w:.4f}")
    print(f"  Cross-subject            exact-match:  {m_ex_c:.4f} ± {s_ex_c:.4f}")
    print(f"  Within-subject (re-sub)  balanced-acc: {m_bal_w:.4f} ± {s_bal_w:.4f}")
    print(f"  Cross-subject            balanced-acc: {m_bal_c:.4f} ± {s_bal_c:.4f}")
    print(f"  Within-subject (re-sub)  Hamming-loss: {m_hm_w:.4f} ± {s_hm_w:.4f}")
    print(f"  Cross-subject            Hamming-loss: {m_hm_c:.4f} ± {s_hm_c:.4f}")
    print(f"  Avg valid labels per fold: {np.mean(n_valid_labels):.1f} / {n_labels}")
    print(f"  Chance (balanced-acc):        0.5000")
    print(f"  Chance (exact-match):         {0.5 ** n_labels:.6f}")
    print(f"{'='*70}")

    return {
        'exact_match_accuracy': m_ex_c,
        'balanced_accuracy': m_bal_c,
        'exact_match_std': s_ex_c,
        'balanced_accuracy_std': s_bal_c,
        'per_subject_exact': exact_cross,
        'per_subject_balanced': bal_cross,
        'within_exact': exact_within,
        'within_balanced': bal_within,
        'n_valid_labels_per_fold': n_valid_labels,
        'f1_micro': m_bal_c,
        'hamming_loss_cross': m_hm_c,
        'hamming_loss_within': m_hm_w,
        'per_subject_hamming_cross': hamming_cross,
        'per_subject_hamming_within': hamming_within
    }

def calculate_reconstruction_quality(X_orig, X_recon, i):
    # ---- 1. voxel-wise metrics (identical to VAE helper) ----
    m = compute_fmri_recon_metrics(X_orig, X_recon, voxel_thresh=0.1)
    
    # ---- 2. global correlation (flatten everything) ----
    global_corr = np.corrcoef(X_orig.flatten(), X_recon.flatten())[0, 1]

    # ---- console report ----
    print(f"  Subject {i+1}:")
    print(f"    VOXEL-WISE mean correlation [equivalent to VAE's]: {m['voxel_correlation_mean']:.4f}")
    print(f"    SAMPLE-WISE mean correlation [equivalent to VAE's]: {m['sample_correlation_mean']:.4f}")
    print(f"    Global correlation:    {global_corr:.4f}")
    print(f"    R² (variance expl.):   {m['r2']:.4f}")

    


    return m, global_corr

def evaluate_reconstruction_quality(aligned_data, W_list, original_data, method_name=None, pca_models=None, data_before_pca=None,
                                     output_dir="results/other"):
    """
    Evaluate reconstruction using the *same* metric helper as the VAE pipeline
    but still print the extra global-correlation line.

    If data_before_pca and pca_models are supplied, metrics are computed
    in the original (pre-PCA) voxel space; otherwise they are computed
    in the PCA space (old behaviour).
    """
    import numpy as np

    n_subjects = len(aligned_data)

    print(f"\n{'='*70}")
    print(f"RECONSTRUCTION QUALITY: {method_name}")
    print(f"{'='*70}")

    subject_correlations = []   # voxel-wise mean per subject
    subject_r2_scores = []
    subject_global_corr = []
    per_voxel_correlations = []  # NEW: Store full per-voxel correlation arrays

    for i in range(n_subjects):
        X_orig = original_data[i]                       # (n_samples, n_voxels)
        X_recon = aligned_data[i] @ W_list[i].T         # (n_samples, n_voxels) 

        if data_before_pca is not None:
            if pca_models is None:
                raise ValueError("pca_models must be provided when data_before_pca is given.")
            
            X_orig = data_before_pca[i] 
            X_recon = pca_models[i].inverse_transform(X_recon)

        m, global_corr = calculate_reconstruction_quality(X_orig, X_recon, i)
        subject_correlations.append(m['voxel_correlation_mean'])
        subject_r2_scores.append(m['r2'])
        subject_global_corr.append(global_corr)
        per_voxel_correlations.append(m['voxel_correlation_list'])  # NEW: Save full list

    avg_corr = np.mean(subject_correlations)
    avg_r2 = np.mean(subject_r2_scores)
    avg_global_corr = np.mean(subject_global_corr)

    print(f"\n  Mean voxel correlation: {avg_corr:.4f}")
    print(f"  Mean global correlation: {avg_global_corr:.4f}")
    print(f"  Mean R²: {avg_r2:.4f}")

    # Updated save function call with per-voxel correlations
    save_subject_metrics(subject_correlations,
                         subject_r2_scores,
                         subject_global_corr,
                         per_voxel_correlations,  # NEW parameter
                         tag=method_name,
                         out_dir=output_dir) 

    return {
        'mean_voxel_correlation': avg_corr,
        'mean_r2': avg_r2,
        'per_subject_correlation': subject_correlations,
        'per_subject_r2': subject_r2_scores,
        'per_voxel_correlations': per_voxel_correlations,  # NEW: Include in return
    }




def _reduce_if_needed(x, n=10, random_state=42, reduction="umap"):
    """UMAP → PCA fallback down to ≤ n components."""
    if reduction == "umap":
        if UMAP is None:
            print("  Warning: UMAP unavailable (numba/numpy conflict), falling back to PCA")
            return PCA(n_components=min(n, x.shape[1]), random_state=random_state).fit_transform(x)
        return UMAP(n_components=n, random_state=random_state).fit_transform(x)
    else:
        return PCA(n_components=min(n, x.shape[1]), random_state=random_state).fit_transform(x)

def calculate_silhouette_scores(latents: np.ndarray, labels: np.ndarray, reduce: bool) -> float:
    """
    Silhouette score for a ready-made latent matrix.

    Parameters
    ----------
    latents : np.ndarray, shape (n_samples, n_features)
    labels  : np.ndarray, shape (n_samples,), dtype int

    Returns
    -------
    silhouette : float
    """
    subjects_silh = []

    n_subjects =  len(latents)
    print("================ SILHOUETTE CALCULATION: each subject separately")

    for i in range(n_subjects):
        print("silhouette calculation for subject: ", i)
        if isinstance(labels, np.ndarray):
            label = labels
        else:
            label = labels[i]

        if reduce:
            latent = _reduce_if_needed(latents[i])
        else:
            latent = latents[i]

        silh_indiv = evaluate_latent_space_hybrid(latent, label)['silhouette']
        subjects_silh.append(silh_indiv)
        print("Silhouette for subject:", i, silh_indiv)
        


    print("================ SILHOUETTE CALCULATION: stacking representations from all subjects")
    
    latents = np.vstack(latents)
    
    if isinstance(labels, np.ndarray):
        labels = np.vstack([labels] * n_subjects)
    else:
        print("Labels is already a list of np arrays, one for each subject")
        labels = np.vstack(labels)
    
    if reduce:
        latents = _reduce_if_needed(latents)
    
    silh_combined = evaluate_latent_space_hybrid(latents, labels)['silhouette']
    
    return silh_combined, subjects_silh


import pathlib
import pickle
import time
from typing import Optional, List

def save_prediction_matrix(prediction_matrix: np.ndarray,
                          per_voxel_corrs: Dict,
                          diagonal_mean: float,
                          offdiagonal_mean: float,
                          space: str = 'pca',
                          tag: Optional[str] = None,
                          out_dir: str = "results/other") -> None:  
    """Pickle the cross-subject prediction matrix and statistics."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    
    stamp = tag if tag is not None else str(int(time.time()))
    
    results = {
        'prediction_matrix': prediction_matrix,
        'per_voxel_correlations': per_voxel_corrs,
        'diagonal_mean': diagonal_mean,
        'offdiagonal_mean': offdiagonal_mean,
        'space': space
    }
    
    with open(out_path / f"cross_subject_prediction_{stamp}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f" Saved prediction matrix ({space} space) to: {out_path} (tag='{stamp}')")



def save_subject_metrics(voxel_corr: List[float],
                         r2_scores: List[float],
                         global_corr: List[float],
                         per_voxel_corr: List[List[float]],  # NEW parameter
                         tag: Optional[str] = None,
                         out_dir: str = "results/other") -> None:
    """
    Pickle the three subject-level metric lists plus per-voxel correlations.

    Parameters
    ----------
    voxel_corr      : list of voxel-correlation means per subject
    r2_scores       : list of R² scores per subject
    global_corr     : list of global correlations per subject
    per_voxel_corr  : list of arrays, each containing per-voxel correlations for one subject
    tag             : optional string appended to file names (defaults to timestamp)
    out_dir         : directory in which to save the files (created if missing)
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    stamp = tag if tag is not None else str(int(time.time()))

    with open(out_path / f"voxel_corr_mean_{stamp}.pkl", "wb") as f:
        pickle.dump(voxel_corr, f)

    with open(out_path / f"r2_scores_{stamp}.pkl", "wb") as f:
        pickle.dump(r2_scores, f)

    with open(out_path / f"global_corr_{stamp}.pkl", "wb") as f:
        pickle.dump(global_corr, f)
    
    # NEW: Save per-voxel correlations
    with open(out_path / f"per_voxel_corr_{stamp}.pkl", "wb") as f:
        pickle.dump(per_voxel_corr, f)
    
    print(f" Saved reconstruction metrics to: {out_path} (tag='{stamp}')")
    print(f"   Including per-voxel correlations for {len(per_voxel_corr)} subjects")
    for i, voxel_list in enumerate(per_voxel_corr):
        print(f"   Subject {i+1}: {len(voxel_list)} voxels")


def save_within_metrics(within_exact: List[float],
                        within_balanced: List[float],
                        tag: Optional[str] = None,
                        subset_tag: Optional[str] = None,
                        out_dir: str = "results/other") -> None: 
    """Pickle the within-subject decoding metrics."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    stamp = tag if tag is not None else str(int(time.time()))
    subset_str = f"_{subset_tag}" if subset_tag else ""
    with open(out_path / f"within_exact_{stamp}{subset_str}.pkl", "wb") as f:
        pickle.dump(within_exact, f)

    with open(out_path / f"within_balanced_{stamp}{subset_str}.pkl", "wb") as f:
        pickle.dump(within_balanced, f)

    print(f" Saved within-subject metrics to: {out_path} (tag='{stamp}{subset_str}')")


def save_silhouette_scores(silh_score: float,
                           silh_score_indiv: List[float],
                           tag: Optional[str] = None,
                           out_dir: str = "results/other") -> None:  
    """Pickle the overall and individual silhouette scores."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    stamp = tag if tag is not None else str(int(time.time()))

    with open(out_path / f"silh_score_{stamp}.pkl", "wb") as f:
        pickle.dump(silh_score, f)

    with open(out_path / f"silh_score_indiv_{stamp}.pkl", "wb") as f:
        pickle.dump(silh_score_indiv, f)
    
    print(f" Saved silhouette scores to: {out_path} (tag='{stamp}')")


# ---------- reusable helpers (put these in utils/metrics.py if you want) ----------
def compute_fmri_recon_metrics(orig: np.ndarray,
                               recon: np.ndarray,
                               voxel_thresh: float = 0.1) -> Dict[str, Any]:
    """Compute all metrics exactly like FUNCTION 1 did."""
    if orig.shape != recon.shape:
        raise ValueError("Shape mismatch")
    n_s, n_v = orig.shape

    # 1. sample-wise (image-based)
    sample_corr, sample_mse = [], []
    for s in range(n_s):
        o, r = orig[s], recon[s]
        if np.any(np.isnan(o)) or np.any(np.isnan(r)):
            continue
        if np.std(o) > 0 and np.std(r) > 0:
            c = np.corrcoef(o, r)[0, 1]
            if not np.isnan(c):
                sample_corr.append(float(c))
        sample_mse.append(float(np.mean((o - r) ** 2)))

    # 2. voxel-wise (competition standard)
    voxel_corr = []
    for v in range(n_v):
        o, r = orig[:, v], recon[:, v]
        if np.any(np.isnan(o)) or np.any(np.isnan(r)):
            continue
        if np.std(o) > 0 and np.std(r) > 0:
            c = np.corrcoef(o, r)[0, 1]
            if not np.isnan(c):
                voxel_corr.append(float(c))

    # 3. overall R² (grand-mean centring)
    flat_o, flat_r = orig.flatten(), recon.flatten()
    ss_res = np.sum((flat_o - flat_r) ** 2)
    ss_tot = np.sum((flat_o - np.mean(flat_o)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # 4. approach correlation
    appr_corr = None
    if len(sample_corr) == len(voxel_corr) and len(sample_corr) > 2:
        try:
            appr_corr = float(np.corrcoef(sample_corr, voxel_corr)[0, 1])
        except Exception:
            pass

    return {
        "num_samples": n_s,
        "num_voxels": n_v,
        "sample_correlation_list": sample_corr,
        "sample_correlation_mean": float(np.mean(sample_corr)) if sample_corr else np.nan,
        "sample_correlation_std": float(np.std(sample_corr)) if sample_corr else np.nan,
        "sample_correlation_min": float(np.min(sample_corr)) if sample_corr else np.nan,
        "sample_correlation_max": float(np.max(sample_corr)) if sample_corr else np.nan,
        "mse": float(np.mean(sample_mse)) if sample_mse else np.nan,
        "voxel_correlation_list": voxel_corr,
        "voxel_correlation_mean": float(np.mean(voxel_corr)) if voxel_corr else np.nan,
        "voxel_correlation_std": float(np.std(voxel_corr)) if voxel_corr else np.nan,
        "voxel_correlation_min": float(np.min(voxel_corr)) if voxel_corr else np.nan,
        "voxel_correlation_max": float(np.max(voxel_corr)) if voxel_corr else np.nan,
        "voxel_corr_above_01": float(np.mean(np.array(voxel_corr) > voxel_thresh)),
        "voxel_corr_above_02": float(np.mean(np.array(voxel_corr) > 0.2)),
        "voxel_corr_above_05": float(np.mean(np.array(voxel_corr) > 0.5)),
        "r2": r2,
        "approach_correlation": appr_corr,
    }



def extract_pairwise_common_samples(data_loader, n_subjects=8, min_subjects=2):
    """
    Extract samples seen by AT LEAST min_subjects (but not necessarily all).
    
    Returns samples where 2+ subjects have data, tracking which subjects saw which images.
    This gives you the images that have partial overlap across subjects.
    
    Parameters:
    -----------
    data_loader : DataLoader
    n_subjects : int
        Number of fMRI subjects (default 8, excluding NN)
    min_subjects : int
        Minimum number of subjects that must have seen an image (default 2)
    
    Returns:
    --------
    pairwise_data_dict : dict
        {
            sample_idx: {
                'data': {subj_idx: array, ...},  # Only subjects that saw it
                'labels': array,
                'subject_mask': boolean array (n_subjects,)
            }
        }
    subject_sample_lists : list of lists
        [subj_0_samples, subj_1_samples, ...] where each is list of sample_idx
    """
    print(f"\nExtracting pairwise common samples (≥{min_subjects} subjects per image)...")
    
    all_batches = []
    all_labels = []
    all_masks = []
    
    # First pass: collect all data
    for encoder_inputs, labels, nn_target, masks in tqdm(data_loader, desc="Scanning batches"):
        m = masks.numpy()[:, :n_subjects]  # Only fMRI subjects, exclude NN
        all_batches.append(encoder_inputs)
        all_labels.append(labels.numpy())
        all_masks.append(m)
    
    # Build dictionary mapping each sample to its data and subject availability
    pairwise_data_dict = {}
    sample_idx = 0
    
    for batch_encoder_inputs, batch_labels, batch_masks in zip(all_batches, all_labels, all_masks):
        batch_size = batch_masks.shape[0]
        
        for local_idx in range(batch_size):
            subject_mask = batch_masks[local_idx]  # (n_subjects,)
            n_available = subject_mask.sum()
            
            # Keep only if at least min_subjects have this image
            if n_available >= min_subjects:
                sample_data = {}
                
                # Store data for each subject that has it
                for subj in range(n_subjects):
                    if subject_mask[subj]:
                        sample_data[subj] = batch_encoder_inputs[subj].numpy()[local_idx]
                
                pairwise_data_dict[sample_idx] = {
                    'data': sample_data,
                    'labels': batch_labels[local_idx],
                    'subject_mask': subject_mask,
                    'n_subjects': n_available
                }
            
            sample_idx += 1
    
    # Create subject-wise sample lists
    subject_sample_lists = [[] for _ in range(n_subjects)]
    for sample_idx, sample_info in pairwise_data_dict.items():
        for subj in sample_info['data'].keys():
            subject_sample_lists[subj].append(sample_idx)
    
    # Statistics
    n_samples_by_count = {}
    for sample_info in pairwise_data_dict.values():
        count = sample_info['n_subjects']
        n_samples_by_count[count] = n_samples_by_count.get(count, 0) + 1
    
    print(f"\nExtracted {len(pairwise_data_dict)} samples with ≥{min_subjects} subjects")
    print(f"\nDistribution of samples by number of subjects:")
    for n_subj in sorted(n_samples_by_count.keys()):
        print(f"  {n_subj} subjects: {n_samples_by_count[n_subj]} samples")
    
    print(f"\nSamples per subject:")
    for subj, samples in enumerate(subject_sample_lists):
        print(f"  Subject {subj+1}: {len(samples)} samples")
    
    return pairwise_data_dict, subject_sample_lists


# ----------------------------------------------------------------------------
# Latent-space silhouette helpers (moved here from multiencoder_latent_hybrid_new.py;
# this module was their only consumer). Used by calculate_silhouette_scores above.
# ----------------------------------------------------------------------------
def evaluate_latent_space_hybrid(latent_vectors, labels_matrix):
    """
    Evaluate latent space quality using multiple metrics.
    Adapted from multiencoder_latent.py for hybrid model.
    
    Parameters:
    latent_vectors : array-like of shape (n_samples, n_latent_dims)
        The latent space representations
    labels_matrix : array-like of shape (n_samples, n_classes) or (n_samples,)
        One-hot encoded labels matrix or single labels
        
    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    metrics = {}
    
    # Calculate Silhouette score
    metrics['silhouette'] = calculate_multilabel_silhouette_hybrid(latent_vectors, labels_matrix)

    return metrics


def calculate_multilabel_silhouette_hybrid(latent_vectors, labels_matrix):
    """
    Calculate modified silhouette score for multi-label data with optimized performance.
    Adapted from multiencoder_latent.py for hybrid model.
    
    Parameters:
    latent_vectors : array-like of shape (n_samples, n_latent_dims)
        The latent space representations
    labels_matrix : array-like of shape (n_samples, n_classes) or (n_samples,)
        One-hot encoded labels matrix or single labels
        
    Returns:
    float: Modified silhouette score
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr
    from time import time
    
    # Convert to numpy if they're tensors
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.cpu().numpy()
    if isinstance(labels_matrix, torch.Tensor):
        labels_matrix = labels_matrix.cpu().numpy()
    
    # Convert to float32 for faster computation
    latent_vectors = latent_vectors.astype(np.float32)
    labels_matrix = labels_matrix.astype(np.float32)
    
    n_samples = latent_vectors.shape[0]
    print(f"Computing silhouette for {n_samples} samples...")
    
    # Handle both single labels and multi-hot labels
    if len(labels_matrix.shape) == 1:
        # Single labels - convert to multi-hot
        max_label = int(np.max(labels_matrix))
        multi_hot = np.zeros((len(labels_matrix), max_label + 1))
        for i, label in enumerate(labels_matrix):
            if 0 <= label <= max_label:
                multi_hot[i, int(label)] = 1
        labels_matrix = multi_hot
    
    # Step 1: Pre-compute all pairwise distances
    t0 = time()
    distances = squareform(pdist(latent_vectors, 'euclidean'))
    print(f"Distances computed in {time() - t0:.2f}s")
    
    # Step 2: Pre-compute pairwise label similarities matrix
    t0 = time()
    # Efficiently compute intersection and union matrices
    dot_product = np.dot(labels_matrix, labels_matrix.T)  # intersection counts
    
    # Sum of 1s in each row
    row_sums = np.sum(labels_matrix, axis=1, keepdims=True)
    
    # For each pair (i,j), calculate: sum_i + sum_j - intersection_ij
    union_counts = row_sums + row_sums.T - dot_product
    
    # Calculate Jaccard similarity
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    similarity_matrix = dot_product / (union_counts + epsilon)
    print(f"Similarity matrix computed in {time() - t0:.2f}s")
    
    # Step 3: Calculate silhouette scores vectorized
    t0 = time()
    silhouette_scores = []
    
    # Optional: Use a subset of samples if the full dataset is too large
    max_samples = min(n_samples, 5000)  # Cap to prevent memory issues
    if n_samples > max_samples:
        print(f"Using {max_samples} samples for silhouette calculation")
        # Use a LOCAL seeded generator so the subsample (and hence the combined
        # silhouette for >5000-sample runs) is reproducible, independent of the
        # global RNG state. The original used the unseeded global np.random,
        # which made the combined silhouette vary run-to-run for large N.
        indices = np.random.default_rng(0).choice(n_samples, max_samples, replace=False)
    else:
        indices = np.arange(n_samples)
    
    # Calculate median similarity for each sample (used as threshold).
    # Exclude each sample's self-similarity (the diagonal == 1.0) from its own
    # median: the per-sample a/b loop below excludes self (mask[i]=False), so the
    # threshold it is compared against must exclude self too. Textbook silhouette
    # is defined over *other* samples. (Effect is ~1e-5 at these N, but this is
    # the correct definition.)
    _sim_block = similarity_matrix[np.ix_(indices, indices)].astype(np.float32, copy=True)
    np.fill_diagonal(_sim_block, np.nan)
    median_similarities = np.nanmedian(_sim_block, axis=1)
    
    for i in indices:
        # Get all other samples
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        
        # Get distances and similarities to all other samples
        sample_distances = distances[i, mask]
        sample_similarities = similarity_matrix[i, mask]
        
        # Find similar and dissimilar samples
        similar_mask = sample_similarities > median_similarities[np.where(indices == i)[0][0]]
        dissimilar_mask = ~similar_mask
        
        # Calculate a and b only if we have both similar and dissimilar samples
        if np.any(similar_mask) and np.any(dissimilar_mask):
            a = np.mean(sample_distances[similar_mask])
            b = np.mean(sample_distances[dissimilar_mask])
            
            # Calculate silhouette score for this sample
            s = (b - a) / max(a, b)
            silhouette_scores.append(s)
    
    print(f"Silhouette scores computed in {time() - t0:.2f}s")
    
    return np.mean(silhouette_scores)


# =============================================================================
# RSA helpers (folded in from the former remove_ann_latent.py)
# =============================================================================
def cross_subj_metrics(latents_clean, valid_idx_list, method,
                       subject_labels=None, novel_subject=None, save=True):
    """
    Cross-subject alignment metrics (component correlation + RSA) on the images
    present in *every* subject.

    latents_clean   : list of (n_i, d) arrays
    valid_idx_list  : list of 1-D arrays with the *original* image indices kept
                      for each subject (so the common set can be intersected)
    subject_labels  : optional label per subject; with ``novel_subject`` (a label)
                      every average is restricted to pairs involving that subject
                      (the add-a-subject eval). Default: average over all pairs.
    """
    # 1. find image indices present in *every* subject
    common = valid_idx_list[0]
    for v in valid_idx_list[1:]:
        common = np.intersect1d(common, v)

    # 2. slice each subject to common images
    lat_common = []
    for z, v in zip(latents_clean, valid_idx_list):
        rows = np.searchsorted(v, common)        # v is sorted → indices into z
        lat_common.append(z[rows])
    # lat_common is now list of (n_common, d) arrays

    n_subj, latent_dim = len(lat_common), lat_common[0].shape[1]

    # Which subject pairs to average over (all i<j, or only pairs involving the
    # novel subject). novel_subject=None → all pairs → identical to before.
    if subject_labels is None:
        subject_labels = list(range(n_subj))
    novel_idx = (subject_labels.index(novel_subject)
                 if novel_subject is not None and novel_subject in subject_labels else None)
    pairs = [(i, j) for i in range(n_subj) for j in range(i + 1, n_subj)
             if novel_idx is None or i == novel_idx or j == novel_idx]

    # component-wise correlation: full pairwise matrix per component, avg over pairs
    comp_corr = np.zeros(latent_dim)
    for d in range(latent_dim):
        cmat = np.corrcoef([subj[:, d] for subj in lat_common])
        comp_corr[d] = np.mean([cmat[i, j] for i, j in pairs])

    # RSA - Euclidean (full matrix, avg over pairs)
    rdms = [pdist(z, 'euclidean') for z in lat_common]
    rsa_mat_euclidean = np.eye(n_subj)
    for i in range(n_subj):
        for j in range(i + 1, n_subj):
            r = spearmanr(rdms[i], rdms[j])[0]
            rsa_mat_euclidean[i, j] = rsa_mat_euclidean[j, i] = r
    avg_rsa_euclidean = np.mean([rsa_mat_euclidean[i, j] for i, j in pairs])

    # RSA - Pearson. Headline = RAW (uncentered); centered kept as legacy. For the
    # PCA-aligned baselines centering is a numerical no-op (zero column means).
    rsa_mat_pearson, _ = rsa_latent_pearson(lat_common, center=False)
    mean_rsa_pearson = np.mean([rsa_mat_pearson[i, j] for i, j in pairs])
    rsa_mat_pearson_centered, _ = rsa_latent_pearson(lat_common, center=True)
    mean_rsa_pearson_centered = np.mean([rsa_mat_pearson_centered[i, j] for i, j in pairs])

    print("mean Euclidean RSA:", avg_rsa_euclidean)
    print("mean Pearson-RSA (raw):", mean_rsa_pearson)
    print("mean Pearson-RSA (centered/legacy):", mean_rsa_pearson_centered)
    print("Avg Comp corr", comp_corr.mean())

    # Save to files (keep backward compatibility)
    if save:
        out_dir = pathlib.Path("results")
        out_dir.mkdir(exist_ok=True, parents=True)
        with open(out_dir / f"comp_corr_{method}.pkl", "wb") as fh:
            pickle.dump(comp_corr, fh)
        with open(out_dir / f"rsa_vals_{method}.pkl", "wb") as fh:
            pickle.dump(rsa_mat_pearson, fh)

    return dict(
        avg_comp_corr=comp_corr.mean(),
        avg_rsa=mean_rsa_pearson,  # default to Pearson (raw)
        avg_rsa_pearson=mean_rsa_pearson,                       # raw (headline)
        avg_rsa_pearson_centered=mean_rsa_pearson_centered,     # legacy centered
        avg_rsa_euclidean=avg_rsa_euclidean,
        per_component_corr=comp_corr,
        rsa_matrix_pearson=rsa_mat_pearson,
        rsa_matrix_pearson_centered=rsa_mat_pearson_centered,
        rsa_matrix_euclidean=rsa_mat_euclidean,
        comp_corrs=comp_corr,  # alias for backward compatibility
        n_common=len(common),
    )


def rsa_latent_pearson(latents_common, center=True):
    """
    RSA on latent space using 1−Pearson dissimilarity  +  Pearson RDM comparison
    latents_common : list of (n_images, 32) arrays (ANN-nulled, same images)
    returns (rsa_matrix, mean_rsa)

    center : bool
        Column-center each latent dimension before building the RDM. The
        **reported headline RSA uses center=False** (raw 1−Pearson); center=True
        is the legacy/centered value. For the PCA-aligned baselines the columns
        are already zero-mean, so centering is a numerical no-op there; it only
        changes un-centered (e.g. VAE mu) latents.
    """
    n_subj = len(latents_common)

    # 1. optionally centre each dimension (row = image, col = latent_dim)
    if center:
        latents_common = [z - z.mean(axis=0, keepdims=True) for z in latents_common]

    # 2. build RDM: 1 − Pearson correlation
    rdms = [pdist(z, metric='correlation') for z in latents_common]

    # 3. pairwise Pearson between RDMs
    rsa = np.eye(n_subj)
    for i in range(n_subj):
        for j in range(i+1, n_subj):
            r, _ = pearsonr(rdms[i], rdms[j])
            rsa[i, j] = rsa[j, i] = r

    mean_rsa = rsa[np.triu_indices_from(rsa, k=1)].mean()
    return rsa, mean_rsa
