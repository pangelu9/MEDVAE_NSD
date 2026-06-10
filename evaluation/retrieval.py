"""Shared cross-subject retrieval evaluation.

One implementation used by BOTH the MED-VAE eval
(`evaluate_vae.py`) and the SRM/Procrustes baseline
(`baseline_retrieval.py`), so the two are measured the same way.

Protocol — fixed gallery size (default 128) so retrieval difficulty is
comparable regardless of how many shared images a split happens to have:

  * n_total <= gallery_size  (e.g. the 128-image pairwise/exclusive split):
        take ALL images, a single deterministic pass (n_reps forced to 1).
  * n_total >  gallery_size  (e.g. the 872-image all-common split):
        subsample `gallery_size` images per repetition, `n_reps` repetitions,
        and average — so the gallery is the same size (128) as the small split.

A query embedding is "retrieved correctly" if its paired image (same index) is
within the top-k nearest gallery embeddings. With a 128-image gallery, random
chance is 100/128 ≈ 0.78% for top-1.

`seed` fixes the subsampled image sets, so the MED-VAE and baseline runs draw
the SAME 128-image subsets and are directly comparable.
"""
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist

DEFAULT_GALLERY_SIZE = 128
DEFAULT_N_REPS = 30


def compute_retrieval_accuracy(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    top_k: List[int] = [1, 2, 3, 5],
    metric: str = 'euclidean',
) -> Dict[int, float]:
    """
    Top-k retrieval accuracy. For each query embedding, find the k nearest
    neighbours in the gallery; query and gallery are paired (same images, same
    order), so the correct match is at the same index (the diagonal).

    Parameters
    ----------
    query_embeddings : (n_samples, n_features)
    gallery_embeddings : (n_samples, n_features)   (same images, same order)
    top_k : list of int
    metric : 'euclidean' | 'cosine' | 'correlation' | ...

    Returns
    -------
    accuracies : dict {k: accuracy in percent}
    """
    n_samples = query_embeddings.shape[0]

    if metric == 'cosine':
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
        distances = 1 - query_norm @ gallery_norm.T
    else:
        distances = cdist(query_embeddings, gallery_embeddings, metric=metric)

    sorted_indices = np.argsort(distances, axis=1)

    accuracies = {}
    for k in top_k:
        top_k_matches = sorted_indices[:, :k]
        correct = np.array([i in top_k_matches[i] for i in range(n_samples)])
        accuracies[k] = correct.mean() * 100  # percentage

    return accuracies


def run_retrieval_with_repetitions(
    latent_space_list: List[np.ndarray],
    subject_labels: Optional[List[int]] = None,
    gallery_size: int = DEFAULT_GALLERY_SIZE,
    n_reps: int = DEFAULT_N_REPS,
    top_k: List[int] = [1, 2, 3, 5],
    metric: str = 'euclidean',
    seed: int = 42,
) -> Dict:
    """
    Cross-subject retrieval with the fixed-gallery protocol (see module docstring).

    Parameters
    ----------
    latent_space_list : list of (n_total, n_features) arrays, one per subject
        Aligned representations (MED-VAE latents or SRM/Procrustes-aligned data).
    subject_labels : list, optional
        Label for each subject (defaults to 0..n_subjects-1). Only used to key
        the ``per_pair`` view; the index-keyed views always use 0..n-1.
    gallery_size : int
        Target gallery size. n_total is capped to this per repetition.
    n_reps : int
        Repetitions when subsampling (ignored — forced to 1 — when n_total fits
        the gallery, since that pass is deterministic).
    top_k, metric, seed : see module docstring.

    Returns
    -------
    results : dict with both index-keyed views (``pairwise`` / ``per_subject`` /
        ``overall``) and label-keyed views (``per_pair`` / ``mean`` / ``std``),
        plus protocol metadata (``n_samples`` = gallery actually used, ``n_reps``).
        For a complete subject graph ``overall[k]['mean'] == mean[k]``.
    """
    top_k = list(top_k)
    n_subjects = len(latent_space_list)
    if subject_labels is None:
        subject_labels = list(range(n_subjects))
    n_total = latent_space_list[0].shape[0]
    for i, data in enumerate(latent_space_list):
        assert data.shape[0] == n_total, \
            f"Subject {i} has {data.shape[0]} samples, expected {n_total}"

    # Fixed-gallery protocol: cap the gallery, and skip repetition when the
    # whole split already fits (that pass is deterministic).
    if n_total <= gallery_size:
        size, reps = n_total, 1
    else:
        size, reps = gallery_size, n_reps

    np.random.seed(seed)

    pair_idx = defaultdict(lambda: defaultdict(list))       # [qi][gj] -> list of acc-dicts (per rep)
    subj_to_others = defaultdict(lambda: defaultdict(list))  # [qi][k]  -> per-rep mean over galleries

    for _rep in range(reps):
        if size < n_total:
            indices = np.random.choice(n_total, size=size, replace=False)
            sampled = [s[indices] for s in latent_space_list]
        else:
            sampled = latent_space_list

        for qi in range(n_subjects):
            gallery_accs = {k: [] for k in top_k}
            for gj in range(n_subjects):
                if qi == gj:
                    continue
                accs = compute_retrieval_accuracy(
                    sampled[qi], sampled[gj], top_k=top_k, metric=metric
                )
                pair_idx[qi][gj].append(accs)
                for k in top_k:
                    gallery_accs[k].append(accs[k])
            for k in top_k:
                subj_to_others[qi][k].append(np.mean(gallery_accs[k]))

    results = {
        'n_samples': size,
        'n_reps': reps,
        'n_subjects': n_subjects,
        'n_total_samples': n_total,
        'gallery_size': size,
        'top_k': top_k,
        'metric': metric,
        'pairwise': {},      # index-keyed: [qi][gj][k] -> {mean,std}
        'per_subject': {},   # index-keyed: [qi][k]     -> {mean,std}
        'overall': {},       # [k] -> {mean,std}  (subject-averaged)
        'per_pair': {},      # label-keyed: (src,tgt)[k] -> {mean,std}
        'mean': {},          # [k] -> float  (pair-averaged; == overall mean for a complete graph)
        'std': {},           # [k] -> float
    }

    # Pairwise (index) + per_pair (label)
    for qi in range(n_subjects):
        results['pairwise'][qi] = {}
        for gj in range(n_subjects):
            if qi == gj:
                continue
            reps_accs = pair_idx[qi][gj]
            stat = {
                k: {'mean': float(np.mean([r[k] for r in reps_accs])),
                    'std': float(np.std([r[k] for r in reps_accs]))}
                for k in top_k
            }
            results['pairwise'][qi][gj] = stat
            results['per_pair'][(subject_labels[qi], subject_labels[gj])] = stat

    # Per-subject (query -> all others)
    for qi in range(n_subjects):
        results['per_subject'][qi] = {
            k: {'mean': float(np.mean(subj_to_others[qi][k])),
                'std': float(np.std(subj_to_others[qi][k]))}
            for k in top_k
        }

    # Overall (subject-averaged)
    all_acc = {k: [] for k in top_k}
    for qi in range(n_subjects):
        for k in top_k:
            all_acc[k].extend(subj_to_others[qi][k])
    results['overall'] = {
        k: {'mean': float(np.mean(all_acc[k])), 'std': float(np.std(all_acc[k]))}
        for k in top_k
    }

    # Pair-averaged mean/std (VAE-style view)
    for k in top_k:
        pair_means = [results['per_pair'][p][k]['mean'] for p in results['per_pair']]
        results['mean'][k] = float(np.mean(pair_means))
        results['std'][k] = float(np.std(pair_means))

    return results
