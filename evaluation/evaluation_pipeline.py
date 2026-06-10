"""Common cross-subject evaluation on aligned latents.

The MED-VAE eval and the SRM/Procrustes baseline build per-subject *aligned
latents* in different ways (VAE encoder vs PCA + learned rotation), but the
downstream metrics that act on those latents are identical. This module is that
shared downstream: alignment quality, latent-space cross-prediction, and
retrieval, plus the optional decoding / silhouette metrics — one implementation
called by both, so the numbers are directly comparable.

The actual metric *functions* live in ``metrics.py`` / ``retrieval.py``; this
module only orchestrates them. (ISC was dropped — it was not reported in the
paper.)
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from ccn_config import setup_paths
setup_paths()

import numpy as np
from scipy.stats import pearsonr

from metrics import (cross_subj_metrics,
                     test_multilabel_decoding_balanced,
                     test_multilabel_decoding_single_subject_generalization,
                     calculate_silhouette_scores,
                     save_silhouette_scores,
                     save_alignment_metrics,
                     evaluate_reconstruction_quality,
                     save_prediction_matrix)
from retrieval import run_retrieval_with_repetitions


def _novel_pairs(subject_labels, novel_subject):
    """Ordered subject-index pairs to average over (all i<j, or — if novel_subject
    is given — only pairs involving it)."""
    n = len(subject_labels)
    novel_idx = (subject_labels.index(novel_subject)
                 if novel_subject is not None and novel_subject in subject_labels else None)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if novel_idx is None or i == novel_idx or j == novel_idx]


def compute_alignment_metrics(latent_space_list, subject_labels, novel_subject=None,
                              method_tag='vae'):
    """
    Alignment metrics on a list of per-subject latents (same images, same order):
    component-wise correlation + RSA (Pearson raw/centered, Euclidean) via the
    shared ``cross_subj_metrics``, plus the latent-space cross-prediction matrix.
    ``novel_subject`` restricts every average to pairs involving that subject.
    """
    n_subj = len(latent_space_list)
    n_components = latent_space_list[0].shape[1]
    n_images = latent_space_list[0].shape[0]

    print(f"\n{'='*60}\nCOMPUTING ALIGNMENT METRICS\n{'='*60}")
    print(f"  Subjects: {subject_labels}  Images: {n_images}  Latent dim: {n_components}")
    if novel_subject is not None:
        print(f"  Novel subject: {novel_subject} (averaging only pairs involving it)")

    # Component correlation + RSA (shared engine; no stray pkls on the VAE path).
    align = cross_subj_metrics(
        latent_space_list, [np.arange(n_images) for _ in latent_space_list],
        method=method_tag, subject_labels=subject_labels, novel_subject=novel_subject,
        save=False,
    )

    # Component correlation as a subject x subject matrix: mean per-component
    # Pearson r between subjects' latents. Symmetric (pearsonr(a,b)==pearsonr(b,a));
    # this is a CORRELATION, not a prediction -- see fmri_prediction_matrix
    # (evaluate_vae.py) for the actual decoded-fMRI cross-prediction.
    pairs = _novel_pairs(subject_labels, novel_subject)
    comp_corr_matrix = np.zeros((n_subj, n_subj))
    for i in range(n_subj):
        for j in range(n_subj):
            corr_vals = []
            for c in range(n_components):
                r, _ = pearsonr(latent_space_list[i][:, c], latent_space_list[j][:, c])
                if not np.isnan(r):
                    corr_vals.append(r)
            comp_corr_matrix[i, j] = np.mean(corr_vals) if corr_vals else 0
    diag_mean = np.diag(comp_corr_matrix).mean()
    offdiag_mean = np.mean([comp_corr_matrix[i, j] for i, j in pairs])

    print(f"  Component correlation: {align['avg_comp_corr']:.4f}")
    print(f"  RSA Pearson (raw): {align['avg_rsa_pearson']:.4f}   "
          f"RSA Euclidean: {align['avg_rsa_euclidean']:.4f}")
    print(f"  Component-corr matrix off-diag: {offdiag_mean:.4f}")

    return {
        'avg_comp_corr': align['avg_comp_corr'],
        'per_component_corr': align['per_component_corr'],
        'avg_rsa_euclidean': align['avg_rsa_euclidean'],
        'rsa_matrix_euclidean': align['rsa_matrix_euclidean'],
        'avg_rsa_pearson': align['avg_rsa_pearson'],                   # raw (headline)
        'rsa_matrix_pearson': align['rsa_matrix_pearson'],
        'avg_rsa_pearson_centered': align['avg_rsa_pearson_centered'],  # legacy
        'rsa_matrix_pearson_centered': align['rsa_matrix_pearson_centered'],
        'component_corr_matrix': comp_corr_matrix,   # renamed from 'prediction_matrix' (it is a correlation, not a prediction)
        'comp_corr_diagonal_mean': diag_mean,        # renamed from 'pred_diagonal_mean'
        'comp_corr_offdiag_mean': offdiag_mean,      # renamed from 'pred_offdiag_mean'
        'n_images': n_images,
        'n_components': n_components,
        'subjects': subject_labels,
    }


def compute_cross_subject_retrieval(latent_space_list, subject_labels,
                                    metrics=['euclidean', 'cosine'], novel_subject=None):
    """
    Cross-subject retrieval in the latent space, per distance metric, using the
    shared fixed-128-gallery protocol (``retrieval.run_retrieval_with_repetitions``).
    With ``novel_subject`` the overall mean is recomputed over only its pairs.
    """
    print(f"\n{'='*60}\nCROSS-SUBJECT RETRIEVAL EVALUATION\n{'='*60}")
    top_k = [1, 2, 3, 5, 10]
    all_results = {}

    for metric in metrics:
        print(f"\n--- Metric: {metric} ---")
        # Fixed 128-image gallery so difficulty is comparable across splits
        # (deterministic single pass when <=128 images; subsample 128 x 30 reps
        # otherwise). Same protocol as the baseline.
        retrieval_results = run_retrieval_with_repetitions(
            latent_space_list, subject_labels,
            gallery_size=128, n_reps=30, top_k=top_k, metric=metric,
        )

        if novel_subject is not None:
            novel_pairs = [(src, tgt) for (src, tgt) in retrieval_results['per_pair']
                           if src == novel_subject or tgt == novel_subject]
            for k in top_k:
                retrieval_results['mean'][k] = np.mean([
                    retrieval_results['per_pair'][p][k]['mean'] for p in novel_pairs])

        all_results[metric] = retrieval_results

        print("\n  Overall Average Retrieval Accuracy:")
        for k in top_k:
            print(f"    Top-{k}: {retrieval_results['mean'][k]:.2f}%")
        print("\n  Per-pair Top-1 Accuracy:")
        for i, src in enumerate(subject_labels):
            for j, tgt in enumerate(subject_labels):
                if i != j:
                    acc = retrieval_results['per_pair'][(src, tgt)][1]['mean']
                    print(f"    S{src}→S{tgt}: {acc:.2f}%")

    print("\n  Retrieval Matrix (Top-1, Euclidean):")
    print("     " + "  ".join([f"S{s:>4}" for s in subject_labels]))
    for i, src in enumerate(subject_labels):
        row_vals = []
        for j, tgt in enumerate(subject_labels):
            row_vals.append("  ---" if i == j
                            else f"{all_results['euclidean']['per_pair'][(src, tgt)][1]['mean']:5.1f}")
        print(f"S{src}: " + " ".join(row_vals))

    return all_results


def evaluate_aligned_latents(latents, labels=None, subject_labels=None, novel_subject=None,
                             compute_retrieval=True, compute_decoding=False,
                             compute_silhouette=False,
                             retrieval_metrics=('euclidean', 'cosine'),
                             method_tag='', output_dir='results/other'):
    """
    Run the full set of latent-only metrics both methods share — alignment +
    (optionally) retrieval / decoding / silhouette — and return one dict.
    """
    if subject_labels is None:
        subject_labels = list(range(len(latents)))

    out = {'alignment': compute_alignment_metrics(
        latents, subject_labels, novel_subject=novel_subject, method_tag=method_tag or 'eval')}

    if compute_retrieval:
        out['retrieval'] = compute_cross_subject_retrieval(
            latents, subject_labels, metrics=list(retrieval_metrics), novel_subject=novel_subject)

    if compute_decoding and labels is not None:
        out['decoding'] = test_multilabel_decoding_balanced(latents, labels, method_tag,
                                                            output_dir=output_dir)

    if compute_silhouette and labels is not None:
        silh_raw, _ = calculate_silhouette_scores(latents, labels, reduce=False)
        silh_umap, silh_indiv = calculate_silhouette_scores(latents, labels, reduce=True)
        save_silhouette_scores(silh_umap, silh_indiv, tag=method_tag)
        out['silhouette'] = {'raw': silh_raw, 'umap10d': silh_umap}

    return out


def comprehensive_evaluation(aligned_data, W_list, original_data, labels, method_name, 
                               pca_models=None, data_before_pca=None, method=None,
                               output_dir="results/other",
                               compute_alignment=True, compute_reconstruction=True,
                               compute_cross_pred=True, compute_decoding=True,
                               compute_silhouette=True):
    """
    Complete evaluation with optional metric selection.
    
    Parameters:
    -----------
    compute_alignment : bool
        Whether to compute alignment metrics (comp corr, RSA)
    compute_reconstruction : bool
        Whether to compute reconstruction quality
    compute_cross_pred : bool
        Whether to compute cross-subject prediction matrix
    compute_decoding : bool
        Whether to compute decoding metrics
    compute_silhouette : bool
        Whether to compute silhouette scores
    """
    results = {}
    
    # 1. Alignment quality (component-wise correlation and RSA)
    if compute_alignment:
        print(f"\n{'='*80}")
        print(f"1. ALIGNMENT QUALITY: {method_name}")
        print(f"{'='*80}")
        
        n_subjects = len(aligned_data)
        n_components = aligned_data[0].shape[1]
        n_common = aligned_data[0].shape[0]
        valid_idx_list = [np.arange(n_common) for _ in aligned_data]
        
        metrics = cross_subj_metrics(
        latents_clean=aligned_data,
        valid_idx_list=valid_idx_list,
        method=method
        )
        
        # Extract all the data we need
        comp_corr_mean = metrics['avg_comp_corr']
        comp_corr_per_comp = metrics.get('per_component_corr', None)  # ← NEW
        
        rsa_pearson_mean = metrics.get('avg_rsa_pearson', metrics['avg_rsa'])
        rsa_euclidean_mean = metrics.get('avg_rsa_euclidean', metrics['avg_rsa'])
        
        rsa_matrix_pearson = metrics.get('rsa_matrix_pearson', None)  # ← NEW
        rsa_matrix_euclidean = metrics.get('rsa_matrix_euclidean', None)  # ← NEW
        
        print(f"  Component-wise correlation: {comp_corr_mean:.4f}")
        print(f"  RSA Pearson: {rsa_pearson_mean:.4f}")
        print(f"  RSA Euclidean: {rsa_euclidean_mean:.4f}")

        save_alignment_metrics(
        comp_corr=comp_corr_mean,
        rsa_euclidean=rsa_euclidean_mean,
        rsa_pearson=rsa_pearson_mean,
        comp_corr_per_component=comp_corr_per_comp,  # ← PASS THIS
        rsa_matrix=rsa_matrix_pearson,  # ← PASS THIS (default to Pearson)
        tag=method,
        out_dir=output_dir,
        rsa_pearson_centered=metrics.get('avg_rsa_pearson_centered', None),
        rsa_matrix_centered=metrics.get('rsa_matrix_pearson_centered', None),
        )
        
        results['alignment_comp_corr'] = comp_corr_mean
        results['alignment_rsa'] = rsa_pearson_mean  # raw (headline)
        # Centered (legacy) RSA on the same aligned latents, kept for reference.
        results['alignment_rsa_centered'] = metrics.get('avg_rsa_pearson_centered', None)
        results['alignment_rsa_matrix'] = rsa_matrix_pearson
        results['alignment_rsa_matrix_centered'] = metrics.get('rsa_matrix_pearson_centered', None)
        if results['alignment_rsa_centered'] is not None:
            print(f"  RSA Pearson (centered/legacy): {results['alignment_rsa_centered']:.4f}")

    # 2. Reconstruction quality
    if compute_reconstruction:
        recon_results = evaluate_reconstruction_quality(
            aligned_data, W_list, original_data, method_name, pca_models, data_before_pca, output_dir=output_dir 
        )
        results.update(recon_results)
    
    # 3. Cross-subject prediction matrix — always disabled here (compute_cross_pred=False).
    # The reported cross-subject prediction is produced by prediction.run_pairwise_analysis_pipeline
    # (training-mean centering), used by evaluation/evaluate_methods.py and baselines/fit_baselines.py.
    if compute_cross_pred:
        raise NotImplementedError(
            "compute_cross_pred is not supported in comprehensive_evaluation; "
            "use prediction.run_pairwise_analysis_pipeline / evaluate_methods.py instead.")

    # 4. Multi-label decoding
    if compute_decoding:
        decoding_results = test_multilabel_decoding_balanced(
            aligned_data, labels, method
        )
        results.update(decoding_results)
        
        print(f"\n{'='*80}")
        print(f"SINGLE SUBJECT DECODING")
        print(f"{'='*80}")
        
        decoding_results = test_multilabel_decoding_single_subject_generalization(
            aligned_data, labels, method
        )
    
    # 5. Silhouette
    if compute_silhouette:
        silh_score, silh_score_indiv = calculate_silhouette_scores(
            aligned_data, labels, reduce=False
        )
        print(f"Silhouette (original dim): {silh_score:.4f}")
        
        silh_score, silh_score_indiv = calculate_silhouette_scores(
            aligned_data, labels, reduce=True
        )
        print(f"Silhouette (UMAP 10D): {silh_score:.4f}")
        
        save_silhouette_scores(silh_score, silh_score_indiv, tag=method)
    
    return results
