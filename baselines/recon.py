"""
Corrected reconstruction-quality metric for the baselines.

Fixes a centering bug in ``evaluation/metrics.evaluate_reconstruction_quality``
(and the equivalent inside ``prediction.run_pairwise_analysis_pipeline``). Those
compute:

    X_recon_pca   = aligned @ W.T                 # = (X_pca - training_mean) @ W W.T
    X_recon_voxel = pca.inverse_transform(X_recon_pca)

but ``aligned`` lives in TRAINING-MEAN-CENTERED PCA space, while
``inverse_transform`` only re-adds ``pca.mean_`` -- so the per-subject
``training_mean`` is never added back. The reconstruction is therefore off by a
per-voxel constant ``training_mean @ pca.components_``.

PRECISE IMPACT (verified): per-voxel and per-sample **Pearson correlation** are
invariant to a per-voxel constant offset, so the correlation reconstruction
metrics are UNAFFECTED. Only **R^2 and MSE** are biased (they depend on absolute
values). This wrapper re-adds ``training_mean`` before ``inverse_transform`` so
R^2/MSE are also correct. It does NOT modify ``evaluation/metrics.py``; it just
calls the existing per-subject metric on a correctly-centered reconstruction.
"""

from __future__ import annotations

import numpy as np


def evaluate_reconstruction(aligned, W_list, training_means, pca_models,
                                  data_before_pca, verbose=True):
    """Correctly-centered self-reconstruction quality, per subject.

    aligned[i]          : (n, k)  = (X_pca - training_mean[i]) @ W_list[i]
    W_list[i]           : (n_pca, k) orthonormal columns
    training_means[i]   : (n_pca,) per-subject mean used at fit time
    pca_models[i]       : fitted sklearn PCA
    data_before_pca[i]  : (n, n_voxels) ground-truth voxels

    Returns dict with per-subject and mean voxel-correlation / R^2.
    """
    from metrics import calculate_reconstruction_quality  # lazy: needs setup_paths()

    voxel_corr, r2 = [], []
    for i in range(len(aligned)):
        tmean = np.asarray(training_means[i]).reshape(1, -1)
        X_recon_pca = aligned[i] @ W_list[i].T + tmean          # <- re-add the centering
        X_recon_vox = pca_models[i].inverse_transform(X_recon_pca)
        m, _ = calculate_reconstruction_quality(data_before_pca[i], X_recon_vox, i)
        voxel_corr.append(m['voxel_correlation_mean'])
        r2.append(m['r2'])
        if verbose:
            print(f"  [recon] subj {i + 1}: voxel_corr={m['voxel_correlation_mean']:.4f} "
                  f"R2={m['r2']:.4f}")
    out = {
        'per_subject_voxel_corr': voxel_corr,
        'per_subject_r2': r2,
        'mean_voxel_corr': float(np.mean(voxel_corr)),
        'mean_r2': float(np.mean(r2)),
    }
    print(f"  [recon] mean voxel_corr={out['mean_voxel_corr']:.4f} "
          f"mean R2={out['mean_r2']:.4f}")
    return out
