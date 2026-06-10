"""
Clean PCA pre-reduction for the baselines.

Replaces ``baselines/data_processing.apply_dimensionality_reduction_robust`` and
fixes its problems:

  * NO unconditional ``import umap``.  The original imported umap at function top
    even when ``n_umap == 0``, so baseline *fitting* could not run in an
    environment where umap/numba is broken (the repo's ``ns`` env, numpy 2.x) --
    despite UMAP being unused.
  * NO UMAP applied to the common set only.  The original UMAP'd ``common`` but not
    the full/generalization data, so (with the argparse default ``n_umap=10``) the
    alignment was fit at a different dimensionality than the data it is later
    applied to -- a latent dim-mismatch / "README example crashes" footgun.
  * Per-method ``n_pca`` is explicit (SRM: large, e.g. 5000; Procrustes-A: k).
  * No stray result files written as a side effect.

PCA is fit on each subject's FULL data (every image that subject saw) and then
applied to the shared/common images and to the held-out generalization images --
identical to the paper, and leakage-free for the *basis* (the common-set
statistics are never used to choose the PCA directions).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class IdentityPCA:
    """No-op stand-in for a fitted PCA, for the --no_pca path.

    transform / inverse_transform are the identity, so the evaluation's PCA
    round-trip becomes a no-op and SRM operates directly on raw voxels
    (W_i maps voxels -> shared 32-d, with no PCA pre-reduction).
    """

    def __init__(self, n_features):
        self.n_components_ = int(n_features)
        self.mean_ = 0.0

    def transform(self, X):
        return np.asarray(X)

    def inverse_transform(self, X):
        return np.asarray(X)


def reduce_subjects(full_data_list, common_data_list, n_pca,
                    gen_data_list=None, seed=42, verbose=True):
    """Fit per-subject PCA on full data; transform the common (+ optional gen) set.

    Parameters
    ----------
    full_data_list   : list of (n_full_i, n_voxels_i)   -- fit PCA on this (ALL of a
                       subject's data; train/test split is irrelevant to the
                       872->128 protocol, so pass the merged full data).
    common_data_list : list of (n_common, n_voxels_i)   -- the 872 shared images.
    n_pca            : int (capped per subject at min(n_samples, n_voxels)).
    gen_data_list    : optional list of (n_gen_i, n_voxels_i) to also transform.

    Returns
    -------
    reduced_common : list of (n_common, n_pca_i)
    reduced_gen    : list of (n_gen_i, n_pca_i)  or  None
    pca_models     : list of fitted sklearn PCA (for inverse_transform in eval)
    """
    pca_models, reduced_common = [], []
    reduced_gen = [] if gen_data_list is not None else None
    for i, (full, common) in enumerate(zip(full_data_list, common_data_list)):
        npc = int(min(n_pca, full.shape[0], full.shape[1]))
        pca = PCA(n_components=npc, random_state=seed).fit(full)
        pca_models.append(pca)
        reduced_common.append(pca.transform(common))
        if gen_data_list is not None:
            reduced_gen.append(pca.transform(gen_data_list[i]))
        if verbose:
            ev = float(pca.explained_variance_ratio_.sum())
            print(f"  subject {i + 1}: PCA fit on {full.shape} -> {npc} comps "
                  f"(explains {ev:.1%}); common {common.shape[0]} rows")
    # SRM/Procrustes require an equal feature dimension across subjects. If a subject's
    # full data is rank-limited, n_pca silently shrinks below the request -- catch it.
    npcs = [m.n_components_ for m in pca_models]
    assert len(set(npcs)) == 1, (
        f"effective n_pca differs across subjects: {npcs}. Lower n_pca so every "
        f"subject reaches it (capped at min(n_pca, n_rows, n_voxels)).")
    if verbose and npcs[0] < n_pca:
        print(f"  NOTE: effective n_pca = {npcs[0]} (< requested {n_pca}; rank-capped).")
    return reduced_common, reduced_gen, pca_models
