"""
Generalized Procrustes Analysis (GPA) for cross-subject alignment, done right.

WHY THIS EXISTS
---------------
Plain Procrustes is an *orthogonal rotation*: the solution R of
``min_R ||X R - T||_F  s.t.  R^T R = I`` is a square (p x p) orthogonal matrix.
A rotation is dimension-preserving -- it can NEVER turn a 5000-D input into a 32-D
output on its own.  So to produce a k-D Procrustes representation you must pair it
with a reduction step.  This module offers the two principled ways:

  Variant A ("pre-reduction", the paper's choice):
      PCA-reduce each subject to k features BEFORE alignment (caller passes
      data already at n_features == k), then GPA rotates within that k-D space.
          W_i = R_i        (k x k, orthogonal)
      Simple and standard, but each subject's k-D is its own top-k PCA, and any
      shared signal in PCA comps > k is discarded before alignment.

  Variant B ("post-reduction", stronger -- matches SRM's input richness):
      PCA-reduce to a LARGE n_features (paper SRM uses 5000), GPA-rotate in that
      full space, then reduce the *aligned consensus* to k via PCA and project:
          V_k = top-k principal directions of  mean_i (X_i - mu_i) R_i      (p x k)
          W_i = R_i @ V_k                                                    (p x k)
      W_i has ORTHONORMAL COLUMNS (W_i^T W_i = V_k^T R_i^T R_i V_k = I_k), so it is
      a genuine rank-k bottleneck and is drop-in compatible with the evaluation.
      This lets Procrustes exploit all 5000 input dims like SRM does.

Note: a dimensionality-reducing orthonormal alignment is, mathematically, almost
exactly SRM (SRM's E-step *is* "orthogonal Procrustes onto a shared k-D target").
Variant B therefore converges toward SRM -- that is expected, not a bug.

THE BUG THIS REPLACES
---------------------
The original ``baselines/procrustes.py`` computed ``aligned_trim = x[:, :n_cca]``
and then *returned the full-dim aligned data and threw the trim away* (its
docstring even claimed 4 return values; it returned 3).  That silently forced
``n_pca == n_components`` for any comparable evaluation -- which is exactly why the
paper had to run Procrustes at n_pca=32 instead of giving it 5000 like SRM.

EVAL CONTRACT
-------------
Both variants return ``W_list`` with orthonormal columns (n_features, k) and
``aligned = (X - mu) @ W``, plus the per-subject centering means, matching
``baselines.alignment.apply_trained_alignment``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes


def fit_gpa(data_list, n_iter=200, tol=1e-6, center=True, scaling=False,
            init_seed=None, verbose=False):
    """Generalized Procrustes alignment in the input dimensionality.

    Iteratively rotates every subject onto the running consensus (mean of the
    aligned configurations) until the consensus stops moving.

    Parameters
    ----------
    data_list : list of (n_samples, p) arrays (same rows across subjects).
    scaling   : if True, also fit an isotropic scale per subject.  WARNING: this
                makes R_i non-orthonormal, which breaks the rank-k reconstruction
                contract used by the eval; only correlation/RSA/retrieval metrics
                stay valid.  Default False (keep transforms orthogonal).
    init_seed : if not None, pre-rotate each subject by a random orthogonal matrix
                (seeded) before GPA -- an init-robustness probe. Standard GPA starts
                from the data itself (init_seed=None), which is the canonical and
                deterministic start; this option exists only to confirm GPA converges
                to the same alignment regardless of starting orientation.

    Returns
    -------
    aligned_full : list of (n_samples, p) arrays.
    R_list       : list of (p, p) transforms (orthogonal unless scaling=True).
    means        : list of (p,) per-subject centering means.
    info         : dict of diagnostics.
    """
    n_subj = len(data_list)
    means = [x.mean(axis=0, keepdims=True) if center else np.zeros((1, x.shape[1]))
             for x in data_list]
    aligned = [x - m for x, m in zip(data_list, means)]
    p = aligned[0].shape[1]
    R_list = [np.eye(p) for _ in range(n_subj)]

    if init_seed is not None:
        # Random orthogonal pre-rotation per subject (QR of a Gaussian) -- changes the
        # starting orientation; GPA should still converge to the same alignment.
        rng = np.random.default_rng(init_seed)
        for i in range(n_subj):
            Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
            aligned[i] = aligned[i] @ Q
            R_list[i] = Q

    centroid = np.mean(aligned, axis=0)
    prev_disp = None
    disp = float(np.sum([np.linalg.norm(x - centroid) ** 2 for x in aligned]))
    it = 0
    for it in range(n_iter):
        for i in range(n_subj):
            R, _ = orthogonal_procrustes(aligned[i], centroid)
            aligned[i] = aligned[i] @ R
            R_list[i] = R_list[i] @ R
            if scaling:
                num = float(np.sum(aligned[i] * centroid))
                den = float(np.sum(aligned[i] ** 2))
                s = num / den if den > 0 else 1.0
                aligned[i] = aligned[i] * s
                R_list[i] = R_list[i] * s        # fold scale into the transform
        centroid = np.mean(aligned, axis=0)
        # Converge on DISPARITY, not centroid movement. Orthogonal GPA's disparity
        # (sum of squared distances to the consensus) decreases monotonically and is
        # the correct stopping signal; the centroid itself can keep drifting slightly
        # (GPA is identified only up to a global rotation of the consensus), so a
        # centroid-move criterion can oscillate forever in high dimension -- which is
        # exactly what made Variant B run for hours without stopping.
        prev_disp = disp
        disp = float(np.sum([np.linalg.norm(x - centroid) ** 2 for x in aligned]))
        if verbose:
            print(f"  [gpa] iter {it + 1:02d}: disparity = {disp:.6e}")
        if abs(prev_disp - disp) <= tol * max(1.0, prev_disp):
            break

    disparity = disp
    info = {"method": "gpa", "n_iter_run": it + 1, "disparity": disparity,
            "scaling": scaling, "n_features": p, "n_subjects": n_subj,
            "converged": (it + 1) < n_iter}
    return aligned, R_list, [m.ravel() for m in means], info


def align_procrustes(data_list, n_components, reduce_after=False,
                     n_iter=200, tol=1e-6, scaling=False, init_seed=None, verbose=False):
    """Procrustes alignment producing a k-D representation.

    reduce_after=False -> Variant A (expects n_features == k; pre-reduced by PCA).
    reduce_after=True  -> Variant B (GPA in full dim, then PCA-reduce consensus to k).

    Returns
    -------
    aligned : list of (n_samples, k) arrays.
    W_list  : list of (n_features, k) arrays with orthonormal columns
              (unless scaling=True, in which case columns are orthogonal*scale).
    means   : list of (n_features,) per-subject centering means.
    info    : dict.
    """
    aligned_full, R_list, means, info = fit_gpa(
        data_list, n_iter=n_iter, tol=tol, scaling=scaling, init_seed=init_seed, verbose=verbose)
    p = data_list[0].shape[1]
    k = int(min(n_components, p))

    if not reduce_after:
        # Variant A: rotation lives in the (already-reduced) k-D space.
        if p != k:
            print(f"[procrustes:A] WARNING: n_features={p} != n_components={k}. "
                  f"Trimming the first {k} GPA axes is NOT a principled reduction "
                  f"(GPA axes are unordered). For n_features>k use reduce_after=True.")
        W_list = [R[:, :k] for R in R_list]
        aligned = [af[:, :k] for af in aligned_full]
        info["variant"] = "A_pre_reduction"
    else:
        # Variant B: reduce the aligned consensus to k, project everyone onto it.
        consensus = np.mean(aligned_full, axis=0)
        cc = consensus - consensus.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(cc, full_matrices=False)
        V = Vt[:k].T                                   # (p, k), orthonormal columns
        W_list = [R @ V for R in R_list]               # (p, k), orthonormal columns
        aligned = [af @ V for af in aligned_full]       # (n_samples, k)
        info["variant"] = "B_post_reduction"

    info["n_components"] = k
    return aligned, W_list, means, info


def apply_alignment(data_list_new, W_list, training_means):
    """aligned_i = (X_new_i - training_means[i]) @ W_list[i].

    Same convention as ``baselines.alignment.apply_trained_alignment`` and
    ``srm.apply_alignment``, so the existing eval/prediction code is unchanged.
    """
    out = []
    for i, X in enumerate(data_list_new):
        mu = np.asarray(training_means[i]).reshape(1, -1)
        out.append((X - mu) @ W_list[i])
    return out
