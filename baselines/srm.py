"""
Clean, correct Shared Response Model (SRM) for cross-subject alignment.

This is a rewrite of ``baselines/srm.py`` (``fit_detsrm``) for the MEDVAE paper
baselines.  The *algorithm* (deterministic SRM, Chen et al. 2015) is unchanged in
spirit, but the implementation is cleaned up and the contract is made explicit.

WHAT SRM DOES
-------------
Given per-subject data ``X_i`` of shape ``(n_samples, n_features)`` (same images,
same row order; features are typically PCA-reduced voxels -- the paper uses
``n_features = n_pca = 5000``), SRM finds, for each subject, an orthonormal basis
``W_i`` of shape ``(n_features, k)`` and a single shared response ``S`` of shape
``(n_samples, k)`` minimising

    sum_i || X_i - S W_i^T ||_F^2     s.t.   W_i^T W_i = I_k .

It is a *dimensionality-reducing* orthonormal alignment: ``5000 -> k`` is a genuine
rank-``k`` bottleneck (unlike Procrustes, which is a pure rotation -- see
``procrustes.py``).

EVAL CONTRACT (drop-in compatible with evaluation/)
---------------------------------------------------
We return ``W_list`` with ORTHONORMAL COLUMNS (``W_i^T W_i = I_k``).  The shared
evaluation code relies on this:
  * reconstruction:      ``X_hat = (X @ W) @ W.T``         (rank-k projector W W^T)
  * cross-prediction:    ``X_i -> (X_i-mu_i) @ W_i @ W_j.T``  maps subject i -> j
  * aligned latent:      ``aligned_i = (X_i - mu_i) @ W_i``  of shape (n_samples, k)
We also return the per-subject centering means so the caller can store them as
``training_means`` -- guaranteeing that fit-time and apply-time centering match
(``apply_alignment`` below replays exactly the same transform on new data).

FIXES vs the original ``baselines/srm.py``
------------------------------------------
  * Removed the noisy "try 3 initialisations, pick the lowest error" block and the
    heavy DEBUG printing.  Init is now a single, documented, deterministic SVD
    (PCA scores of the cross-subject grand mean) -- the standard DetSRM warm start.
  * Added a real convergence check on the objective (the original ran a fixed
    ``n_iter`` with no stopping criterion).
  * The centering mean is returned (not recomputed elsewhere), so train<->apply
    centering can never silently drift.
  * No global-RNG dependence: PCA/SVD here are deterministic; an explicit ``seed``
    is accepted only for the optional random init path.
"""

from __future__ import annotations

import numpy as np


def _orthonormal_procrustes(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """argmin_W ||X - S W^T||_F  s.t.  W^T W = I.

    Solution: with A = X^T S, SVD A = U Sigma V^T, then W = U V^T.
    X: (n_samples, n_features), S: (n_samples, k)  ->  W: (n_features, k).
    """
    A = X.T @ S                      # (n_features, k)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    return U @ Vt                    # (n_features, k), orthonormal columns


def _init_shared(X_centered, k, method="grand_mean", seed=42):
    """Deterministic warm start for the shared response S (n_samples, k)."""
    if method == "grand_mean":
        grand = np.mean(X_centered, axis=0)            # (n_samples, n_features)
        grand = grand - grand.mean(axis=0, keepdims=True)
        U, sv, _ = np.linalg.svd(grand, full_matrices=False)
        S = U[:, :k] * sv[:k]                          # PCA scores of the grand mean
    elif method == "concat":
        Xc = np.vstack(X_centered)                     # (n_subj*n_samples, n_features)
        U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
        n = X_centered[0].shape[0]
        S = (U[:, :k] * sv[:k])[:n]
    elif method == "random":
        rng = np.random.default_rng(seed)
        S = rng.standard_normal((X_centered[0].shape[0], k))
    else:
        raise ValueError(f"unknown init method {method!r}")
    return S - S.mean(axis=0, keepdims=True)


def fit_detsrm(data_list, n_components, n_iter=200, tol=1e-6,
               init="grand_mean", seed=42, n_init=1, verbose=False):
    """Fit deterministic SRM.

    Parameters
    ----------
    data_list : list of (n_samples, n_features) arrays, one per subject (same rows).
    n_components : int   -- shared dimensionality k (paper: 32).
    n_iter, tol : EM iteration budget and relative-objective stopping tolerance.
    init : 'grand_mean' (default) | 'random' | 'concat'.
        SRM's EM is non-convex, BUT a diagnostic sweep showed the init has NO effect on
        the rotation-INVARIANT metrics: recon_error and RSA are identical for grand_mean
        vs random (vs no-PCA). The large spread in *component-wise correlation* across
        inits is purely a ROTATION ARTIFACT -- comp-corr is rotation-sensitive and SRM
        is identified only up to a global rotation of the shared space. So grand_mean
        (deterministic, reproduces the paper to ~1e-5) is the default. best-of-N random
        (n_init>1, init='random') is available but UNNECESSARY for quality; use a
        rotation-invariant metric (RSA / decoding / retrieval), not comp-corr, to
        compare alignments.
    n_init : random restarts selected by recon_error (only meaningful for init='random';
        default 1, since the init does not change alignment quality).

    Returns
    -------
    aligned : list of (n_samples, k) arrays   -- (X_i - mu_i) @ W_i.
    W_list  : list of (n_features, k) arrays   -- orthonormal columns.
    means   : list of (n_features,) arrays     -- per-subject centering means.
    info    : dict with convergence diagnostics.
    """
    # best-of-N random restarts, selected by the SRM objective (recon_error).
    if n_init > 1 and init == "random":
        best, best_r = None, 0
        for r in range(n_init):
            res = fit_detsrm(data_list, n_components, n_iter=n_iter, tol=tol,
                             init="random", seed=seed + r, n_init=1, verbose=False)
            if best is None or res[3]["recon_error"] < best[3]["recon_error"]:
                best, best_r = res, r
        best[3]["n_init"] = n_init
        best[3]["selected_seed"] = seed + best_r
        if verbose:
            print(f"  [detsrm] best-of-{n_init} random restarts: selected seed "
                  f"{seed + best_r}, recon_error={best[3]['recon_error']:.6e}")
        return best

    n_subj = len(data_list)
    means = [x.mean(axis=0, keepdims=True) for x in data_list]
    X = [x - m for x, m in zip(data_list, means)]
    n_feat = X[0].shape[1]
    k = int(min(n_components, n_feat))
    # SRM is only meaningful as a bottleneck: k must be < n_features. At k == n_feat,
    # W is square orthogonal and X @ W @ W.T == X exactly (no reduction, no denoising)
    # -- SRM degenerates to a pure rotation (i.e. Procrustes-on-PCA). Run with n_pca >> k.
    if k >= n_feat:
        print(f"  [detsrm] WARNING: n_components ({n_components}) >= n_features ({n_feat}). "
              f"SRM has NO bottleneck here and degenerates to a pure rotation. Use n_pca >> k.")
    # The EM itself and the 'random' init work with PER-SUBJECT feature counts (W_i can
    # differ in rows) -- this is what the --no_pca path needs (raw voxels: 20732, 20735,
    # ...). Only the grand_mean/concat inits average/stack across subjects, so only those
    # require equal n_features.
    if init in ("grand_mean", "concat"):
        assert all(x.shape[1] == n_feat for x in X), \
            "grand_mean/concat init requires equal n_features across subjects; use init='random'"

    S = _init_shared(X, k, method=init, seed=seed)
    W = [None] * n_subj

    prev = None          
    it = 0
    for it in range(n_iter):
        # E-step: per-subject orthonormal basis given the shared response.
        for i in range(n_subj):
            W[i] = _orthonormal_procrustes(X[i], S)
        # M-step: shared response is the mean projection (re-centered).
        S = np.mean([X[i] @ W[i] for i in range(n_subj)], axis=0)
        S = S - S.mean(axis=0, keepdims=True)

        err = float(sum(np.linalg.norm(X[i] - S @ W[i].T, "fro") ** 2
                        for i in range(n_subj)))
        if verbose:
            print(f"  [detsrm] iter {it + 1:02d}: recon_error = {err:.6e}")
        # Check the relative objective change only once we have a previous value
        # (so we always run at least two full EM sweeps before declaring convergence).
        if prev is not None and abs(prev - err) <= tol * max(1.0, prev):
            break
        prev = err

    aligned = [X[i] @ W[i] for i in range(n_subj)]
    info = {
        "method": "detsrm",
        "n_components": k,
        "n_features": n_feat,
        "n_subjects": n_subj,
        "n_iter_run": it + 1,
        "recon_error": err,
        "converged": (it + 1) < n_iter,
    }
    return aligned, W, [m.ravel() for m in means], info


def apply_alignment(data_list_new, W_list, training_means):
    """Apply a fitted SRM (or any orthonormal-W alignment) to new data.

    aligned_i = (X_new_i - training_means[i]) @ W_list[i].
    Identical convention to ``baselines.alignment.apply_trained_alignment`` so the
    existing evaluation/prediction code consumes it unchanged.
    """
    out = []
    for i, X in enumerate(data_list_new):
        mu = np.asarray(training_means[i]).reshape(1, -1)
        out.append((X - mu) @ W_list[i])
    return out
