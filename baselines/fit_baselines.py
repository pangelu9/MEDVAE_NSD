"""
Cluster driver for the improved baselines.

Mirrors the data flow of ``baselines/fit_baselines.py`` + ``baselines/alignment.py``
but uses the corrected components in this folder and the *existing* shared
evaluation (so the numbers are directly comparable to the MED-VAE eval).

PROTOCOL (confirmed)
--------------------
  * FIT the alignment on the 872 images seen by ALL 8 subjects.
  * TEST generalization on the 128 images shared by subjects 1,2,5,7 (seen by <=7
    subjects, hence DISJOINT from the 872 -> no fit/test leakage).
  * ALL generalization metrics -- alignment, retrieval, decoding, silhouette
    (latent-space, via the shared ``evaluate_aligned_latents``) AND cross-subject
    prediction + reconstruction (voxel-space, via ``run_pairwise_analysis_pipeline``)
    -- are computed on that SAME 128-image set, matching the VAE's not_all8 eval.
  * The VAE's 90/10 train/test split is irrelevant here, so we do NOT separate
    train_common / test_common; we use ALL 872 and fit PCA on ALL per-subject data.

Methods
-------
  --method srm           : DetSRM, n_pca (large, e.g. 5000) -> n_components (32)
  --method procrustes_a  : PCA -> n_components, then GPA rotate   (paper's Procrustes)
  --method procrustes_b  : PCA -> n_pca (large), GPA rotate, reduce consensus -> n_components

Fixes vs the original: Procrustes actually reduces to k; no UMAP coupling; SRM can
run at the paper's n_pca=5000; and decoding/silhouette generalization is on the 128
(the original computed them on the full per-subject test set -- a different image set
from the 128, inconsistent with the VAE).

Example (SLURM)
---------------
  cd baselines
  sbatch -A gpu_costa.prj -p gpu_rtx8000_48gb,gpu_a100_80gb --gres gpu:1 \
      --qos gpu_bmrc_4hr --cpus-per-gpu 4 --mem-per-gpu 160G --time 3:55:00 \
      -o logs/fit_%j.out -e logs/fit_%j.out --export=ALL,REPO_ROOT=/path/to/medvae_release \
      --wrap='source $REPO_ROOT/scripts/_env.sh; cd $REPO_ROOT/baselines; \
              python3 -u fit_baselines.py --method srm --n_pca 5000 --n_components 32 \
                  --filename aligned_all_activations_fair_resnet50_hendrycks.npy \
                  --model_dir fitted_models --output_dir results_improved'
"""

from __future__ import annotations

import os
import sys
import argparse
import importlib.util

import numpy as np

# --- load THIS folder's modules unambiguously (baselines/ is also on sys.path and
#     has same-named srm.py / procrustes.py) -------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_local(name):
    spec = importlib.util.spec_from_file_location(f"bi_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


srm_mod = _load_local("srm")
proc_mod = _load_local("procrustes")
reduction = _load_local("reduction")
recon_mod = _load_local("recon")
model_io = _load_local("model_io")  # vendored copy -> folder is self-contained

# --- repo paths + shared eval ---------------------------------------------------
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from ccn_config import setup_paths, NSD_DIR  # noqa: E402
setup_paths()

_dp = _load_local("data_processing")  # vendored copy -> folder is self-contained
extract_full_subject_data = _dp.extract_full_subject_data
extract_common_fmri_samples = _dp.extract_common_fmri_samples
extract_pairwise_common_samples = _dp.extract_pairwise_common_samples
from evaluation_pipeline import (comprehensive_evaluation,      # noqa: E402
                                 evaluate_aligned_latents)
from metrics import save_prediction_matrix                        # noqa: E402
from prediction import (compute_training_means_pca,               # noqa: E402
                        save_training_means,
                        evaluate_cross_subject_prediction,
                        run_pairwise_analysis_pipeline)

# subjects 1,2,5,7 are encoder slots 0,1,4,6 (0-indexed); the 128 = images all four
# of these "hero" subjects share among the partial-overlap (<=7-subject) images.
HERO_IDX = [0, 1, 4, 6]
HERO_LABELS = [1, 2, 5, 7]


def _align(method, reduced_common, n_components, seed, srm_init="grand_mean"):
    if method == "srm":
        # init has no effect on alignment quality (recon_error/RSA identical across
        # inits; only the rotation-sensitive comp-corr moves). grand_mean is the
        # deterministic, paper-reproducing default; the no-PCA path passes 'random'
        # (grand_mean needs equal per-subject dims).
        aligned, W, _m, info = srm_mod.fit_detsrm(
            reduced_common, n_components=n_components, seed=seed, init=srm_init, verbose=True)
    elif method == "procrustes_a":
        aligned, W, _m, info = proc_mod.align_procrustes(
            reduced_common, n_components=n_components, reduce_after=False, verbose=True)
    elif method == "procrustes_b":
        aligned, W, _m, info = proc_mod.align_procrustes(
            reduced_common, n_components=n_components, reduce_after=True, verbose=True)
    else:
        raise ValueError(f"unknown method {method!r}")
    print(f"  alignment info: {info}")
    return aligned, W


def _extract_hero_common(pairwise_dict, subject_lists):
    """The 128 generalization set: images all four hero subjects share.

    Returns (image_ids, {hero_slot: (n128, n_voxels)}, labels (n128, n_cat))."""
    common = set(subject_lists[HERO_IDX[0]])
    for s in HERO_IDX[1:]:
        common &= set(subject_lists[s])
    common = sorted(common)
    data = {s: np.stack([pairwise_dict[i]['data'][s] for i in common]) for s in HERO_IDX}
    labels = np.stack([pairwise_dict[i]['labels'] for i in common])
    print(f"  hero-common (128) generalization set: {len(common)} images")
    return common, data, labels


def run_one(method, reduced_common, common_data_list, common_labels, pca_models,
            pairwise_dict, subject_lists, n_components, output_dir, model_dir, seed,
            srm_init="grand_mean", save_model=True, tag=None, load_model_path=None):
    tag = tag or f"{method}_{n_components}d"
    print("\n" + "=" * 80)
    print(f"{method.upper()} (improved)  tag={tag}")
    print("=" * 80)

    if load_model_path:
        import pickle
        print(f"Loading saved model from {load_model_path}")
        with open(load_model_path, "rb") as f:
            M = pickle.load(f)
        W = M["W_list"]
        pca_models = M["pca_models"]
        training_means = M["training_means"]
        reduced_common = [pca_models[s].transform(common_data_list[s]) for s in range(len(common_data_list))]
        aligned = [(reduced_common[i] - np.asarray(training_means[i]).reshape(1, -1)) @ W[i]
                    for i in range(len(reduced_common))]
    else:
        aligned, W = _align(method, reduced_common, n_components, seed, srm_init=srm_init)
        training_means = compute_training_means_pca(reduced_common)

        if save_model:
            model_io.save_alignment_model(W_list=[w.copy() for w in W], pca_models=pca_models,
                                 training_means=training_means,
                                 method_name=f"{method}_shared872_{n_components}d",
                                 model_dir=model_dir)

    # ---- (1) FIT-SET (872) alignment quality + reconstruction + cross-prediction ----
    comprehensive_evaluation(
        aligned, W_list=W, original_data=reduced_common, labels=common_labels,
        method_name=f"{tag}_shared872", pca_models=pca_models,
        data_before_pca=common_data_list, method=f"{tag}_shared872",
        output_dir=output_dir,
        compute_alignment=True, compute_reconstruction=True,
        compute_cross_pred=False, compute_decoding=False, compute_silhouette=False)

    pred = evaluate_cross_subject_prediction(
        aligned_data=aligned, W_list=[w.copy() for w in W],
        original_data=reduced_common, method_name=f"{tag}_shared872",
        training_means=training_means, pca_models=pca_models,
        data_before_pca=common_data_list)
    save_prediction_matrix(pred['prediction_matrix'], pred['per_voxel_correlations'],
                           pred['diagonal_mean'], pred['offdiagonal_mean'],
                           pred['space'], tag=f"{tag}_shared872", out_dir=output_dir)
    save_training_means(training_means, tag=f"{tag}_shared872", out_dir=output_dir)

    # ---- (2) GENERALIZATION on the 128 (subjects 1,2,5,7) ----
    # 2a. voxel-space cross-subject prediction + reconstruction + alignment on the 128
    run_pairwise_analysis_pipeline(
        pairwise_data_dict=pairwise_dict, subject_sample_lists=subject_lists,
        W_list=[w.copy() for w in W], pca_models=pca_models,
        training_means=training_means, method_name=f"{tag}_gen128",
        min_shared_images=5, compute_original_space=True,
        # indices are 0-indexed: slot 0 = subject 1, slot 1 = subject 2, ... slot 7 = subject 8.
        # exclude {2,3,5,7} -> drop subjects 3,4,6,8 -> KEEP slots {0,1,4,6} = subjects 1,2,5,7.
        exclude_subjects=[2, 3, 5, 7], output_dir=output_dir)

    # 2b. latent-space alignment + retrieval + DECODING + SILHOUETTE on the SAME 128
    #     (this is what the original code computed on the full test set instead).
    _ids, data128, labels128 = _extract_hero_common(pairwise_dict, subject_lists)
    aligned128, W_hero, tmean_hero, pca_hero, vox_hero = [], [], [], [], []
    for s in HERO_IDX:
        Xpca = pca_models[s].transform(data128[s])
        mu = np.asarray(training_means[s]).reshape(1, -1)
        aligned128.append((Xpca - mu) @ W[s])
        W_hero.append(W[s]); tmean_hero.append(training_means[s])
        pca_hero.append(pca_models[s]); vox_hero.append(data128[s])
    evaluate_aligned_latents(
        aligned128, labels=labels128, subject_labels=HERO_LABELS,
        compute_retrieval=True, compute_decoding=True, compute_silhouette=True,
        method_tag=f"{tag}_gen128", output_dir=output_dir)

    # 2c. correctly-centered self-reconstruction on the 128 (fixes the R2/MSE
    #     centering bug in the shared eval; correlation metrics are offset-invariant).
    print("  --- corrected reconstruction (128) ---")
    recon_mod.evaluate_reconstruction(
        aligned128, W_hero, tmean_hero, pca_hero, vox_hero)

    print(f"\nDONE: {method} -> {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Improved SRM / Procrustes baselines")
    p.add_argument('--dataset', type=str, default='streams')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--test_size', type=float, default=0.1)
    p.add_argument('--filename', type=str, required=True)
    p.add_argument('--data_dir', type=str, default=NSD_DIR + os.sep)
    p.add_argument('--output_dim', type=int, nargs='+',
                   default=[20732, 20735, 20736, 20733, 20733, 20734, 20726, 20733])
    p.add_argument('--nn_output_dim', type=int, default=51060)
    p.add_argument('--input_dim', type=int, nargs='+',
                   default=[20732, 20735, 20736, 20733, 20733, 20734, 20726, 20733, 51060])
    p.add_argument('--n_pca', type=int, default=5000,
                   help='PCA comps for SRM before alignment (large, e.g. 5000).')
    p.add_argument('--n_pca_procb', type=int, default=200,
                   help='PCA comps for Procrustes Variant B. Kept MODERATE (default 200): '
                        'full-dimensional GPA needs an O(n_pca^3) SVD per subject per '
                        'iteration, so 5000 is intractable (it ran for hours without '
                        'converging). 200 >> k=32 still gives the richer-than-A input the '
                        'variant exists for.')
    p.add_argument('--n_components', type=int, default=32)
    p.add_argument('--method', type=str, default='srm',
                   choices=['srm', 'procrustes_a', 'procrustes_b', 'all'])
    p.add_argument('--no_pca', action='store_true', default=False,
                   help='SRM directly on raw voxels (NO PCA pre-reduction). SRM-only '
                        '(full-voxel GPA is intractable). The E-step SVD is thin (voxels x k) '
                        'so this is tractable; uses random init (grand-mean init needs equal '
                        'per-subject dims) and an identity reducer so the eval round-trip is a no-op.')
    # data-loading compat flags (mirror baselines/fit_baselines.py)
    p.add_argument('--noise_level', type=float, default=0.0)
    p.add_argument('--max_memory_gb', type=float, default=150.0)
    p.add_argument('--remove_overlaps', action='store_true', default=False)
    p.add_argument('--remove_all_overlaps', action='store_true', default=False)
    p.add_argument('--shuffle_fmri', action='store_true', default=False)
    p.add_argument('--only_nn_encoder', action='store_true', default=False)
    p.add_argument('--only_fmri_encoders', action='store_true', default=False)
    p.add_argument('--framework', type=str, default='multiencoder')
    p.add_argument('--brain_to_brain', action='store_true', default=False)
    p.add_argument('--keep_percent', type=float, nargs='+',
                   default=[100, 100, 100, 100, 100, 100, 100, 100])
    p.add_argument('--model_dir', type=str, default='fitted_models')
    p.add_argument('--output_dir', type=str, default='results_improved')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--srm_init', type=str, default='grand_mean',
                   choices=['grand_mean', 'random'],
                   help='SRM initialisation (grand_mean = deterministic, random = random).')
    p.add_argument('--load_model', type=str, default=None,
                   help='Path to a saved .pkl model — skip fitting, just run eval.')
    p.add_argument('--tag', type=str, default=None,
                   help='Override the result tag (default: method_Nd).')
    args = p.parse_args()

    import random as _random
    import torch
    from pipeline import load_data
    _random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    methods = ['srm', 'procrustes_a', 'procrustes_b'] if args.method == 'all' else [args.method]
    if args.no_pca:
        if any(m != 'srm' for m in methods):
            print("[no_pca] full-voxel GPA is intractable; restricting to SRM only.")
        methods = ['srm']

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    args.train = False
    train_loader, test_loader, _, _, _ = load_data(args, kwargs)

    # 8 fMRI subjects (extractors use n_subjects=7 -> 7+1 slots = the 8 fMRI encoders;
    # NN slot excluded). We MERGE both loaders everywhere: the train/test split is
    # irrelevant to the 872->128 protocol.
    if not args.load_model:
        ft_tr, _ = extract_full_subject_data(train_loader, n_subjects=7)
        ft_te, _ = extract_full_subject_data(test_loader, n_subjects=7)
        full_data = [np.vstack([a, b]) for a, b in zip(ft_tr, ft_te)]      # ALL data -> PCA fit

    tr_common, tr_lbl = extract_common_fmri_samples(train_loader, n_subjects=7)
    te_common, te_lbl = extract_common_fmri_samples(test_loader, n_subjects=7)
    common_data_list = [np.vstack([a, b]) for a, b in zip(tr_common, te_common)]  # 872
    common_labels = np.concatenate([tr_lbl, te_lbl])

    pw_tr, sl_tr = extract_pairwise_common_samples(train_loader, min_subjects=2)
    pw_te, sl_te = extract_pairwise_common_samples(test_loader, min_subjects=2)
    off = (max(pw_tr.keys()) + 1) if pw_tr else 0
    pairwise = dict(pw_tr)
    for idx, payload in pw_te.items():
        pairwise[idx + off] = payload
    subject_lists = [a + [i + off for i in b] for a, b in zip(sl_tr, sl_te)]

    for method in methods:
        if args.load_model:
            run_one(method, None, common_data_list, common_labels, None,
                    pairwise, subject_lists, n_components=args.n_components,
                    output_dir=args.output_dir, model_dir=args.model_dir, seed=args.seed,
                    save_model=False, tag=args.tag, load_model_path=args.load_model)
            continue
        if args.no_pca:
            reduced_common = [np.asarray(c) for c in common_data_list]
            pca_models = [reduction.IdentityPCA(c.shape[1]) for c in common_data_list]
            print(f"[srm:no_pca] raw voxels per subj = {[c.shape[1] for c in common_data_list]} "
                  f"-> n_components = {args.n_components}")
            run_one("srm", reduced_common, common_data_list, common_labels, pca_models,
                    pairwise, subject_lists, n_components=args.n_components,
                    output_dir=args.output_dir, model_dir=args.model_dir, seed=args.seed,
                    srm_init="random", save_model=False, tag=f"srm_nopca_{args.n_components}d")
            continue
        if method == 'procrustes_a':
            n_pca = args.n_components
        elif method == 'procrustes_b':
            n_pca = args.n_pca_procb
        else:
            n_pca = args.n_pca
        print(f"[{method}] n_pca = {n_pca} -> n_components = {args.n_components}")
        reduced_common, _, pca_models = reduction.reduce_subjects(
            full_data, common_data_list, n_pca=n_pca, seed=args.seed)
        srm_init = args.srm_init if method == 'srm' else 'grand_mean'
        tag = args.tag
        if tag is None and method == 'srm' and args.srm_init == 'random':
            tag = f"srm_rndinit_{args.n_components}d"
        run_one(method, reduced_common, common_data_list, common_labels, pca_models,
                pairwise, subject_lists, n_components=args.n_components,
                output_dir=args.output_dir, model_dir=args.model_dir, seed=args.seed,
                srm_init=srm_init, tag=tag)


if __name__ == "__main__":
    main()
