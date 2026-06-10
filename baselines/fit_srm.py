"""
Fit and SAVE a random-init SRM model (PCA-5000 -> 32), deterministic (seed 42).

Reuses the per-subject PCA from the existing grand_mean model (PCA is init-independent),
so no PCA refit -- just re-fit SRM's EM from a random start and save. Saved as
srm_rndinit_shared872_32d_model.pkl (does NOT overwrite the grand_mean srm_shared872_32d).

NOTE: random vs grand_mean is the SAME alignment (identical RSA/recon); they differ only
in the rotation-sensitive comp-corr (random lands in a ~0.50 orientation on the 128,
grand_mean ~0.37). This model just persists the random orientation. Set N_PCA env to
refit PCA at a different n_pca (e.g. 8200) instead of reusing 5000.
"""
import os
import sys
import argparse
import pickle
import importlib.util

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _ll(n):
    s = importlib.util.spec_from_file_location(f"bi_{n}", os.path.join(_HERE, f"{n}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


srm_mod = _ll("srm")
reduction = _ll("reduction")
model_io = _ll("model_io")  # vendored copy -> folder is self-contained

_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from ccn_config import setup_paths, NSD_DIR  # noqa: E402
setup_paths()
_dp = _ll("data_processing")  # vendored copy -> folder is self-contained
extract_full_subject_data = _dp.extract_full_subject_data
extract_common_fmri_samples = _dp.extract_common_fmri_samples
extract_pairwise_common_samples = _dp.extract_pairwise_common_samples
from prediction import compute_training_means_pca                  # noqa: E402
from metrics import cross_subj_metrics                             # noqa: E402

MODEL_DIR = os.path.join(_HERE, "fitted_models")
SEED = 42
REFIT_NPCA = int(os.environ.get("N_PCA", "0"))   # 0 => reuse existing PCA-5000; else refit
HERO = [0, 1, 4, 6]


def cc(aligned):
    n = aligned[0].shape[0]
    m = cross_subj_metrics(aligned, [np.arange(n) for _ in aligned], method="x", save=False)
    return float(m["avg_comp_corr"]), float(m["avg_rsa_euclidean"])


def main():
    import torch
    from pipeline import load_data
    args = argparse.Namespace(
        dataset="streams", batch_size=64, test_size=0.1,
        filename="aligned_all_activations_fair_resnet50_hendrycks.npy",
        output_dim=[20732, 20735, 20736, 20733, 20733, 20734, 20726, 20733], nn_output_dim=51060,
        input_dim=[20732, 20735, 20736, 20733, 20733, 20734, 20726, 20733, 51060],
        noise_level=0.0, max_memory_gb=150.0, remove_overlaps=False, remove_all_overlaps=False,
        shuffle_fmri=False, only_nn_encoder=False, only_fmri_encoders=False,
        framework="multiencoder", brain_to_brain=False, keep_percent=[100] * 8, train=False)
    np.random.seed(42); torch.manual_seed(42)
    kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    tr, te, _, _, _ = load_data(args, kwargs)
    trc, _ = extract_common_fmri_samples(tr, n_subjects=7)
    tec, _ = extract_common_fmri_samples(te, n_subjects=7)
    common = [np.vstack([a, b]) for a, b in zip(trc, tec)]

    if REFIT_NPCA > 0:
        ftr, _ = extract_full_subject_data(tr, n_subjects=7)
        fte, _ = extract_full_subject_data(te, n_subjects=7)
        full = [np.vstack([a, b]) for a, b in zip(ftr, fte)]
        reduced_common, _, pca = reduction.reduce_subjects(full, common, n_pca=REFIT_NPCA, seed=42)
        tag = f"srm_rndinit_npca{REFIT_NPCA}_32d"
        print(f"Refit PCA at n_pca={REFIT_NPCA}")
    else:
        with open(os.path.join(MODEL_DIR, "srm_shared872_32d_model.pkl"), "rb") as f:
            M = pickle.load(f)
        pca = M["pca_models"]
        reduced_common = [pca[i].transform(common[i]) for i in range(len(pca))]
        tag = "srm_rndinit_shared872_32d"
        print(f"Reused existing PCA (n_pca={pca[0].n_components_}) from grand_mean model")

    aligned, W, means, info = srm_mod.fit_detsrm(reduced_common, 32, init="random", seed=SEED)
    print(f"SRM random-init (seed {SEED}): {info}")
    tmeans = compute_training_means_pca(reduced_common)

    c8, r8 = cc(aligned)
    print(f"  comp-corr 872 (8 subj) = {c8:.4f} | RSA-E = {r8:.4f}")
    # hero 128 for context
    pwt, slt = extract_pairwise_common_samples(tr, min_subjects=2)
    pwte, slte = extract_pairwise_common_samples(te, min_subjects=2)
    off = (max(pwt) + 1) if pwt else 0
    pw = dict(pwt)
    for i, v in pwte.items():
        pw[i + off] = v
    sl = [a + [i + off for i in b] for a, b in zip(slt, slte)]
    h = set(sl[HERO[0]])
    for s in HERO[1:]:
        h &= set(sl[s])
    h = sorted(h)
    al128 = [(pca[s].transform(np.stack([pw[i]["data"][s] for i in h]))
              - np.asarray(tmeans[s]).reshape(1, -1)) @ W[s] for s in HERO]
    c1, r1 = cc(al128)
    print(f"  comp-corr 128 (hero) = {c1:.4f} | RSA-E = {r1:.4f}")

    model_io.save_alignment_model(W_list=[w.copy() for w in W], pca_models=pca,
                         training_means=tmeans, method_name=tag, model_dir=MODEL_DIR)
    print("SRMRND_SENTINEL_DONE")


if __name__ == "__main__":
    main()
