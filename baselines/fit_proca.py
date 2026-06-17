"""
Fit and SAVE a random-START Procrustes-A model (PCA-32 -> GPA with init_seed) -- the
Procrustes analog of SRM's random init, giving a HIGH-comp-corr orientation (~0.45 on
the 128) instead of standard GPA's ~0.32. SAME alignment quality (RSA unchanged ~0.405),
different gauge -- so its comp-corr is recipe-matched to srm_rndinit (random start).

Reuses the PCA-32 from the existing procrustes_a model (init-independent), so only the
GPA starting orientation changes. Saved as procrustes_a_rndinit_shared872_32d_model.pkl.
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


proc_mod = _ll("procrustes")
model_io = _ll("model_io")  # vendored copy -> folder is self-contained

_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from ccn_config import setup_paths, NSD_DIR, ANN_FEATURES_FILE  # noqa: E402
setup_paths()
_dp = _ll("data_processing")  # vendored copy -> folder is self-contained
extract_common_fmri_samples = _dp.extract_common_fmri_samples
extract_pairwise_common_samples = _dp.extract_pairwise_common_samples
from prediction import compute_training_means_pca                  # noqa: E402
from metrics import cross_subj_metrics                             # noqa: E402

MODEL_DIR = os.path.join(_HERE, "fitted_models")
INIT_SEED = 0
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
        filename=ANN_FEATURES_FILE,
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

    with open(os.path.join(MODEL_DIR, "procrustes_a_shared872_32d_model.pkl"), "rb") as f:
        M = pickle.load(f)
    pca = M["pca_models"]
    print(f"Reused PCA from procrustes_a (n_pca={pca[0].n_components_})")
    reduced_common = [pca[i].transform(common[i]) for i in range(len(pca))]

    aligned, W, means, info = proc_mod.align_procrustes(
        reduced_common, 32, reduce_after=False, init_seed=INIT_SEED, verbose=True)
    print(f"Procrustes-A random-start (init_seed={INIT_SEED}): {info}")
    tmeans = compute_training_means_pca(reduced_common)

    c8, r8 = cc(aligned)
    print(f"  comp-corr 872 (8 subj) = {c8:.4f} | RSA-E = {r8:.4f}")

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
                         training_means=tmeans,
                         method_name="procrustes_a_rndinit_shared872_32d", model_dir=MODEL_DIR)
    print("PROCARND_SENTINEL_DONE")


if __name__ == "__main__":
    main()
