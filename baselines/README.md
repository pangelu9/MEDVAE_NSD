# `baselines/` — corrected & strengthened SRM / Procrustes

> **See [FINDINGS.md](FINDINGS.md) for the current status, results, and the key
> methodological finding** (comp-corr is orientation-arbitrary for these baselines →
> use RSA/decoding/retrieval; init & PCA are immaterial). This README documents the code.

A clean re-implementation of the MEDVAE classical alignment baselines (SRM and
Procrustes), written to be **as correct and as strong as possible** so the
MED-VAE-vs-baseline comparison is fair.

**Nothing in the existing `baselines_old/`, `evaluation/`, or `medvae/` is modified.**
This folder is additive and stays drop-in compatible with the existing evaluation
pipeline (same per-subject aligned-latent contract, same `(n_features, k)`
orthonormal `W`, same `training_means` / apply convention).

> Status: algorithms + synthetic correctness test are done and **pass**
> (`test_correctness.py`, all 4 checks at cross-subject r = 1.000). The cluster
> `fit_baselines.py` driver wires them to the real data + the shared eval; it
> still needs one real run to regenerate the comparison numbers.

---

## TL;DR of what was wrong / weak in `baselines_old/`

| # | Severity | Issue | Where | Effect |
|---|----------|-------|-------|--------|
| 1 | High | **Procrustes never trims to `k`.** `align_generalized_procrustes` computes `aligned_trim = x[:, :n_cca]` and then returns the *full-dim* `aligned_full`, discarding the trim (docstring even claims 4 returns; it returns 3). | `baselines_old/procrustes.py:62-67` | Forces `n_pca == n_components` for any comparable eval. This is *why* Procrustes had to be run at `n_pca=32` instead of being given 5000 like SRM. Present in the original paper code too. |
| 2 | Med-High | **UMAP applied to the common set only**, not to full/generalization data; argparse default is `n_umap=10`. | `baselines_old/data_processing.py:130-142` | With the default, the alignment is fit at a different dim than the data it's later applied to → dim mismatch. The README example (no `--n_umap`) would crash. Masked only because the SLURM script passes `--n_umap 0`. |
| 3 | Med | **Unconditional `import umap`** inside the reduction fn, even when `n_umap=0`. | `baselines_old/data_processing.py:94` | Couples baseline *fitting* to a working umap/numba install (broken under numpy 2.x `ns` env) although umap is unused. |
| 4 | ~~Med~~ → **corrected: not a bug** | **No leakage in the intended protocol.** Methods are fit on the **872** (images all 8 subjects saw) and generalization is tested on the **128** (images shared by subjects 1,2,5,7 but seen by ≤7 subjects). The 128 are disjoint from the 872 *by construction*, so fit-872 / test-128 has **no leakage**. `vstack(train_common, test_common)` merely reassembles the full 872 from the two data loaders — it is **not** leakage. | `fit_baselines.py:137` | — |
| 4b | Low (caveat) | The **separate full-test-set** recon/decoding/silhouette path (`alignment.py` §3, `full_data_reduced_test`) mixes in the VAE's 10% test split and *does* overlap the 872's test-split rows. If those numbers are reported as "generalization," that path is not a clean held-out — the **128 pairwise** path is. | `baselines_old/alignment.py:189-251` | Only matters if §3 (not the 128) is the reported generalization metric. |
| 5 | Med | **Repo ≠ paper for SRM.** The repo's `fit_baselines.sh` defaults `N_PCA→N_COMPONENTS` (=32); its logs + saved model (`detsrm_shared872_32d_model.pkl` is 21 MB = 32-comp float32 PCA) show SRM was re-run at `n_pca=32`, not the paper's 5000. | `scripts/fit_baselines.sh` | SRM@32 is degenerate (n_pca==k ⇒ square orthogonal rotation, **no bottleneck**). The regenerated SRM numbers understate the paper's SRM. |
| 6 | Low | SRM had a noisy "try 3 inits, pick best" block + heavy DEBUG prints, **no convergence check** (fixed `n_iter`), and re-derived centering means in two places. | `baselines_old/srm.py` | Cosmetic + reproducibility; cleaned up here. |
| 7 | Med (R²/MSE only) | **Reconstruction centering bug.** `X_recon = aligned @ W.T` is in *training-mean-centered* PCA space, but `inverse_transform` only re-adds `pca.mean_` — `training_mean` is never added back, so the reconstruction is off by a per-voxel constant. | `evaluation/metrics.py:403-410`; `prediction.py:108-113` | Pearson correlation is invariant to a per-voxel offset, so **voxel/sample-correlation metrics are UNAFFECTED**; only **R²/MSE** are biased. Fixed for the 128 via `recon_fixed.py` (existing code untouched). |

> The 78-agent audit independently **confirmed** findings 1–7 and the methodology
> points below, and **refuted** several false candidates (see next block). It also
> verified the *core math* of both methods is correct (SVD E-step, GPA centroid,
> centering consistency, subject-exclusion logic).

**Refuted (verified false — do not trust these earlier claims):**
- An automated claim that SRM init "Method 1" `np.mean(X, axis=0)` vstacks to `(6976,5000)` — **false**; numpy stacks the list to `(8,872,5000)` and means to `(872,5000)`. The 3-init heuristic was still removed here for cleanliness, but it was not a numeric bug.
- `exclude_subjects=[2,3,5,7]` in `baselines_old/alignment.py:184` **correctly keeps** the hero subjects 1,2,5,7 (0-indexed exclude `{2,3,5,7}` → include `{0,1,4,6}`). Only the docstrings ("excluding 3 & 6") are stale; the behaviour is right.

---

## The key methodological point: can Procrustes do 5000 → 32?

**No — plain Procrustes cannot reduce dimensionality.** Orthogonal Procrustes solves
`min_R ‖XR − T‖_F s.t. RᵀR = I`; the solution `R = UVᵀ` is a *square* orthogonal
matrix, i.e. a pure rotation. It maps `p → p`. To get a `k`-D Procrustes
representation you must bolt on a reduction step. There are two principled ways,
both implemented here:

- **Variant A — reduce *before* (the paper's choice).** PCA each subject `5000→32`,
  then GPA rotates inside the 32-D space. `Wᵢ = Rᵢ` (32×32 orthogonal). Standard, but
  each subject's 32-D is its *own* top-32 PCA, and shared signal in PCA comps > 32 is
  discarded *before* alignment.
- **Variant B — reduce *after* (stronger, matches SRM's input richness).** PCA `5000`,
  GPA-rotate in 5000-D, then take the top-`k` principal directions `V_k` of the
  *aligned consensus* and project: `Wᵢ = Rᵢ V_k` (5000×32, **orthonormal columns**).
  This lets Procrustes exploit all 5000 dims, like SRM.

A dimensionality-reducing orthonormal alignment is, mathematically, almost exactly
**SRM** (SRM's per-subject E-step *is* "orthogonal Procrustes onto a shared `k`-D
target"). So Variant B converges toward SRM — expected, and a useful sanity check.

By contrast, **SRM is natively reducing**: it learns `Wᵢ ∈ ℝ^{5000×32}` with
`WᵢᵀWᵢ = I₃₂` and a shared response `S ∈ ℝ^{T×32}` minimising `Σᵢ‖Xᵢ − S Wᵢᵀ‖²`. That
is the right tool for `5000 → 32`.

So everything lands at **32-D** for the comparison with the 32-D MED-VAE latent:
MED-VAE encoder → 32; SRM `5000 → 32`; Procrustes A (`PCA→32`, rotate) and Procrustes
B (`PCA→5000`, rotate, reduce → 32).

---

## Files

| File | What |
|---|---|
| `srm.py` | Clean deterministic SRM (`fit_detsrm`) + `apply_alignment`. Orthonormal `(n_features,k)` `W`; documented SVD init; real convergence check; returns centering means. |
| `procrustes.py` | `fit_gpa` (GPA in input dim) + `align_procrustes(..., reduce_after=)` for variants **A** and **B**; `apply_alignment`. Both return orthonormal-column `W`. |
| `reduction.py` | `reduce_subjects` — per-subject PCA fit-on-full / transform-common. No umap coupling, no stray files, per-method `n_pca`; asserts equal effective `n_pca` across subjects. |
| `recon_fixed.py` | `evaluate_reconstruction_fixed` — correctly-centered self-reconstruction (re-adds `training_mean` before `inverse_transform`), fixing the R²/MSE bug without touching `evaluation/`. |
| `test_correctness.py` | Self-contained synthetic tests (no data/GPU/cluster). Validates SRM signal recovery, Procrustes A & B alignment + reduction, orthonormality, and apply-to-new-data. |
| `fit_baselines.py` | Cluster driver: data → reduction → align (srm / procrustes_a / procrustes_b) → the **existing** shared evaluation, so numbers are directly comparable to the VAE. All generalization metrics on the 128; corrected reconstruction via `recon_fixed`. |

### The eval contract (why this is drop-in)
The shared evaluation consumes, per subject: an aligned latent `(n_images, k)`, a
transform `W` of shape `(n_features, k)` with **orthonormal columns**, the PCA model,
and a `training_mean`. It then uses `aligned @ Wᵀ` (rank-`k` reconstruction) and
`Wᵢ Wⱼᵀ` (cross-prediction). Every method here returns exactly that, and
`apply_alignment` replays `(X − training_mean) @ W`, identical to
`baselines.alignment.apply_trained_alignment`.

---

## How to run

**1. Correctness test (fast, CPU):**
```bash
cd baselines
sbatch -A costa.prj -p short -c 2 --mem 4G --time 00:05:00 \
  -o test_%j.out -e test_%j.out \
  --wrap='source /well/costa/users/odx145/miniconda3/etc/profile.d/conda.sh; \
          conda activate ns; cd '"$PWD"'; python3 -u test_correctness.py'
```
Expect: `ALL PASSED` (4/4).

**2. Fit a baseline on the real data (regenerate comparison numbers):**
```bash
# SRM: 5000 -> 32 (paper config)
METHOD=srm        N_PCA=5000 N_COMPONENTS=32 sbatch ... fit_baselines.py ...
# Procrustes A: PCA->32, rotate
METHOD=procrustes_a            N_COMPONENTS=32 sbatch ... fit_baselines.py ...
# Procrustes B: PCA->5000, rotate, reduce->32
METHOD=procrustes_b N_PCA=5000 N_COMPONENTS=32 sbatch ... fit_baselines.py ...
```
(See `fit_baselines.py --help` for the exact flags; it mirrors the existing
`baselines_old/fit_baselines.py` data flags.)

---

## Protocol (confirmed, implemented in `fit_baselines.py`)
- **Fit** the alignment on the **872** images (seen by all 8 subjects).
- **Test** generalization on the **128** images shared by subjects 1,2,5,7 (seen by
  ≤7 subjects, hence disjoint from the 872 → no fit/test leakage).
- **Every** generalization metric is on that **same 128**: alignment, retrieval,
  decoding, silhouette (latent-space via the shared `evaluate_aligned_latents`) **and**
  cross-subject prediction + reconstruction (voxel-space via `run_pairwise_analysis_pipeline`)
  — matching the VAE's `not_all8` eval.
- The VAE's 90/10 split is **not used**: we merge both loaders, fit PCA on **all**
  per-subject data, and use the full 872 / full 128.

> **Inconsistency this fixes (vs `baselines_old/`):** the original computes decoding &
> silhouette generalization on the **full per-subject test set** (`alignment.py` §3),
> not on the 128 — a *different* image set from the one the VAE uses (`not_all8`),
> so those two metrics were not comparable between baseline and VAE. Here they're on
> the 128, like the VAE.

## Open items (need a real cluster run to close)
- Regenerate SRM at the **paper's `n_pca=5000`** (not the repo's degenerate 32) and
  confirm it matches / improves on the published SRM.
- Report Procrustes **A vs B** side by side to show the baseline isn't crippled by
  pre-reduction (expectation: B ≥ A, and B ≈ SRM).
- Validate the full driver end-to-end on real data (the synthetic test covers the
  algorithms; the data plumbing — 872/128 extraction, `evaluate_aligned_latents`
  wiring — only runs on the cluster).
- **VAE-parity audit (important):** confirm the VAE eval uses the *identical* 128
  held-out set, the *same* decoder (`test_multilabel_decoding_balanced`, same hyper-
  params), and the same RSA convention (raw vs centered) — otherwise the fairness
  fixes here are undone on the VAE side. (The baselines' PCA columns are zero-mean so
  raw==centered RSA, but VAE μ are not, so the convention matters for the VAE.)
- Report **mean ± std over ≥3 seeds/splits** (SRM-EM and GPA are iterative; a single
  seed=42 run can mis-rank methods). Submit via `sbatch --wrap` with different `--seed`.
- Add **anchors**: a no-alignment lower bound and a within-subject noise-ceiling upper
  bound, so the VAE-vs-baseline gap is contextualized (Bazeille et al. 2021/2024).
- Confirm the **effective `n_pca`** for SRM at 5000 (PCA is fit on each subject's full
  data, ~9k rows, so 5000 should be reachable — the driver now asserts equality across
  subjects and logs the effective value).
- Optional, to pre-empt the "weak baselines" critique: add an **Optimal Transport**
  alignment (fmralign / POT), the current classical SOTA per the Bazeille benchmark.
- Optional: a probabilistic SRM (Chen 2015) variant — the original had
  `apply_probabilistic_srm` (dropped in the cleanup). Its aligned rep is the
  posterior shared response, not `X @ W`, so it needs a small contract shim; left out
  for now to keep the orthonormal-`W` contract clean.
