# `baselines/`: SRM and Procrustes

The paper's two classical cross-subject alignment baselines: the shared response model
(SRM) and generalised Procrustes analysis. Each maps every subject's fMRI to a 32-d shared
space, like the MED-VAE latent space. Both align the subjects on the 872 images that all
eight subjects saw; the fitted models are saved to `fitted_models/` and evaluated on the 128
test images (seen by subjects 1, 2, 5 and 7 but not by all eight) by
`evaluation/evaluate_methods.py` (`scripts/eval_medvae.sh`).

## The models

Each subject's responses are first reduced by PCA, fitted on every image that subject saw
(fMRI only, no labels; for subjects 1, 2, 5 and 7 this includes the 128 test images). The
alignment is then fitted on the PCA scores of the 872 shared images:

- **SRM** (`fit_baselines.py --method srm`, `srm.py`): deterministic SRM (Chen et al.,
  2015) on 5,000 PCA components. It learns, for each subject, a 5000 × 32 matrix `W_i` with
  orthonormal columns, and one shared response `S` (872 × 32) minimising
  `Σ_i ‖X_i − S W_iᵀ‖²`, alternating the two updates until the objective converges (at
  most 200 iterations, relative tolerance 1e-6).
- **Procrustes** (`--method procrustes_a`, `procrustes.py`): generalised Procrustes
  analysis on 32 PCA components. A Procrustes transform is a rotation and keeps the
  dimensionality, so each subject is reduced to 32 components first ("variant A"); each
  subject's 32 × 32 rotation `W_i` is then fitted iteratively onto the mean of the rotated
  subjects until their summed squared distance to it converges (same limits).

A subject's fMRI `X` maps to the shared space as `z = (pca_i.transform(X) − μ_i) W_i`, where
`μ_i` is the subject's mean PCA score over the 872 images; the evaluation maps back with
`pca_j.inverse_transform(z W_jᵀ + μ_j)`, for reconstruction (j = i) and cross-subject
prediction (j ≠ i).

`fit_baselines.py` also offers options not used in the paper: `--method procrustes_b`
(rotate in `--n_pca_procb` = 200 PCA dimensions, then reduce the consensus to 32),
`--method all`, `--no_pca` (SRM on the raw vertices; saves no model), `--srm_init random`
and `--load_model <pkl>` (re-evaluate a saved model). `python fit_baselines.py --help`
lists all flags.

## Fitting

```bash
sbatch scripts/fit_baselines.sh      # from the repository root (without SLURM: bash scripts/fit_baselines.sh)
```

runs four fits in two stages:

```bash
cd baselines
# 1. standard fits: per-subject PCA, then the alignment from the standard start
python fit_baselines.py --method srm          --n_pca 5000 --n_components 32 --model_dir fitted_models
python fit_baselines.py --method procrustes_a --n_pca 32   --n_components 32 --model_dir fitted_models
# 2. refits from a random start that reuse those PCA bases: the paper's baselines
N_PCA=0 python fit_srm.py
python fit_proca.py
```

(The script also passes `--dataset streams`, `--filename` and `--output_dir`; see the script.)

1. `fit_baselines.py` loads the data with the MED-VAE data loader: the fMRI of all eight
   subjects and the category labels from `$CCN_DATA_ROOT/nsd/data/`, and the ANN features
   file (`--filename`, default `ann_features.npy`), which the baselines load but do not
   use. It fits the per-subject PCA and the alignment (SRM starting from the PCA scores of
   the cross-subject mean response, Procrustes from the data as they are), saves
   `fitted_models/<method>_shared872_32d_model.pkl`, and evaluates the fit on the 872 and on
   the 128. For `procrustes_a` the number of PCA components is `--n_components` (`--n_pca`
   is ignored). This evaluation writes to `--output_dir` (default `results_improved/`;
   the script uses `ccn_results_evaluation/`) and to `results/` in the working directory;
   neither the paper's evaluation nor the figures read these files.
2. `fit_srm.py` takes the PCA of `fitted_models/srm_shared872_32d_model.pkl` and refits SRM
   from a random shared response; `fit_proca.py` takes the PCA of
   `fitted_models/procrustes_a_shared872_32d_model.pkl` and refits Procrustes after a
   random rotation of each subject. Both are fixed to 32 components, print the component
   correlation and the Euclidean RSA on the 872 and on the 128, and save their model to
   `baselines/fitted_models/`, a fixed path (so stage 1 must save there too). With
   `N_PCA` > 0 in the environment, `fit_srm.py` would instead refit the PCA with that many
   components and save `srm_rndinit_npca<N_PCA>_32d_model.pkl`, hence `N_PCA=0`.

The random start changes the orientation in which the shared space ends up, not the quality
of the alignment (see the docstrings of `fit_srm.py` and `fit_proca.py`, and
[Rotation](#rotation)). The paper reports the random-start model of both methods, so both
orientations come from the same kind of start.

### Outputs

`fitted_models/` is not part of the repository (it is gitignored); the fits create it:

| File | Written by | Size | Used for |
|---|---|---|---|
| `srm_shared872_32d_model.pkl` | `fit_baselines.py --method srm` | 3.3 GB | its PCA, reused by `fit_srm.py` |
| `procrustes_a_shared872_32d_model.pkl` | `fit_baselines.py --method procrustes_a` | 22 MB | its PCA, reused by `fit_proca.py` |
| `srm_rndinit_shared872_32d_model.pkl` | `fit_srm.py` | 3.3 GB | the paper's SRM |
| `procrustes_a_rndinit_shared872_32d_model.pkl` | `fit_proca.py` | 22 MB | the paper's Procrustes |

Each file pickles a dict with the per-subject `W_list`, `pca_models` (fitted scikit-learn
PCA) and `training_means` (the `μ_i`), plus `method_name`, `n_subjects`, `n_components` and
`n_pca_features`. The two `rndinit` files are the defaults of `scripts/eval_medvae.sh`
(`evaluation/evaluate_methods.py --alignment_model_path`) and the files to pass to
`visualisation/visualise_latent.py --alignment_model`. The SRM files are large because they
hold the 5,000-component PCA bases.

### Determinism and resources

The fits are seeded. `fit_baselines.py --seed` (default 42) seeds Python, NumPy and PyTorch
(which fixes the order in which the shuffled training loader yields the images) and the PCA;
`fit_srm.py` and `fit_proca.py` seed NumPy and PyTorch with 42, and draw the random SRM
start with seed 42 and the random Procrustes rotations with seed 0 (`INIT_SEED`). The same
data and software environment therefore give the same models.

The fits run on the CPU: PyTorch only serves the data loader (a visible GPU just switches on
loader workers and pinned memory). The loader holds the fMRI of all eight subjects and the
ANN features in RAM. On our cluster the four fits, run one after the other, peaked at
~160 GiB of RAM and took ~45 min in total on 11 CPU cores; `scripts/fit_baselines.sh`
requests `--mem 192G`.

## Rotation

SRM and Procrustes determine the shared space only up to a rotation: rotating every `W_i`
(and SRM's shared response) by the same orthogonal matrix leaves their objectives
unchanged, and the rotation a fit ends in depends on where it starts. Component correlation
compares the shared dimensions one at a time, so for these models its value depends on that
arbitrary rotation: the standard and the random-start fits give different values for the
same alignment. Metrics computed from distances between images in the shared space do not
depend on it: Euclidean RSA, retrieval (Euclidean and cosine distances) and the silhouette
(Euclidean distances); neither do the cross-subject decoding (an L2-regularised logistic
regression), reconstruction and cross-subject prediction (`W_i W_jᵀ` is unchanged). The
paper's RSA builds its dissimilarities from 1 − Pearson correlation between images, which
removes each image's mean over the 32 dimensions, so it is not exactly rotation-invariant.

## Files

| File | Content |
|---|---|
| `fit_baselines.py` | stage 1: data, PCA, alignment, saved model, evaluation |
| `fit_srm.py`, `fit_proca.py` | stage 2: random-start refits on the stage-1 PCA |
| `srm.py` | `fit_detsrm` (deterministic SRM), `apply_alignment` |
| `procrustes.py` | `fit_gpa`, `align_procrustes` (variants A and B), `apply_alignment` |
| `reduction.py` | `reduce_subjects` (per-subject PCA), `IdentityPCA` (for `--no_pca`) |
| `data_processing.py` | extracts each subject's images, the 872 shared images and the partially shared images from the data loader |
| `recon.py` | self-reconstruction on the 128 with `μ_i` added back before the inverse PCA (printed by `fit_baselines.py`) |
| `model_io.py` | `save_alignment_model`, which writes the model files |
