# MED-VAE: cross-subject fMRI latent alignment

Code for the paper "Task-guided cross-subject latent alignment: a multi-encoder-decoder VAE", Angeliki Papathanasiou, Jascha Achterberg, Thomas E. Nichols, Rui Ponte Costa. In Proceedings of the 9th Conference on Cognitive Computational Neuroscience, New York, NY, USA, 2026.

https://arxiv.org/abs/2606.15989

MED-VAE learns a shared latent space across subjects
from fMRI by jointly training per-subject encoders/decoders and a shared 
(ANN) encoder. This repository reproduces the cross-subject **alignment**,
**category-encoding**, and **reconstruction** results for MED-VAE and the two
linear baselines (SRM, Procrustes).

```
medvae_release/
├── ccn_config.py        paths/config (set CCN_DATA_ROOT, CCN_RESULTS_DIR)
├── requirements.txt, environment.yml
├── medvae/              MED-VAE model + training pipeline   (main.py = entry)
├── baselines/           SRM / Procrustes fitting            (fit_*.py)
├── evaluation/          unified evaluator + metrics         (evaluate_methods.py)
├── visualisation/       figure pipeline                     (render_figure.py)
└── scripts/             SLURM submission wrappers
```

## Data

The pipeline expects, under `$CCN_DATA_ROOT/nsd/`:
- per-subject fMRI:        `data/fmri_subject{01..08}_streams_overl_NEW.npz`
- ANN (ResNet-50) features: `data/aligned_all_activations_fair_resnet50_hendrycks.npy`
- multi-label labels:       `labels_all_aligned.npy`

> **TODO (to be added):** instructions for generating the ANN activations and the
> fMRI (`streams`) data from the source NSD dataset.

## 1. Setup

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
export CCN_DATA_ROOT=/path/to/data        # dir containing nsd/ (fMRI npz + ANN activations)
export CCN_RESULTS_DIR=/path/to/results   # where checkpoints + eval outputs are written
```
All paths derive from these two variables (see `ccn_config.py`); the cluster
wrappers in `scripts/` also read `CONDA_ROOT`/`CONDA_ENV` (edit `scripts/_env.sh`).

> **Environment note.** The pipeline needs a working UMAP for the silhouette metric,
> which requires `numpy <= 2.2` with a compatible `numba`/`umap-learn`
> (e.g. numpy 2.2, numba 0.61, umap-learn 0.5.7). `environment.yml` pins these — a
> too-new numpy (≥ 2.4) breaks numba/UMAP and the silhouette silently falls back to PCA.

## 2. Train MED-VAE

```bash
sbatch scripts/train_medvae.sh            # wraps medvae/main.py --train ...
```
Writes a checkpoint `medvae_*.pt` to `$CCN_RESULTS_DIR`. (See `medvae/args.py` for
all hyper-parameters; `scripts/train_medvae.sh` sets the paper configuration.)

## 3. Fit the baselines (SRM, Procrustes)

The reported baselines are the **random-initialised** variants, fit in two stages:
first the standard models (which build the shared PCA bases), then the random-init
refits that reuse those PCAs. All models are written to `baselines/fitted_models/`.
Every fit is deterministic (seed 42).

```bash
cd baselines
ANN=aligned_all_activations_fair_resnet50_hendrycks.npy   # ANN features file (under $CCN_DATA_ROOT/nsd/data/)
# (a) standard models — build the PCA bases on the 872 all-subject-shared images
python fit_baselines.py --method srm          --filename $ANN --n_pca 5000 --n_components 32 --model_dir fitted_models
python fit_baselines.py --method procrustes_a --filename $ANN --n_pca 32   --n_components 32 --model_dir fitted_models
# (b) random-init refits — reuse the PCA from the standard models (the paper baselines)
N_PCA=0 python fit_srm.py     # -> fitted_models/srm_rndinit_shared872_32d_model.pkl
python fit_proca.py           # -> fitted_models/procrustes_a_rndinit_shared872_32d_model.pkl
```
Each model stores `W_list`, `pca_models`, `training_means`. Note that component correlation
is rotation-arbitrary for the linear baselines; RSA, silhouette, and decoding are the
rotation-invariant measures.

## 4. Evaluate all three methods

Run the unified evaluator once per method — this produces the files the figures read,
`alignment_eval_<method>_32d_not_all8_streams.pkl` (+ `silh_latents_*`), in
`evaluation/results/`:

```bash
METHOD=vae        CKPT=$CCN_RESULTS_DIR/medvae_<...>.pt                          sbatch scripts/eval_medvae.sh
METHOD=srm        MODEL=baselines/fitted_models/srm_rndinit_shared872_32d_model.pkl        sbatch scripts/eval_medvae.sh
METHOD=procrustes MODEL=baselines/fitted_models/procrustes_a_rndinit_shared872_32d_model.pkl sbatch scripts/eval_medvae.sh
```
(Direct call: `python evaluation/evaluate_methods.py --method <m> --mode not_all8
--dataset streams --subjects 1 2 5 7 --test_size 1.0 [--vae_checkpoint|--alignment_model_path] ...`.)
Each pkl holds the alignment (component correlation, RSA), reconstruction
(within/cross-trial voxel correlation), decoding, silhouette, and the 4×4
cross-subject `fmri_prediction_matrix`.

## 5. Make the figure (Panels A–C)

Regenerating the panels from scratch needs the trained MED-VAE checkpoint (§2),
the fitted baseline models (§3), and the eval outputs (§4). To just reproduce the
**published** figure without re-running anything, skip to the note at the end of
this section — the required outputs are bundled.

```bash
# Panel A — UMAP of the shared latents, one run per method
python visualisation/visualise_latent.py --method vae \
    --vae_checkpoint $CCN_RESULTS_DIR/medvae_<...>.pt \
    --out visualisation/data/encoder_data_2d_VAE.pkl
python visualisation/visualise_latent.py --method srm \
    --vae_checkpoint <...>.pt --alignment_model baselines/fitted_models/srm_rndinit_shared872_32d_model.pkl \
    --out visualisation/data/encoder_data_2d_srm.pkl
python visualisation/visualise_latent.py --method procrustes \
    --vae_checkpoint <...>.pt --alignment_model baselines/fitted_models/procrustes_a_rndinit_shared872_32d_model.pkl \
    --out visualisation/data/encoder_data_2d_procrustes.pkl

# Significance tests -> visualisation/data/figure_stats.json  (needs working umap/sklearn)
python visualisation/stat_tests.py

# Render Panels A, B, C -> visualisation/figure_abc.{eps,pdf,png}
python visualisation/render_figure.py
```

**Reproduce the figure directly:** the eval outputs and `figure_stats.json` for the
paper are bundled (`evaluation/results/`, `visualisation/data/`), so
`python visualisation/render_figure.py` renders the published figure without
re-running the pipeline.

## Figure contents
- **Panel A** — UMAP of the shared latent space (MED-VAE / SRM / Procrustes), coloured by stimulus category.
- **Panel B** — combined multi-label silhouette (bar + bootstrap 95% CI) and leave-one-subject-out decoding (exact-match, balanced).
- **Panel C** — alignment (component correlation, RSA) and reconstruction (within/cross-trial voxel correlation).

Statistics: six metrics by per-subject paired t-tests (n=4) with Benjamini–Hochberg
FDR within each pairwise contrast; the pooled silhouette by a paired image bootstrap
(B=5000). See `visualisation/stat_tests.py`.
