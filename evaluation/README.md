# `evaluation/`: evaluating MED-VAE, SRM and Procrustes

`evaluate_methods.py` evaluates one method per run on the held-out test images and writes
all its metrics (Fig. 2B/C, Fig. 3, Table 1 and more) to one results file. The other
modules hold the metric code, shared with `baselines/`.

## Running the evaluator

The paper's calls (`scripts/eval_medvae.sh` runs them, one per `METHOD=vae|srm|procrustes`):

```bash
cd evaluation
python evaluate_methods.py --method vae --vae_checkpoint $CCN_RESULTS_DIR/<medvae checkpoint>.pt \
    --ann_activations rn50_streams \
    --n_dims 32 --subjects 1 2 5 7 --mode not_all8 --test_size 1.0 --output_dir $CCN_RESULTS_DIR
python evaluate_methods.py --method srm \
    --alignment_model_path ../baselines/fitted_models/srm_rndinit_shared872_32d_model.pkl \
    --n_dims 32 --subjects 1 2 5 7 --mode not_all8 --test_size 1.0 --output_dir $CCN_RESULTS_DIR
python evaluate_methods.py --method procrustes \
    --alignment_model_path ../baselines/fitted_models/procrustes_a_rndinit_shared872_32d_model.pkl \
    --n_dims 32 --subjects 1 2 5 7 --mode not_all8 --test_size 1.0 --output_dir $CCN_RESULTS_DIR
```

| Argument | Paper | Notes |
|---|---|---|
| `--method` | `vae`, `srm`, `procrustes` | one method per run |
| `--vae_checkpoint` | the trained MED-VAE `.pt` | default: `VAE_CHECKPOINT_TEMPLATE` in `RESULTS_DIR` (`ccn_config.py`) |
| `--alignment_model_path` | `{srm,procrustes_a}_rndinit_shared872_32d_model.pkl` | a `.pkl` from `baselines/`, or its directory: then this method's random-init refit is preferred over the standard fit, and the file used is printed |
| `--n_dims` | `32` | default |
| `--subjects` | `1 2 5 7` | default |
| `--mode` | `not_all8` | default; see [Test images](#test-images) |
| `--test_size` | `1.0` | default; below 1 a random subset (seed 42) of the candidate images is kept, i.e. a different test set |
| `--ann_activations` | `rn50_streams` | MED-VAE only: the ANN features of the ANN-pathway metrics, a key of `ccn_config.ANN_ACTIVATIONS` or a `.npy` path; default: inferred from the checkpoint |
| `--output_dir` | `$CCN_RESULTS_DIR` | default: `ccn_config.RESULTS_DIR` (`$CCN_RESULTS_DIR`, else `medvae/results`) |
| `--silhouette_reduce` | `true` | default: 10-d UMAP before the silhouette |
| `--seed` | `42` | default |

## Test images

The data files have one row per NSD image. The candidate images are those with fMRI for
every subject in `--subjects`; for subjects 1, 2, 5 and 7 these are the 1,000 NSD shared
images. `--mode` then selects (counts for `--subjects 1 2 5 7`):

- `not_all8` (paper): the **128** candidates that not all of subjects 3, 4, 6 and 8 saw,
  i.e. the images seen by subjects 1, 2, 5 and 7 but not by all eight subjects;
- `all_common`: all 1,000 candidates;
- `shared_only`: the 872 candidates all eight subjects saw (the baselines' fit images);
- `exclusive`: the candidates none of subjects 3, 4, 6 and 8 saw;
- `vae_test`: legacy, not supported in this release (it needs a feature file that
  `data_preparation/` does not build).

Subjects 3, 4, 6 and 8 are fixed in the code, so `not_all8`, `shared_only` and `exclusive`
read the fMRI files of all eight subjects; the evaluator stops with an error if one cannot
be loaded. MED-VAE never saw the 128 test images; the baselines' alignment was fitted on
the 872 images all eight subjects saw, while their unsupervised per-subject PCA used all of
a subject's images (see `baselines/README.md`).

## Metrics

Each run writes one dict to `alignment_eval_<method>_<n_dims>d_<mode>_streams.pkl`. The
latents are the MED-VAE encoder means, or for SRM/Procrustes the shared-space projections
(PCA, minus the training mean, times the subject's transform).

| Metric | Computation | Keys | Paper |
|---|---|---|---|
| Component correlation | per latent dimension, Pearson r across images between two subjects' latents, averaged over subject pairs | `per_component_corr`, `avg_comp_corr`, `component_corr_matrix` | Fig. 2C |
| RSA | per subject, an image × image dissimilarity matrix (1 − Pearson r of the latents); Pearson r between subjects' matrices, averaged over pairs (also a Euclidean/Spearman variant) | `rsa_matrix_pearson`, `avg_rsa_pearson`, `avg_rsa_euclidean` | Fig. 2C |
| Cross-subject fMRI prediction | subject i's fMRI encoded and decoded into subject j's vertices (MED-VAE: j's decoder; baselines: j's inverse transform and PCA); Pearson r across images per vertex, averaged over vertices | `fmri_prediction_matrix` (row = source, column = target; diagonal = self-reconstruction), `fmri_prediction_matrix_image` | Fig. 3 (off-diagonal pairs) |
| Within-trial reconstruction | each subject's fMRI reconstructed through its own encoder and decoder; vertex correlation as above, R² | `recon_per_subject`, `recon_avg_voxel_corr`, `recon_avg_r2` | Fig. 2C |
| Cross-trial reconstruction (optional data) | for images with ≥ 2 single trials, one trial reconstructed and correlated with another trial of the same image; ceiling = the two measured trials' correlation | `cross_trial` (`cross_trial_per_subject[s]['mean_voxel_corr']`, ...) | Fig. 2C |
| Retrieval | for each ordered subject pair, % of images whose latent in one subject has the same image's latent in the other among its k nearest neighbours (Euclidean and cosine distance; protocol below) | `retrieval['euclidean']['mean'][k]`, k = 1, 2, 3, 5, 10; `per_pair` | Table 1 (Euclidean) |
| Category decoding | leave one subject out: one logistic regression per category (balanced class weights) trained on the other subjects' latents, tested on the left-out subject; exact match (all labels right) and balanced accuracy (mean over labels) | `decoding['per_subject_exact']`, `decoding['per_subject_balanced']` | Fig. 2B |
| Silhouette | multi-label silhouette of the 12 category labels in a 10-d UMAP of the latents, per subject and for all subjects pooled | `silhouette['silhouette_combined']`, `silhouette['silhouette_per_subject']` | Fig. 2B |
| ANN pathway (MED-VAE only) | ANN → fMRI prediction, ANN → ANN and fMRI → ANN reconstruction | `ann2fmri_*`, `ann2ann_*`, `fmri2ann_*` | appendix |

Fig. 2B's silhouette bars and bootstrap CIs are recomputed by `visualisation/stat_tests.py`
from `silh_latents_*.pkl`. The silhouette needs `umap-learn` (pinned environment): without
it the evaluator falls back to PCA with a warning, and the values do not match the paper.

## Outputs

In `--output_dir`:

- `alignment_eval_<method>_<n_dims>d_<mode>_streams.pkl`: all metrics above;
- `silh_latents_<method>_<n_dims>d_<mode>_streams.pkl`: the per-subject test latents and
  labels behind the silhouette (`$DUMP_SILH_LATENTS` overrides this path);
- `other/within_{exact,balanced}_<method>_<n_dims>d_<mode>_streams.pkl`: decoding accuracy on
  each fold's training subjects;
- `results/{cross_subj_prediction,alignment_metrics,retrieval}_<method>_<n_dims>d_<mode>_streams.csv`:
  CSV summaries.

## Optional inputs

Two metrics need data that `data_preparation/` does not build. Without it the evaluator
prints that the metric is skipped; all other metrics are unaffected.

- **Cross-trial reconstruction**: single-trial responses
  `$CCN_DATA_ROOT/vae/cross_trial_data/fmri_subject{01..08}_trials.npz`
  (`CROSS_TRIAL_SUBPATH`, `FMRI_TRIAL_FILE_TEMPLATE`), with `fmri_data` (one row per trial,
  same vertices as the main files) and `image_ids` (the image of each row, consistent
  across subjects). `not_all8` needs all eight subjects' files: the images are selected
  from them by the test-set rule, then those with at least two trials are kept. Without
  them `cross_trial` is `None`.
- **Noise-ceiling normalisation**: per-vertex noise ceilings (fraction of variance) of each
  evaluated subject,
  `$CCN_DATA_ROOT/nsd_classification/data/subjXX/ncsnr_processed/subjXX_noiseceiling.npy`
  (`NOISE_CEILING_DIR`, `NOISE_CEILING_TEMPLATE`). They add the `*nc_norm*` keys.

## Bundled paper outputs (`results/`)

`results/` holds the evaluation of the paper's models (the MED-VAE checkpoint and the
random-init SRM and Procrustes models) on the 128 test images:
`alignment_eval_{vae,srm,procrustes}_32d_not_all8_streams.pkl`, the matching
`silh_latents_*.pkl` and `other/within_*.pkl`. The scripts in `visualisation/` regenerate
the figures, tables and statistics from them without any data (top-level README, Quick
start). The evaluator writes there only if `--output_dir` (default `$CCN_RESULTS_DIR`)
points there.

The MED-VAE file was written by an earlier version of the evaluator: it stores the
component-correlation matrix under the older key `prediction_matrix` (the visualisation
scripts read both names), and its stored silhouette (0.306) was computed in a different
software environment. The figure's silhouette (0.34) is the one `visualisation/stat_tests.py`
recomputes from `silh_latents_vae_32d_not_all8_streams.pkl`, so the warning it prints about
the stored value is expected.

## Modules

| Module | Role |
|---|---|
| `evaluate_methods.py` | the evaluator |
| `evaluation_pipeline.py` | shared orchestration: `compute_alignment_metrics`, `compute_cross_subject_retrieval` (used by `evaluate_methods.py`), `evaluate_aligned_latents`, `comprehensive_evaluation` (used by `baselines/fit_baselines.py`) |
| `metrics.py` | metric functions: alignment (`cross_subj_metrics`), decoding, silhouette (with the UMAP reduction), reconstruction helpers |
| `retrieval.py` | cross-subject retrieval with the fixed 128-image gallery |
| `prediction.py` | cross-subject prediction and training means for the PCA-based models, used by `baselines/` |
| `savemetrics.py` | pickles the alignment metrics |

**Retrieval protocol (`retrieval.py`).** A fixed **128-image gallery**, so difficulty is
comparable across test sets: a set of ≤ 128 images (e.g. `not_all8`) is used whole in one
deterministic pass; a larger one is subsampled to 128 images × 30 repetitions and averaged.
The seed fixes the subsets, so all methods draw the same galleries. Chance is 1/128 ≈ 0.78 %
(top-1).
