# MED-VAE: cross-subject fMRI latent alignment

Code for the paper "Task-guided cross-subject latent alignment: a multi-encoder-decoder VAE", Angeliki Papathanasiou, Jascha Achterberg, Thomas E. Nichols, Rui Ponte Costa. In Proceedings of the 9th Conference on Cognitive Computational Neuroscience, New York, NY, USA, 2026.

https://arxiv.org/abs/2606.15989



https://github.com/user-attachments/assets/4362f53d-d7a1-467b-a4e1-e2346b21e86c



MED-VAE learns a shared latent space across subjects from fMRI by jointly training
per-subject fMRI encoders/decoders and a shared encoder/decoder for the activations of
an ImageNet-trained ResNet-50 (the "ANN scaffold"). Subjects never need to have seen the
same images: the scaffold supplies the correspondence.

This repository reproduces the paper's main comparison of MED-VAE with the two linear
baselines (SRM and Procrustes), from the public data to the figures:

| Paper result | Produced by |
|---|---|
| Fig. 2A: latent spaces, coloured by category | `visualisation/visualise_latent.py` → `render_figure.py` |
| Fig. 2B: silhouette, leave-one-subject-out category decoding | `evaluation/evaluate_methods.py` → `stat_tests.py` → `render_figure.py` |
| Fig. 2C: component correlation, RSA, within-/cross-trial reconstruction | `evaluate_methods.py` → `stat_tests.py` → `render_figure.py` |
| Fig. 3: cross-subject neural prediction (12 subject pairs) | `evaluate_methods.py` → `visualisation/render_figure3.py` |
| Table 1: cross-subject retrieval top-k | `evaluate_methods.py` → `visualisation/render_table1.py` |
| Statistics in §5.1–5.3 (paired t-tests, bootstrap CIs) | `visualisation/stat_tests.py` |

The evaluation outputs of the paper's models are bundled, so the figures and tables can be
regenerated in a minute without any data ([Quick start](#quick-start-the-published-figures-from-the-bundled-results)).
The rest of this README describes the full pipeline from scratch. What is not covered is
listed at the [end](#not-included).

```
MEDVAE_NSD/
├── ccn_config.py          all paths/file names (set CCN_DATA_ROOT, CCN_RESULTS_DIR)
├── requirements.txt, environment.yml
├── data_preparation/      build the four data files from the public downloads
├── medvae/                MED-VAE model and training            (main.py)
├── baselines/             SRM / Procrustes fitting               (fit_*.py)
├── evaluation/            unified evaluator + metrics            (evaluate_methods.py)
│   └── results/           bundled evaluation outputs of the paper's models
├── visualisation/         figures, tables and statistics         (render_*.py, stat_tests.py)
│   └── data/              bundled Panel A data + figure statistics
├── scripts/               SLURM job scripts (train, fit baselines, evaluate)
└── logs/                  SLURM logs
```

## Quick start: the published figures from the bundled results

```bash
git clone https://github.com/pangelu9/MEDVAE_NSD.git && cd MEDVAE_NSD
pip install -r requirements.txt            # see "Software" below for the exact environment

python visualisation/render_figure.py  --results evaluation/results --data_dir visualisation/data \
                                       --out figure_abc.eps                                          # Fig. 2 A–C
python visualisation/render_figure3.py --results evaluation/results --out figure3_crosssubj.eps     # Fig. 3
python visualisation/render_table1.py  --results evaluation/results                                  # Table 1
python visualisation/stat_tests.py     --results evaluation/results --out figure_stats_check.json    # §5.1–5.3 statistics
```

The two figures are written as `.eps`, `.pdf` and `.png` to the current directory
(compare with the bundled `visualisation/figure_abc.pdf`). `stat_tests.py` (5–15 min) prints
every test of Fig. 2B/C and writes the silhouette bars and significance brackets that
`render_figure.py` draws; the paper's are bundled in `visualisation/data/figure_stats.json`.
`evaluation/results/` holds one `alignment_eval_<method>_32d_not_all8_streams.pkl` per
method with every metric, plus the latents the silhouette bootstrap uses.

## Full pipeline from scratch

| Step | Section | Hardware | Time (measured on our cluster) | Peak RAM |
|---|---|---|---|---|
| Download the data | [1](#1-data) | internet, ~60 GB disk | depends on bandwidth | – |
| Build the data files | [1](#1-data) | CPU; GPU for the ANN features | ~25 min CPU + GPU time for the features | < 64 GB |
| Train MED-VAE | [2](#2-train-med-vae) | 1 GPU | ~1 h 15 min on an A100 | ~125 GiB |
| Fit SRM and Procrustes | [3](#3-fit-the-baselines) | CPU | ~45 min for all four fits (11 cores) | ~160 GiB |
| Evaluate the three methods | [4](#4-evaluate) | 1 GPU | 7–11 min per method | ~61 GiB |
| Panel A latents | [5](#5-figures-tables-and-statistics) | 1 GPU (optional) | 10–55 min per method | ~150 GB |
| Statistics, figures, tables | [5](#5-figures-tables-and-statistics) | CPU | 5–15 min, then ~1 min per figure | ~16 GB |

Host memory, not GPU memory, is the constraint: the model has ~0.9 GB of parameters, but
training and evaluation hold the fMRI of all eight subjects and the 14 GB ANN feature
matrix in RAM.

### Software

The release was verified end to end with Python 3.13 and the package versions pinned in
`requirements.txt` (PyTorch 2.7.0, NumPy 2.2.5, scikit-learn 1.6.1, umap-learn 0.5.7, …).
Either

```bash
conda env create -f environment.yml && conda activate medvae
```

or

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Keep NumPy at 2.2: the silhouette metric uses UMAP, whose numba dependency does not support
newer NumPy, and the bundled result files need NumPy ≥ 2 to load.

### Paths

All paths derive from two environment variables (see `ccn_config.py`); set both, to
absolute paths, before every step:

```bash
export CCN_DATA_ROOT=/path/to/data        # the data files live in $CCN_DATA_ROOT/nsd/data/
export CCN_RESULTS_DIR=/path/to/results   # checkpoints, evaluation outputs, figures
```

Unless a block below changes directory, run its commands from the repository root.

### Running the job scripts

`scripts/` holds SLURM scripts for the three heavy steps (`train_medvae.sh`,
`fit_baselines.sh`, `eval_medvae.sh`). They source `scripts/_env.sh`, which

- activates a conda env only if you set `CONDA_ROOT` (and optionally `CONDA_ENV`,
  default `medvae`); otherwise it uses the `python3` on your `PATH`, e.g. your activated
  venv, since `sbatch` passes your environment to the job;
- uses `CCN_DATA_ROOT` and `CCN_RESULTS_DIR` (default `<repo>/medvae/results`).

Submit them **from the repository root** (the log files go to `logs/`). The `#SBATCH`
lines only request resources; add your cluster's account, partition or QoS on the
command line, e.g. `sbatch -A <account> -p <partition> scripts/train_medvae.sh`. Without
SLURM, run them with `bash` on a machine with enough memory, e.g. `bash scripts/train_medvae.sh`.
Each script documents its settings and environment-variable overrides in its header.

## 1. Data

All data are public and derived from the Natural Scenes Dataset (NSD; Allen et al., 2022).
The pipeline reads four files from `$CCN_DATA_ROOT/nsd/data/`, all with one row per NSD
image, in the same order:

| File | Content |
|---|---|
| `fmri_subject{01..08}.npz` | `fmri_data`: (70502, ~20,730) float32 responses of one subject (NaN for images it did not see) |
| `ann_features.npy` | (70502, 51048) float32 ResNet-50 features (the ANN scaffold) |
| `category_labels.npy` | (70502, 12) float32 multi-hot COCO super-categories |
| `nsd_ids.npy` | (70502,) NSD image id (0-based) of each row |

The scripts in `data_preparation/` build them from three public sources. We rebuilt the
files from the public downloads with these scripts: the index, the labels and all eight
fMRI files are identical to the ones used for the paper.

### Download

1. **fMRI — Algonauts 2023 challenge data** (Gifford et al., 2023). Download the
   `train_data/` and `test_data/` folders of the
   [challenge data folder](https://drive.google.com/drive/folders/17RyBAnvDhrrt18Js2VZqSVi_nZ7bn3G3)
   (8 `subjXX.zip` each, ~36 GB in total; see also the
   [challenge devkit](https://github.com/gifale95/algonauts_2023)). The archives in the two
   folders have the same names, so keep the folders apart, then unzip all 16 into one
   directory (`-o`: the test images are in both archives of a subject, identical):
   ```bash
   mkdir -p /path/to/algonauts_2023_challenge_data
   ( cd /path/to/algonauts_2023_challenge_data &&
     for z in /path/to/train_data/subj0*.zip /path/to/test_data/subj0*.zip; do
         UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -o "$z"
     done )
   ```
   (Some builds of `unzip` wrongly reject the large train archives of subjects 1, 2 and 7
   as a "possible zip bomb"; the variable switches that check off.)
   This gives `subjXX/{roi_masks,training_split,test_split}`.
   The responses are NSD single-trial betas on the fsaverage surface, z-scored within
   each scan session and averaged over the three repeats of each image. The test-split
   fMRI is needed too: the images seen by several subjects are spread over both splits.
2. **Images for the ANN features — MindEye2** (Scotti et al., 2024): the 73,000 NSD
   stimuli as one HDF5 file (22 GB; SHA-256
   `778fabfa5cc1e08896b8af3541279f8bc11b3f407d7372956622cffc2bf3f270`),
   ```bash
   hf download pscotti/mindeyev2 coco_images_224_float16.hdf5 --repo-type dataset --local-dir /path/to/mindeyev2
   ```
3. **Category labels — NSD-Flat** (HuggingFace `clane9/NSD-Flat`): read directly by
   `fetch_labels.py` below (only the label columns are downloaded, ~10 min).

Use of the data is subject to the [NSD Terms and Conditions](https://cvnlab.slite.com/api/s/note/9dgh5HCqgZYhMoAESZBS86/Terms-and-Conditions);
please also cite the Algonauts 2023, MindEye2 and NSD-Flat sources.

### Build

```bash
cd data_preparation
OUT=$CCN_DATA_ROOT/nsd/data
ALG=/path/to/algonauts_2023_challenge_data

python fetch_labels.py           --out $OUT/nsd_flat_labels.json
python build_index_and_labels.py --algonauts_dir $ALG --labels_json $OUT/nsd_flat_labels.json --out_dir $OUT
python build_fmri.py             --algonauts_dir $ALG --nsd_ids $OUT/nsd_ids.npy --out_dir $OUT
python extract_ann_features.py   --images_hdf5 /path/to/mindeyev2/coco_images_224_float16.hdf5 \
                                 --nsd_ids $OUT/nsd_ids.npy --out $OUT/ann_features.npy   # GPU recommended
cd ..
```

`fetch_labels.py` and the first run of `extract_ann_features.py` (which downloads the
torchvision ResNet-50 weights, ~100 MB) need internet access. If your compute nodes have
none, run `fetch_labels.py` on a login node and fetch the weights there once:
`python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"`.
`build_fmri.py` takes ~10 min and < 64 GB of RAM.

What the steps do:
- **Image index** (`build_index_and_labels.py`): all NSD images any subject saw in the
  Algonauts training or test split (70,566), minus the 64 without COCO annotations, gives
  70,502 rows. The rows are sorted by NSD id **as strings** ('0', '1', '10', '100', ...),
  the order of the paper's files. The 12 label columns are alphabetical (accessory,
  animal, appliance, electronic, food, furniture, indoor, kitchen, outdoor, person,
  sports, vehicle).
- **fMRI** (`build_fmri.py`): left then right hemisphere, restricted to the `streams`
  ROIs midventral, midlateral, ventral and lateral (labels 2, 3, 5, 6), giving 20,726–20,736
  vertices per subject. No further normalisation.
- **ANN features** (`extract_ann_features.py`): ImageNet-trained ResNet-50
  (torchvision `IMAGENET1K_V1`), every 8th Conv2d and ReLU layer plus the global average
  pool. Each of the 14 Conv2d/ReLU layers is reduced to 3,500 dimensions by a fixed sparse
  random projection (seed 42) and the 2,048-d pool output is kept: 14 × 3,500 + 2,048 =
  51,048 features. The training code z-scores each column.

If your files are named differently, don't edit the tracked code: map the generic names
above to your filenames in an untracked override, `ccn_config_local.py` next to
`ccn_config.py` (reassign the names listed at the top of `ccn_config.py`), and, for the job
scripts, `scripts/_env.local.sh` (export `CCN_ANN_FEATURES`, `CCN_DATA_ROOT`, `CONDA_ROOT`, …).
Both are gitignored and picked up automatically.

## 2. Train MED-VAE

```bash
sbatch scripts/train_medvae.sh
```

This trains the paper's model: eight subject-specific fMRI encoders/decoders plus the ANN
encoder/decoder, 32-d latent space, hidden size 256, scaffold weight w = 5, KL weight
β = 1, 30 epochs, batch size 128, learning rate 1e-4. The 1,000 images seen by several
subjects are left out of training: their fMRI is removed (`--remove_all_overlaps`) and
images without fMRI are dropped, so MED-VAE never sees the test images. Every other image
was seen by one subject only, i.e. no two subjects share a training image. All settings
are in `scripts/train_medvae.sh`; the defaults in `medvae/args.py` are not the paper's
settings, so call `medvae/main.py` through the script (or copy its arguments).

Output in `$CCN_RESULTS_DIR`: `medvae_ALLsubjects_RN50_streams_rvoverl_all_nnw5_annw5_32_b1.0.pt`
(3.8 GB) and `medvae_ALLsubjects_RN50_streams_rvoverl_all_nnw5_annw5_32_1.0_losses.csv`.
The seed is fixed (1); a retrained model still differs slightly from ours (see
[expected results](#expected-results)).

## 3. Fit the baselines

```bash
sbatch scripts/fit_baselines.sh
```

The paper's baselines are SRM and (generalised) Procrustes. Each subject's fMRI is first
reduced by PCA (fitted on all images the subject saw, without labels); the alignment
between subjects is then fitted on the 872 images that all eight subjects saw. The script
runs two stages (details in `baselines/README.md`):

```bash
cd baselines
# (a) standard fits: per-subject PCA, then SRM (5,000 PCA components) / Procrustes (32)
python fit_baselines.py --method srm          --n_pca 5000 --n_components 32 --model_dir fitted_models
python fit_baselines.py --method procrustes_a --n_pca 32   --n_components 32 --model_dir fitted_models
# (b) refits from a random start that reuse those PCA bases: the paper's baselines
N_PCA=0 python fit_srm.py      # -> fitted_models/srm_rndinit_shared872_32d_model.pkl
python fit_proca.py            # -> fitted_models/procrustes_a_rndinit_shared872_32d_model.pkl
cd ..
```

The fits are deterministic (seeded) and run on the CPU. SRM and Procrustes fix their
shared space only up to a rotation, so their component correlation depends on where the
fit starts; see `baselines/README.md` for which metrics do not.

## 4. Evaluate

```bash
METHOD=vae        sbatch scripts/eval_medvae.sh
METHOD=srm        sbatch scripts/eval_medvae.sh
METHOD=procrustes sbatch scripts/eval_medvae.sh
```

By default the script evaluates the checkpoint from step 2 and the random-initialised
baselines from step 3; set `CKPT=...` or `MODEL=...` to evaluate other files. All methods
are evaluated on the paper's test set: the 128 of the 1,000 shared NSD images that
subjects 1, 2, 5 and 7 all saw but not all eight subjects did (subjects 3, 4, 6 and 8
did not complete all sessions). MED-VAE never saw these images, and the baselines'
alignment was fitted on the other 872 (their unsupervised per-subject PCA used all of a
subject's images). The equivalent direct call is

```bash
cd evaluation
python evaluate_methods.py --method vae --vae_checkpoint $CCN_RESULTS_DIR/medvae_ALLsubjects_RN50_streams_rvoverl_all_nnw5_annw5_32_b1.0.pt \
    --ann_activations rn50_streams --dataset streams --n_dims 32 --mode not_all8 \
    --subjects 1 2 5 7 --test_size 1.0 --output_dir $CCN_RESULTS_DIR
# baselines: --method srm|procrustes --alignment_model_path ../baselines/fitted_models/<model>.pkl
cd ..
```

Each run writes, to `$CCN_RESULTS_DIR`, `alignment_eval_<method>_32d_not_all8_streams.pkl`
(all metrics: component correlation, RSA, the 4 × 4 cross-subject prediction matrix,
within- and cross-trial reconstruction, retrieval, decoding, silhouette) and
`silh_latents_<method>_32d_not_all8_streams.pkl` (input of the silhouette bootstrap), plus
CSV summaries in `$CCN_RESULTS_DIR/results/` and per-fold decoding results in
`$CCN_RESULTS_DIR/other/`. The evaluator needs the fMRI files of all eight subjects
(subjects 3, 4, 6 and 8 define the test set). See `evaluation/README.md` for the metrics.

Two metrics need data this pipeline does not build; the evaluator skips them with a
message when the files are absent:
- **cross-trial reconstruction** (Fig. 2C, last group) needs NSD's single-trial betas;
- **noise-ceiling-normalised reconstruction** (§5.3 and the appendix) needs per-vertex
  noise ceilings.

Without the single-trial data, Fig. 2C has no cross-trial group, and the
Benjamini–Hochberg q-values printed by `stat_tests.py` cover five metrics instead of six
(the figure's stars use uncorrected p and are unaffected).

## 5. Figures, tables and statistics

```bash
export FIG=$CCN_RESULTS_DIR/figure_data
export CKPT=$CCN_RESULTS_DIR/medvae_ALLsubjects_RN50_streams_rvoverl_all_nnw5_annw5_32_b1.0.pt
export FITTED=$PWD/baselines/fitted_models

# Panel A: 2-D UMAP of a label-balanced set of up to 3,000 held-out validation images
# (the same images for all methods; each run needs ~150 GB of RAM)
python visualisation/visualise_latent.py --method vae        --vae_checkpoint $CKPT --out $FIG/encoder_data_2d_VAE.pkl
python visualisation/visualise_latent.py --method srm        --vae_checkpoint $CKPT \
    --alignment_model $FITTED/srm_rndinit_shared872_32d_model.pkl          --out $FIG/encoder_data_2d_srm.pkl
python visualisation/visualise_latent.py --method procrustes --vae_checkpoint $CKPT \
    --alignment_model $FITTED/procrustes_a_rndinit_shared872_32d_model.pkl --out $FIG/encoder_data_2d_procrustes.pkl

# Statistics (prints all tests; writes the significance brackets + silhouette bars)
python visualisation/stat_tests.py --results $CCN_RESULTS_DIR --out $FIG/figure_stats.json

# Fig. 2 A–C, Fig. 3, Table 1
python visualisation/render_figure.py  --results $CCN_RESULTS_DIR --data_dir $FIG --out $CCN_RESULTS_DIR/figure_abc.eps
python visualisation/render_figure3.py --results $CCN_RESULTS_DIR --out $CCN_RESULTS_DIR/figure3_crosssubj.eps
python visualisation/render_table1.py  --results $CCN_RESULTS_DIR
```

`visualise_latent.py` needs the MED-VAE checkpoint for every method: it rebuilds the
data loader from the checkpoint's configuration. Run these three commands on a node with
enough memory, e.g. as jobs (the double quotes expand the variables when you submit):

```bash
sbatch --gres=gpu:1 --mem=160G -o logs/panelA-%j.out \
    --wrap "python visualisation/visualise_latent.py --method vae --vae_checkpoint $CKPT --out $FIG/encoder_data_2d_VAE.pkl"
```

Statistics (as in the paper): component correlation, RSA, within- and cross-trial
reconstruction, exact-match and balanced decoding accuracy are compared with two-tailed
paired t-tests over the four subjects (one value per subject); the pooled silhouette is
compared with a paired image bootstrap (B = 5000) in a 10-d UMAP embedding. The stars in
the figure use uncorrected p, as in the paper; `stat_tests.py` also prints the effect
sizes (d_z) and Benjamini–Hochberg-adjusted q-values across the metrics.

### Expected results

On the 128 test images (cross-trial: those of them with at least two single trials), as
reported in the paper. The bundled results give the same values up to rounding (e.g. SRM
retrieval top-2 is 29.95 %); the silhouette is the one `stat_tests.py` recomputes
(bundled in `visualisation/data/figure_stats.json`).

| Metric | MED-VAE | SRM | Procrustes |
|---|---|---|---|
| Component correlation | 0.637 | 0.516 | 0.446 |
| RSA | 0.627 | 0.405 | 0.411 |
| Silhouette (Fig. 2B) | 0.34 | 0.22 | 0.21 |
| Decoding, exact match | 0.184 | 0.104 | 0.074 |
| Decoding, balanced accuracy | 0.833 | 0.758 | 0.754 |
| Within-trial reconstruction r | 0.606 | 0.707 | 0.746 |
| Cross-trial reconstruction r | 0.188 | 0.175 | 0.174 |
| Retrieval top-1 / 2 / 5 / 10 (%) | 46.2 / 62.0 / 80.3 / 91.1 | 21.8 / 30.0 / 44.5 / 57.8 | 18.4 / 25.6 / 40.9 / 54.1 |

What to expect from a rerun:
- **SRM and Procrustes** are deterministic: the same data and environment give the same
  models and numbers.
- **MED-VAE** retrained from scratch differs slightly from the paper's model. A retrain
  with the paper's settings gave component correlation 0.641, RSA 0.627, retrieval
  top-1/2/5/10 46.0 / 61.9 / 80.1 / 90.8 %, within-trial r 0.608, exact match 0.189,
  balanced accuracy 0.841, cross-trial r 0.187.
- **Silhouette** depends on the UMAP/numba versions; use the pinned environment.

## Not included

- Fig. 2D (generalisation gap across latent sizes), the image-decoding experiments with
  MindEye2 (§5.5, Table 2, Fig. 4, Appendix J), the robustness check across sparse
  random projection seeds (§4) and the appendix analyses (untrained and shuffled scaffold
  controls, scaffold-weight sweep, per-layer and CLIP scaffolds, the ANN-only VAE of
  Appendix H, integrating new subjects).
- Building the single-trial (cross-trial) data and the noise ceilings, and hence the
  cross-trial and noise-ceiling-normalised values (see [section 4](#4-evaluate)); the
  paper's values of both are in the bundled results.

## Citation

```bibtex
@inproceedings{papathanasiou2026medvae,
  title     = {Task-guided cross-subject latent alignment: a multi-encoder-decoder {VAE}},
  author    = {Papathanasiou, Angeliki and Achterberg, Jascha and Nichols, Thomas E. and Costa, Rui Ponte},
  booktitle = {Proceedings of the 9th Conference on Cognitive Computational Neuroscience},
  year      = {2026},
  note      = {arXiv:2606.15989}
}
```
