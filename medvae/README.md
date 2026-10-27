# `medvae/` — MED-VAE model, training, and data loading

The Multi-Encoder Multi-Decoder VAE. Per-subject fMRI encoders + one shared ANN
encoder map into a shared latent space; per-subject fMRI decoders + one shared
ANN decoder reconstruct out of it. The shared ANN decoder is the primary
cross-subject alignment mechanism.

## Training
`main.py` is the training entry point (`--train`); it has no analysis sub-commands.
Run it through [`../scripts/train_medvae.sh`](../scripts/train_medvae.sh), which holds
the paper configuration. The defaults in `args.py` are **not** the paper settings:

| Flag | `args.py` default | `train_medvae.sh` (paper) |
|---|---|---|
| `--dataset` | `algonauts` (not usable in this release) | `streams` |
| `--latent_dim` / `--hidden_dim` | 256 / 512 | 32 / 256 |
| `--KL_beta` / `--epochs` | 0.1 / 50 | 1 / 30 |
| `--nn_weight` / `--ann_pathway_weight` | 1 / unset (= `--nn_weight`) | 5 / 5 |
| `--nn_output_dim` and last `--input_dim` | 7168 | 51048 (the ANN feature width) |

The switches `--train --hybrid_vae --use_nn_decoder --use_fmri_decoders --nn_to_fmri
--eval_only_fmri --remove_all_overlaps` are all off by default; the script turns them on
(`--hybrid_vae` is required). `--load_name <ckpt>` (looked up in `$CCN_RESULTS_DIR`)
starts from a trained checkpoint with its ANN encoder and decoder frozen, and
`--finetune` then saves only the unfrozen weights; neither is used for the paper.

## Outputs
Written to `$CCN_RESULTS_DIR` (default `medvae/results/`, see `ccn_config.py`):
- `medvae_<save_name>_<latent_dim>_b<KL_beta>.pt` — final checkpoint: `model_state_dict`,
  `optimizer_state_dict`, final losses and `config` (the arguments; `evaluation/` and
  `visualisation/` rebuild the model from it);
- `medvae_<save_name>_<latent_dim>_<KL_beta>_losses.csv` — per-epoch train/test losses. Its
  alignment and silhouette columns are always 0: those metrics are computed afterwards by
  [`../evaluation/evaluate_methods.py`](../evaluation/evaluate_methods.py);
- with `--checkpoint_every N`, also `..._b<KL_beta>_epoch<k>.pt` every N epochs.

`KL_beta` is a float, so the paper run writes `..._32_b1.0.pt` and `..._32_1.0_losses.csv`.

## Key modules
| File | Role |
|---|---|
| `main.py` | training entry: loads data, builds the model, runs the epochs, saves the outputs above |
| `args.py` | the command-line parser (`build_parser` / `parse_args`) for `main.py` |
| `model.py` | `HybridMultiEncoderVAE` + the Encoder / Decoder MLPs |
| `train.py` | `train_hybrid_vae` / `test_hybrid_vae` (one epoch each): weights the KL and the 4 reconstruction pathways (fMRI→ANN, ANN→ANN, fMRI→fMRI, ANN→fMRI); optional KL annealing (`--kl_annealing`, off in the paper) |
| `loss.py` | per-encoder KL and ANN / fMRI reconstruction losses |
| `pipeline.py` | `create_vae_model`, `load_data` and the per-epoch train/test step (used by `main.py`, the baselines and `visualisation/visualise_latent.py`) |
| `dataset.py` | loads the 8 subjects' fMRI + ANN features + labels, removes overlapping images (`--remove_all_overlaps`), drops images left without fMRI (`--filter_no_fmri`, on by default), splits train/test (`--test_size`, seed 42), builds the torch datasets / loaders |
| `load_data.py` | ANN-feature loading (**each column z-scored on load**), label loading, and the fMRI loader used by `common_samples.py` |
| `common_samples.py` | `find_common_samples` — the images all 8 subjects saw; `main.py` attaches them to `args` (so they are saved in the checkpoint `config`), training does not use them |
| `memory_utils.py` | RSS / garbage-collection helpers used while loading |

The post-hoc evaluation metrics (alignment, reconstruction, silhouette, decoding)
live in [`../evaluation/`](../evaluation/) — `medvae/` is just the model, training
and data loading.

## Paths
No absolute paths are hard-coded; data locations come from
[`../ccn_config.py`](../ccn_config.py). Training reads `fmri_subject{01..08}.npz`, the ANN
features (`--filename`, default `ccn_config.ANN_FEATURES_FILE`) and `category_labels.npy`
from `$CCN_DATA_ROOT/nsd/data/`. All 8 subjects must load: a missing fMRI file, or the
process memory exceeding `--max_memory_gb` (default 100) before a subject, stops with an error.
