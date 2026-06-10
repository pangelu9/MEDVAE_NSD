# `medvae/` — MED-VAE model, training, and data loading

The Multi-Encoder Multi-Decoder VAE. Per-subject fMRI encoders + one shared ANN
encoder map into a shared latent space; per-subject fMRI decoders + one shared
ANN decoder reconstruct out of it. The shared ANN decoder is the primary
cross-subject alignment mechanism.

## Entry point
`main.py` — training and several analysis sub-commands. See the
top-level [README](../README.md) for the full training command and flags.

## Key modules
| File | Role |
|---|---|
| `main.py` | training entry: train from scratch, or `--load_name … --finetune` to freeze a model and add/train a new subject's encoder/decoder |
| `args.py` | the command-line parser (`build_parser` / `parse_args`) for `main.py` |
| `model.py` | `HybridMultiEncoderVAE` + the Encoder / Decoder MLPs |
| `train.py` | `train_hybrid_vae` / `test_hybrid_vae` epoch loop + cyclical-β KL annealing |
| `loss.py` | per-encoder reconstruction + KL loss (the 4 pathways) |
| `pipeline.py` | training runner: builds the model + data loaders and drives the epoch train/test loop + logging (used by `main.py` + the baselines) |
| `dataset.py` | torch dataset + overlap-removal modes (`--remove_all_overlaps`) |
| `load_data.py` | fMRI / ANN-activation loading, **per-feature z-score on load**, train/test split |
| `common_samples.py` | `find_common_samples` — shared fMRI / ANN / label loading across subjects |
| `memory_utils.py` | loader memory helpers |

The post-hoc evaluation metrics (alignment, reconstruction, silhouette, decoding)
live in [`../evaluation/`](../evaluation/) — `medvae/` is just the model, training
and data loading.

## Paths
No absolute paths are hard-coded; data locations come from
[`../ccn_config.py`](../ccn_config.py). Modules add the repo root to `sys.path`
and import `ccn_config` for `NSD_DIR` / `NSD_DATA_DIR` / etc.
