# `evaluation/` — MED-VAE evaluation pipeline

## Main script
`evaluate_vae.py` computes, in one run on the held-out
images:

- **Latent alignment** — component-wise correlation, ISC, RSA (Fig. 2C)
- **Cross-subject neural prediction** — subject *i* latent → subject *j* decoder
  → subject *j* voxels (Fig. 3)
- **Within-subject reconstruction** + **cross-trial** correlation + NC-normalised
  reconstruction (Fig. 2C, Appendix D)
- **ANN→fMRI / ANN→ANN / fMRI→ANN** reconstruction (Appendix F/H)
- **Cross-subject retrieval** (Table 1)
- **Category decoding** (leave-one-subject-out, multi-label) + **silhouette**
  (Fig. 2B)

```bash
python evaluate_vae.py \
  --dataset streams --vae_checkpoint <ckpt.pt> \
  --latent_dim 32 --hidden_dim 256 --subjects 1 2 5 7 --mode all_common \
  --ann_activations rn50_streams --output_dir results_vae
```

`--ann_activations` takes a logical key from `ccn_config.ANN_ACTIVATIONS` or an
absolute `.npy` path. If omitted, the scaffold is **auto-detected from the
checkpoint name** (the original behaviour, preserved verbatim).

Writes `alignment_eval_vae_<latent>d_<mode>_streams.pkl` + CSVs under
`--output_dir`.

## `untrained/` — untrained-ResNet control (Appendix C)
Latent-geometry visualisations showing that with an untrained / shuffled
scaffold the latent space fragments by subject (no cross-subject category
structure):
- `eval_untrained_rn50_ann_clustering.py` — per-subject eval + PCA/UMAP by encoder
- `eval_untrained_rn50_clustering_full.py` — full-N cluster plots by encoder
- `plot_latent_by_category.py` — PCA/UMAP coloured by image super-category
- `_untrained_utils.py` — shared `encode_fmri_full` / `encode_ann_sample`
  helpers (factored out of the two clustering scripts; `encode_ann_sample` takes
  a `sort_indices` flag so each caller's original behaviour is preserved exactly)

## Shared metrics toolkit
The whole post-hoc evaluation toolkit lives here and is shared by both this
MED-VAE eval entry and the SRM/Procrustes baselines (a single canonical copy):

| Module | Role |
|---|---|
| `metrics.py` | the metric **functions**: alignment (`cross_subj_metrics`: component correlation + RSA), decoding, silhouette, reconstruction-quality |
| `evaluation_pipeline.py` | the shared evaluation **orchestration**, imported by **both** entries: `comprehensive_evaluation_v2` (the baseline's full eval) + `compute_alignment_metrics` / `compute_cross_subject_retrieval` / `evaluate_aligned_latents` (used by the MED-VAE eval) — so both run the same metric code |
| `prediction.py` | cross-subject neural prediction (subject *i* latent → subject *j* voxels) + pairwise pipeline |
| `retrieval.py` | shared cross-subject retrieval (Table 1) — one implementation used by **both** the MED-VAE eval and the SRM/Procrustes baseline |
| `baseline_retrieval.py` | retrieval entry for an SRM/Procrustes alignment model (loads the `.pkl`, calls `retrieval.py`) |
| `savemetrics.py` | pickle/CSV metric saving |

ISC (inter-subject correlation) was dropped from the alignment metrics — it was
not reported in the paper. The MED-VAE alignment numbers are otherwise unchanged
(verified Δ=0 against the previous inline implementation).

**Retrieval protocol (`retrieval.py`).** A fixed **128-image gallery** so
difficulty is comparable across splits: a split with ≤128 shared images uses all
of them in one deterministic pass; a larger split (e.g. the 872 all-common
images) subsamples 128 images × 30 repetitions and averages. The `seed` fixes the
subsets, so the MED-VAE and baseline runs draw the *same* 128-image galleries and
their numbers are directly comparable. Random chance ≈ 0.78 % (top-1).
