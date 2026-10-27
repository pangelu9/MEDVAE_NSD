#!/bin/bash
#SBATCH -J medvae_train
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 160G
#SBATCH --time=3:55:00
#SBATCH -o logs/medvae_train-%j.out
#SBATCH -e logs/medvae_train-%j.out
#
# These #SBATCH lines request resources only; add your cluster's account/partition/QoS
# on the command line, e.g. "sbatch -A <account> -p <partition> scripts/train_medvae.sh"
# (command-line options override #SBATCH lines). Submit from the repository root (the
# logs/ path is relative to it).
#
# Train MED-VAE. Defaults reproduce the main paper model (ResNet-50 scaffold,
# 51048-d features from data_preparation/extract_ann_features.py, w=5, 32-d latent).
# Writes to $CCN_RESULTS_DIR the checkpoint medvae_${SAVE_NAME}_${LATENT_DIM}_b<KL_BETA>.pt
# and the losses medvae_${SAVE_NAME}_${LATENT_DIM}_<KL_BETA>_losses.csv, with KL_BETA
# printed as a float (default: medvae_ALLsubjects_RN50_streams_rvoverl_all_nnw5_annw5_32_b1.0.pt).
# Override via environment variables, e.g.:
#
#   sbatch scripts/train_medvae.sh                              # main model
#   FILENAME=my_features.npy NN_OUTPUT_DIM=<its width> SAVE_NAME=my_model \
#     sbatch scripts/train_medvae.sh                            # other ANN features (in $CCN_DATA_ROOT/nsd/data)
#

# Locate the repository. Under sbatch the script runs from a spool copy, so fall
# back to the submission directory: submit from the repository root.
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    [ -f "${REPO_ROOT}/ccn_config.py" ] || REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
fi
if [ ! -f "${REPO_ROOT}/ccn_config.py" ]; then
    echo "Cannot find the repository (no ccn_config.py in ${REPO_ROOT}); run from the repository root or export REPO_ROOT." >&2
    exit 1
fi
export REPO_ROOT
source "${REPO_ROOT}/scripts/_env.sh"

# ANN features file. Generic default; set CCN_ANN_FEATURES (e.g. in the gitignored
# scripts/_env.local.sh) to your real filename.
FILENAME="${FILENAME:-${CCN_ANN_FEATURES:-ann_features.npy}}"
NN_OUTPUT_DIM="${NN_OUTPUT_DIM:-51048}"
SAVE_NAME="${SAVE_NAME:-ALLsubjects_RN50_streams_rvoverl_all_nnw5_annw5}"
LATENT_DIM="${LATENT_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
KL_BETA="${KL_BETA:-1}"
EPOCHS="${EPOCHS:-30}"
NN_WEIGHT="${NN_WEIGHT:-5}"
ANN_PATHWAY_WEIGHT="${ANN_PATHWAY_WEIGHT:-5}"

cd "${REPO_ROOT}/medvae"
python3 -u main.py \
    --train --dataset streams --hybrid_vae \
    --use_nn_decoder --use_fmri_decoders --nn_to_fmri --eval_only_fmri \
    --remove_all_overlaps \
    --latent_dim "${LATENT_DIM}" --hidden_dim "${HIDDEN_DIM}" \
    --batch_size 128 --epochs "${EPOCHS}" --KL_beta "${KL_BETA}" --learning_rate 1e-4 \
    --nn_weight "${NN_WEIGHT}" --ann_pathway_weight "${ANN_PATHWAY_WEIGHT}" --fmri_weight 1 \
    --input_dim 20732 20735 20736 20733 20733 20734 20726 20733 "${NN_OUTPUT_DIM}" \
    --output_dim 20732 20735 20736 20733 20733 20734 20726 20733 \
    --nn_output_dim "${NN_OUTPUT_DIM}" \
    --keep_percent 100 100 100 100 100 100 100 100 \
    --filename "${FILENAME}" \
    --save_name "${SAVE_NAME}"
