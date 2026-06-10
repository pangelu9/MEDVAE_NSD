#!/bin/bash
#SBATCH -A gpu_costa.prj
#SBATCH -J medvae_train
#SBATCH -p gpu_rtx8000_48gb,gpu_a100_40gb,gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --qos gpu_bmrc_4hr
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 160G
#SBATCH --time=3:55:00
#SBATCH -o logs/medvae_train-%j.out
#SBATCH -e logs/medvae_train-%j.out
#
# Train MED-VAE. Defaults reproduce the main paper model (ResNet-50 scaffold,
# 51060-d, w=5, 32-d latent). Override via environment variables, e.g.:
#
#   sbatch scripts/train_medvae.sh                              # main model
#   FILENAME=final_datasets_mindeye2_untrained_resnet/activations_all.npy \
#     SAVE_NAME=8subj_RN50untrained_51060_streams_rvoverl_nnw5_annw5 \
#     sbatch scripts/train_medvae.sh                            # untrained control
#
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

FILENAME="${FILENAME:-aligned_all_activations_fair_resnet50_hendrycks.npy}"
NN_OUTPUT_DIM="${NN_OUTPUT_DIM:-51060}"
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
