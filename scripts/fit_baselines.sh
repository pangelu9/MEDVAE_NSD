#!/bin/bash
#SBATCH -A gpu_costa.prj
#SBATCH -J fit_baselines
#SBATCH -p gpu_rtx8000_48gb,gpu_a100_40gb,gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --qos gpu_bmrc_4hr
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 160G
#SBATCH --time=3:55:00
#SBATCH -o logs/fit_baselines-%j.out
#SBATCH -e logs/fit_baselines-%j.out
#
# Fit a classical alignment baseline (SRM / Procrustes / CCA / GCCA). Override:
#
#   METHOD=procrustes N_COMPONENTS=32 sbatch scripts/fit_baselines.sh
#   METHOD=srm        N_COMPONENTS=512 sbatch scripts/fit_baselines.sh
#
# Under sbatch the script is copied to a spool dir, so BASH_SOURCE no longer
# points at scripts/. Honor an explicit REPO_ROOT (pass via --export) and fall
# back to BASH_SOURCE for direct execution.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_ROOT
source "${REPO_ROOT}/scripts/_env.sh"

METHOD="${METHOD:-procrustes}"               # srm | procrustes | cca | gcca | all
N_COMPONENTS="${N_COMPONENTS:-32}"
N_PCA="${N_PCA:-${N_COMPONENTS}}"
FILENAME="${FILENAME:-${CCN_ANN_FEATURES:-ann_features.npy}}"   # set CCN_ANN_FEATURES (scripts/_env.local.sh) to your real file
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/baselines/fitted_models}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/baselines/ccn_results_evaluation}"

# Fits the STANDARD model (builds the shared PCA basis). The paper baselines are
# the random-init refits — run baselines/fit_srm.py + fit_proca.py afterwards,
# which reuse the PCA from these standard models (see README section 3).
cd "${REPO_ROOT}/baselines"
python3 -u fit_baselines.py \
    --dataset streams --method "${METHOD}" \
    --filename "${FILENAME}" \
    --n_pca "${N_PCA}" --n_components "${N_COMPONENTS}" \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}"
