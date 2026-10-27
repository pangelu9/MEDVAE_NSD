#!/bin/bash
#SBATCH -J fit_baselines
#SBATCH --cpus-per-task 11
#SBATCH --mem 192G
#SBATCH --time=4:00:00
#SBATCH -o logs/fit_baselines-%j.out
#SBATCH -e logs/fit_baselines-%j.out
#
# These #SBATCH lines request resources only; add your cluster's account/partition/QoS
# on the command line, e.g. "sbatch -A <account> -p <partition> scripts/fit_baselines.sh"
# (command-line options override #SBATCH lines). Submit from the repository root (the
# logs/ path is relative to it).
#
# Fit the paper's SRM and Procrustes baselines (32-d; per-subject PCA on all of a subject's
# images, alignment on the 872 images all eight subjects saw). Runs in baselines/ and
# writes to baselines/fitted_models/:
#   1. fit_baselines.py --method srm          (PCA 5000 -> SRM)  srm_shared872_32d_model.pkl
#   2. fit_baselines.py --method procrustes_a (PCA 32 -> GPA)    procrustes_a_shared872_32d_model.pkl
#   3. fit_srm.py   (random-init SRM, PCA of 1)                  srm_rndinit_shared872_32d_model.pkl
#   4. fit_proca.py (random-start GPA, PCA of 2)                 procrustes_a_rndinit_shared872_32d_model.pkl
# 1-2 are the standard models whose PCA bases 3-4 reuse; 3-4 are the paper's baselines
# (the defaults of scripts/eval_medvae.sh). The fits run on the CPU (a visible GPU would
# only switch on DataLoader workers and pinned memory), so no GPU is requested, but they
# hold the fMRI of all eight subjects and the ANN features in RAM.
#
#   sbatch -A <account> -p <partition> scripts/fit_baselines.sh
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

# ANN features file, loaded with the fMRI (same default as train_medvae.sh; steps 3-4
# read ccn_config.ANN_FEATURES_FILE).
FILENAME="${FILENAME:-${CCN_ANN_FEATURES:-ann_features.npy}}"   # set CCN_ANN_FEATURES (scripts/_env.local.sh) to your real file
# fit_srm.py / fit_proca.py read the standard models from, and save their models to,
# baselines/fitted_models (hard-coded), so steps 1-2 must save there too.
MODEL_DIR="${REPO_ROOT}/baselines/fitted_models"
# Auxiliary metric dumps of steps 1-2 (not read by the evaluation or the figures).
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/baselines/ccn_results_evaluation}"

cd "${REPO_ROOT}/baselines"
python3 -u fit_baselines.py --method srm          --n_pca 5000 --n_components 32 \
    --dataset streams --filename "${FILENAME}" --model_dir "${MODEL_DIR}" --output_dir "${OUTPUT_DIR}"
python3 -u fit_baselines.py --method procrustes_a --n_pca 32   --n_components 32 \
    --dataset streams --filename "${FILENAME}" --model_dir "${MODEL_DIR}" --output_dir "${OUTPUT_DIR}"
# N_PCA=0: fit_srm.py reuses the PCA-5000 of step 1 (a positive N_PCA in the
# environment would refit PCA and save under another name).
N_PCA=0 python3 -u fit_srm.py
python3 -u fit_proca.py
