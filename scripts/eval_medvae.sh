#!/bin/bash
#SBATCH -J medvae_eval
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 160G
#SBATCH --time=3:55:00
#SBATCH -o logs/medvae_eval-%j.out
#SBATCH -e logs/medvae_eval-%j.out
#
# These #SBATCH lines request resources only; add your cluster's account/partition/QoS
# on the command line, e.g. "sbatch -A <account> -p <partition> scripts/eval_medvae.sh"
# (command-line options override #SBATCH lines). Submit from the repository root (the
# logs/ path is relative to it).
#
# Evaluate MED-VAE / SRM / Procrustes with the unified evaluator
# (evaluation/evaluate_methods.py). Writes, into OUTPUT_DIR (default $CCN_RESULTS_DIR),
#   alignment_eval_<method>_<n_dims>d_<mode>_streams.pkl   (Panels B/C + Fig.3 data)
#   silh_latents_<method>_<n_dims>d_<mode>_streams.pkl      (silhouette bootstrap input)
# and CSV summaries in OUTPUT_DIR/results/. Run once per method; the figures expect all three:
#
#   METHOD=vae        sbatch scripts/eval_medvae.sh
#   METHOD=srm        sbatch scripts/eval_medvae.sh
#   METHOD=procrustes sbatch scripts/eval_medvae.sh
#
# By default these evaluate the outputs of scripts/train_medvae.sh and scripts/fit_baselines.sh:
#   vae         CKPT=$CCN_RESULTS_DIR/medvae_${SAVE_NAME}_${N_DIMS}_b1.0.pt
#               (SAVE_NAME defaults to train_medvae.sh's ALLsubjects_RN50_streams_rvoverl_all_nnw5_annw5)
#   srm         MODEL=baselines/fitted_models/srm_rndinit_shared872_${N_DIMS}d_model.pkl
#   procrustes  MODEL=baselines/fitted_models/procrustes_a_rndinit_shared872_${N_DIMS}d_model.pkl
# Set CKPT=... or MODEL=... to evaluate other files; relative CKPT/MODEL/OUTPUT_DIR are
# taken from the repository root. For vae, ANN_ACTIVATIONS selects the ANN features of the
# ANN metrics (ANN->fMRI, ANN->ANN, fMRI->ANN): a key of ccn_config.ANN_ACTIVATIONS or an
# absolute .npy path. The default, rn50_streams, is $CCN_DATA_ROOT/nsd/data/ann_features.npy
# (ccn_config.ANN_FEATURES_FILE), the file train_medvae.sh trains on by default; set it to
# the features your checkpoint was trained on. EXTRA="..." passes further evaluator flags.
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

METHOD="${METHOD:-vae}"                # vae | srm | procrustes
N_DIMS="${N_DIMS:-32}"
MODE="${MODE:-not_all8}"               # 128 held-out images used in the paper's main figure
SUBJECTS="${SUBJECTS:-1 2 5 7}"
# Eval outputs go to CCN_RESULTS_DIR (set in _env.sh). The figure scripts read
# $CCN_RESULTS_DIR too (without it they read the bundled evaluation/results).
OUTPUT_DIR="${OUTPUT_DIR:-${CCN_RESULTS_DIR}}"

# Default inputs: the outputs of train_medvae.sh / fit_baselines.sh (see header).
case "${METHOD}" in
    vae)
        CKPT="${CKPT:-${CCN_RESULTS_DIR}/medvae_${SAVE_NAME:-ALLsubjects_RN50_streams_rvoverl_all_nnw5_annw5}_${N_DIMS}_b1.0.pt}" ;;
    srm)
        MODEL="${MODEL:-${REPO_ROOT}/baselines/fitted_models/srm_rndinit_shared872_${N_DIMS}d_model.pkl}" ;;
    procrustes)
        MODEL="${MODEL:-${REPO_ROOT}/baselines/fitted_models/procrustes_a_rndinit_shared872_${N_DIMS}d_model.pkl}" ;;
    *)
        echo "METHOD must be vae, srm or procrustes (got '${METHOD}')." >&2
        exit 1 ;;
esac

# Relative paths are taken from the repository root (the evaluator runs in evaluation/).
from_repo() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "${REPO_ROOT}/$1" ;; esac; }
OUTPUT_DIR="$(from_repo "${OUTPUT_DIR}")"
if [ "${METHOD}" = "vae" ]; then
    CKPT="$(from_repo "${CKPT}")"
    if [ ! -f "${CKPT}" ]; then
        echo "MED-VAE checkpoint not found: ${CKPT}. Run scripts/train_medvae.sh first, or set CKPT=..." >&2
        exit 1
    fi
else
    MODEL="$(from_repo "${MODEL}")"
    if [ ! -e "${MODEL}" ]; then
        echo "${METHOD} model not found: ${MODEL}. Run scripts/fit_baselines.sh first, or set MODEL=..." >&2
        exit 1
    fi
fi

cd "${REPO_ROOT}/evaluation"
ARGS=(--method "${METHOD}" --dataset streams --n_dims "${N_DIMS}"
      --mode "${MODE}" --subjects ${SUBJECTS} --test_size 1.0
      --output_dir "${OUTPUT_DIR}")
if [ "${METHOD}" = "vae" ]; then
    ARGS+=(--vae_checkpoint "${CKPT}" --ann_activations "${ANN_ACTIVATIONS:-rn50_streams}")
else
    ARGS+=(--alignment_model_path "${MODEL}")
fi
python3 -u evaluate_methods.py "${ARGS[@]}" ${EXTRA:-}
