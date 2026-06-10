#!/bin/bash
#SBATCH -A gpu_costa.prj
#SBATCH -J medvae_eval
#SBATCH -p gpu_rtx8000_48gb,gpu_a100_40gb,gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --qos gpu_bmrc_4hr
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 160G
#SBATCH --time=3:55:00
#SBATCH -o logs/medvae_eval-%j.out
#SBATCH -e logs/medvae_eval-%j.out
#
# Evaluate MED-VAE / SRM / Procrustes with the unified evaluator
# (evaluation/evaluate_methods.py). Writes, into OUTPUT_DIR,
#   alignment_eval_<method>_<n_dims>d_<mode>_streams.pkl   (Panels B/C + Fig.3 data)
#   silh_latents_<method>_<n_dims>d_<mode>_streams.pkl      (silhouette bootstrap input)
# Run once per method (vae, srm, procrustes); the figures expect all three.
#
#   METHOD=vae        CKPT=/path/to/medvae.pt          sbatch scripts/eval_medvae.sh
#   METHOD=srm        MODEL=/path/to/srm_model.pkl     sbatch scripts/eval_medvae.sh
#   METHOD=procrustes MODEL=/path/to/procrustes.pkl    sbatch scripts/eval_medvae.sh
#
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_ROOT
source "${REPO_ROOT}/scripts/_env.sh"

METHOD="${METHOD:-vae}"
N_DIMS="${N_DIMS:-32}"
MODE="${MODE:-not_all8}"               # 128 held-out images used in the paper's main figure
SUBJECTS="${SUBJECTS:-1 2 5 7}"
# Eval outputs go where the figure scripts read them: CCN_RESULTS_DIR if set
# (matches stat_tests.py / render_figure.py), else the bundled evaluation/results.
OUTPUT_DIR="${OUTPUT_DIR:-${CCN_RESULTS_DIR:-${REPO_ROOT}/evaluation/results}}"

cd "${REPO_ROOT}/evaluation"
ARGS=(--method "${METHOD}" --dataset streams --n_dims "${N_DIMS}"
      --mode "${MODE}" --subjects ${SUBJECTS} --test_size 1.0
      --output_dir "${OUTPUT_DIR}")
if [ "${METHOD}" = "vae" ]; then
    ARGS+=(--vae_checkpoint "${CKPT:?set CKPT=/path/to/medvae_checkpoint.pt for --method vae}")
else
    ARGS+=(--alignment_model_path "${MODEL:?set MODEL=/path/to/${METHOD}_model.pkl}")
fi
python3 -u evaluate_methods.py "${ARGS[@]}" ${EXTRA:-}
