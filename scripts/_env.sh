#!/bin/bash
# Common environment setup for the MED-VAE job scripts (train/eval/fit), which
# source it. Python comes from a conda env if CONDA_ROOT is set (CONDA_ENV, default
# medvae, is activated); otherwise the python3 already on PATH is used, e.g. an
# activated venv (sbatch passes the submitting environment to the job by default).
set -euo pipefail

# Local, uncommitted site overrides (gitignored): create scripts/_env.local.sh
# exporting CONDA_ROOT, CCN_DATA_ROOT, CCN_ANN_FEATURES, etc. to keep
# machine-specific values out of git. Sourced here if present.
_LOCAL_ENV="$(dirname "${BASH_SOURCE[0]}")/_env.local.sh"
if [ -f "${_LOCAL_ENV}" ]; then source "${_LOCAL_ENV}"; fi

CONDA_ROOT="${CONDA_ROOT:-}"   # e.g. /path/to/miniconda3; empty = no conda
if [ -n "${CONDA_ROOT}" ]; then
    module purge || true
    export PATH="${CONDA_ROOT}/bin:${PATH}"
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
    conda activate "${CONDA_ENV:-medvae}"   # the env created from environment.yml (needs working UMAP)
fi

# Repo root: set by the job scripts, else the parent of this scripts/ dir (made absolute).
REPO_ROOT="$(cd "${REPO_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
export REPO_ROOT

# Where the data lives (nsd/data/ with the files built by data_preparation/). Override
# by exporting CCN_DATA_ROOT before sbatch, or edit the default here.
export CCN_DATA_ROOT="${CCN_DATA_ROOT:-/path/to/data}"
# Write checkpoints into the repo's medvae/results by default (override with CCN_RESULTS_DIR).
export CCN_RESULTS_DIR="${CCN_RESULTS_DIR:-${REPO_ROOT}/medvae/results}"
# Relative paths are taken from the repository root (the jobs run python in its subdirectories).
case "${CCN_DATA_ROOT}" in /*) ;; *) CCN_DATA_ROOT="${REPO_ROOT}/${CCN_DATA_ROOT}" ;; esac
case "${CCN_RESULTS_DIR}" in /*) ;; *) CCN_RESULTS_DIR="${REPO_ROOT}/${CCN_RESULTS_DIR}" ;; esac
mkdir -p "${CCN_RESULTS_DIR}"

echo "Repo:        ${REPO_ROOT}"
echo "Data root:   ${CCN_DATA_ROOT}"
echo "Results dir: ${CCN_RESULTS_DIR}"
python3 -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"
