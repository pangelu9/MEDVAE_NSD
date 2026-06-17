#!/bin/bash
# Common environment setup for MEDVAE SLURM jobs (Oxford BMRC cluster).
# Sourced by the train/eval/fit scripts. Edit CONDA_ROOT / CONDA_ENV / the data
# root for your site.
set -euo pipefail

# Local, uncommitted site overrides (gitignored): create scripts/_env.local.sh
# exporting CONDA_ROOT, CCN_DATA_ROOT, CCN_ANN_FEATURES, etc. to keep
# machine-specific values out of git. Sourced here if present.
_LOCAL_ENV="$(dirname "${BASH_SOURCE[0]}")/_env.local.sh"
if [ -f "${_LOCAL_ENV}" ]; then source "${_LOCAL_ENV}"; fi

CONDA_ROOT="${CONDA_ROOT:-/path/to/miniconda3}"   # <-- edit for your site (or export CONDA_ROOT)
CONDA_ENV="${CONDA_ENV:-medvae}"   # the env created from environment.yml (needs working UMAP)

module purge || true
export PATH="${CONDA_ROOT}/bin:$PATH"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Where the data (nsd/, nsd_classification/) lives. Override by exporting
# CCN_DATA_ROOT before sbatch, or edit the default here.
export CCN_DATA_ROOT="${CCN_DATA_ROOT:-/path/to/data}"

# Repo root = parent of this scripts/ dir (honor a pre-set REPO_ROOT, since under
# sbatch BASH_SOURCE points at the spool copy, not scripts/).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_ROOT
# Write checkpoints into the repo's medvae/results by default (override with CCN_RESULTS_DIR).
export CCN_RESULTS_DIR="${CCN_RESULTS_DIR:-${REPO_ROOT}/medvae/results}"
mkdir -p "${CCN_RESULTS_DIR}"

echo "Repo:        ${REPO_ROOT}"
echo "Data root:   ${CCN_DATA_ROOT}"
echo "Results dir: ${CCN_RESULTS_DIR}"
python3 -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"
