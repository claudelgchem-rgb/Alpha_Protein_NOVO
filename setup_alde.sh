#!/usr/bin/env bash
# Set up the ALDE environment (.venv-alde) for machine-learning-guided enzyme
# activity optimisation.
#
# ALDE (Active Learning-assisted Directed Evolution) lives in tools/ALDE, which
# is a copy of https://github.com/jsunn-y/ALDE (MIT licence). Upstream ships a
# conda environment file pinning pytorch-cuda; this script installs the same
# pinned versions against CPU wheels instead, because ALDE's own README puts a
# production round at "minutes" on several CPUs.
#
# Usage:
#   ./setup_alde.sh                  # CPU wheels (default)
#   ALDE_CUDA=1 ./setup_alde.sh      # CUDA 12.1 wheels, matching alde.yml
#
# After setup, generate the design space once per campaign and then run one
# round per wet-lab cycle -- see tools/ALDE/SETUP_NOTES.md.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv-alde"
ALDE_DIR="$REPO_DIR/tools/ALDE"

command -v uv >/dev/null 2>&1 || {
  echo "error: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
[[ -d "$ALDE_DIR" ]] || {
  echo "error: $ALDE_DIR not found" >&2; exit 1; }

echo "==> Creating Python 3.11 virtual environment at $VENV"
uv venv --python 3.11 "$VENV"
PY="$VENV/bin/python"

# Versions follow tools/ALDE/alde.yml. pandas 1.5 requires numpy < 2, so numpy
# is pinned below 2 as well.
if [[ "${ALDE_CUDA:-0}" == "1" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
  echo "==> Installing dependencies (CUDA 12.1 wheels)"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  echo "==> Installing dependencies (CPU wheels)"
fi

uv pip install --python "$PY" \
  --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
  "torch==2.1.1" \
  "numpy<2" \
  "pandas==1.5.3" \
  "gpytorch==1.11" \
  "botorch==0.9.4" \
  "pyro-ppl==1.8.6" \
  scipy xgboost matplotlib seaborn tqdm openpyxl xlrd

echo "==> Verifying installation"
cd "$ALDE_DIR"
"$PY" - <<'PYCODE'
import pandas as pd, torch, gpytorch, botorch, xgboost
import src.objectives as objectives
from src.optimize import BO_ARGS  # noqa: F401

print(f"  torch {torch.__version__} | gpytorch {gpytorch.__version__} "
      f"| botorch {botorch.__version__} | xgboost {xgboost.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

df = pd.read_csv("data/ParPgb/fitness_round1_training.csv")
print(f"  example campaign: {len(df)} labelled variants")
PYCODE

cat <<'EOF'

ALDE environment is ready.

Next steps (run from tools/ALDE, using ../../.venv-alde/bin/python):

  1. Generate the design space once per campaign. `nsites` is the number of
     residues mutated simultaneously; the space is 20^nsites variants, so a
     5-site library is 3.2M rows and ~1.2 GB of one-hot encodings:

       ../../.venv-alde/bin/python generate_domain.py --name=MyEnzyme --nsites=5

  2. Put measured activities in data/MyEnzyme/fitness_roundN.csv with a `Combo`
     column (the residues at the mutated positions, e.g. "FRMNY") and a column
     holding the activity to maximise.

  3. Run one round per wet-lab cycle. Pass --device=cpu unless you have a GPU
     (the script defaults to cuda):

       ../../.venv-alde/bin/python execute_production.py \
         --device=cpu --name=MyEnzyme \
         --data_csv=fitness_round1.csv --obj_col=<activity column> \
         --output_path=results/MyEnzyme/round1/ --batch_size=90

  4. Map the returned indices back to sequences with output2variants.ipynb,
     screen that batch in the lab, append the results, and repeat from step 3.

EOF
