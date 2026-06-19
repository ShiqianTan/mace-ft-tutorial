#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cpu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

case "$MODE" in
  cpu)
    ENV_DIR="${MACE_FT_ENV_DIR:-$REPO_DIR/.conda/mace-ft-cpu}"
    REQUIREMENTS="$REPO_DIR/requirements-cpu.txt"
    ;;
  gpu)
    ENV_DIR="${MACE_FT_ENV_DIR:-$REPO_DIR/.conda/mace-ft-gpu}"
    REQUIREMENTS="$REPO_DIR/requirements-gpu.txt"
    ;;
  *)
    echo "Usage: bash scripts/setup_env.sh [cpu|gpu]"
    exit 2
    ;;
esac

mkdir -p "$REPO_DIR/.conda" "$REPO_DIR/.cache/matplotlib"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create -y -p "$ENV_DIR" python=3.11 pip
fi

PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip

if [[ "$MODE" == "gpu" ]]; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  echo "Installing CUDA PyTorch from: $TORCH_INDEX_URL"
  "$PY" -m pip install torch --index-url "$TORCH_INDEX_URL"
fi

"$PY" -m pip install -r "$REQUIREMENTS"

cat <<EOF

Environment ready: $ENV_DIR

Activate it with:
  source scripts/activate_mace_ft.sh

For a GPU environment on a cluster:
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_env.sh gpu

EOF
