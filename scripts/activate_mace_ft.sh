#!/usr/bin/env bash
# Source this file from the repository root:
#   source scripts/activate_mace_ft.sh

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  SCRIPT_PATH="${(%):-%x}"
else
  SCRIPT_PATH="$0"
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="${MACE_FT_ENV_DIR:-$REPO_DIR/.conda/mace-ft-cpu}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Environment not found: $ENV_DIR"
  echo "Create it with: bash scripts/setup_env.sh cpu"
  return 1 2>/dev/null || exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
export CONDA_PREFIX="$ENV_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
echo "Activated MACE tutorial environment: $ENV_DIR"
