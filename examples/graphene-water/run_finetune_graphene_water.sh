#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAIN_FILE="${TRAIN_FILE:-$SCRIPT_DIR/mace_train.xyz}"
VALID_FILE="${VALID_FILE:-$SCRIPT_DIR/mace_test.xyz}"
FOUNDATION_MODEL="${FOUNDATION_MODEL:-$REPO_DIR/models/mace-mpa-0-medium.model}"
MODEL_NAME="${MODEL_NAME:-graphene-water-mace-mpa0}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/$MODEL_NAME}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DRY_RUN="${DRY_RUN:-0}"
E0S="${E0S:-foundation}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VALID_FILE" ]]; then
  echo "Missing train/validation files:"
  echo "  $TRAIN_FILE"
  echo "  $VALID_FILE"
  exit 1
fi

if [[ ! -f "$FOUNDATION_MODEL" ]]; then
  echo "Missing foundation model: $FOUNDATION_MODEL"
  echo "Set FOUNDATION_MODEL=/path/to/mace-mpa-0-medium.model"
  exit 1
fi

if [[ "$DEVICE" == "auto" ]]; then
  DETECT_PY="${PYTHON:-python3}"
  if [[ -x "$REPO_DIR/.conda/mace-ft-cpu/bin/python" ]]; then
    DETECT_PY="$REPO_DIR/.conda/mace-ft-cpu/bin/python"
  fi
  DEVICE="$("$DETECT_PY" "$REPO_DIR/scripts/detect_device.py")"
fi

mkdir -p "$RESULTS_DIR"

if [[ -n "${MACE_RUN_TRAIN:-}" ]]; then
  read -r -a TRAIN_CMD <<< "$MACE_RUN_TRAIN"
elif command -v mace_run_train >/dev/null 2>&1; then
  TRAIN_CMD=(mace_run_train)
else
  TRAIN_CMD=(python3 -m mace.cli.run_train)
fi

CMD=("${TRAIN_CMD[@]}" \
  --name="$MODEL_NAME" \
  --foundation_model="$FOUNDATION_MODEL" \
  --model_dir="$RESULTS_DIR" \
  --log_dir="$RESULTS_DIR" \
  --checkpoints_dir="$RESULTS_DIR" \
  --results_dir="$RESULTS_DIR" \
  --train_file="$TRAIN_FILE" \
  --valid_file="$VALID_FILE" \
  --energy_key=energy \
  --forces_key=forces \
  --stress_key=stress \
  --E0s="$E0S" \
  --energy_weight=1.0 \
  --forces_weight=10.0 \
  --stress_weight=1.0 \
  --loss=universal \
  --lr=0.0005 \
  --scaling=rms_forces_scaling \
  --batch_size="$BATCH_SIZE" \
  --max_num_epochs="$MAX_EPOCHS" \
  --ema \
  --ema_decay=0.99 \
  --weight_decay=1e-6 \
  --amsgrad \
  --default_dtype=float64 \
  --clip_grad=10 \
  --device="$DEVICE" \
  --seed="$SEED" \
  --num_samples_pt=500 \
  --swa \
  --swa_lr=1e-4 \
  --swa_energy_weight=100.0 \
  --swa_forces_weight=10.0 \
  --swa_stress_weight=1.0)

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  CMD+=(--dry_run)
fi

"${CMD[@]}"
