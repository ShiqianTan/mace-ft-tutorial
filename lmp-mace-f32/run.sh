#!/bin/bash
#SBATCH -p cpu-5218
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

LMP_BIN="${LMP_BIN:-lmp}"
INPUT="${INPUT:-mace-lmp.in}"
USE_KOKKOS="${USE_KOKKOS:-1}"

cd "$(dirname "$0")"

if [[ "$USE_KOKKOS" == "1" ]]; then
  "$LMP_BIN" -k on g 1 -sf kk -in "$INPUT"
else
  "$LMP_BIN" -in "$INPUT"
fi
