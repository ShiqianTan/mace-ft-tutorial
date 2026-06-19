# Environment setup

This repository now includes a project-local MACE environment that supports the
Li10GeP2S12 and graphene-water examples on CPU, plus a matching GPU setup path
for Linux/CUDA machines.

## Local CPU environment

The CPU environment has already been created in this checkout at:

```text
.conda/mace-ft-cpu
```

Activate it from the repository root:

```bash
source scripts/activate_mace_ft.sh
```

Recreate it if needed:

```bash
bash scripts/setup_env.sh cpu
```

## GPU environment

On a Linux CUDA node, create a separate GPU environment:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_env.sh gpu
MACE_FT_ENV_DIR=.conda/mace-ft-gpu source scripts/activate_mace_ft.sh
```

Change `TORCH_INDEX_URL` to match your CUDA/PyTorch stack if your cluster uses a
different CUDA version.

Optional cuEquivariance acceleration can be installed after PyTorch if your
CUDA/PyTorch versions are compatible:

```bash
python -m pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
```

## Verification

```bash
python scripts/detect_device.py
python -m mace.cli.run_train --help

cd examples/LiGePS-SSE-PBE
python prepare_ligps_data.py --max-frames 5 --output-dir /tmp/ligps_verify

cd ../graphene-water
python run.py --config graphene-128.xyz --steps 0 --optimize-steps 0 --device cpu --output-dir /tmp/gw_verify
```

The repository's foundation models in `models/` load on CPU. The included
`examples/graphene-water/mace-ft-tutorial-main-3.model` checkpoint appears to
contain CUDA-serialized TorchScript state, so use it on CUDA or replace it with
a model fine-tuned/exported in your active environment.
