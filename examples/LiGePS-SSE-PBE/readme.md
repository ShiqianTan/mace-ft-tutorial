# Example A: Li10GeP2S12 solid-state electrolyte

This example reconstructs the paper's LGPS case study with the data already
stored in this repository. The paper uses Li10GeP2S12 (LGPS) to evaluate force
accuracy after fine-tuning MACE foundation models on a DPAISquare dataset. The
reported best ablation setting for this case is:

- `forces_weight=100`
- `lr=0.001`
- `ema_decay=0.999`
- `stress_weight=0`

The scripts below prepare a MACE `extxyz` split and launch fine-tuning from a
foundation model.

## Files

- `data.init/`, `iter.000000/`, `iter.000001/`, `iter.000002/`: DeepMD-style
  source data and converted `output.extxyz` files.
- `deepmd2mace.py`: optional converter from DeepMD npy/raw data to `extxyz`.
- `prepare_ligps_data.py`: combines existing `output.extxyz` files and writes a
  reproducible train/test split.
- `run_finetune_ligps.sh`: MACE fine-tuning entry point for the LGPS paper
  setting.

## 1. Prepare train/test data

For a quick smoke test:

```bash
cd examples/LiGePS-SSE-PBE
python3 prepare_ligps_data.py --max-frames 20
```

For the full example:

```bash
cd examples/LiGePS-SSE-PBE
python3 prepare_ligps_data.py
```

This writes:

- `prepared/total_mace.xyz`
- `prepared/mace_train.xyz`
- `prepared/mace_test.xyz`

## 2. Fine-tune

Activate an environment with MACE, ASE, PyTorch, and CUDA if available. Then run:

```bash
cd examples/LiGePS-SSE-PBE
bash run_finetune_ligps.sh
```

Useful overrides:

```bash
DEVICE=cpu MAX_EPOCHS=2 bash run_finetune_ligps.sh
FOUNDATION_MODEL=/path/to/mace-mp-0b3-medium.model bash run_finetune_ligps.sh
MACE_RUN_TRAIN="python3 /path/to/mace/cli/run_train.py" bash run_finetune_ligps.sh
```

By default, the script uses:

```text
../../models/mace-mp-0b3-medium.model
prepared/mace_train.xyz
prepared/mace_test.xyz
E0s=foundation
```

Training outputs are written to `results/lgps-mace-mp0b3-fw100/`.

## Notes

The paper reports LGPS test RMSEs near 0.28 meV/atom for energy and
15-16 meV/A for forces after fine-tuning, depending on the foundation model and
training stage. Exact numbers can vary with MACE version, hardware, seed, and
the exact train/test split.
