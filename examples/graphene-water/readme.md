# Example D: Graphene-water interface

This example reconstructs the paper's graphene-water interface case study. The
paper uses a pre-equilibrated periodic graphene-water interface, DFT single-point
labels, MACE fine-tuning, and MD validation at 300 K. The key validation target
is the oxygen density profile along `z`; the first oxygen peak gives a
graphene-water distance near 0.35 nm, close to the cited experimental value of
0.36 nm.

## Files

- `mace_train.xyz`, `mace_test.xyz`, `totalMACE.xyz`: prepared MACE datasets.
- `graphene-128.xyz`, `graphene-372.xyz`, `graphene-1038.xyz`: MD starting
  configurations.
- `mace-ft-tutorial-main-3.model`: fine-tuned model included with the example.
- `run_finetune_graphene_water.sh`: MACE fine-tuning entry point.
- `run.py`: ASE/Langevin MD runner for one or more models/configurations.
- `analyze_oxygen_density.py`: oxygen density profile and interface gap
  analysis.

## 1. Fine-tune

Activate an environment with MACE, ASE, PyTorch, and CUDA if available. Then run:

```bash
cd examples/graphene-water
bash run_finetune_graphene_water.sh
```

Useful overrides:

```bash
DEVICE=cpu MAX_EPOCHS=2 bash run_finetune_graphene_water.sh
FOUNDATION_MODEL=/path/to/mace-mpa-0-medium.model bash run_finetune_graphene_water.sh
MACE_RUN_TRAIN="python3 /path/to/mace/cli/run_train.py" bash run_finetune_graphene_water.sh
```

By default, the script uses `../../models/mace-mpa-0-medium.model` and writes
training outputs to `results/graphene-water-mace-mpa0/`. It uses
`E0s=foundation` by default; override with `E0S=/path/to/e0s.json` or a MACE
E0 dictionary if you have isolated-atom energies from matching DFT settings.

## 2. Run MD

Quick smoke test with the included fine-tuned model:

```bash
cd examples/graphene-water
python3 run.py --config graphene-128.xyz --steps 10 --optimize-steps 0 --device cpu
```

On CPU, `run.py` defaults to the CPU-loadable `../../models/mace-mpa-0-medium.model`.
The included `mace-ft-tutorial-main-3.model` checkpoint was serialized with CUDA
TorchScript state and should be used on a CUDA machine, or replaced by a model
fine-tuned/exported in the current CPU environment.

Paper-style run for the 372-atom interface:

```bash
cd examples/graphene-water
python3 run.py \
  --config graphene-372.xyz \
  --model mace-ft=mace-ft-tutorial-main-3.model \
  --steps 10000 \
  --temperature 300 \
  --time-step-fs 1
```

Compare the fine-tuned model with the foundation model:

```bash
cd examples/graphene-water
python3 run.py \
  --config graphene-372.xyz \
  --model mace-ft=mace-ft-tutorial-main-3.model \
  --model mace-mpa0=../../models/mace-mpa-0-medium.model \
  --steps 10000
```

Add `--enable-cueq` only in an environment where cuEquivariance is installed and
compatible with the selected CUDA/PyTorch build.

## 3. Analyze oxygen density

After MD writes an xyz trajectory under `results/`, compute the oxygen density
profile and graphene-water gap:

```bash
cd examples/graphene-water
python3 analyze_oxygen_density.py \
  results/graphene-372_mace-ft_300K_10000.xyz \
  --output results/oxygen_density_z.csv
```

The script reports the first positive-z oxygen-density peak in Angstrom and nm,
and writes a CSV profile for plotting.
