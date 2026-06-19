#!/usr/bin/env python3
"""Run graphene-water MACE MD with ASE.

Defaults follow the tutorial case: 300 K Langevin dynamics with a 1 fs time
step for 10000 steps. Use fewer steps for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.optimize import LBFGS
from mace.calculators import MACECalculator


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def print_energy(atoms) -> None:
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)
    temp = ekin / (1.5 * units.kB)
    print(
        f"Energy per atom: Epot={epot:.6f} eV "
        f"Ekin={ekin:.6f} eV T={temp:.1f} K Etot={epot + ekin:.6f} eV"
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model as name=/path/model or /path/model. May be repeated.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Input xyz file. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--time-step-fs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--optimize-steps", type=int, default=100)
    parser.add_argument("--fmax", type=float, default=0.01)
    args = parser.parse_args()

    if args.model:
        model_specs = args.model
    elif args.device == "cpu":
        model_specs = [f"mace-mpa0={here / '../../models/mace-mpa-0-medium.model'}"]
        print("No --model supplied on CPU; using the CPU-loadable MACE-MPA-0 foundation model.")
    else:
        model_specs = [f"mace-ft={here / 'mace-ft-tutorial-main-3.model'}"]
    config_paths = [Path(p) for p in (args.config or [here / "graphene-372.xyz"])]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_result: dict[str, float] = {}

    for model_arg in model_specs:
        model_name, model_path = parse_model(model_arg)
        if not model_path.is_absolute():
            model_path = (here / model_path).resolve()
        if not model_path.exists():
            raise SystemExit(f"Missing model file: {model_path}")

        try:
            calculator = MACECalculator(
                model_paths=str(model_path),
                enable_cueq=args.enable_cueq,
                device=args.device,
            )
        except NotImplementedError as exc:
            if args.device == "cpu" and "CUDA" in str(exc):
                raise SystemExit(
                    "This checkpoint appears to contain CUDA-serialized TorchScript "
                    "state and cannot be loaded in a CPU-only PyTorch build. Use "
                    "--model mace-mpa0=../../models/mace-mpa-0-medium.model for a "
                    "CPU smoke test, run this checkpoint on a CUDA machine, or "
                    "fine-tune/export a CPU-loadable model in this environment."
                ) from exc
            raise

        for config_path in config_paths:
            if not config_path.is_absolute():
                config_path = (here / config_path).resolve()
            if not config_path.exists():
                raise SystemExit(f"Missing configuration file: {config_path}")

            config_name = config_path.stem
            run_name = f"{config_name}_{model_name}_{int(args.temperature)}K_{args.steps}"
            traj_path = args.output_dir / f"{run_name}.traj"
            xyz_path = args.output_dir / f"{run_name}.xyz"

            print(f"Running {run_name} on {args.device}")
            atoms = read(config_path)
            atoms.calc = calculator

            if args.optimize_steps > 0:
                opt = LBFGS(atoms)
                opt.run(fmax=args.fmax, steps=args.optimize_steps)

            dyn = Langevin(
                atoms,
                args.time_step_fs * units.fs,
                args.temperature * units.kB,
                args.friction,
            )
            dyn.attach(lambda atoms=atoms: print_energy(atoms), interval=args.log_interval)
            traj = Trajectory(traj_path, "w", atoms)
            dyn.attach(traj.write, interval=args.log_interval)

            start = time.time()
            traj.write()
            dyn.run(args.steps)
            if args.steps % args.log_interval != 0:
                traj.write()
            traj.close()
            elapsed = time.time() - start

            trajectory_frames = read(traj_path, index=":")
            write(xyz_path, trajectory_frames)
            timing_result[run_name] = elapsed
            print(f"Finished {run_name} in {elapsed:.2f} s")
            print(f"Wrote {traj_path}")
            print(f"Wrote {xyz_path}")

    json_path = args.output_dir / f"timing_results_{int(args.temperature)}K_{args.steps}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(timing_result, handle, indent=2)
    print(f"Timing results saved to {json_path}")


if __name__ == "__main__":
    main()
