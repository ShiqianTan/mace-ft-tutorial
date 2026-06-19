#!/usr/bin/env python3
"""Compute an oxygen density profile and graphene-water gap from a trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read


def minimum_image_dz(z: np.ndarray, z0: float, cell_z: float) -> np.ndarray:
    dz = z - z0
    if cell_z > 0:
        dz -= np.rint(dz / cell_z) * cell_z
    return dz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("oxygen_density_z.csv"))
    args = parser.parse_args()

    frames = read(args.trajectory, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise SystemExit("No frames found")

    first = frames[0]
    symbols = np.array(first.get_chemical_symbols())
    if "C" not in symbols or "O" not in symbols:
        raise SystemExit("Trajectory must contain C atoms for graphene and O atoms for water")

    cell_z = float(first.cell.lengths()[2])
    graphene_z = float(np.mean(first.positions[symbols == "C", 2]))
    half_width = cell_z / 2.0
    edges = np.linspace(-half_width, half_width, args.bins + 1)
    counts = np.zeros(args.bins, dtype=float)

    area = float(np.linalg.norm(np.cross(first.cell[0], first.cell[1])))
    for atoms in frames:
        frame_symbols = np.array(atoms.get_chemical_symbols())
        oxygen_z = atoms.positions[frame_symbols == "O", 2]
        dz = minimum_image_dz(oxygen_z, graphene_z, cell_z)
        counts += np.histogram(dz, bins=edges)[0]

    bin_width = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (len(frames) * area * bin_width)

    positive = centers > 0
    if not np.any(positive):
        raise SystemExit("Could not find positive-z side of the interface")
    peak_idx = np.argmax(density[positive])
    gap_angstrom = float(centers[positive][peak_idx])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("z_from_graphene_A,oxygen_number_density_A^-3\n")
        for z, rho in zip(centers, density):
            handle.write(f"{z:.8f},{rho:.12e}\n")

    print(f"Frames: {len(frames)}")
    print(f"Graphene z reference: {graphene_z:.4f} A")
    print(f"First oxygen density peak: {gap_angstrom:.4f} A ({gap_angstrom / 10.0:.4f} nm)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
