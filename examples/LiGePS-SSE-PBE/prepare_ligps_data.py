#!/usr/bin/env python3
"""Collect LGPS extxyz frames and create MACE train/validation files.

The repository already contains many converted ``output.extxyz`` files. This
script combines them without requiring ASE or dpdata, then writes a reproducible
train/test split for MACE fine-tuning.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def iter_extxyz_frames(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        while True:
            first = handle.readline()
            if not first:
                return
            if not first.strip():
                continue
            try:
                natoms = int(first.strip())
            except ValueError as exc:
                raise ValueError(f"{path}: expected atom count, got {first!r}") from exc
            comment = handle.readline()
            if not comment:
                raise ValueError(f"{path}: truncated frame after atom count")
            atom_lines = [handle.readline() for _ in range(natoms)]
            if any(line == "" for line in atom_lines):
                raise ValueError(f"{path}: truncated atom block")
            yield first + comment + "".join(atom_lines)


def discover_inputs(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("output.extxyz"):
        if path.parts[-2:] in [("prepared", "output.extxyz")]:
            continue
        if "prepared" in path.parts:
            continue
        paths.append(path)
    return sorted(paths)


def write_frames(path: Path, frames: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for smoke tests; omit to use every available frame.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "prepared").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_inputs(root)
    if not input_files:
        raise SystemExit(f"No output.extxyz files found below {root}")

    frames: list[str] = []
    for path in input_files:
        for frame in iter_extxyz_frames(path):
            frames.append(frame)
            if args.max_frames is not None and len(frames) >= args.max_frames:
                break
        if args.max_frames is not None and len(frames) >= args.max_frames:
            break

    if len(frames) < 2:
        raise SystemExit("Need at least two frames to create a train/test split")

    rng = random.Random(args.seed)
    rng.shuffle(frames)

    test_count = max(1, int(round(len(frames) * args.test_fraction)))
    test_count = min(test_count, len(frames) - 1)
    test_frames = frames[:test_count]
    train_frames = frames[test_count:]

    total_path = output_dir / "total_mace.xyz"
    train_path = output_dir / "mace_train.xyz"
    test_path = output_dir / "mace_test.xyz"

    write_frames(total_path, frames)
    write_frames(train_path, train_frames)
    write_frames(test_path, test_frames)

    print(f"Read {len(input_files)} extxyz files")
    print(f"Frames: total={len(frames)} train={len(train_frames)} test={len(test_frames)}")
    print(f"Wrote {total_path}")
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
