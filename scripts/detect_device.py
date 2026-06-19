#!/usr/bin/env python3
"""Print the best torch device available for these examples."""

try:
    import torch

    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
