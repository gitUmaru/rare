import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent


def load_case(name, split, device=None, dtype=torch.float64):
    d = np.load(ROOT / "data" / split / f"{name}.npz")
    out = {
        "means": torch.as_tensor(d["means"], dtype=dtype, device=device),
        "weights": torch.as_tensor(d["weights"], dtype=dtype, device=device),
        "base_variance": float(d["base_variance"]),
        "terminal_variance": float(d["terminal_variance"]),
        "train": torch.as_tensor(d["train"], dtype=dtype, device=device),
        "reference": torch.as_tensor(d["reference"], dtype=dtype, device=device),
        "terminal": torch.as_tensor(d["terminal"], dtype=dtype, device=device),
    }
    return out


def manifest():
    return json.loads((ROOT / "data" / "manifest.json").read_text())["cases"]


class GaussianMixture:
    def __init__(self, means, weights, base_variance):
        self.means = torch.as_tensor(means)
        self.weights = torch.as_tensor(weights)
        self.base_variance = float(base_variance)

    def score(self, x, v):
        var = self.base_variance + v
        delta = x[:, None, :] - self.means[None, :, :]
        logmass = torch.log(self.weights)[None, :] - 0.5 * (delta ** 2).sum(-1) / var
        posterior = torch.softmax(logmass, dim=1)
        return -(posterior[:, :, None] * delta).sum(dim=1) / var

    def sample(self, terminal, grid):
        x = terminal.clone()
        for v, next_v in zip(grid[:-1], grid[1:]):
            x = x + (next_v - v) * (-0.5 * self.score(x, v))
        return x

    def rare_region(self, x):
        return rare_region(x, self.means)


def rare_region(x, means):
    means = torch.as_tensor(means)
    d2 = ((x[:, None, :] - means[None, :, :]) ** 2).sum(dim=2)
    return d2.argmin(dim=1) == 2


def uniform_grid(terminal_variance, steps):
    return float(terminal_variance) * torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)


def quadratic_grid(terminal_variance, steps):
    u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return float(terminal_variance) * u ** 2


def wilson(hits, n):
    z = 1.96
    p = hits / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [float(mid - half), float(mid + half)]
