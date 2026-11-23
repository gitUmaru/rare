import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter

from models import GaussianMixture, ROOT, quadratic_grid, rare_region, uniform_grid

from .experiment_abc import Experiment


def default_model_factory(means, weights, base_variance):
    return GaussianMixture(means, weights, base_variance)


def normal_cdf(x):
    return torch.special.ndtr(torch.as_tensor(x, dtype=torch.float64))


def exact_region_probability(separation, weight, sigma):
    a = separation / (math.sqrt(2.0) * sigma)
    upper, lower = normal_cdf(a), normal_cdf(-a)
    return float((1.0 - weight) * upper * lower + weight * upper ** 2)


def build_mixture(separation, weight):
    means = torch.tensor([[-separation, 0.0], [separation, 0.0], [0.0, separation]], dtype=torch.float64)
    weights = torch.tensor([(1.0 - weight) / 2, (1.0 - weight) / 2, weight], dtype=torch.float64)
    return means, weights


def draw_components(weights, n, gen):
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    return torch.searchsorted(torch.cumsum(weights, 0), u).clamp(max=2)


def to_numpy(t):
    return t.detach().cpu().numpy()


class FactorSweepExperiment(Experiment):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(config)
        self.separations = config.get("separations", [0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 3.2, 4.0])
        self.weights = config.get("weights", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
        self.budgets = config.get("budgets", [4, 8, 16, 32, 64, 128, 256, 512])
        self.schedules = config.get("schedules", ["uniform", "quadratic"])
        self.n = config.get("n", 100000)
        self.n_margin = config.get("n_margin", 5000)
        self.sigma = config.get("sigma", 0.35)
        self.terminal_variance = config.get("terminal_variance", 16.0)
        self.seed = config.get("seed", 0)
        self.reuse = config.get("reuse", False)
        self.analysis_budgets = config.get("analysis_budgets", [8, 32])
        self.model_factory = config.get("model_factory", default_model_factory)
        self.out_dir = Path(config.get("out_dir", ROOT / "results" / "factor_sweep"))
        self.fig_dir = Path(config.get("fig_dir", ROOT / "figures"))

    def setup(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        return None

    def _grid(self, schedule, steps):
        if schedule == "uniform":
            return uniform_grid(self.terminal_variance, steps)
        return quadratic_grid(self.terminal_variance, steps)

    def _cell(self, i, j, separation, weight):
        gen = torch.Generator().manual_seed(self.seed + 1000 * i + j)
        means, weights = build_mixture(separation, weight)
        component = draw_components(weights, self.n, gen)
        noise = torch.randn(self.n, 2, generator=gen, dtype=torch.float64)
        start = means[component] + math.sqrt(self.sigma ** 2 + self.terminal_variance) * noise
        model = self.model_factory(means, weights, self.sigma ** 2)
        p = exact_region_probability(separation, weight, self.sigma)
        rows = []
        for schedule in self.schedules:
            for steps in self.budgets:
                out = model.sample(start, self._grid(schedule, steps))
                hits = int(rare_region(out, means).sum())
                rows.append({"separation": separation, "weight": weight, "schedule": schedule,
                             "steps": steps, "p": p, "hits": hits, "n": self.n})
        return rows

    def validate(self):
        path = self.out_dir / "records.json"
        if self.reuse and path.exists():
            return json.loads(path.read_text())
        records = []
        for i, s in enumerate(self.separations):
            for j, w in enumerate(self.weights):
                records.extend(self._cell(i, j, s, w))
            print(f"  separation {s:g} done")
        path.write_text(json.dumps(records))
        return records

    def test(self, n=2_000_000, cells=((0.8, 0.02), (2.0, 0.08), (1.2, 0.005), (3.2, 0.2))):
        checks = []
        gen = torch.Generator().manual_seed(self.seed + 777)
        for s, w in cells:
            means, weights = build_mixture(s, w)
            component = draw_components(weights, n, gen)
            x = means[component] + self.sigma * torch.randn(n, 2, generator=gen, dtype=torch.float64)
            p = exact_region_probability(s, w, self.sigma)
            q = float(rare_region(x, means).double().mean())
            se = math.sqrt(p * (1 - p) / n)
            checks.append({"separation": s, "weight": w, "closed_form": p, "monte_carlo": q,
                           "z": (q - p) / se})
            assert abs(q - p) <= 4 * se
        (self.out_dir / "closed_form_check.json").write_text(json.dumps(checks, indent=2))
        return checks

    def _index(self, records):
        return {(r["separation"], r["weight"], r["schedule"], r["steps"]): r for r in records}

    def _array(self, index, budget, fn):
        S, W, K = len(self.separations), len(self.weights), len(self.schedules)
        out = torch.zeros(S, W, K, dtype=torch.float64)
        for i, s in enumerate(self.separations):
            for j, w in enumerate(self.weights):
                for k, sched in enumerate(self.schedules):
                    r = index[(s, w, sched, budget)]
                    out[i, j, k] = fn(r["hits"] / r["n"], r["p"])
        return out

    def _shares(self, values):
        flat = values.reshape(-1)
        total = ((flat - flat.mean()) ** 2).sum()
        shares = {}
        for name, dim in (("separation", 0), ("weight", 1), ("schedule", 2)):
            others = [d for d in range(3) if d != dim]
            level_means = values.mean(dim=others)
            count = flat.numel() / values.shape[dim]
            shares[name] = float(count * ((level_means - flat.mean()) ** 2).sum() / total)
        shares["interactions"] = 1.0 - sum(shares.values())
        return shares

    def _budget_needed(self, index, s, w, schedule):
        ok = []
        for steps in self.budgets:
            r = index[(s, w, schedule, steps)]
            p, q = r["p"], r["hits"] / r["n"]
            margin = 1.96 * math.sqrt(p * (1 - p) / self.n_margin)
            ok.append(abs(q - p) <= margin)
        for i in range(len(self.budgets)):
            if all(ok[i:]):
                return self.budgets[i]
        return None

    def analyze(self, records):
        index = self._index(records)
        out = {"variance_explained": {}, "peak_separation": {}, "weight_spread": {}, "budget_needed": {}}
        for budget in self.analysis_budgets:
            absolute = self._array(index, budget, lambda q, p: 100.0 * (q - p))
            log_ratio = self._array(index, budget, lambda q, p: math.log(max(q, 1e-12) / p))
            out["variance_explained"][str(budget)] = {
                "absolute_error": self._shares(absolute),
                "log_relative_error": self._shares(log_ratio),
            }
            uniform = absolute[:, :, self.schedules.index("uniform")]
            peaks = {}
            for j, w in enumerate(self.weights):
                peaks[str(w)] = self.separations[int(uniform[:, j].argmax())]
            out["peak_separation"][str(budget)] = peaks
            rare = [j for j, w in enumerate(self.weights) if w <= 0.05]
            spreads = []
            for i in range(len(self.separations)):
                column = uniform[i, rare]
                spreads.append(float((column.max() - column.min()) / column.mean()))
            out["weight_spread"][str(budget)] = {
                "per_separation": dict(zip(map(str, self.separations), spreads)),
                "median": float(torch.tensor(spreads).median()),
            }
        for schedule in self.schedules:
            table = {}
            for s in self.separations:
                for w in self.weights:
                    table[f"{s}|{w}"] = self._budget_needed(index, s, w, schedule)
            out["budget_needed"][schedule] = table
        out["share_robustness"] = {}
        slices = {
            "full grid": (lambda s: True, lambda w: True),
            "s in [0.8, 2.4]": (lambda s: 0.8 <= s <= 2.4, lambda w: True),
            "w <= 0.05": (lambda s: True, lambda w: w <= 0.05),
            "s in [0.8, 2.4] and w <= 0.05": (lambda s: 0.8 <= s <= 2.4, lambda w: w <= 0.05),
        }
        for budget in self.analysis_budgets:
            absolute = self._array(index, budget, lambda q, p: 100.0 * (q - p))
            log_ratio = self._array(index, budget, lambda q, p: math.log(max(q, 1e-12) / p))
            table = {}
            for name, (keep_s, keep_w) in slices.items():
                si = [i for i, s in enumerate(self.separations) if keep_s(s)]
                wi = [j for j, w in enumerate(self.weights) if keep_w(w)]
                table[name] = {
                    "absolute_error": self._shares(absolute[si][:, wi]),
                    "log_relative_error": self._shares(log_ratio[si][:, wi]),
                }
            out["share_robustness"][str(budget)] = table
        (self.out_dir / "analysis.json").write_text(json.dumps(out, indent=2))
        return out

    def _annotate(self, ax, array, norm, fmt, flip=0.6):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                v = array[i, j]
                if np.isnan(v):
                    continue
                shade = norm(v)
                color = "black" if (np.ma.is_masked(shade) or shade > flip) else "white"
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7, color=color)

    def _axes(self, ax):
        ax.set_xticks(range(len(self.separations)))
        ax.set_xticklabels([f"{s:g}" for s in self.separations])
        ax.set_yticks(range(len(self.weights)))
        ax.set_yticklabels([f"{w:g}" for w in self.weights])
        ax.set_xlabel("separation s")
        ax.set_ylabel("rare weight w")

    def figure_heatmaps(self, records, budget=8):
        index = self._index(records)
        k = self.schedules.index("uniform")
        absolute = to_numpy(self._array(index, budget, lambda q, p: 100.0 * (q - p))[:, :, k]).T
        relative = to_numpy(self._array(index, budget, lambda q, p: 100.0 * (q - p) / p)[:, :, k]).T
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        im = axes[0].imshow(absolute, origin="lower", aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=axes[0], label="percentage points")
        self._annotate(axes[0], absolute, im.norm, lambda v: f"{v:.1f}")
        axes[0].set_title(f"absolute error, q - p, N = {budget}")
        relative = np.clip(relative, 1.0, None)
        im = axes[1].imshow(relative, origin="lower", aspect="auto", interpolation="nearest",
                            norm=LogNorm(vmin=relative.min(), vmax=relative.max()))
        fig.colorbar(im, ax=axes[1], label="percent")
        self._annotate(axes[1], relative, im.norm, lambda v: f"{v:.0f}")
        axes[1].set_title(f"relative error, (q - p) / p, N = {budget}")
        for ax in axes:
            self._axes(ax)
        fig.tight_layout()
        out = self.fig_dir / "factor_sweep_heatmaps.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def figure_curves(self, records):
        index = self._index(records)
        k = self.schedules.index("uniform")
        budgets = self.analysis_budgets
        fig, axes = plt.subplots(2, len(budgets), figsize=(6.5 * len(budgets), 8.5), sharex=True, squeeze=False)
        colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(self.weights)))
        for col, budget in enumerate(budgets):
            for color, w in zip(colors, self.weights):
                rows = [index[(s, w, "uniform", budget)] for s in self.separations]
                axes[0, col].plot(self.separations, [100.0 * (r["hits"] / r["n"] - r["p"]) for r in rows],
                                  marker="o", color=color, label=f"w = {w:g}")
                axes[1, col].plot(self.separations, [(r["hits"] / r["n"]) / r["p"] for r in rows],
                                  marker="o", color=color, label=f"w = {w:g}")
            axes[0, col].set_title(f"absolute error, q - p, N = {budget}")
            axes[0, col].set_ylabel("percentage points")
            axes[1, col].set_title(f"ratio of probabilities, q / p, N = {budget}")
            axes[1, col].set_ylabel("q / p")
            axes[1, col].set_yscale("log")
            axes[1, col].set_yticks([1, 2, 3, 5, 10, 20])
            axes[1, col].yaxis.set_major_formatter(ScalarFormatter())
            axes[1, col].axhline(1.0, color="gray", linestyle="--")
            for row in (0, 1):
                axes[row, col].set_xscale("log")
                axes[row, col].set_xticks(self.separations)
                axes[row, col].set_xticklabels([f"{s:g}" for s in self.separations])
                axes[row, col].minorticks_off()
                axes[row, col].grid(True)
            axes[1, col].set_xlabel("separation s")
        axes[0, 0].legend(title="rare weight")
        fig.tight_layout()
        out = self.fig_dir / "factor_sweep_curves.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def figure_budget(self, analysis):
        fig, axes = plt.subplots(1, len(self.schedules), figsize=(7.5 * len(self.schedules), 5.5))
        axes = np.atleast_1d(axes)
        cmap = plt.get_cmap().copy()
        cmap.set_bad("lightgray")
        for ax, schedule in zip(axes, self.schedules):
            table = analysis["budget_needed"][schedule]
            array = np.full((len(self.weights), len(self.separations)), np.nan)
            for i, s in enumerate(self.separations):
                for j, w in enumerate(self.weights):
                    needed = table[f"{s}|{w}"]
                    if needed is not None:
                        array[j, i] = needed
            masked = np.ma.masked_invalid(array)
            im = ax.imshow(masked, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
                           norm=LogNorm(vmin=min(self.budgets), vmax=max(self.budgets)))
            for j in range(array.shape[0]):
                for i in range(array.shape[1]):
                    v = array[j, i]
                    if np.isnan(v):
                        ax.text(i, j, f">{max(self.budgets)}", ha="center", va="center", fontsize=7)
                    else:
                        shade = im.norm(v)
                        ax.text(i, j, f"{int(v)}", ha="center", va="center", fontsize=7,
                                color="black" if shade > 0.6 else "white")
            fig.colorbar(im, ax=ax, label="NFE")
            ax.set_title(f"NFE needed to be within the margin of error, {schedule} schedule")
            self._axes(ax)
        fig.tight_layout()
        out = self.fig_dir / "factor_sweep_budget.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def run(self):
        self.setup()
        self.train()
        checks = self.test()
        print("closed form vs Monte Carlo, z scores:", [round(c["z"], 2) for c in checks])
        records = self.validate()
        analysis = self.analyze(records)
        for fig in (self.figure_heatmaps(records), self.figure_curves(records), self.figure_budget(analysis)):
            print("wrote", fig.name)
        return analysis


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--budgets", type=int, nargs="+")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--out-dir")
    a = p.parse_args()
    config = {"n": a.n, "reuse": a.reuse}
    if a.budgets:
        config["budgets"] = a.budgets
    if a.out_dir:
        config["out_dir"] = a.out_dir
    FactorSweepExperiment(config).run()


if __name__ == "__main__":
    main()
