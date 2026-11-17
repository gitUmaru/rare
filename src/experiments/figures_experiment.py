from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import GaussianMixture, ROOT, load_case, manifest, rare_region, uniform_grid

from .experiment_abc import Experiment

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)


def default_model_factory(d):
    return GaussianMixture(d["means"], d["weights"], d["base_variance"])


def _np(t):
    return t.detach().cpu().numpy()


def blob(ax, pts, color, alpha):
    pts = _np(pts)
    ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha, c=color, edgecolors="none")


def style(ax, title):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)


class FiguresExperiment(Experiment):
    """Figure pipeline, model-agnostic.

    Depends only on a model factory that maps a loaded case dict to an object with
    `sample(terminal, grid)`, plus the geometric `rare_region(x, means)` labeling.
    Swap `model_factory` in the config to run the same figures with a learned score
    or any other sampler. No training is needed for the exact-score default.
    """

    def __init__(self, config=None):
        config = config or {}
        super().__init__(config)
        self.device = config.get("device", "cpu")
        self.model_factory = config.get("model_factory", default_model_factory)
        self.prefix = config.get("prefix", "torch")
        self.highlight = config.get(
            "highlight",
            [
                "development_r0.02_s0.8_seed940",
                "development_r0.02_s2.0_seed940",
                "development_r0.08_s0.8_seed940",
                "development_r0.08_s2.0_seed940",
                "final_r0.04_s1.4_seed940",
                "final_r0.01_s2.8_seed940",
            ],
        )

    def setup(self):
        self.cases = manifest()
        self.by_name = {c["name"]: c for c in self.cases}

    def train(self):
        return None

    def _load(self, name):
        split = self.by_name[name]["split"]
        return load_case(name, split, device=self.device, dtype=torch.float64)

    def _model(self, d):
        return self.model_factory(d)

    def _fig(self, stem):
        return FIGDIR / f"{self.prefix}_{stem}.png"

    def story_figure(self, name, steps=32):
        d = self._load(name)
        model = self._model(d)
        means = d["means"]
        grid = uniform_grid(d["terminal_variance"], steps)
        generated = model.sample(d["terminal"], grid)
        ref_rare = rare_region(d["reference"], means)
        gen_rare = rare_region(generated, means)
        true_pct = ref_rare.float().mean().item() * 100
        got_pct = gen_rare.float().mean().item() * 100

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        blob(axes[0], d["reference"][~ref_rare], "#9bbcd6", 0.12)
        blob(axes[0], d["reference"][ref_rare], "#e06666", 0.2)
        axes[0].scatter(*_np(means).T, marker="*", c="k", zorder=5)
        style(axes[0], f"1. The real picture  (red = {true_pct:.0f}%)")
        blob(axes[1], d["terminal"], "#b4a7d6", 0.18)
        style(axes[1], "2. Pure noise (the starting point)")
        blob(axes[2], generated[~gen_rare], "#9bbcd6", 0.18)
        blob(axes[2], generated[gen_rare], "#e06666", 0.3)
        axes[2].scatter(*_np(means).T, marker="*", c="k", zorder=5)
        style(axes[2], f"3. What the fast sampler made  (red = {got_pct:.0f}%)")
        fig.tight_layout()
        out = self._fig(name)
        fig.savefig(out)
        plt.close(fig)
        return out, true_pct, got_pct

    def fast_vs_slow_figure(self, name, fast=4, slow=128):
        d = self._load(name)
        model = self._model(d)
        vmax = d["terminal_variance"]
        means = d["means"]
        ref_rare = rare_region(d["reference"], means)
        gen_fast = model.sample(d["terminal"], uniform_grid(vmax, fast))
        gen_slow = model.sample(d["terminal"], uniform_grid(vmax, slow))
        fr, sr = rare_region(gen_fast, means), rare_region(gen_slow, means)
        t_pct = ref_rare.float().mean().item() * 100
        f_pct = fr.float().mean().item() * 100
        s_pct = sr.float().mean().item() * 100

        ref = _np(d["reference"])
        lo, hi = ref.min(0), ref.max(0)
        pad = 0.15 * (hi - lo)
        xlim = (lo[0] - pad[0], hi[0] + pad[0])
        ylim = (lo[1] - pad[1], hi[1] + pad[1])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        blob(axes[0], d["reference"][~ref_rare], "#9bbcd6", 0.12)
        blob(axes[0], d["reference"][ref_rare], "#e06666", 0.2)
        axes[0].scatter(*_np(means).T, marker="*", c="k", zorder=5)
        style(axes[0], f"truth  (red = {t_pct:.0f}%)")
        blob(axes[1], gen_fast[~fr], "#9bbcd6", 0.18)
        blob(axes[1], gen_fast[fr], "#e06666", 0.3)
        axes[1].scatter(*_np(means).T, marker="*", c="k", zorder=5)
        style(axes[1], f"fast: {fast} steps  (red = {f_pct:.0f}%)")
        blob(axes[2], gen_slow[~sr], "#9bbcd6", 0.18)
        blob(axes[2], gen_slow[sr], "#e06666", 0.3)
        axes[2].scatter(*_np(means).T, marker="*", c="k", zorder=5)
        style(axes[2], f"slow: {slow} steps  (red = {s_pct:.0f}%)")
        for ax in axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        fig.tight_layout()
        out = self._fig(f"fast_vs_slow_{name}")
        fig.savefig(out)
        plt.close(fig)
        return out, t_pct, f_pct, s_pct

    def stepsize_figure(self, steps=(4, 8, 16, 32, 64, 128)):
        steps = list(steps)
        cols = 3
        rows = (len(self.highlight) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes, self.highlight):
            d = self._load(name)
            model = self._model(d)
            means = d["means"]
            vmax = d["terminal_variance"]
            target = rare_region(d["reference"], means).float().mean().item() * 100
            got = []
            for s in steps:
                gen = model.sample(d["terminal"], uniform_grid(vmax, s))
                got.append(rare_region(gen, means).float().mean().item() * 100)
            ax.axhline(target, color="#444444", linestyle="--")
            ax.plot(steps, got, marker="o", color="#e06666")
            ax.set_xscale("log", base=2)
            ax.set_xticks(steps)
            ax.set_xticklabels([str(s) for s in steps])
            ax.set_xlabel("number of ODE steps")
            ax.set_ylabel("red fraction (%)")
            ax.set_title(f"{name}\ntrue target = {target:.0f}%")
            ax.grid(True)
        for ax in axes[len(self.highlight):]:
            ax.axis("off")
        fig.suptitle("Few steps overshoot the red fraction; more steps settle onto the dashed target")
        fig.tight_layout()
        out = self._fig("stepsize_all_cases")
        fig.savefig(out)
        plt.close(fig)
        return out

    def overview_figure(self, mode="truth", steps=None):
        order = sorted(self.cases, key=lambda c: (c["separation"], c["rare_weight"], c["seed"]))
        cols = 6
        rows = (len(order) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, c in zip(axes, order):
            d = self._load(c["name"])
            model = self._model(d)
            means = d["means"]
            if mode == "truth":
                pts = d["reference"]
                a_bg, a_rare = 0.08, 0.25
            else:
                pts = model.sample(d["terminal"], uniform_grid(d["terminal_variance"], steps))
                a_bg, a_rare = 0.18, 0.35
            rare = rare_region(pts, means)
            blob(ax, pts[~rare], "#9bbcd6", a_bg)
            blob(ax, pts[rare], "#e06666", a_rare)
            ax.scatter(*_np(means).T, marker="*", c="k")
            spread = "far apart" if c["separation"] >= 1.4 else "overlap"
            pct = rare.float().mean().item() * 100
            lim = float(means.abs().max().item()) + 2.5
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            style(ax, f"w={c['rare_weight']*100:.0f}% sep {c['separation']} ({spread})\nred = {pct:.0f}%")
        for ax in axes[len(order):]:
            ax.axis("off")
        titles = {
            "truth": "Truth: red = the rare group. Left/down = harder (rarer, more overlap)",
            "fast": f"Fast sampler ({steps} steps): red fraction overshoots, worst where clusters overlap",
            "slow": f"Slow sampler ({steps} steps): red fraction close to truth across all cases",
        }
        fig.suptitle(titles[mode])
        fig.tight_layout()
        suffix = {"truth": "all_cases", "fast": f"fast_{steps}steps", "slow": f"slow_{steps}steps"}[mode]
        out = self._fig(f"overview_{suffix}")
        fig.savefig(out)
        plt.close(fig)
        return out

    def validate(self):
        made = []
        for name in self.highlight:
            out, t, g = self.story_figure(name)
            made.append(out.name)
            print(f"  {out.name:48s} truth {t:4.1f}%  ->  fast {g:4.1f}%")
        return made

    def test(self):
        made = []
        for name in ["development_r0.02_s0.8_seed940", "development_r0.02_s2.0_seed940"]:
            out, t, f, s = self.fast_vs_slow_figure(name)
            made.append(out.name)
            print(f"  {out.name:48s} truth {t:4.1f}%  fast {f:4.1f}%  slow {s:4.1f}%")
        made.append(self.stepsize_figure().name)
        made.append(self.overview_figure("truth").name)
        made.append(self.overview_figure("fast", steps=4).name)
        made.append(self.overview_figure("slow", steps=128).name)
        for n in made[-4:]:
            print(f"  {n}")
        return made

    def run(self):
        self.setup()
        self.train()
        print("Story figures:")
        self.validate()
        print("Comparison and grid figures:")
        self.test()
        print(f"\nAll torch figures saved under: {FIGDIR}")


def main():
    FiguresExperiment({}).run()


if __name__ == "__main__":
    main()
