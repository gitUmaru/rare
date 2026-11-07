from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starter import ROOT, sample, rare_region

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)


def load_case(name, split):
    return np.load(ROOT / "data" / split / f"{name}.npz")


def blob(ax, pts, color, alpha):
    ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha, c=color, edgecolors="none")


def style(ax, title):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)


def story_figure(name, split, steps=32):
    d = load_case(name, split)
    means, weights = d["means"], d["weights"]
    reference, terminal = d["reference"], d["terminal"]

    u = np.linspace(1, 0, steps + 1)
    grid = float(d["terminal_variance"]) * u
    generated = sample(terminal, grid, means, weights, float(d["base_variance"]))

    ref_rare = rare_region(reference, means)
    gen_rare = rare_region(generated, means)
    true_pct = ref_rare.mean() * 100
    got_pct = gen_rare.mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    blob(ax, reference[~ref_rare], "#9bbcd6", alpha=0.12)
    blob(ax, reference[ref_rare], "#e06666", alpha=0.2)
    ax.scatter(means[:, 0], means[:, 1], marker="*", c="k", zorder=5)
    style(ax, f"1. The real picture  (red = {true_pct:.0f}%)")

    ax = axes[1]
    blob(ax, terminal, "#b4a7d6", alpha=0.18)
    style(ax, "2. Pure noise (the starting point)")

    ax = axes[2]
    blob(ax, generated[~gen_rare], "#9bbcd6", alpha=0.18)
    blob(ax, generated[gen_rare], "#e06666", alpha=0.3)
    ax.scatter(means[:, 0], means[:, 1], marker="*", c="k", zorder=5)
    style(ax, f"3. What the fast sampler made  (red = {got_pct:.0f}%)")

    fig.tight_layout()
    out = FIGDIR / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out, true_pct, got_pct


def process_figure(name, split, n_steps=6, n_points=4000, seed=0):
    d = load_case(name, split)
    means = d["means"]
    train = d["train"]
    vmax = float(d["terminal_variance"])

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=min(n_points, len(train)), replace=False)
    clean = train[idx]
    noise = rng.standard_normal(clean.shape)

    levels = np.linspace(0.0, vmax, n_steps)

    fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, v in zip(axes, levels):
        pts = clean + np.sqrt(v) * noise
        rare = rare_region(pts, means)
        blob(ax, pts[~rare], "#9bbcd6", alpha=0.3)
        blob(ax, pts[rare], "#e06666", alpha=0.4)
        if v == 0:
            label = "data (v=0)"
        elif np.isclose(v, vmax):
            label = f"prior (v={vmax:.0f})"
        else:
            label = f"v={v:.1f}"
        style(ax, label)
    fig.suptitle(f"{name}: data (left) -> noised over time -> prior (right)")
    fig.tight_layout()
    out = FIGDIR / f"process_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def stepsize_figure(named_cases, steps=(4, 8, 16, 32, 64, 128)):
    steps = list(steps)
    cols = 3
    rows = (len(named_cases) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, split) in zip(axes, named_cases):
        d = load_case(name, split)
        means, weights = d["means"], d["weights"]
        reference, terminal = d["reference"], d["terminal"]
        base = float(d["base_variance"])
        vmax = float(d["terminal_variance"])

        target = rare_region(reference, means).mean() * 100
        got = []
        for s in steps:
            grid = vmax * np.linspace(1, 0, s + 1)
            generated = sample(terminal, grid, means, weights, base)
            got.append(rare_region(generated, means).mean() * 100)

        ax.axhline(target, color="#444444", linestyle="--")
        ax.plot(steps, got, marker="o", color="#e06666")
        ax.set_xscale("log", base=2)
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel("number of ODE steps")
        ax.set_ylabel("red fraction (%)")
        ax.set_title(f"{name}\ntrue target = {target:.0f}%")
        ax.grid(True)
    for ax in axes[len(named_cases):]:
        ax.axis("off")
    fig.suptitle("Few steps overshoot the red fraction; more steps settle onto the dashed target")
    fig.tight_layout()
    out = FIGDIR / "stepsize_all_cases.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def overview_figure(cases):
    order = sorted(cases, key=lambda c: (c["separation"], c["rare_weight"], c["seed"]))
    cols = 6
    rows = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, order):
        d = load_case(c["name"], c["split"])
        ref = d["reference"]
        rare = rare_region(ref, d["means"])
        blob(ax, ref[~rare], "#9bbcd6", alpha=0.08)
        blob(ax, ref[rare], "#e06666", alpha=0.25)
        ax.scatter(d["means"][:, 0], d["means"][:, 1], marker="*", c="k")
        spread = "clusters far apart" if c["separation"] >= 1.4 else "clusters overlap"
        style(ax, f"red group = {c['rare_weight']*100:.0f}%\n{spread} (sep {c['separation']})")
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.suptitle("Every scenario: red = the rare group. Left/down = harder (rarer, more overlap)")
    fig.tight_layout()
    out = FIGDIR / "overview_all_cases.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    cases = json.loads((ROOT / "data" / "manifest.json").read_text())["cases"]
    highlight = [
        "development_r0.02_s0.8_seed940",
        "development_r0.02_s2.0_seed940",
        "development_r0.08_s0.8_seed940",
        "development_r0.08_s2.0_seed940",
        "final_r0.04_s1.4_seed940",
        "final_r0.01_s2.8_seed940",
    ]
    by_name = {c["name"]: c for c in cases}
    print("Per-scenario figures:")
    for name in highlight:
        c = by_name[name]
        out, true_pct, got_pct = story_figure(name, c["split"])
        print(f"  {out.name:40s} should be {true_pct:4.1f}%  ->  sampler gave {got_pct:4.1f}%")

    print("Process (data -> noise) figures:")
    for name in ["development_r0.02_s0.8_seed940", "development_r0.02_s2.0_seed940"]:
        c = by_name[name]
        out = process_figure(name, c["split"])
        print(f"  {out.name}")

    out = overview_figure(cases)
    print(f"Overview figure: {out.name}")

    named = [(name, by_name[name]["split"]) for name in highlight]
    out = stepsize_figure(named)
    print(f"Step-size figure: {out.name}")
    print(f"\nAll figures saved under: {FIGDIR}")


if __name__ == "__main__":
    main()
