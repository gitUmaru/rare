from pathlib import Path
import sys
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))
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


def fast_vs_slow_figure(name, split, fast=4, slow=128):
    d = load_case(name, split)
    means, weights = d["means"], d["weights"]
    reference, terminal = d["reference"], d["terminal"]
    base = float(d["base_variance"])
    vmax = float(d["terminal_variance"])

    ref_rare = rare_region(reference, means)
    true_pct = ref_rare.mean() * 100

    gen_fast = sample(terminal, vmax * np.linspace(1, 0, fast + 1), means, weights, base)
    gen_slow = sample(terminal, vmax * np.linspace(1, 0, slow + 1), means, weights, base)
    fast_rare = rare_region(gen_fast, means)
    slow_rare = rare_region(gen_slow, means)
    fast_pct = fast_rare.mean() * 100
    slow_pct = slow_rare.mean() * 100

    lo = reference.min(0)
    hi = reference.max(0)
    pad = 0.15 * (hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    blob(ax, reference[~ref_rare], "#9bbcd6", alpha=0.12)
    blob(ax, reference[ref_rare], "#e06666", alpha=0.2)
    ax.scatter(means[:, 0], means[:, 1], marker="*", c="k", zorder=5)
    style(ax, f"truth  (red = {true_pct:.0f}%)")

    ax = axes[1]
    blob(ax, gen_fast[~fast_rare], "#9bbcd6", alpha=0.18)
    blob(ax, gen_fast[fast_rare], "#e06666", alpha=0.3)
    ax.scatter(means[:, 0], means[:, 1], marker="*", c="k", zorder=5)
    style(ax, f"fast: {fast} steps  (red = {fast_pct:.0f}%)")

    ax = axes[2]
    blob(ax, gen_slow[~slow_rare], "#9bbcd6", alpha=0.18)
    blob(ax, gen_slow[slow_rare], "#e06666", alpha=0.3)
    ax.scatter(means[:, 0], means[:, 1], marker="*", c="k", zorder=5)
    style(ax, f"slow: {slow} steps  (red = {slow_pct:.0f}%)")

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()
    out = FIGDIR / f"fast_vs_slow_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out, true_pct, fast_pct, slow_pct


def displacement_figure(name, split, fast=4, slow=128, n_arrows=300, seed=0):
    d = load_case(name, split)
    means, weights = d["means"], d["weights"]
    terminal = d["terminal"]
    base = float(d["base_variance"])
    vmax = float(d["terminal_variance"])

    gen_fast = sample(terminal, vmax * np.linspace(1, 0, fast + 1), means, weights, base)
    gen_slow = sample(terminal, vmax * np.linspace(1, 0, slow + 1), means, weights, base)
    gap = np.linalg.norm(gen_fast - gen_slow, axis=1)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(terminal), size=min(n_arrows, len(terminal)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for i in idx:
        ax.annotate(
            "",
            xy=(gen_slow[i, 0], gen_slow[i, 1]),
            xytext=(gen_fast[i, 0], gen_fast[i, 1]),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.5, alpha=0.6),
        )
    ax.scatter(gen_slow[idx, 0], gen_slow[idx, 1], s=6, c="#2d6a4f", zorder=5)
    ax.scatter(means[:, 0], means[:, 1], marker="*", c="k", s=120, zorder=6)
    style(ax, f"arrow: fast ({fast}) -> slow ({slow}) landing spot")

    ax = axes[1]
    sc = ax.scatter(gen_slow[:, 0], gen_slow[:, 1], c=gap, s=5, cmap="inferno")
    ax.scatter(means[:, 0], means[:, 1], marker="*", c="c", s=120, zorder=6, edgecolors="k")
    fig.colorbar(sc, ax=ax, label="distance fast moved from slow")
    style(ax, "how far each point was off")

    fig.suptitle(f"{name}: per-point gap between fast and slow")
    fig.tight_layout()
    out = FIGDIR / f"displacement_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out, float(gap.mean()), float(gap.max())


def quality_vs_steps_figure(named_cases, steps=(4, 8, 16, 32, 64, 128)):
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
        ref_mean = reference.mean(0)
        ref_cov = np.cov(reference.T)

        mean_err, cov_err = [], []
        for s in steps:
            gen = sample(terminal, vmax * np.linspace(1, 0, s + 1), means, weights, base)
            mean_err.append(np.linalg.norm(gen.mean(0) - ref_mean))
            cov_err.append(np.linalg.norm(np.cov(gen.T) - ref_cov))

        ax.plot(steps, mean_err, marker="o", color="#1f77b4", label="mean L2 error")
        ax.plot(steps, cov_err, marker="s", color="#d62728", label="cov Frobenius error")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel("number of ODE steps")
        ax.set_ylabel("error")
        ax.set_title(name)
        ax.grid(True, which="both")
        ax.legend()
    for ax in axes[len(named_cases):]:
        ax.axis("off")
    fig.suptitle("Whole-distribution error (mean, covariance) shrinks as steps grow")
    fig.tight_layout()
    out = FIGDIR / "quality_vs_steps.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def density_difference_figure(name, split, fast=4, slow=128, bins=80, span=3.0):
    d = load_case(name, split)
    means, weights = d["means"], d["weights"]
    terminal = d["terminal"]
    base = float(d["base_variance"])
    vmax = float(d["terminal_variance"])

    gen_fast = sample(terminal, vmax * np.linspace(1, 0, fast + 1), means, weights, base)
    gen_slow = sample(terminal, vmax * np.linspace(1, 0, slow + 1), means, weights, base)

    edges = np.linspace(-span, span, bins + 1)
    hf, _, _ = np.histogram2d(gen_fast[:, 0], gen_fast[:, 1], bins=[edges, edges], density=True)
    hs, _, _ = np.histogram2d(gen_slow[:, 0], gen_slow[:, 1], bins=[edges, edges], density=True)
    diff = (hf - hs).T
    limit = float(np.abs(diff).max())

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(
        diff,
        origin="lower",
        extent=[-span, span, -span, span],
        cmap="bwr",
        vmin=-limit,
        vmax=limit,
    )
    ax.scatter(means[:, 0], means[:, 1], marker="*", c="k", s=120, zorder=5)
    fig.colorbar(im, ax=ax, label="fast density minus slow density")
    style(ax, f"{name}: where fast puts extra mass (red) vs less (blue)")
    fig.tight_layout()
    out = FIGDIR / f"density_diff_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def error_vs_overlap_figure(cases, fast=4, slow=128):
    by_geom = {}
    for c in cases:
        by_geom.setdefault((c["separation"], c["rare_weight"]), []).append(c)

    rows = []
    for (sep, w), group in sorted(by_geom.items()):
        fast_errs, slow_errs = [], []
        for c in group:
            d = load_case(c["name"], c["split"])
            means, weights = d["means"], d["weights"]
            reference, terminal = d["reference"], d["terminal"]
            base = float(d["base_variance"])
            vmax = float(d["terminal_variance"])
            target = rare_region(reference, means).mean()
            gf = sample(terminal, vmax * np.linspace(1, 0, fast + 1), means, weights, base)
            gs = sample(terminal, vmax * np.linspace(1, 0, slow + 1), means, weights, base)
            fast_errs.append(rare_region(gf, means).mean() - target)
            slow_errs.append(rare_region(gs, means).mean() - target)
        rows.append((sep, w, np.mean(fast_errs) * 100, np.mean(slow_errs) * 100))

    rows.sort()
    seps = [r[0] for r in rows]
    fast_e = [r[2] for r in rows]
    slow_e = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="#888888", lw=1)
    ax.scatter(seps, fast_e, c="#e06666", s=60, zorder=5, label=f"fast ({fast} steps)")
    ax.scatter(seps, slow_e, c="#2d6a4f", s=60, zorder=5, label=f"slow ({slow} steps)")
    ax.set_xlabel("separation s (smaller = more overlap)")
    ax.set_ylabel("region error: sampler minus target (percentage points)")
    ax.set_title("Fast-sampler region error collapses as clusters separate")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out = FIGDIR / "error_vs_overlap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


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


def overview_figure(cases, mode="truth", steps=None):
    order = sorted(cases, key=lambda c: (c["separation"], c["rare_weight"], c["seed"]))
    cols = 6
    rows = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, order):
        d = load_case(c["name"], c["split"])
        means = d["means"]
        if mode == "truth":
            pts = d["reference"]
            alpha_bg, alpha_rare = 0.08, 0.25
        else:
            grid = float(d["terminal_variance"]) * np.linspace(1, 0, steps + 1)
            pts = sample(d["terminal"], grid, means, d["weights"], float(d["base_variance"]))
            alpha_bg, alpha_rare = 0.18, 0.35
        rare = rare_region(pts, means)
        blob(ax, pts[~rare], "#9bbcd6", alpha=alpha_bg)
        blob(ax, pts[rare], "#e06666", alpha=alpha_rare)
        ax.scatter(means[:, 0], means[:, 1], marker="*", c="k")
        spread = "far apart" if c["separation"] >= 1.4 else "overlap"
        pct = rare.mean() * 100
        lim = float(np.abs(means).max()) + 2.5
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
    out = FIGDIR / f"overview_{suffix}.png"
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
    out = overview_figure(cases, mode="fast", steps=4)
    print(f"Overview (fast sampler): {out.name}")
    out = overview_figure(cases, mode="slow", steps=128)
    print(f"Overview (slow sampler): {out.name}")

    named = [(name, by_name[name]["split"]) for name in highlight]
    out = stepsize_figure(named)
    print(f"Step-size figure: {out.name}")

    print("Fast-vs-slow figures:")
    for name in ["development_r0.02_s0.8_seed940", "development_r0.02_s2.0_seed940"]:
        c = by_name[name]
        out, true_pct, fast_pct, slow_pct = fast_vs_slow_figure(name, c["split"])
        print(f"  {out.name:45s} truth {true_pct:4.1f}%  fast {fast_pct:4.1f}%  slow {slow_pct:4.1f}%")

    print("Per-point displacement figures:")
    for name in ["development_r0.02_s0.8_seed940", "development_r0.02_s2.0_seed940"]:
        c = by_name[name]
        out, gap_mean, gap_max = displacement_figure(name, c["split"])
        print(f"  {out.name:45s} mean gap {gap_mean:.3f}  max gap {gap_max:.3f}")

    print("Density-difference figures:")
    for name in ["development_r0.02_s0.8_seed940", "development_r0.02_s2.0_seed940"]:
        c = by_name[name]
        out = density_difference_figure(name, c["split"])
        print(f"  {out.name}")

    named = [(name, by_name[name]["split"]) for name in highlight]
    out = quality_vs_steps_figure(named)
    print(f"Distribution-quality figure: {out.name}")

    out = error_vs_overlap_figure(cases)
    print(f"Error-vs-overlap figure: {out.name}")
    print(f"\nAll figures saved under: {FIGDIR}")


if __name__ == "__main__":
    main()
