import json
import math
from pathlib import Path

import torch

from models import GaussianMixture, ROOT, load_case, manifest, quadratic_grid, rare_region, uniform_grid

from .experiment_abc import Experiment


def default_model_factory(d):
    return GaussianMixture(d["means"], d["weights"], d["base_variance"])


def auroc(score, label):
    score = score.double()
    label = label.bool()
    npos = int(label.sum())
    nneg = label.numel() - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inverse, counts = torch.unique(score, return_inverse=True, return_counts=True)
    average_rank = counts.cumsum(0) - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def bootstrap_interval(score, label, n_boot, gen):
    n = score.numel()
    draws = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=gen)
        value = auroc(score[idx], label[idx])
        if not math.isnan(value):
            draws.append(value)
    if len(draws) < max(2, n_boot // 2):
        return [float("nan"), float("nan")]
    t = torch.tensor(draws, dtype=torch.float64)
    return [float(t.quantile(0.025)), float(t.quantile(0.975))]


def clean(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def nanmean(values):
    t = torch.tensor([v for v in values if v is not None and not math.isnan(v)], dtype=torch.float64)
    return float(t.mean()) if t.numel() else float("nan")


class SensitivityExperiment(Experiment):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(config)
        self.device = config.get("device", "cpu")
        self.model_factory = config.get("model_factory", default_model_factory)
        self.n_fast = config.get("n_fast", 8)
        self.ref_steps = config.get("ref_steps", 1024)
        self.cheap = config.get("cheap", {"K": 4, "finish": 16})
        self.expensive = config.get("expensive", {"K": 8, "finish": 128})
        self.use_expensive = config.get("use_expensive", True)
        self.rel = config.get("rel", 0.2)
        self.checkpoint = config.get("checkpoint", 4)
        self.refine_steps = config.get("refine_steps", 128)
        self.taus = config.get("taus", [0.25, 0.5, 0.75])
        self.baseline_steps = config.get("baseline_steps", [8, 16, 32, 64, 128])
        self.n_boot = config.get("n_boot", 100)
        self.seed = config.get("seed", 0)
        self.window_edges = config.get("window_edges", [0, 0.25, 0.5, 1, 2, 4, 8, 16])
        self.frozen = config.get("frozen", False)
        self.only = config.get("cases")
        self.out_dir = Path(config.get("out_dir", ROOT / "results" / "sensitivity"))

    def setup(self):
        self.cases = manifest()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        return None

    def _select(self, split):
        chosen = [c for c in self.cases if c["split"] == split]
        if self.only:
            chosen = [c for c in chosen if c["name"] in self.only]
        return chosen

    def _sensitivity(self, model, means, x0, v0, var, spec, gen):
        grid = uniform_grid(v0, spec["finish"])
        base_label = rare_region(model.sample(x0, grid), means)
        z = torch.randn(spec["K"] * x0.shape[0], x0.shape[1], dtype=x0.dtype, generator=gen)
        kicked = x0.repeat(spec["K"], 1) + z * self.rel * var ** 0.5
        labels = rare_region(model.sample(kicked, grid), means).reshape(spec["K"], -1)
        return (labels != base_label[None]).double().mean(0)

    def _strategy(self, label, fine, extra_evals):
        return {
            "evals": self.n_fast + float(extra_evals),
            "disagree": int((label != fine).sum()),
            "false_rare": int((label & ~fine).sum()),
            "missed_rare": int((~label & fine).sum()),
            "rate_error_pct": float((label.double().mean() - fine.double().mean()) * 100),
        }

    def window_damage(self, model, means, terminal, T, fine):
        v = uniform_grid(T, self.ref_steps)
        rows = []
        for lo, hi in zip(self.window_edges[:-1], self.window_edges[1:]):
            keep = ~((v > lo + 1e-9) & (v < hi - 1e-9))
            label = rare_region(model.sample(terminal, v[keep]), means)
            rows.append({
                "window": [lo, hi],
                "flips": int((label != fine).sum()),
                "rate_shift_pct": float((label.double().mean() - fine.double().mean()) * 100),
            })
        return rows

    def analyze_case(self, c):
        d = load_case(c["name"], c["split"], device=self.device, dtype=torch.float64)
        model = self.model_factory(d)
        means, weights = d["means"], d["weights"]
        terminal = d["terminal"]
        T, base = d["terminal_variance"], d["base_variance"]
        gen = torch.Generator().manual_seed(self.seed + 1000 * c["seed"] + 7)
        boot_gen = torch.Generator().manual_seed(self.seed + 31)

        fine = rare_region(model.sample(terminal, uniform_grid(T, self.ref_steps)), means)
        grid = uniform_grid(T, self.n_fast)
        xs = [terminal]
        for k in range(self.n_fast):
            xs.append(model.sample(xs[-1], grid[k:k + 2]))
        lfast = rare_region(xs[-1], means)
        disagree = lfast != fine
        in_group = lfast

        out = {
            "case": c["name"],
            "n": int(terminal.shape[0]),
            "fine_rare": int(fine.sum()),
            "fast_rare": int(lfast.sum()),
            "disagree": int(disagree.sum()),
            "false_rare": int((lfast & ~fine).sum()),
            "missed_rare": int((~lfast & fine).sum()),
            "label_alone_auroc": auroc(lfast.double(), disagree),
            "auroc": {},
        }

        out["windows"] = self.window_damage(model, means, terminal, T, fine)
        kept = {}
        for k in range(1, self.n_fast):
            x0 = xs[k]
            v0 = float(grid[k])
            var = base + v0
            delta = x0[:, None, :] - means[None]
            p = torch.softmax(torch.log(weights)[None] - 0.5 * (delta ** 2).sum(-1) / var, dim=1)
            scores = {
                "sens_cheap": self._sensitivity(model, means, x0, v0, var, self.cheap, gen),
                "entropy": -(p * torch.log(p + 1e-300)).sum(1),
                "score_norm": (model.score(x0, v0) ** 2).sum(1),
                "radius": x0.norm(dim=1),
            }
            if self.use_expensive:
                scores["sens_exp"] = self._sensitivity(model, means, x0, v0, var, self.expensive, gen)
            row = {"v0": v0}
            for name, s in scores.items():
                row[name] = {
                    "all": auroc(s, disagree),
                    "within": auroc(s[in_group], disagree[in_group]),
                    "within_ci": bootstrap_interval(s[in_group], disagree[in_group], self.n_boot, boot_gen),
                }
            out["auroc"][str(k)] = row
            if k == self.checkpoint:
                kept = {"x0": x0, "v0": v0, "sens": scores["sens_cheap"]}

        strategies = {}
        f = float(lfast.double().mean())
        for n in self.baseline_steps:
            label = rare_region(model.sample(terminal, uniform_grid(T, n)), means)
            strategies[f"uniform_{n}"] = self._strategy(label, fine, n - self.n_fast)
            label = rare_region(model.sample(terminal, quadratic_grid(T, n)), means)
            strategies[f"quadratic_{n}"] = self._strategy(label, fine, n - self.n_fast)

        if kept:
            ref_label = rare_region(model.sample(kept["x0"], uniform_grid(kept["v0"], self.refine_steps)), means)
            detector_cost = self.cheap["finish"] * (1 + self.cheap["K"])
            strategies["refine_all"] = self._strategy(
                torch.where(lfast, ref_label, lfast), fine, f * self.refine_steps
            )
            for tau in self.taus:
                flagged = lfast & (kept["sens"] >= tau)
                p_flag = float(flagged.double().mean())
                extra = f * detector_cost + p_flag * self.refine_steps
                strategies[f"gate_{tau}"] = self._strategy(torch.where(flagged, ref_label, lfast), fine, extra)
                n_flag = int(flagged.sum())
                pick = torch.zeros_like(lfast)
                idx = torch.nonzero(lfast).flatten()
                chosen = idx[torch.randperm(idx.numel(), generator=gen)[:n_flag]]
                pick[chosen] = True
                strategies[f"random_{tau}"] = self._strategy(torch.where(pick, ref_label, lfast), fine, extra)
                matched = max(self.n_fast, int(round(self.n_fast + extra)))
                label = rare_region(model.sample(terminal, uniform_grid(T, matched)), means)
                strategies[f"uniform_matched_{tau}"] = self._strategy(label, fine, matched - self.n_fast)
                strategies[f"uniform_matched_{tau}"]["steps"] = matched
        out["strategies"] = strategies
        return out

    def _aggregate(self, results):
        summary = {"n_cases": len(results), "strategies": {}, "auroc": {}}
        for key in results[0]["strategies"]:
            rows = [r["strategies"][key] for r in results]
            summary["strategies"][key] = {
                m: nanmean([row[m] for row in rows])
                for m in ("evals", "disagree", "false_rare", "missed_rare", "rate_error_pct")
            }
        for k in results[0]["auroc"]:
            summary["auroc"][k] = {"v0": results[0]["auroc"][k]["v0"]}
            for name in results[0]["auroc"][k]:
                if name == "v0":
                    continue
                rows = [r["auroc"][k][name] for r in results]
                above = [
                    row["within_ci"][0] > 0.5 for row in rows if not math.isnan(row["within_ci"][0])
                ]
                summary["auroc"][k][name] = {
                    "all": nanmean([row["all"] for row in rows]),
                    "within": nanmean([row["within"] for row in rows]),
                    "cases_ci_above_half": sum(above),
                    "cases_scored": len(above),
                }
        by_geometry = {}
        for r in results:
            by_geometry.setdefault(r["case"].rsplit("_seed", 1)[0], []).append(r)
        summary["window_edges"] = self.window_edges
        summary["windows"] = {
            g: [nanmean([r["windows"][i]["flips"] for r in rr]) for i in range(len(self.window_edges) - 1)]
            for g, rr in by_geometry.items()
        }
        summary["label_alone_auroc"] = nanmean([r["label_alone_auroc"] for r in results])
        summary["totals"] = {
            m: sum(r[m] for r in results)
            for m in ("n", "fine_rare", "fast_rare", "disagree", "false_rare", "missed_rare")
        }
        return summary

    def _evaluate(self, split, tag):
        results = []
        for c in self._select(split):
            r = self.analyze_case(c)
            results.append(r)
            (self.out_dir / f"{c['name']}.json").write_text(json.dumps(clean(r), indent=2))
            print(f"  {c['name']}: fast-rare {r['fast_rare']}, spurious {r['false_rare']}, missed {r['missed_rare']}")
        summary = self._aggregate(results)
        (self.out_dir / f"summary_{tag}.json").write_text(json.dumps(clean(summary), indent=2))
        return summary

    def validate(self):
        return self._evaluate("development", "development")

    def test(self):
        if not self.frozen:
            raise RuntimeError("final cases are locked: freeze the protocol, then pass frozen=True")
        return self._evaluate("final", "final")

    def report(self, summary):
        t = summary["totals"]
        print(f"\n{summary['n_cases']} cases, {t['n']} trajectories: fast-rare {t['fast_rare']}, "
              f"spurious {t['false_rare']}, missed {t['missed_rare']}; "
              f"fast label alone AUROC {summary['label_alone_auroc']:.3f}")
        print("\nWithin the fast-rare group, mean AUROC for predicting a spurious label (cases whose 95% CI clears 0.5):")
        names = [n for n in summary["auroc"][next(iter(summary["auroc"]))] if n != "v0"]
        print("  v0    " + "  ".join(f"{n:>14s}" for n in names))
        for k, row in summary["auroc"].items():
            cells = [f"{row[n]['within']:.3f} ({row[n]['cases_ci_above_half']}/{row[n]['cases_scored']})" for n in names]
            print(f"  {row['v0']:5.1f} " + "  ".join(f"{c:>14s}" for c in cells))
        print("\nStrategies at matched cost (means over cases; reference is the "
              f"{self.ref_steps}-step run):")
        print(f"  {'strategy':<22s} {'evals':>8s} {'disagree':>9s} {'false_rare':>10s} {'rate err %':>10s}")
        for key, row in summary["strategies"].items():
            print(f"  {key:<22s} {row['evals']:8.1f} {row['disagree']:9.1f} {row['false_rare']:10.1f} {row['rate_error_pct']:10.2f}")
        edges = summary["window_edges"]
        print("\nLabel flips when one window of v is coarsened to a single step (mean over seeds):")
        print("  " + f"{'geometry':<26s}" + "".join(f"{f'{lo:g}-{hi:g}':>9s}" for lo, hi in zip(edges[:-1], edges[1:])))
        for g, flips in summary["windows"].items():
            print("  " + f"{g:<26s}" + "".join(f"{x:9.0f}" for x in flips))

    def run(self):
        self.setup()
        self.train()
        print("Development cases:")
        summary = self.validate()
        self.report(summary)
        if self.frozen:
            print("\nFinal cases:")
            self.report(self.test())
        return summary


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+")
    p.add_argument("--n-boot", type=int, default=100)
    p.add_argument("--checkpoint", type=int, default=4)
    p.add_argument("--no-expensive", action="store_true")
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--out-dir")
    a = p.parse_args()
    config = {"cases": a.cases, "n_boot": a.n_boot, "checkpoint": a.checkpoint,
              "use_expensive": not a.no_expensive, "frozen": a.frozen}
    if a.out_dir:
        config["out_dir"] = a.out_dir
    SensitivityExperiment(config).run()


if __name__ == "__main__":
    main()
