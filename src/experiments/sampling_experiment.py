import json
from pathlib import Path

import numpy as np
import torch

from models import GaussianMixture, load_case, manifest, quadratic_grid, uniform_grid, wilson

from .experiment_abc import Experiment


class SamplingExperiment(Experiment):
    """Exact-score probability-flow ODE sampling, torch replica of the starter.

    No training: the score is closed form, so train() is a no-op. validate() and
    test() measure the rare-region probability and distribution-quality metrics
    at each requested step count.
    """

    def __init__(self, config):
        super().__init__(config)
        self.case = config["case"]
        self.steps = config.get("steps", [8, 16, 32])
        self.schedule = config.get("schedule", "uniform")
        self.device = config.get("device", "cpu")
        self.output = config.get("output")
        self.info = next(c for c in manifest() if c["name"] == self.case)

    def setup(self):
        c = load_case(self.case, self.info["split"], device=self.device, dtype=torch.float64)
        self.data = c
        self.model = GaussianMixture(c["means"], c["weights"], c["base_variance"])
        self.reference = c["reference"]
        self.ref_rare = self.model.rare_region(self.reference)
        self.ref_mean = self.reference.mean(0)
        self.ref_cov = torch.cov(self.reference.T)

    def train(self):
        return None

    def _grid(self, steps):
        vmax = self.data["terminal_variance"]
        if self.schedule == "uniform":
            return uniform_grid(vmax, steps)
        return quadratic_grid(vmax, steps)

    def _run_steps(self, steps):
        generated = self.model.sample(self.data["terminal"], self._grid(steps))
        hit = self.model.rare_region(generated)
        target = self.ref_rare.float().mean().item()
        got = hit.float().mean().item()
        cov_err = torch.linalg.norm(torch.cov(generated.T) - self.ref_cov).item()
        return {
            "score_evaluations": steps,
            "schedule": self.schedule,
            "rare_region_probability": got,
            "rare_region_error": got - target,
            "generated_wilson_95": wilson(int(hit.sum().item()), len(hit)),
            "mean_error_l2": torch.linalg.norm(generated.mean(0) - self.ref_mean).item(),
            "covariance_error_frobenius": cov_err,
            "finite_samples": bool(torch.isfinite(generated).all().item()),
        }

    def validate(self):
        return [self._run_steps(s) for s in self.steps]

    def test(self):
        return self.validate()

    def run(self):
        self.setup()
        self.train()
        target = self.ref_rare.float().mean().item()
        out = {
            "case": self.case,
            "definition": "Probability of nearest-mean region 2, not latent mixture membership",
            "reference_rare_region_probability": target,
            "reference_wilson_95": wilson(int(self.ref_rare.sum().item()), len(self.ref_rare)),
            "latent_rare_weight": float(self.data["weights"][2].item()),
            "backend": "torch",
            "runs": self.test(),
        }
        if self.output:
            Path(self.output).write_text(json.dumps(out, indent=2))
        return out


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--case", default="development_r0.02_s0.8_seed940")
    p.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--schedule", choices=["uniform", "quadratic"], default="uniform")
    p.add_argument("--output", default="results_torch.json")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.steps = [8, 16]
    exp = SamplingExperiment(
        {"case": a.case, "steps": a.steps, "schedule": a.schedule, "output": a.output}
    )
    out = exp.run()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
