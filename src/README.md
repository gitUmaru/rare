# src/ layout

- `starter/` — course-provided starter files, unchanged. numpy/scipy. Run from the
  project root, for example `python src/starter/starter.py --quick`.
- `models/` — torch reimplementation of the mixture, score, and probability-flow ODE
  sampler (`mixture.py`). Matches the starter numerics to machine precision.
- `experiments/` — `Experiment` base class (`experiment_abc.py`) plus two torch
  experiments that replicate the starter results and the figures. No training is
  needed since the score is exact, so `train()` is a no-op.
- `plot_data.py` — the original numpy/matplotlib figure script.

## Running the torch replica

Put `src` on the path, then run either experiment:

```
PYTHONPATH=src python -m experiments.sampling_experiment --case development_r0.02_s0.8_seed940 --steps 8 16 32
PYTHONPATH=src python -m experiments.figures_experiment
```

The sampling experiment writes a results JSON that matches the starter's numbers; the
figures experiment writes `figures/torch_*.png`, replicas of the numpy figures.

## Backend notes

The torch code defaults to CPU and float64 to match the numpy starter exactly. It runs
on Apple Silicon (MPS is available) by passing `device="mps"`, though float64 on MPS is
limited, so CPU float64 is the default for numerical parity.
