# ACMA: Delay-Augmented Clustering and Model Approximation

Official implementation of the NeurIPS 2026 paper "Delay-Embedded Representations for Robust Saccade Classification in Noisy Oculographic Signals" by Vladimir Antipov, Artem Badarin, and Alexander Hramov.

This repository contains the implementation of the proposed saccade detector (ACMA), the benchmarking pipeline against nine baseline detectors, and the data and pretrained weights required to reproduce the reported tables.

## Repository layout

```
.
├── acma.py                # ACMA implementation (PCA / clustering variants + waveform fitting)
├── test_tools.py          # Event-level evaluation utilities (1-to-1 matching, label unification)
├── Benchmark.ipynb        # Main benchmark: noise sweep, all detectors, all datasets
├── MakeTables.ipynb       # Aggregates results_metrics.csv into the paper's tables
├── datasets/
│   ├── andersson_data.csv # Andersson et al. (2017) hand-labeled recordings
│   ├── eyelink_data.csv   # EyeLink hand-labeled recordings
│   └── sim_data.csv       # Synthetic recordings with ground-truth events
└── training/
    ├── weights_andersson  # U'n'Eye weights, trained on andersson
    ├── weights_eyelink    # U'n'Eye weights, trained on eyelink
    └── weights_sim        # U'n'Eye weights, trained on sim
```

All CSV files share the schema `t,x,y,label`, where `t` is time in seconds, `x,y` are gaze coordinates in pixels, and `label ∈ {fix, sac, blink}`.

## Setup

Tested with Python 3.10. Install dependencies into a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `peyes`, `I2MC`, and `uneye` pull in their own dependencies (PyTorch, matplotlib, etc.).
- A CUDA-capable GPU is **not** required; U'n'Eye runs on CPU within minutes per dataset.

## Reproducing the results

1. Run `Benchmark.ipynb` end-to-end. It iterates every detector × every dataset × every RMS noise level and writes `results_metrics.csv`.
2. Run `MakeTables.ipynb` to obtain the aggregate, per-dataset, per-class, and per-noise-regime tables, plus the Friedman + Wilcoxon tests reported in the paper.

The full sweep takes on the order of an hour on a modern laptop CPU; the hyperparameter grid in the last cell of `Benchmark.ipynb` is parallelised via `joblib`.

## License and data attribution

The Andersson and EyeLink datasets are redistributed under their original licenses; please cite the original publications when using them. The synthetic dataset and all code in this repository are released under the MIT license upon de-anonymization.
