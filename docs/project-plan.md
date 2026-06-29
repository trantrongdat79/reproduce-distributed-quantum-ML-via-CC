# 1. Project Aim

This project aims to reproduce and understand the experimental results from
_Distributed quantum machine learning via classical communication_, with the
long-term reproduction target focused on Figure 4 and Table 1.

The project will begin with the CC-DQML setting because classical communication
is the central contribution of the paper and gives us the most useful first
prototype. The first sprint will prioritize a small, working CC-DQML training
loop over full paper-scale accuracy. Later work will expand that prototype into
the complete comparison against non-DQML, NC-DQML, and QC-DQML.

Long-term target:

- Reproduce Figure 4: synthetic 8-dimensional binary classification, DQML model
  behavior, training convergence, and validation accuracy comparisons.
- Reproduce Table 1: validation accuracy mean and standard deviation across
  communication schemes, convolutional sub-layer counts, datasets, and random
  initializations.
- Keep notebooks as line-by-line learning companions for the scripts, not as
  the primary implementation source.

# 2. Directory Structure

The structure below is designed for Sprint 1 and Sprint 2. Later sprints may add
more experiment and reporting folders as the reproduction becomes more complete.

```text
.
├── README.md
├── requirements.txt
├── main.py
├── config/
│   └── Local run configs. Ignored by git.
├── config-example/
│   └── cc_dqml.yaml
├── docs/
│   ├── project-plan.md
│   └── Paper notes, source materials, and reproduction documentation.
├── notebooks/
│   ├── 01_cc_dqml_walkthrough.ipynb
│   └── 02_baseline_comparison.ipynb
├── src/
│   └── cc_dqml/
│       ├── config.py
│       ├── data.py
│       ├── circuits.py
│       ├── train.py
│       └── evaluate.py
└── results/
    └── Generated experiment outputs. Ignored by git.
```

Directory responsibilities:

- `config/`: machine-local experiment configs, seeds, output paths, and runtime
  overrides. This directory is ignored by git.
- `config-example/`: tracked config templates that document expected settings.
- `docs/`: project planning, paper notes, reproduction findings, and generated
  written documentation.
- `notebooks/`: `.ipynb` walkthroughs that run the main script logic step by
  step so we can inspect the implementation details.
- `src/`: reusable Python implementation for data generation, circuits,
  training, and evaluation.
- `results/`: generated metrics, plots, and intermediate experiment outputs.
  This directory is ignored by git.

# 3. Execution Plan (in Sprints)

## Sprint 1: CC-DQML First Run

Goal: build the smallest useful CC-DQML reproduction path.

Main tasks:

- Implement synthetic 8-dimensional binary classification data generation based
  on the paper appendix:
  - 2048 samples.
  - 32 clusters.
  - labels in `{-1, 1}`.
  - train/validation split of 1536/512.
- Implement the first CC-DQML model:
  - two 4-qubit QPUs.
  - embedding layer for two 4-feature partitions.
  - convolutional sub-layers.
  - pooling layers with mid-circuit measurement and feedforward-style classical
    communication.
  - interpret function `w0 P[00] + w1 P[01] + w2 P[10] + w3 P[11]`.
- Add a config-driven local run using `config/cc_dqml.yaml`, copied from
  `config-example/cc_dqml.yaml`.
- Keep default Sprint 1 values small enough for local iteration:
  - model: `cc_dqml`.
  - `L = 3`.
  - one dataset seed.
  - one initialization seed.
  - reduced iteration count for smoke testing.
- Add `notebooks/01_cc_dqml_walkthrough.ipynb` as a step-by-step notebook mirror
  of the CC-DQML script.
- Save basic metrics to `results/`, including loss and validation accuracy.

Success criteria:

- A local CC-DQML run completes from the command line.
- The run writes metrics to `results/`.
- The notebook can walk through the same major steps as the script.

## Sprint 2: Figure 4 and Table 1 Reproduction Framework

Goal: turn the CC-DQML prototype into a comparison framework aimed at Figure 4
and Table 1.

Main tasks:

- Add shared experiment orchestration for multiple models, datasets, and random
  initializations.
- Add baselines:
  - non-DQML with one 4-qubit QPU.
  - NC-DQML with two 4-qubit QPUs and no communication.
  - CC-DQML from Sprint 1.
- Add QC-DQML as either a Sprint 2 stretch goal or the first Sprint 3 task,
  depending on local runtime and implementation complexity.
- Support Table 1-style experiment settings:
  - `L` values: `3`, `5`, `7`, `9`, `15`, `20`.
  - multiple synthetic datasets.
  - multiple random initializations.
  - 1000 training iterations for full reproduction runs.
- Add metrics aggregation:
  - validation accuracy mean.
  - validation accuracy standard deviation.
  - per-run metrics for debugging.
- Add plotting utilities for Figure 4-style outputs:
  - training convergence curves.
  - validation accuracy comparisons.
- Add `notebooks/02_baseline_comparison.ipynb` for line-by-line comparison of
  CC-DQML against the baselines.

Success criteria:

- The project can run a small comparison across non-DQML, NC-DQML, and CC-DQML.
- Results can be aggregated into a Table 1-like format.
- Plotting utilities can produce Figure 4-style training and accuracy views.

## Sprint 3: QC-DQML and Full Table 1 Runs

Placeholder: implement or finalize QC-DQML, then run the full Table 1 experiment
matrix with paper-scale settings.

## Sprint 4: Figure 4 Polish and Reproduction Notes

Placeholder: generate polished Figure 4-style plots, compare deviations from
the paper, and document reproduction assumptions.

## Sprint 5: Effective Dimension and Additional Appendices

Placeholder: reproduce supporting analyses such as effective dimension and
Fisher information spectrum if needed after Figure 4 and Table 1.
