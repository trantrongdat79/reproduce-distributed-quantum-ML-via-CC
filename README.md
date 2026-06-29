# Reproduce Distributed Quantum ML via Classical Communication

This repository is a reproduction project for _Distributed quantum machine
learning via classical communication_. The long-term target is to reproduce the
paper's Figure 4 and Table 1, starting with a local CC-DQML prototype and then
expanding into comparisons against non-DQML, NC-DQML, and QC-DQML.

## Project Layout

- `docs/`: project plans, paper notes, and reproduction documentation.
- `notebooks/`: `.ipynb` walkthroughs for running the main script logic line by
  line.
- `config/`: local run configs ignored by git.
- `config-example/`: tracked config templates and guidance.
- `results/`: generated metrics and plots ignored by git.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local config from the tracked example:

```bash
mkdir -p config
cp config-example/cc_dqml.yaml config/cc_dqml.yaml
```

## Run

Run the Sprint 1 CC-DQML prototype:

```bash
python main.py --config config/cc_dqml.yaml
```

The tracked example config is intentionally modest, but still uses the paper's
8-dimensional synthetic dataset shape. For a very quick local smoke test, create
a smaller ignored config under `config/` using the same keys as
`config-example/cc_dqml.yaml`.

For exploratory work, open the notebooks in `notebooks/` and run the matching
script logic step by step.

The run writes metrics and a config snapshot under the configured `results/`
directory.
