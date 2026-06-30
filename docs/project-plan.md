# 1. Project Aim

Reproduce and understand the results of _Distributed quantum machine learning via
classical communication_ using the authors' original `DQML/` source release as
the baseline implementation. Project code should support inspection,
verification, aggregation, and reproduction rather than replace the original
implementation.

# 2. Directory Structure

```text
.
├── DQML/
│   ├── DQML/                 # Original author source code
│   ├── Dataset/              # Original datasets
│   ├── DatafromPaper/        # Extracted saved paper-result data
│   ├── DatafromPaper.zip
│   ├── Demo.ipynb
│   ├── README.md
│   └── LICENSE
├── config/                   # Local reproduction configs, ignored by git
├── config-example/           # Tracked example configs
├── docs/                     # Plans, paper notes, and reproduction notes
├── notebooks/                # Our exploratory/reproduction notebooks
├── results/                  # Our generated reproduction outputs
├── results-example/          # Small tracked example outputs, if useful
├── README.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

Directory responsibilities:

- `DQML/`: original research artifact from the authors. Keep this as the
  baseline source for reproduction.
- `config/`: machine-local reproduction configs, runtime overrides, and output
  paths. This directory is ignored by git.
- `config-example/`: tracked config templates for reproducible workflows.
- `docs/`: project planning, paper notes, source materials, and reproduction
  findings.
- `notebooks/`: exploratory notebooks for inspecting the original code, data,
  and generated results.
- `results/`: generated metrics, plots, logs, and intermediate reproduction
  outputs. This directory is ignored by git.
- `results-example/`: small tracked result examples when they are useful for
  documenting expected output shape.

# 3. Execution Plan

## Sprint 1: Reproduce From Original Source

- Treat `DQML/` as the source of truth.
- Verify the original demo and core functions run in the project environment.
- Inspect `DatafromPaper/` and document what each result group contains.
- Add minimal scripts or notebooks only when needed to summarize or replot
  original results.
- Avoid modifying the original source unless a compatibility patch is required
  and documented.

## Sprint 2: Placeholder

To be defined after Sprint 1 results are understood.
