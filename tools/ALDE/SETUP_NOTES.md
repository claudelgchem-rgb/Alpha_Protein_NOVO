# ALDE in this repository

## Provenance

`tools/ALDE` is a copy of [jsunn-y/ALDE](https://github.com/jsunn-y/ALDE) at
upstream commit `d0b3593dd17987c9864830d36f5976b1ad04a619`, vendored rather than
added as a submodule so this checkout is self-contained.

- **Licence:** MIT (`tools/ALDE/LICENSE`, Copyright (c) 2025 Jason Yang). No
  non-commercial clause — commercial use is permitted. The licence file is kept
  unmodified.
- **Paper:** Yang et al., *Active learning-assisted directed evolution*,
  Nature Communications (2025). DOI
  [10.5281/zenodo.14194551](https://doi.org/10.5281/zenodo.14194551) for the code.

Two upstream files are **not** copied because they are large and are
regenerated locally by `generate_domain.py`:

| Path | Size | How to recreate |
|---|---|---|
| `data/ParPgb/all_combos.csv` | 19 MB | `generate_domain.py --name=ParPgb --nsites=5` |
| `data/ParPgb/onehot_x.pt` | 1.2 GB | same command |

Both are listed in `.gitignore`. Everything else from upstream is present,
including the ParPgb wet-lab campaign data used in the paper.

## What ALDE is for

ALDE optimises the **activity of an existing protein** over a combinatorial
site-saturation library. It is not a de novo design tool and it does not
predict activity from sequence alone — it is the model half of a closed loop:

```
  pick mutable residues  ->  generate_domain.py (once per campaign)
          |
          v
  screen a batch in the lab  ->  fitness.csv (Combo + activity column)
          |
          v
  execute_production.py  ->  indices of the next batch to screen
          |
          +---------------- repeat, 3-4 rounds ----------------+
```

Each round fits 4 model families (`BOOSTING_ENSEMBLE`, `GP_BOTORCH`,
`DNN_ENSEMBLE`, `DKL_BOTORCH`) under 3 acquisition functions (`GREEDY`, `UCB`,
`TS`) and returns the variant indices to screen next. The model/acquisition
grid is hard-coded in `execute_production.py`, so narrowing it means editing
that loop.

## Environment

`../../setup_alde.sh` (at the repo root) creates `.venv-alde` with Python 3.11.

Upstream's `alde.yml` is a conda file pinning `pytorch-cuda=12.1`. The setup
script installs the same pinned versions from PyPI/PyTorch wheels instead, CPU
by default:

| Package | Version | Note |
|---|---|---|
| torch | 2.1.1+cpu | `ALDE_CUDA=1` installs the cu121 build |
| gpytorch | 1.11 | as `alde.yml` |
| botorch | 0.9.4 | as `alde.yml` |
| pyro-ppl | 1.8.6 | as `alde.yml` |
| pandas | 1.5.3 | as `alde.yml`; forces numpy < 2 |

This env is separate from the repo's `.venv` (AlphaProtein Novo / JAX). They
pin incompatible numpy majors, so do not merge them.

## Gotchas

- **`--device` defaults to `cuda`.** On a machine without an NVIDIA GPU you must
  pass `--device=cpu` to `execute_production.py` or it fails at startup.
- **The design space is 20^nsites.** 5 sites is 3.2M variants and ~1.2 GB of
  one-hot encodings; `generate_domain.py` took ~1.5 min and the tensor is held
  in memory during a round. 6 sites would be 64M variants — check RAM first.
- **`fitness.csv` needs a `Combo` column** holding only the residues at the
  mutated positions (e.g. `FRMNY` for a 5-site library), plus a column with the
  value to maximise, named via `--obj_col`.
- Unseen variants in the domain are filled with 0 as placeholders, so the
  "regret" numbers printed during a production run are meaningless. Only the
  queried indices matter.

## Verified here

- Dependencies install and import cleanly (`torch 2.1.1+cpu`, `gpytorch 1.11`,
  `botorch 0.9.4`, `xgboost 3.2.0`).
- `generate_domain.py --name=ParPgb --nsites=5` produced the 3.2M-variant
  design space.
- `objectives.Production` loads the 216-variant ParPgb round-1 training set
  against that domain (encoding dim 100, ymax 3.82).

See the repo root `SETUP_NOTES.md` for the AlphaProtein Novo environment, which
is a separate matter — note that AP Novo's *model weights* are licensed for
non-commercial use only, whereas ALDE carries no such restriction.
