# Setup notes for this checkout

The upstream `README.md` is the authoritative guide. This file records what was
actually done in this repository and the two things that differ from the
README's happy path.

## What is installed

`./setup_env.sh` reproduces the environment:

- `.venv/` — Python 3.12, created with `uv`
- `jax` (CPU build by default; `AP_NOVO_CUDA=1 ./setup_env.sh` for CUDA 12)
- `alphafold3`, built from source
- `alphaprotein_novo`, installed editable from this checkout
- AF3 chemical component data compiled via `build_data`

Verified working: `import alphafold3`, `import alphaprotein_novo`, and
`python run_pipeline.py --help`.

## Deviation 1: AlphaFold 3 dependency fetching

AF3's `CMakeLists.txt` fetches abseil, pybind11, pybind11_abseil, boost.regex,
eigen, libmcfp, libcifpp and dssp as **archive tarballs** over HTTPS. In a
sandboxed or proxied network these GitHub archive URLs can be refused (HTTP 403)
while `git clone` of the same repositories still succeeds, and the build fails
with:

```
CMake Error ... Build step for pybind11 failed: 1
*** CMake configuration failed
```

`setup_env.sh` works around this by `git fetch --depth 1`-ing each dependency at
the exact commit or tag AF3 pins (into `~/af3-deps` by default, override with
`AP_NOVO_DEPS_DIR`) and passing `-DFETCHCONTENT_SOURCE_DIR_<NAME>=...` through
`CMAKE_ARGS`. Eigen comes from GitLab, which was not blocked. If you bump the
AF3 revision, re-sync the pins in `setup_env.sh` against its `CMakeLists.txt`.

## Deviation 2: JAX flavour

`pyproject.toml` pins `jax[cuda12]==0.10.2`, so a plain `pip install -e .` drags
in the CUDA plugin. `setup_env.sh` installs the package with `--no-deps` and then
its dependencies explicitly, minus that pin, so the JAX build chosen at the top
of the script is kept.

On a machine with no GPU, the CUDA plugin (pulled in transitively by AF3) prints
a `cuInit(0) failed: Unknown CUDA error 303` traceback at import and then falls
back to CPU. It is noisy but harmless; `jax.devices()` reports `[CpuDevice(id=0)]`.

## Not installed here

- **Model weights.** Not part of the package and subject to separate terms —
  see `WEIGHTS_TERMS_OF_USE.md` and `WEIGHTS_PROHIBITED_USE_POLICY.md`. Scripts
  look under `./models/apnovo_generator` (generator, ~522 MiB) and
  `./models/af3_la` (AF3 leaving-atom, ~973 MiB) by default. Download commands
  are printed at the end of `setup_env.sh` and documented in `README.md`.
- **LigandMPNN environment.** Optional, separate PyTorch venv — see the
  "LigandMPNN Environment" section of `README.md`. Only needed for the optional
  sequence-redesign stage; `run_pipeline.py` can skip it.
- **A GPU.** Required in practice. The pipeline shards stages 1 and 3 across
  visible GPUs; CPU-only is for import and configuration checks, not for
  diffusion or folding.
