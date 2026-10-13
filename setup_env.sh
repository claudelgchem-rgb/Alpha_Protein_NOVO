#!/usr/bin/env bash
# Set up the primary AlphaProtein Novo environment (.venv).
#
# Why this script exists instead of the plain README commands: the AlphaFold 3
# build pulls its C++ dependencies as GitHub archive tarballs over HTTPS, which
# some sandboxed/proxied environments block (HTTP 403) even though `git clone`
# works. This script pre-clones each dependency at the commit AF3 pins and
# points CMake's FetchContent at the local copies.
#
# Usage:
#   ./setup_env.sh                 # CPU-only JAX
#   AP_NOVO_CUDA=1 ./setup_env.sh  # JAX with CUDA 12 (needs an NVIDIA GPU)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${AP_NOVO_DEPS_DIR:-$HOME/af3-deps}"
AF3_DIR="${AP_NOVO_AF3_DIR:-$DEPS_DIR/alphafold3}"
VENV="$REPO_DIR/.venv"

have() { command -v "$1" >/dev/null 2>&1; }
have uv || { echo "error: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
for t in cmake ninja g++ git curl; do
  have "$t" || { echo "error: $t is required to build AlphaFold 3" >&2; exit 1; }
done

echo "==> Creating Python 3.12 virtual environment at $VENV"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"

echo "==> Installing JAX"
if [[ "${AP_NOVO_CUDA:-0}" == "1" ]]; then
  uv pip install --python "$PY" -U "jax[cuda12]>=0.4.30"
else
  uv pip install --python "$PY" -U jax
fi

# Pinned exactly as AlphaFold 3's CMakeLists.txt declares them. Keep in sync
# when bumping the AlphaFold 3 revision.
mkdir -p "$DEPS_DIR"
fetch_dep() { # <dir> <url> <ref>
  local dir="$DEPS_DIR/$1"
  if [[ -d "$dir/.git" ]]; then echo "    $1 already present"; return; fi
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$2"
  git -C "$dir" fetch -q --depth 1 origin "$3"
  git -C "$dir" checkout -q FETCH_HEAD
  echo "    $1 $(git -C "$dir" rev-parse --short HEAD)"
}

echo "==> Fetching AlphaFold 3 C++ dependencies into $DEPS_DIR"
fetch_dep abseil-cpp      https://github.com/abseil/abseil-cpp         d7aaad83b488fd62bd51c81ecf16cd938532cc0a
fetch_dep pybind11        https://github.com/pybind/pybind11           2e0815278cb899b20870a67ca8205996ef47e70f
fetch_dep pybind11_abseil https://github.com/pybind/pybind11_abseil    bddf30141f9fec8e577f515313caec45f559d319
fetch_dep boost-rx        https://github.com/boostorg/regex            boost-1.83.0
fetch_dep libmcfp         https://github.com/mhekkel/libmcfp           v1.3.1
fetch_dep cifpp           https://github.com/pdb-redo/libcifpp         ac98531a2fc8daf21131faa0c3d73766efa46180
fetch_dep dssp            https://github.com/PDB-REDO/dssp             57560472b4260dc41f457706bc45fc6ef0bc0f10

if [[ ! -d "$DEPS_DIR/my-eigen3" ]]; then
  echo "    eigen 3.4.0"
  curl -sSL -o "$DEPS_DIR/eigen.tar.gz" \
    https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  tar -xzf "$DEPS_DIR/eigen.tar.gz" -C "$DEPS_DIR"
  mv "$DEPS_DIR/eigen-3.4.0" "$DEPS_DIR/my-eigen3"
  rm -f "$DEPS_DIR/eigen.tar.gz"
else
  echo "    my-eigen3 already present"
fi

if [[ ! -d "$AF3_DIR/.git" ]]; then
  echo "==> Cloning AlphaFold 3"
  git clone -q --depth 1 https://github.com/google-deepmind/alphafold3.git "$AF3_DIR"
fi

echo "==> Building and installing AlphaFold 3 (this takes several minutes)"
CMAKE_ARGS=""
for pair in "ABSEIL-CPP:abseil-cpp" "PYBIND11:pybind11" "PYBIND11_ABSEIL:pybind11_abseil" \
            "BOOST-RX:boost-rx" "MY-EIGEN3:my-eigen3" "LIBMCFP:libmcfp" \
            "CIFPP:cifpp" "DSSP:dssp"; do
  CMAKE_ARGS="$CMAKE_ARGS -DFETCHCONTENT_SOURCE_DIR_${pair%%:*}=$DEPS_DIR/${pair##*:}"
done
export CMAKE_ARGS
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
uv pip install --python "$PY" "$AF3_DIR"

echo "==> Installing AlphaProtein Novo (editable) and its dependencies"
uv pip install --python "$PY" --no-deps -e "$REPO_DIR"
# Installed without the jax[cuda12] pin from pyproject.toml so the JAX flavour
# chosen above (CPU or CUDA) is preserved.
uv pip install --python "$PY" \
  absl-py chex "dm-haiku==0.0.17" dm-tree "etils[epath]" flax numpy \
  "pydantic>=2.0" "rdkit==2026.3.3" scipy "tokamax==0.0.12" tqdm zstandard

echo "==> Compiling AlphaFold 3 chemical component data"
"$VENV/bin/build_data"

echo "==> Verifying installation"
"$PY" - <<'PYCODE'
import jax
import alphafold3  # noqa: F401
import alphaprotein_novo
from alphaprotein_novo.data import design_manifest  # noqa: F401
from alphaprotein_novo.model import sampling  # noqa: F401
print(f"  jax {jax.__version__}, devices: {jax.devices()}")
print(f"  alphaprotein_novo {alphaprotein_novo.version.__version__}")
PYCODE

cat <<'EOF'

Environment is ready. Remaining steps before a real run:

  1. Download model weights (subject to their own terms of use -- see
     WEIGHTS_TERMS_OF_USE.md and WEIGHTS_PROHIBITED_USE_POLICY.md):

       mkdir -p models/apnovo_generator models/af3_la
       wget -P models/apnovo_generator https://storage.googleapis.com/alphaprotein_novo/generator.bin.zst
       wget -P models/af3_la https://storage.googleapis.com/alphafold3/af3_leaving_atom.bin.zst

  2. Optionally set up the separate LigandMPNN (PyTorch) environment -- see the
     "LigandMPNN Environment" section of README.md.

  3. Smoke test, then run:

       source .venv/bin/activate
       python run_pipeline.py \
         --manifest=examples/kemp_eliminase/kemp_test_manifest.json \
         --output_dir=./kemp_test

A CUDA GPU is required for realistic runtimes; CPU-only works for imports and
configuration checks but diffusion and folding are impractically slow.
EOF
