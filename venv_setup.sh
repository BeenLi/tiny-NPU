#!/usr/bin/env bash
# Create .venv (Python 3.11 from .conda-env) and install the Python deps.
# Network: PyPI direct (through the host's http(s)_proxy); torch from the pytorch CPU index.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT/.conda-env/bin:$PATH"
export UV_HTTP_TIMEOUT=120
cd "$ROOT" || exit 1
LOG="$ROOT/venv_setup.log"
rm -f venv_setup.DONE venv_setup.FAIL
exec >"$LOG" 2>&1
echo "[$(date)] start venv setup (pypi; cpu torch); uv=$(command -v uv)"
set -x
uv venv --clear --python "$ROOT/.conda-env/bin/python3.11" .venv || { touch venv_setup.FAIL; exit 1; }
source .venv/bin/activate
uv pip install "numpy>=1.24" "scipy>=1.10" onnx || { touch venv_setup.FAIL; exit 1; }
# pre-install torch's pure-python deps from PyPI, so the CPU-only torch wheel
# does not have to pull them from the (slower) pytorch index
uv pip install sympy networkx jinja2 filelock fsspec mpmath typing-extensions setuptools || { touch venv_setup.FAIL; exit 1; }
# CPU-only torch: ONLY the pytorch cpu index (adding --extra-index makes the resolver pick the CUDA build)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu || { touch venv_setup.FAIL; exit 1; }
# socksio: only needed if all_proxy points at a SOCKS proxy (httpx then requires it)
uv pip install "transformers>=4.30" huggingface_hub safetensors socksio || { touch venv_setup.FAIL; exit 1; }
set +x
python - <<'PY'
import numpy, scipy, onnx, torch, transformers, huggingface_hub, safetensors
print("numpy", numpy.__version__, "scipy", scipy.__version__, "onnx", onnx.__version__)
print("torch", torch.__version__, "transformers", transformers.__version__, "hub", huggingface_hub.__version__, "safetensors", safetensors.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
PY
rc=$?
if [ $rc -eq 0 ]; then touch venv_setup.DONE; else touch venv_setup.FAIL; fi
echo "[$(date)] venv setup exit rc=$rc"
