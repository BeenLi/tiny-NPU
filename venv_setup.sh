#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export UV_HTTP_TIMEOUT=120
ALI=https://mirrors.aliyun.com/pypi/simple
cd <repo> || exit 1
LOG=<repo>/venv_setup.log
rm -f venv_setup.DONE venv_setup.FAIL
exec >"$LOG" 2>&1
echo "[$(date)] start venv setup (aliyun; cpu torch); uv=$(command -v uv)"
set -x
uv venv --clear --python 3.11 .venv || { touch venv_setup.FAIL; exit 1; }
source .venv/bin/activate
uv pip install --index-url "$ALI" "numpy>=1.24" "scipy>=1.10" onnx || { touch venv_setup.FAIL; exit 1; }
# pre-install torch's pure-python deps from the fast mirror, so the CPU-only torch
# wheel does not have to pull them from the (slow) pytorch index
uv pip install --index-url "$ALI" sympy networkx jinja2 filelock fsspec mpmath typing-extensions setuptools || { touch venv_setup.FAIL; exit 1; }
# CPU-only torch: ONLY the pytorch cpu index (adding --extra-index makes the resolver pick the CUDA build)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu || { touch venv_setup.FAIL; exit 1; }
uv pip install --index-url "$ALI" "transformers>=4.30" huggingface_hub safetensors socksio || { touch venv_setup.FAIL; exit 1; }
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
