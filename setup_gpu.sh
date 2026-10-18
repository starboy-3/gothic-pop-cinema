#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077

fail() { printf '%s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == Linux ]] || fail 'Run setup only on the Linux NVIDIA GPU host.'
[[ $# -eq 0 ]] || fail 'Usage: PERSISTENT_WORKSPACE=/path/to/persistent/workspace bash setup_gpu.sh'
for command in git python3 nvidia-smi; do
    command -v "$command" >/dev/null || fail "Required command missing: $command"
done
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ required")'

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -n "${PERSISTENT_WORKSPACE:-}" ]] || fail 'Set PERSISTENT_WORKSPACE to the mounted persistent workspace containing this project.'
workspace="$(cd -- "$PERSISTENT_WORKSPACE" && pwd -P)"
[[ "$workspace" != / ]] || fail 'PERSISTENT_WORKSPACE must identify the persistent workspace, not /.'
case "$project_root/" in
    "$workspace/"*) ;;
    *) fail 'Copy the project into PERSISTENT_WORKSPACE before running setup.' ;;
esac

export XDG_CACHE_HOME="$project_root/.cache"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$XDG_CACHE_HOME/huggingface/hub"
export HF_XET_CACHE="$XDG_CACHE_HOME/huggingface/xet"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export npm_config_cache="$XDG_CACHE_HOME/npm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR" "$HF_HOME" "$project_root/vendor"
unset PYTHONHOME PYTHONPATH

toolkit="$project_root/vendor/ai-toolkit"
commit=8cf43dd2cc1fe828884001aceeda044649d9abd0
if [[ ! -e "$toolkit" ]]; then
    git clone --no-checkout https://github.com/ostris/ai-toolkit.git "$toolkit"
    git -C "$toolkit" checkout --detach "$commit"
fi
[[ -d "$toolkit/.git" ]] || fail 'vendor/ai-toolkit must be its own upstream Git checkout.'
[[ "$(git -C "$toolkit" rev-parse HEAD)" == "$commit" ]] || fail 'Existing AI Toolkit checkout differs from the reviewed commit; use a clean pinned checkout.'
[[ -z "$(git -C "$toolkit" status --porcelain --untracked-files=no)" ]] || fail 'AI Toolkit has tracked changes; restore a clean pinned checkout before setup.'
git -C "$toolkit" submodule update --init --recursive
export UV_PYTHON_INSTALL_DIR="$toolkit/.uv/python"
cd -- "$toolkit"
python3 -m manager install
toolkit_python="$(python3 -c 'from manager.util import venv_python; print(venv_python())')"
"$toolkit_python" - <<'PY'
import torch
import torchvision
import torchaudio

if not torch.cuda.is_available() or not torch.version.cuda:
    raise SystemExit("The installed PyTorch cannot use CUDA; check the NVIDIA driver and GPU visibility.")
print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}; built arches {torch.cuda.get_arch_list()}")
for device_index in range(torch.cuda.device_count()):
    with torch.cuda.device(device_index):
        capability = torch.cuda.get_device_capability(device_index)
        print(f"GPU {device_index}: {torch.cuda.get_device_name(device_index)}; compute capability {capability}")
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("This training setup requires a GPU with BF16 support.")
        values = torch.ones((32, 32), device=f"cuda:{device_index}", dtype=torch.bfloat16)
        result = values @ values
        torch.cuda.synchronize(device_index)
        if not torch.all(result == 32).item():
            raise SystemExit("CUDA matrix multiplication verification failed.")
print("CUDA runtime and BF16 execution verified.")
PY
printf '%s\n' 'Setup complete. No model weights downloaded and no training started.' \
    'Before training, provide HF_TOKEN through a protected environment or authenticate interactively on the GPU host. Credentials stay outside the project; model caches use the persistent workspace.'
