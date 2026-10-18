#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077
trap 'exit 130' INT
trap 'exit 143' TERM

fail() { printf '%s\n' "$*" >&2; exit 1; }

[[ $# -gt 0 ]] || fail 'Usage: bash run_training.sh STYLE [STYLE ...]'
for style in "$@"; do
    [[ "$style" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || fail 'Style slugs must contain lowercase letters/digits separated by single hyphens.'
done
[[ "$(uname -s)" == Linux ]] || fail 'Run training only on the Linux NVIDIA GPU host.'
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$project_root"
toolkit="$project_root/vendor/ai-toolkit"
[[ -d "$toolkit/.git" ]] || fail 'Run setup_gpu.sh on the GPU host first.'
[[ "$(git -C "$toolkit" rev-parse HEAD)" == 8cf43dd2cc1fe828884001aceeda044649d9abd0 ]] || fail 'AI Toolkit is not at the reviewed commit.'
[[ -z "$(git -C "$toolkit" status --porcelain --untracked-files=no)" ]] || fail 'AI Toolkit has tracked changes; use the clean reviewed checkout.'
toolkit_python="$toolkit/.venv/bin/python3"
if [[ ! -d "$toolkit/.venv" ]]; then
    toolkit_python="$toolkit/venv/bin/python3"
fi
[[ -x "$toolkit_python" ]] || fail 'AI Toolkit Python environment missing; run setup_gpu.sh.'
unset PYTHONHOME PYTHONPATH
export VIRTUAL_ENV="$(dirname -- "$(dirname -- "$toolkit_python")")"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export XDG_CACHE_HOME="$project_root/.cache"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$XDG_CACHE_HOME/huggingface/hub"
export HF_XET_CACHE="$XDG_CACHE_HOME/huggingface/xet"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR"

"$toolkit_python" - "$@" <<'PY'
import subprocess
import sys
import os
from pathlib import Path

import torch
import yaml

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; run setup_gpu.sh and check GPU visibility.")
for style in sys.argv[1:]:
    style_dir = Path("styles") / style
    config_path = style_dir / "train.yaml"
    if not config_path.is_file():
        raise SystemExit(f"Missing configuration: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    processes = config["config"]["process"]
    if len(processes) != 1:
        raise SystemExit(f"{config_path}: expected one training process")
    process = processes[0]
    datasets = process["datasets"]
    if len(datasets) != 1 or datasets[0]["folder_path"] != str(style_dir / "data"):
        raise SystemExit(f"{config_path}: dataset must be {style_dir / 'data'}")
    if process["training_folder"] != str(style_dir / "output"):
        raise SystemExit(f"{config_path}: training_folder must be {style_dir / 'output'}")
    trigger_path = style_dir / "trigger.txt"
    if not trigger_path.is_file():
        raise SystemExit(f"Missing trigger: {trigger_path}")
    trigger = trigger_path.read_text(encoding="utf-8").strip()
    if not trigger or "\n" in trigger or "\r" in trigger:
        raise SystemExit(f"{trigger_path}: expected a nonempty single-line trigger")
    validation_command = [sys.executable, "validate_dataset.py", str(style_dir / "data"), "--trigger", trigger]
    if os.environ.get("ALLOW_LOW_RESOLUTION") == "1":
        validation_command.append("--allow-low-resolution")
    subprocess.run(validation_command, check=True)
PY

for style in "$@"; do
    output_dir="$project_root/styles/$style/output"
    mkdir -p "$output_dir"
    log_file="$output_dir/training-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    printf 'Training style: %s; log: %s\n' "$style" "$log_file"
    "$toolkit_python" -u "$toolkit/run.py" "$project_root/styles/$style/train.yaml" 2>&1 | tee -a "$log_file"
done
