#!/usr/bin/env bash
# Start the shared baseline policy server for LIBERO and LIBERO-plus.
set -euo pipefail
PHYVLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STARVLA_DIR="$PHYVLA_DIR/thirdparty/starVLA"
CKPT="${VLA_CKPT:-$STARVLA_DIR/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
GPU="${GPU:-0}"
PORT="${PORT:-6694}"
cmd=(conda run --no-capture-output -n starVLA python deployment/model_server/server_policy.py
     --ckpt_path "$CKPT" --port "$PORT" --use_bf16)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${cmd[@]}"; printf '\n'
    exit 0
fi
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
cd "$STARVLA_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "${cmd[@]}"
