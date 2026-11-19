#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${VLA_CKPT:-$ROOT/thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
PAIR_DIR="${PAIRED_CHECKPOINT_DIR:-$ROOT/outputs/online_physics_checkpoints_v1}"
if [[ -z "${PROJECTOR_CKPT:-}" && -f "$PAIR_DIR/summary.json" ]]; then
    PROJECTOR="$PAIR_DIR/projector_best.pt"
    TOKENIZER_CKPT="${TOKENIZER_CKPT:-$PAIR_DIR/tokenizer_best.pt}"
else
    PROJECTOR="${PROJECTOR_CKPT:-$ROOT/outputs/p5_checkpoints/projector_40tasks_best.pt}"
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
cmd=(conda run --no-capture-output -n starVLA python "$ROOT/scripts/serve_phyvla.py"
     --checkpoint "$CKPT" --projector "$PROJECTOR" --host "${POLICY_HOST:-127.0.0.1}" --port "${PORT:-9884}")
if [[ -n "${TOKENIZER_CKPT:-}" ]]; then
    cmd+=(--tokenizer-checkpoint "$TOKENIZER_CKPT")
fi
cd "$ROOT"
exec "${cmd[@]}"
