#!/usr/bin/env bash
# Run an upstream baseline client; start the policy server separately.
set -euo pipefail
PHYVLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STARVLA_DIR="$PHYVLA_DIR/thirdparty/starVLA"
BENCHMARK="${BENCHMARK:-LIBERO}"
case "$BENCHMARK" in
    LIBERO) EVAL_ENV="${EVAL_ENV:-libero_replay}" ;;
    LIBERO-plus) EVAL_ENV="${EVAL_ENV:-libero-plus}" ;;
    *) echo "BENCHMARK must be LIBERO or LIBERO-plus" >&2; exit 2 ;;
esac
SUITE="${SUITE:-libero_spatial}"
case "$SUITE" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *) echo "Unsupported SUITE: $SUITE" >&2; exit 2 ;;
esac
CKPT="${VLA_CKPT:-$STARVLA_DIR/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"
OUTPUT_DIR="${OUTPUT_DIR:-$PHYVLA_DIR/outputs/comparison/${BENCHMARK}/baseline/${SUITE}_$(date +%Y%m%d_%H%M%S)}"
CLIENT="examples/simBenchmarks/$BENCHMARK/eval_files/eval_libero.py"
export LIBERO_CONFIG_PATH="$PHYVLA_DIR/thirdparty/$BENCHMARK/libero"
export PYTHONPATH="$PHYVLA_DIR/thirdparty/$BENCHMARK:$STARVLA_DIR:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
cmd=(conda run --no-capture-output -n "$EVAL_ENV" python "$CLIENT"
     --args.pretrained-path "$CKPT" --args.host "$HOST" --args.port "$PORT"
     --args.task-suite-name "$SUITE" --args.num-trials-per-task "$NUM_TRIALS"
     --args.seed "$SEED" --args.video-out-path "$OUTPUT_DIR/videos")
if [[ "$BENCHMARK" == LIBERO-plus ]]; then
    cmd+=(--args.log-path "$OUTPUT_DIR/logs")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${cmd[@]}"; printf '\n'; exit 0
fi
[[ -f "$CKPT" && -f "$STARVLA_DIR/$CLIENT" && -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || {
    echo "Missing checkpoint, client, or LIBERO config" >&2; exit 1;
}
mkdir -p "$OUTPUT_DIR/logs"
cd "$STARVLA_DIR"
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/client.log"
