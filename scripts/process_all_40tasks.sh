#!/usr/bin/env bash
# Export raw original-LIBERO physics only; train the tokenizer separately.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${OUTPUT_DIR:-$ROOT/outputs/physics_reexport_v2}"
cmd=(conda run --no-capture-output -n libero-plus python scripts/export_physics_dataset_v2.py
     --output "$OUT" --workers "${WORKERS:-8}")
# Unset DEMO_INDICES means every recorded demonstration.
if [[ -n "${DEMO_INDICES:-}" ]]; then
  read -r -a demos <<< "$DEMO_INDICES"
  cmd+=(--demo-indices "${demos[@]}")
fi
"${cmd[@]}"
conda run --no-capture-output -n libero-plus python scripts/validate_physics_dataset_v2.py --root "$OUT"
