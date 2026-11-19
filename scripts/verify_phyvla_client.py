"""Check the real LIBERO-plus client/server with saved tokens, not a rollout."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files"),
               str(ROOT / "thirdparty/starVLA")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9884)
    parser.add_argument("--token-root", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "logs/phyvla_gpu1_audit/inference.json")
    args = parser.parse_args()
    from model2libero_interface import ModelClient
    client = ModelClient(policy_ckpt_path="", host=args.host, port=args.port)
    results = []
    try:
        assert client.physics_server, client.server_metadata
        tokenizer_meta = client.server_metadata.get("tokenizer")
        if args.token_root is None:
            args.token_root = Path(tokenizer_meta["checkpoint"]).parent / "tokens" if tokenizer_meta else ROOT / "outputs"
        example = {"image": [np.zeros((224, 224, 3), dtype=np.uint8)] * 2,
                   "lang": "diagnostic: saved physics token inference, no environment rollout"}
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            for task_id in range(10):
                path = args.token_root / suite / f"task_{task_id}" / "physics_tokens_demo0.npz"
                paired_path = path.with_name("demo_0.npz")
                if paired_path.is_file():
                    path = paired_path
                with np.load(path, allow_pickle=False) as data:
                    if tokenizer_meta:
                        if "tokenizer_sha256" not in data or str(data["tokenizer_sha256"]) != tokenizer_meta["sha256"]:
                            raise ValueError("Diagnostic tokens do not match the loaded tokenizer")
                    tokens = data["tokens"][:1].astype(np.float32)
                client.reset(example["lang"])
                client.step(dict(example), step=0, z_phy=tokens)
                first = client.raw_actions.copy()
                # Chunk scheduling must use the already-predicted second action.
                cached = client.step(dict(example), step=1, z_phy=tokens)["raw_action"]
                np.testing.assert_array_equal(cached["world_vector"], first[1, :3])
                assert first.shape == (8, 7) and np.isfinite(first).all()
                # Changing tokens after reset must affect a fresh model call.
                client.reset(example["lang"])
                client.step(dict(example), step=0, z_phy=np.zeros_like(tokens))
                difference = float(np.max(np.abs(first-client.raw_actions)))
                assert difference > 1e-7, "Physics input has no observable effect"
                results.append({"suite": suite, "task_id": task_id,
                                "token_source": str(path), "action_shape": list(first.shape),
                                "max_action_delta_vs_zero_tokens": difference})
        invalid = client.client.predict_action({"examples": [example],
                                                "z_phy": np.zeros((1, 22, 256), np.float32)})
        assert not invalid["ok"] and "Expected finite" in invalid["error"]["message"]
        missing = client.client.predict_action({"examples": [example]})
        assert not missing["ok"] and "exactly one" in missing["error"]["message"]
        if not client.server_metadata["online_physics_ready"]:
            try:
                ModelClient(policy_ckpt_path="", host=args.host, port=args.port, require_physics=True)
            except RuntimeError as exc:
                assert "saved tokenizer" in str(exc)
            else:
                raise AssertionError("Online client must fail without tokenizer")
        report = {"scope": "saved-token interface diagnostic; not closed-loop evaluation",
                  "metadata": client.server_metadata, "results": results,
                  "invalid_payload_rejected": True, "missing_physics_rejected": True,
                  "chunk_cache_and_reset_verified": True}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps({"verified_tasks": len(results), "report": str(args.report)}))
    finally:
        client.client.close()


if __name__ == "__main__":
    main()
