"""Serve baseline VLA plus a saved P5 projector on an independent port."""
from pathlib import Path
import argparse
import logging
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "thirdparty/starVLA"))
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--projector", required=True)
    parser.add_argument("--tokenizer-checkpoint")
    parser.add_argument("--allow-unpaired-tokenizer", action="store_true", help="Experimental legacy projector without a pairing manifest; results are not a restored policy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9884)
    args = parser.parse_args()
    for value in (args.checkpoint, args.projector, args.tokenizer_checkpoint):
        if value and not Path(value).is_file():
            parser.error(f"Checkpoint not found: {value}")
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.projector = str(Path(args.projector).resolve())
    if args.tokenizer_checkpoint:
        args.tokenizer_checkpoint = str(Path(args.tokenizer_checkpoint).resolve())
    # Upstream model config resolves its base VLM relative to the StarVLA checkout.
    os.chdir(ROOT / "thirdparty/starVLA")
    from p5.physics_policy import PhysicsPolicy
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    policy = PhysicsPolicy(args.checkpoint, args.projector, args.tokenizer_checkpoint,
                           allow_unpaired_tokenizer=args.allow_unpaired_tokenizer)
    logging.info("PHYVLA metadata: %s", policy.metadata)
    WebsocketPolicyServer(policy, host=args.host, port=args.port,
                          metadata=policy.metadata, idle_timeout=-1).serve_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
