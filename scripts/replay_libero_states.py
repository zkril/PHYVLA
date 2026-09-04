"""
Replay a LIBERO demonstration by directly restoring MuJoCo states.

Example:
python scripts/replay_libero_states.py \
    --suite libero_spatial \
    --task-id 2 \
    --demo-index 0 \
    --output outputs/replay/task02_demo00_states.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio

from libero_replay_common import (
    load_demo,
    make_env,
    resolve_task,
    restore_recorded_model,
    transform_frame,
    validate_state_dimension,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay LIBERO demonstration states."
    )

    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
    )

    parser.add_argument(
        "--task-id",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--demo-index",
        type=int,
        default=0,
        help="Index in the sorted demonstration list.",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Write one video frame every N recorded states.",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=-1,
        help="-1 means replay every state.",
    )

    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "rot180"],
        default="none",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/replay/task02_states.mp4"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_suite, task, bddl_path, demo_path = resolve_task(
        suite_name=args.suite,
        task_id=args.task_id,
    )

    demo = load_demo(
        demo_path=demo_path,
        demo_index=args.demo_index,
    )

    states = demo["states"]
    model_xml = demo["model_xml"]
    metadata = demo["metadata"]

    print("=" * 80)
    print("State replay")
    print("Suite       :", args.suite)
    print("Task ID     :", args.task_id)
    print("Task name   :", task.name)
    print("Language    :", task.language)
    print("BDDL        :", bddl_path)
    print("HDF5        :", demo_path)
    print("Episode     :", metadata["demo_key"])
    print("States      :", states.shape)
    print("Actions     :", metadata["actions_shape"])
    print("Model XML   :", model_xml is not None)
    print("=" * 80)

    env = make_env(
        bddl_path=bddl_path,
        camera_name=args.camera,
        image_height=args.height,
        image_width=args.width,
        seed=args.seed,
    )

    writer = None

    try:
        restore_recorded_model(
            env=env,
            model_xml=model_xml,
        )

        validate_state_dimension(
            env=env,
            states=states,
        )

        num_states = len(states)

        if args.max_states >= 0:
            num_states = min(num_states, args.max_states)

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = imageio.get_writer(
            str(args.output),
            fps=args.fps,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )

        num_written = 0
        success_ever = False

        for state_index in range(num_states):
            # LIBERO wrapper:
            # set state -> forward -> update observables -> return observation
            observation = env.regenerate_obs_from_state(
                states[state_index]
            )

            success_now = bool(env.check_success())
            success_ever = success_ever or success_now

            if state_index % args.stride != 0:
                continue

            frame = transform_frame(
                observation=observation,
                camera_name=args.camera,
                flip=args.flip,
            )

            writer.append_data(frame)
            num_written += 1

            if state_index % 50 == 0:
                print(
                    "[state] {:4d}/{:4d} | success={}".format(
                        state_index,
                        num_states - 1,
                        success_now,
                    )
                )

        print("\nState replay completed.")
        print("Video frames :", num_written)
        print("Success seen :", success_ever)
        print("Output       :", args.output.resolve())

    finally:
        if writer is not None:
            writer.close()

        env.close()


if __name__ == "__main__":
    main()
