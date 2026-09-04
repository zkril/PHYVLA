from __future__ import annotations

import os
from pathlib import Path

from libero.libero import benchmark, get_libero_path


SUITE_NAME = "libero_spatial"
TASK_ID = 2


def main() -> None:
    suite_cls = benchmark.get_benchmark_dict()[SUITE_NAME]
    suite = suite_cls()

    task = suite.get_task(TASK_ID)

    bddl_path = Path(
        get_libero_path("bddl_files")
    ) / task.problem_folder / task.bddl_file

    demo_relative_path = suite.get_task_demonstration(TASK_ID)
    demo_path = Path(get_libero_path("datasets")) / demo_relative_path

    init_states_path = Path(
        get_libero_path("init_states")
    ) / task.problem_folder / task.init_states_file

    print("suite       :", SUITE_NAME)
    print("task_id     :", TASK_ID)
    print("task_name   :", task.name)
    print("language    :", task.language)

    print("\nbddl_path   :", bddl_path)
    print("bddl exists :", bddl_path.exists())

    print("\ndemo path   :", demo_path)
    print("demo exists :", demo_path.exists())

    print("\ninit states :", init_states_path)
    print("init exists :", init_states_path.exists())

    if not demo_path.exists():
        print("\nSearching possible matching HDF5 files...")

        dataset_root = Path(get_libero_path("datasets"))
        matches = list(
            dataset_root.rglob(f"*{task.name}*.hdf5")
        )

        for match in matches:
            print("candidate   :", match)

        if not matches:
            raise FileNotFoundError(
                f"No demonstration file found under {dataset_root}"
            )


if __name__ == "__main__":
    main()
