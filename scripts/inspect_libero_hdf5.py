from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", type=Path)
    args = parser.parse_args()

    if not args.hdf5_path.exists():
        raise FileNotFoundError(args.hdf5_path)

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            print(
                f"[dataset] {name:60s} "
                f"shape={obj.shape!s:18s} "
                f"dtype={obj.dtype}"
            )
        else:
            print(f"[group]   {name}")

    with h5py.File(args.hdf5_path, "r") as file:
        print("File:", args.hdf5_path)
        print("\nRoot attributes:")
        for key, value in file.attrs.items():
            print(f"  {key}: {value}")

        print("\nStructure:")
        file.visititems(visitor)

        if "data" in file:
            demo_names = sorted(file["data"].keys())
            print("\nNumber of demos:", len(demo_names))
            print("First demos:", demo_names[:10])

            if demo_names:
                demo = file["data"][demo_names[0]]

                print("\nFirst demo attributes:")
                for key, value in demo.attrs.items():
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
