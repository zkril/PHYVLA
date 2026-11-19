# Repository Guidelines

## Project Structure & Module Organization

PHYVLA extracts LIBERO physics observations, encodes physical tokens, and integrates them with StarVLA. `src/adapters/` bridges simulator states; `src/models/` contains tokenizers and projectors; `src/simulation/` implements rollouts and gradients; `src/p5/` handles VLA integration. Shared data loading and helpers live in `src/datasets/` and `src/utils/`.

`scripts/` contains pipeline, training, evaluation, and validation entry points. `configs/` holds task YAML; `envs/` holds Conda specifications. `data/` and `outputs/` contain datasets and generated artifacts. Maintained workflow documentation lives in `docs/`, indexed by `README.md`. External checkouts are under `thirdparty/` on this server; older documentation references `third_party/`.

## Build, Test, and Development Commands

Run from the repository root on Linux with Conda and the required NVIDIA GPU dependencies. There is no separate package build step.

- `bash setup_envs.sh`: provision the three environments from `envs/`.
- `conda activate libero_replay`: select the extraction/replay environment; use `phyvla-p2` for rollouts/gradients and `starVLA` for P5 training.
- `python scripts/run_task_pipeline.py --suite libero_spatial --task-id 2`: run the single-task pipeline in `libero_replay`; interpreter overrides such as `--py-libero` require a Python executable path.
- `conda run -n phyvla-p2 python scripts/test_p3_warp_rollout.py`: exercise the task-02 rollout using existing XML, trace, and physics artifacts.
- `conda run -n phyvla-p2 python scripts/verify_warp_gradients.py`: validate finite-difference and Warp gradient behavior.
- `python scripts/check_40tasks_data.py`: inspect completeness and sequence alignment across suite outputs.

## Coding Style & Naming Conventions

Follow surrounding Python style: four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Add type hints and concise docstrings for interfaces; document tensor shapes and coordinate conventions. Prefer `pathlib.Path` and repository-relative paths. No repository-wide formatter or linter configuration is present.

## Testing Guidelines

Validation uses standalone `scripts/test_*.py` and `scripts/verify_*.py` scripts; no centralized pytest suite or coverage threshold is configured. Add focused checks using these naming patterns. Prepare required datasets and artifacts first. Report commands, environment, task/suite, and numerical results; inspect metrics rather than relying solely on exit status.

## Commit & Pull Request Guidelines

History uses concise prefixes such as `docs:` and `feat:`; follow that pattern. Keep commits focused. PRs should describe behavior, affected stages, configuration changes, and validation evidence. Link relevant issues and attach comparison metrics or visualizations when applicable.

## Configuration & Artifacts

Check legacy absolute paths before launching batch jobs. Keep incompatible stage dependencies separate. Do not commit credentials, datasets, checkpoints, or generated videos; respect `.gitignore` and preserve existing local changes.
