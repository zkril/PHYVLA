"""TaskConfig：多任务通用配置，自动解析 target/goal body 名称。

使用方式：
    from utils.task_config import TaskConfig
    cfg = TaskConfig.from_task_id("libero_spatial", 3)
    # cfg.target_body → "akita_black_bowl_1_main"（自动从语言解析）
    # cfg.goal_body   → "plate_1_main"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROLE_TARGET       = 0
ROLE_GOAL_SURFACE = 1


@dataclass
class TaskConfig:
    suite:       str
    task_id:     int
    language:    str
    target_body: str   # MuJoCo body name，role=TARGET
    goal_body:   str   # MuJoCo body name，role=GOAL_SURFACE
    bddl_path:   Path
    demo_path:   Path
    output_dir:  Path  = field(init=False)

    def __post_init__(self):
        self.output_dir = PROJECT_ROOT / f"outputs/task{self.task_id:02d}"

    @property
    def model_xml(self) -> Path:
        return self.output_dir / "model_raw.xml"

    @property
    def trace_npz(self) -> Path:
        return self.output_dir / "controller_trace_v1.npz"

    @property
    def rollout_features_npz(self) -> Path:
        return self.output_dir / "rollout_features.npz"

    @property
    def physics_npz(self) -> Path:
        """P0 物理数据文件路径（统一格式）。"""
        return self.output_dir / f"task{self.task_id:02d}_demo00_physics.npz"

    # ─────────────────────────────────────────────────────────────────────
    # 工厂方法：从 task_id 自动解析
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def from_task_id(
        cls,
        suite:       str = "libero_spatial",
        task_id:     int = 2,
        demo_index:  int = 0,
        xml_path:    Optional[Path] = None,
    ) -> "TaskConfig":
        """自动解析 target/goal body 名称，无需手动填写。

        解析流程：
        1. LIBERO benchmark → task.language（如 "pick up the black bowl..."）
        2. TaskRoleParser.parse(language, body_names) → {body_name: role}
        3. 取 role=TARGET 的 body 作为 target_body
        4. 取 role=GOAL_SURFACE 的 body 作为 goal_body

        若 xml_path 未指定，则使用 outputs/task{id:02d}/model_raw.xml。
        """
        from libero.libero import benchmark, get_libero_path
        from utils.task_role_parser import TaskRoleParser, ROLE_TARGET, ROLE_GOAL_SURFACE
        import mujoco

        # 获取任务语言与路径
        bd        = benchmark.get_benchmark_dict()
        suite_obj = bd[suite]()
        task      = suite_obj.get_task(task_id)   # 与 run_task_pipeline.py 一致
        language  = task.language

        from pathlib import Path as _P
        bddl_path = _P(get_libero_path("bddl_files")) / suite / \
                    f"{task.problem_folder}/{task.bddl_file}"
        demo_path = _P(get_libero_path("datasets")) / suite / \
                    f"{suite}_demo.hdf5"

        # 解析 body 角色
        xml_file = xml_path or (PROJECT_ROOT / f"outputs/task{task_id:02d}/model_raw.xml")
        if not xml_file.exists():
            raise FileNotFoundError(
                f"model_raw.xml 不存在：{xml_file}\n"
                f"请先运行：python scripts/extract_task_xml.py "
                f"--suite {suite} --task-id {task_id} --demo-index {demo_index} "
                f"--output {xml_file}"
            )

        mj_model   = mujoco.MjModel.from_xml_path(str(xml_file))
        body_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(mj_model.nbody)
        ]

        parser = TaskRoleParser()
        roles  = parser.parse(language, body_names)  # {body_name: role_int}

        # 找 target / goal body（优先选最短匹配名，与 LiberoPhysicsAdapter 一致）
        target_candidates = sorted(
            [b for b, r in roles.items() if r == ROLE_TARGET],
            key=len,
        )
        goal_candidates = sorted(
            [b for b, r in roles.items() if r == ROLE_GOAL_SURFACE],
            key=len,
        )

        if not target_candidates:
            raise ValueError(
                f"TaskRoleParser 未找到 TARGET body（task{task_id}: {language}）\n"
                f"可用 body：{body_names[:20]}"
            )
        if not goal_candidates:
            raise ValueError(
                f"TaskRoleParser 未找到 GOAL_SURFACE body（task{task_id}: {language}）"
            )

        target_body = target_candidates[0]
        goal_body   = goal_candidates[0]

        print(f"[TaskConfig] task{task_id:02d} · {language}")
        print(f"  target_body : {target_body}")
        print(f"  goal_body   : {goal_body}")

        return cls(
            suite=suite,
            task_id=task_id,
            language=language,
            target_body=target_body,
            goal_body=goal_body,
            bddl_path=bddl_path,
            demo_path=demo_path,
        )

    @classmethod
    def from_task_id_phyvla_p2(
        cls,
        suite:    str = "libero_spatial",
        task_id:  int = 2,
        xml_path: Optional[Path] = None,
    ) -> "TaskConfig":
        """phyvla-p2 环境（无 LIBERO）用此方法：从已保存的 task_info.json 加载。

        先在 libero_replay 中运行 save_task_info 保存 JSON，再在 phyvla-p2 中读取。
        """
        import json
        xml_file  = xml_path or (PROJECT_ROOT / f"outputs/task{task_id:02d}/model_raw.xml")
        info_file = xml_file.parent / "task_info.json"

        if not info_file.exists():
            raise FileNotFoundError(
                f"task_info.json 不存在：{info_file}\n"
                f"请先在 libero_replay 环境运行：\n"
                f"  python scripts/save_task_info.py --suite {suite} --task-id {task_id}"
            )

        info = json.loads(info_file.read_text())
        print(f"[TaskConfig] 从 JSON 加载 task{task_id:02d}: "
              f"target={info['target_body']}  goal={info['goal_body']}")

        return cls(
            suite=info["suite"],
            task_id=info["task_id"],
            language=info["language"],
            target_body=info["target_body"],
            goal_body=info["goal_body"],
            bddl_path=Path(info["bddl_path"]),
            demo_path=Path(info["demo_path"]),
        )
