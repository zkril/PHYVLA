"""LIBERO live-state payloads using the same adapter as offline extraction."""
import numpy as np
from adapters.libero_physics_adapter import LiberoPhysicsAdapter
from utils.contact_events import detect_contact_events
from utils.task_role_parser import TaskRoleParser, ROLE_TARGET, ROLE_GOAL_SURFACE


class LivePhysicsState:
    def __init__(self, env, language):
        self.adapter = LiberoPhysicsAdapter(env)
        self.keys = sorted(self.adapter.objects)
        if not 0 < len(self.keys) <= 16:
            raise ValueError(f"Expected 1..16 objects, got {self.keys}")
        roles = TaskRoleParser().parse(language, self.keys)
        self.target = next((k for k in self.keys if roles.get(k) == ROLE_TARGET), self.keys[0])
        self.goal = next((k for k in self.keys if roles.get(k) == ROLE_GOAL_SURFACE), None)
        self.reset()

    def reset(self):
        self.history = []

    def capture(self):
        raw = self.adapter.extract()
        flat = dict(raw["robot"])
        flat["robot_qpos"] = flat.pop("qpos")
        flat["robot_qvel"] = flat.pop("qvel")
        for key, obj in raw["objects"].items():
            flat.update({f"{key}_{name}": value for name, value in obj.items()})
        flat["contacts"] = raw["contacts"]
        flat["contact_mask"] = raw["contact_mask"]
        # Do not expose terminal labels to the policy as an input.
        flat["success"] = np.asarray(0, dtype=np.float32)
        self.history.append(flat)
        history = {k: np.stack([row[k] for row in self.history]) for k in flat}
        target = history[f"{self.target}_position"]
        dist = (np.linalg.norm(target-history[f"{self.goal}_position"], axis=-1)
                if self.goal else np.zeros(len(self.history), dtype=np.float32))
        events = detect_contact_events(history["contacts"], history["contact_mask"],
                                       target, dist, history["gripper_qpos"], history["success"])
        return {"feature_schema": "causal_hold_v1",
                "arrays": {k: np.asarray(v)[None] for k, v in flat.items()},
                "events": {"phase": events.phase[-1:], "events": events.events[-1:]}}
