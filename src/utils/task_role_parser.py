"""Task role parser: assign semantic roles to MuJoCo body names from task language.

Roles:
  0  TARGET        — object to manipulate (pick up, push, open...)
  1  GOAL_SURFACE  — destination surface (plate, shelf, bin...)
  2  DISTRACTOR    — other movable objects in the scene
  3  STATIC        — static environment (table surface, cabinet body...)
  4  GRIPPER       — robot end-effector / fingers
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

ROLE_TARGET       = 0
ROLE_GOAL_SURFACE = 1
ROLE_DISTRACTOR   = 2
ROLE_STATIC       = 3
ROLE_GRIPPER      = 4

ROLE_NAMES = ["target", "goal_surface", "distractor", "static", "gripper"]

# Keywords that indicate static environment bodies
_STATIC_KEYWORDS = (
    "table", "counter", "floor", "wall", "ceiling", "arena",
    "cabinet", "shelf_body", "drawer_frame", "base",
)
# Keywords that indicate gripper bodies
_GRIPPER_KEYWORDS = ("gripper", "finger", "hand", "eef", "robot0_link")

_PICK_PATTERNS = [
    # 在位置介词（on/in/from/next to/between）前停止捕获，避免把场景描述纳入对象名
    r"pick(?:\s+up)?\s+the\s+(.+?)(?:\s+(?:on|in|from|next|between)\s+|\s+and|\s*$)",
    r"grasp\s+the\s+(.+?)(?:\s+(?:on|in|from|next|between)\s+|\s+and|\s*$)",
    r"lift\s+the\s+(.+?)(?:\s+(?:on|in|from|next|between)\s+|\s+and|\s*$)",
    r"grab\s+the\s+(.+?)(?:\s+(?:on|in|from|next|between)\s+|\s+and|\s*$)",
]
_PLACE_PATTERNS = [
    r"place\s+it\s+(?:on|in|into)\s+the\s+(.+?)(?:\s+and|\s*$)",
    r"put\s+it\s+(?:on|in|into)\s+the\s+(.+?)(?:\s+and|\s*$)",
    r"place\s+\S+\s+(?:on|in|into)\s+the\s+(.+?)(?:\s+and|\s*$)",
    r"stack\s+\S+\s+on\s+the\s+(.+?)(?:\s+and|\s*$)",
    r"store\s+\S+\s+in\s+the\s+(.+?)(?:\s+and|\s*$)",
]
_OPEN_PATTERNS = [
    r"open\s+the\s+(.+?)(?:\s+and|\s*$)",
    r"close\s+the\s+(.+?)(?:\s+and|\s*$)",
    r"push\s+the\s+(.+?)(?:\s+to|\s+and|\s*$)",
]


def _fuzzy_match(body_name: str, keyword: str) -> bool:
    """Return True if keyword words appear in body_name (case-insensitive)."""
    body_lower = body_name.lower().replace("_", " ")
    kw_lower   = keyword.lower().replace("_", " ")
    words = [w for w in kw_lower.split() if len(w) > 2]
    return any(w in body_lower for w in words) if words else False


def _contains_any(name: str, keywords: tuple) -> bool:
    name_lower = name.lower()
    return any(k in name_lower for k in keywords)


def _first_match(text: str, patterns: List[str]) -> Optional[str]:
    text_lower = text.lower().strip()
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            return m.group(1).strip()
    return None


class TaskRoleParser:
    """Parse task.language to assign roles to MuJoCo body names."""

    def parse(
        self,
        language: str,
        body_names: List[str],
    ) -> Dict[str, int]:
        """Return {body_name: role_int} for all bodies."""
        target_kw = _first_match(language, _PICK_PATTERNS)
        goal_kw   = _first_match(language, _PLACE_PATTERNS)
        # open/close tasks: same object is target, no separate goal surface
        if target_kw is None:
            target_kw = _first_match(language, _OPEN_PATTERNS)

        roles: Dict[str, int] = {}
        for body in body_names:
            if _contains_any(body, _GRIPPER_KEYWORDS):
                roles[body] = ROLE_GRIPPER
            elif target_kw and _fuzzy_match(body, target_kw):
                roles[body] = ROLE_TARGET
            elif goal_kw and _fuzzy_match(body, goal_kw):
                roles[body] = ROLE_GOAL_SURFACE
            elif _contains_any(body, _STATIC_KEYWORDS):
                roles[body] = ROLE_STATIC
            else:
                roles[body] = ROLE_DISTRACTOR
        return roles

    def parse_with_body_ids(
        self,
        language: str,
        body_names: List[str],
    ) -> Dict[int, int]:
        """Return {body_id: role_int} (id = index in body_names list)."""
        name_roles = self.parse(language, body_names)
        return {i: name_roles[name] for i, name in enumerate(body_names)}
