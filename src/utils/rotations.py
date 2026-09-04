"""Rotation utilities for Physical Tokenizer.

All quaternions are in xyzw convention (matching LiberoPhysicsAdapter output).
6D rotation representation follows Zhou et al. 2019 (first two columns of R).
"""

from __future__ import annotations

import numpy as np


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion(s) to 3x3 rotation matrix.

    Args:
        q: [..., 4] float array in xyzw order.
    Returns:
        [..., 3, 3] rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64)
    shape = q.shape[:-1]
    q = q.reshape(-1, 4)

    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-8)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
        2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w),
        2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y),
    ], axis=-1).reshape(-1, 3, 3)

    return R.reshape(shape + (3, 3))


def quat_xyzw_to_6d(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion(s) to 6D rotation representation.

    The 6D representation is the first two columns of the rotation matrix
    flattened to a 6-element vector: [r1, r2] where r1, r2 are 3D column vectors.

    Args:
        q: [..., 4] float array in xyzw order.
    Returns:
        [..., 6] 6D rotation representation.
    """
    R = quat_xyzw_to_rotmat(q)   # [..., 3, 3]
    return R[..., :2].transpose(0, 2, 1).reshape(q.shape[:-1] + (6,)) \
        if q.ndim > 1 else R[:2, :].flatten()


def rot6d_to_rotmat(r6d: np.ndarray) -> np.ndarray:
    """Recover a rotation matrix from a 6D representation (Gram-Schmidt).

    Args:
        r6d: [..., 6]
    Returns:
        [..., 3, 3]
    """
    r6d = np.asarray(r6d, dtype=np.float64)
    shape = r6d.shape[:-1]
    r6d = r6d.reshape(-1, 6)

    a1 = r6d[:, :3]
    a2 = r6d[:, 3:6]

    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-8)
    b2 = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=-1)  # [N, 3, 3]
    return R.reshape(shape + (3, 3))
