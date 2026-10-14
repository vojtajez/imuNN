"""
https://arxiv.org/pdf/1711.02508
"""

from __future__ import annotations

import numpy as np

_SMALL_ANGLE = 1e-6


def skew(v: np.ndarray) -> np.ndarray:
    """
    Constructs a 3x3 skew-symmetric matrix [v]_x from a 3D vector v (hat operator).
    Satisfies: skew(v) @ u == np.cross(v, u).

    Usage example:
    >>> v = np.array([1.0, 2.0, 3.0])
    >>> K = skew(v)
    >>> K
    array([[ 0., -3.,  2.],
           [ 3.,  0., -1.],
           [-2.,  1.,  0.]])
    >>> # Batch processing for N=2 vectors:
    >>> v_batch = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> skew(v_batch).shape
    (2, 3, 3)
    """
    v = np.asarray(v, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = np.zeros_like(x)
    return np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )


def vee(M: np.ndarray) -> np.ndarray:
    """
    Inverse operator to skew(). Extracts the 3D vector from the skew-symmetric
    part of a 3x3 matrix: vee(0.5 * (M - M.T)).

    Usage example:
    >>> v = np.array([1.0, -2.0, 0.5])
    >>> K = skew(v)
    >>> v_reconstructed = vee(K)
    >>> np.allclose(v, v_reconstructed)
    True
    """
    M = np.asarray(M, dtype=float)
    return np.stack(
        [
            0.5 * (M[..., 2, 1] - M[..., 1, 2]),
            0.5 * (M[..., 0, 2] - M[..., 2, 0]),
            0.5 * (M[..., 1, 0] - M[..., 0, 1]),
        ],
        axis=-1,
    )


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """
    Exponential map Exp: R^3 -> SO(3) (Rodrigues' rotation formula).
    Converts a rotation vector phi = theta * u (where theta is the angle in rad
    and u is the unit axis) into a 3x3 rotation matrix.

    Usage example:
    >>> # 90-degree (pi/2) rotation around the Z-axis:
    >>> phi = np.array([0.0, 0.0, np.pi / 2])
    >>> R = exp_so3(phi)
    >>> np.round(R, 4)
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])
    """
    phi = np.asarray(phi, dtype=float)
    theta2 = np.sum(phi * phi, axis=-1)
    theta = np.sqrt(theta2)
    small = theta < _SMALL_ANGLE
    safe = np.where(small, 1.0, theta)

    A = np.where(small, 1.0 - theta2 / 6.0, np.sin(safe) / safe)
    B = np.where(small, 0.5 - theta2 / 24.0, (1.0 - np.cos(safe)) / safe**2)

    K = skew(phi)
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + A[..., None, None] * K + B[..., None, None] * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """
    Logarithmic map Log: SO(3) -> R^3.
    Converts a 3x3 rotation matrix back into a rotation vector phi = theta * u.
    Handles singularities as theta -> 0 and theta -> pi.

    Usage example:
    >>> phi_in = np.array([0.3, -0.4, 1.2])
    >>> R = exp_so3(phi_in)
    >>> phi_out = log_so3(R)
    >>> np.allclose(phi_in, phi_out)
    True
    >>> # Edge case: 180-degree (pi rad) rotation around the X-axis:
    >>> R_pi = np.diag([1.0, -1.0, -1.0])
    >>> log_so3(R_pi)
    array([3.14159265, 0.        , 0.        ])
    """
    R = np.asarray(R, dtype=float)
    batch_shape = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)

    v = vee(Rf)
    s = np.linalg.norm(v, axis=-1)
    c = np.clip(0.5 * (np.trace(Rf, axis1=-2, axis2=-1) - 1.0), -1.0, 1.0)
    theta = np.arctan2(s, c)

    small = theta < _SMALL_ANGLE
    near_pi = (np.pi - theta) < 1e-3

    safe_s = np.where(small | near_pi, 1.0, s)
    factor = np.where(small, 1.0 + theta**2 / 6.0, theta / safe_s)
    phi = factor[:, None] * v

    for i in np.flatnonzero(near_pi):
        B = 0.5 * (Rf[i] + Rf[i].T) - np.cos(theta[i]) * np.eye(3)
        B = B / max(1.0 - np.cos(theta[i]), 1e-12)
        k = int(np.argmax(np.diag(B)))
        n = B[:, k] / np.sqrt(max(B[k, k], 1e-12))
        n = n / np.linalg.norm(n)
        if np.dot(n, v[i]) < 0.0:
            n = -n
        phi[i] = theta[i] * n

    return phi.reshape(*batch_shape, 3)


def quat_identity(batch_shape: tuple[int, ...] = ()) -> np.ndarray:
    """
    Returns the identity quaternion [1, 0, 0, 0] representing zero rotation.

    Usage example:
    >>> q_id = quat_identity()
    >>> q_id
    array([1., 0., 0., 0.])
    >>> # Batch of 3 identity quaternions:
    >>> quat_identity((3,))
    array([[1., 0., 0., 0.],
           [1., 0., 0., 0.],
           [1., 0., 0., 0.]])
    """
    q = np.zeros((*batch_shape, 4))
    q[..., 0] = 1.0
    return q


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalizes a quaternion to unit length (||q|| = 1) so that it lies on the S^3 manifold.
    """
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """
    Returns the quaternion conjugate q* = [w, -x, -y, -z].
    For unit quaternions, the conjugate equals the inverse rotation (q* == q^{-1}).
    """
    q = np.asarray(q, dtype=float)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions (p ⊗ q). Corresponds to rotation composition:
    quat_to_R(quat_mult(p, q)) == quat_to_R(p) @ quat_to_R(q).

    Usage example:
    >>> # Composing two 90-degree rotations around Z -> 180-degree rotation around Z:
    >>> q_z90 = quat_exp(np.array([0.0, 0.0, np.pi / 2]))
    >>> q_z180 = quat_mult(q_z90, q_z90)
    >>> np.round(q_z180, 4)
    array([0., 0., 0., 1.])
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        axis=-1,
    )


def quat_exp(phi: np.ndarray) -> np.ndarray:
    """
    Exponential map Exp: R^3 -> S^3.
    Converts a rotation vector phi = theta * u into a unit quaternion [cos(theta/2), sin(theta/2) * u].

    Usage example:
    >>> # 90-degree (pi/2) rotation around the X-axis:
    >>> phi = np.array([np.pi / 2, 0.0, 0.0])
    >>> q = quat_exp(phi)
    >>> np.round(q, 4)
    array([0.7071, 0.7071, 0.    , 0.    ])
    """
    phi = np.asarray(phi, dtype=float)
    theta2 = np.sum(phi * phi, axis=-1)
    theta = np.sqrt(theta2)
    small = theta < _SMALL_ANGLE
    safe = np.where(small, 1.0, theta)

    w = np.where(small, 1.0 - theta2 / 8.0, np.cos(0.5 * safe))
    k = np.where(small, 0.5 - theta2 / 48.0, np.sin(0.5 * safe) / safe)
    return np.concatenate([w[..., None], k[..., None] * phi], axis=-1)


def quat_log(q: np.ndarray) -> np.ndarray:
    """
    Logarithmic map Log: S^3 -> R^3.
    Converts a unit quaternion back into a rotation vector phi = theta * u.
    Enforces the canonical hemisphere (w >= 0) to return the shortest rotation in [0, pi].

    Usage example:
    >>> phi_in = np.array([0.1, -0.5, 0.8])
    >>> q = quat_exp(phi_in)
    >>> phi_out = quat_log(q)
    >>> np.allclose(phi_in, phi_out)
    True
    """
    q = np.asarray(q, dtype=float)
    q = np.where(q[..., :1] < 0.0, -q, q)
    w, v = q[..., 0], q[..., 1:]
    n = np.linalg.norm(v, axis=-1)
    small = n < _SMALL_ANGLE
    safe_n = np.where(small, 1.0, n)
    factor = np.where(
        small,
        2.0 / w - 2.0 * n**2 / (3.0 * w**3),
        2.0 * np.arctan2(n, w) / safe_n,
    )
    return factor[..., None] * v


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion into a 3x3 rotation matrix in SO(3).

    Usage example:
    >>> q = quat_exp(np.array([0.0, 0.0, np.pi / 2]))
    >>> R = quat_to_R(q)
    >>> np.round(R, 4)
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])
    """
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix into a unit quaternion (Shepperd's method).
    Always returns a canonical quaternion with a non-negative scalar part (w >= 0).

    Usage example:
    >>> R = exp_so3(np.array([0.2, -0.7, 1.1]))
    >>> q = R_to_quat(R)
    >>> R_reconstructed = quat_to_R(q)
    >>> np.allclose(R, R_reconstructed)
    True
    """
    R = np.asarray(R, dtype=float)
    batch_shape = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    q = np.empty((Rf.shape[0], 4))

    for i, M in enumerate(Rf):
        tr = np.trace(M)
        d = np.array([tr, M[0, 0], M[1, 1], M[2, 2]])
        k = int(np.argmax(d))
        if k == 0:
            s = 2.0 * np.sqrt(1.0 + tr)
            q[i] = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif k == 1:
            s = 2.0 * np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2])
            q[i] = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif k == 2:
            s = 2.0 * np.sqrt(1.0 - M[0, 0] + M[1, 1] - M[2, 2])
            q[i] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 - M[0, 0] - M[1, 1] + M[2, 2])
            q[i] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]

    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[:, :1] < 0.0, -q, q)
    return q.reshape(*batch_shape, 4)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotates a 3D vector v by quaternion q: v_rot = R(q) @ v (equivalent to q ⊗ [0, v] ⊗ q*).

    Usage example:
    >>> # Rotating vector [1, 0, 0] by 90 degrees around Z yields [0, 1, 0]:
    >>> q_z90 = quat_exp(np.array([0.0, 0.0, np.pi / 2]))
    >>> v = np.array([1.0, 0.0, 0.0])
    >>> v_rot = quat_rotate(q_z90, v)
    >>> np.round(v_rot, 4)
    array([0., 1., 0.])
    """
    R = quat_to_R(q)
    return np.einsum("...ij,...j->...i", R, np.asarray(v, dtype=float))


def quat_boxplus(q: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """
    Right-hand manifold addition operator (boxplus, ⊞): q_new = q ⊗ Exp(dtheta).
    In an ESKF, this injects a local angular error perturbation dtheta (expressed
    in the local/body frame) into the nominal state quaternion q.

    Usage example:
    >>> q_nominal = quat_identity()
    >>> dtheta = np.array([0.01, -0.02, 0.005])  # small angular correction from ESKF
    >>> q_updated = quat_boxplus(q_nominal, dtheta)
    >>> np.round(q_updated, 6)
    array([ 0.999934,  0.005   , -0.01    ,  0.0025  ])
    """
    return quat_normalize(quat_mult(q, quat_exp(dtheta)))


def quat_boxminus(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Right-hand manifold subtraction operator (boxminus, ⊟): dtheta = Log(q2^{-1} ⊗ q1).
    Computes the 3D local angular error vector (in the body frame) that transforms
    orientation q2 into orientation q1. Satisfies: quat_boxplus(q2, quat_boxminus(q1, q2)) == q1.

    Usage example:
    >>> q2 = quat_exp(np.array([0.2, 0.1, -0.3]))
    >>> delta_true = np.array([0.05, -0.01, 0.02])
    >>> q1 = quat_boxplus(q2, delta_true)
    >>> # Recovering the local angular difference between q1 and q2:
    >>> delta_est = quat_boxminus(q1, q2)
    >>> np.allclose(delta_true, delta_est)
    True
    """
    return quat_log(quat_mult(quat_conj(q2), q1))