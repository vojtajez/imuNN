import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyR

from imuNN.core.rotations import (
    R_to_quat,
    exp_so3,
    log_so3,
    quat_boxminus,
    quat_boxplus,
    quat_conj,
    quat_exp,
    quat_identity,
    quat_log,
    quat_mult,
    quat_rotate,
    quat_to_R,
    skew,
    vee,
)

rng = np.random.default_rng(0)


def random_phi(n, max_angle=np.pi - 1e-3):
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    return axis * rng.uniform(0.0, max_angle, size=(n, 1))


def to_wxyz(xyzw):
    return np.concatenate([xyzw[..., 3:], xyzw[..., :3]], axis=-1)


def same_rotation(q1, q2):
    return np.allclose(q1, q2, atol=1e-9) or np.allclose(q1, -q2, atol=1e-9)


def test_skew_is_cross_product_and_vee_inverts():
    a, b = rng.normal(size=(2, 3))
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))
    np.testing.assert_allclose(skew(a), -skew(a).T)
    np.testing.assert_allclose(vee(skew(a)), a)


def test_exp_so3_orthogonal_det_one():
    R = exp_so3(random_phi(200, max_angle=10.0))
    np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_exp_matches_scipy():
    phi = random_phi(100, max_angle=10.0)
    np.testing.assert_allclose(exp_so3(phi), ScipyR.from_rotvec(phi).as_matrix(), atol=1e-12)


def test_exp_log_roundtrip():
    phi = random_phi(500)
    np.testing.assert_allclose(log_so3(exp_so3(phi)), phi, atol=1e-9)


def test_log_exp_roundtrip_on_matrices():
    R = ScipyR.random(200, random_state=1).as_matrix()
    np.testing.assert_allclose(exp_so3(log_so3(R)), R, atol=1e-10)


@pytest.mark.parametrize("theta", [0.0, 1e-12, 1e-8, 1e-6, 1e-4])
def test_small_angles(theta):
    phi = theta * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    np.testing.assert_allclose(exp_so3(phi), ScipyR.from_rotvec(phi).as_matrix(), atol=1e-14)
    np.testing.assert_allclose(log_so3(exp_so3(phi)), phi, atol=1e-14)
    np.testing.assert_allclose(quat_log(quat_exp(phi)), phi, atol=1e-14)


@pytest.mark.parametrize("eps", [0.0, 1e-9, 1e-5, 5e-4])
def test_log_near_pi(eps):
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        phi = (np.pi - eps) * axis
        out = log_so3(exp_so3(phi))
        # u úhlu π jsou ±phi stejná rotace, proto porovnáváme rotace i normu
        np.testing.assert_allclose(np.linalg.norm(out), np.pi - eps, atol=1e-7)
        np.testing.assert_allclose(exp_so3(out), exp_so3(phi), atol=1e-7)
        if eps > 0:
            np.testing.assert_allclose(out, phi, atol=1e-6)


def test_batch_shapes():
    phi = random_phi(12).reshape(3, 4, 3)
    R = exp_so3(phi)
    assert R.shape == (3, 4, 3, 3)
    assert log_so3(R).shape == (3, 4, 3)
    q = quat_exp(phi)
    assert q.shape == (3, 4, 4)
    assert quat_to_R(q).shape == (3, 4, 3, 3)
    assert R_to_quat(R).shape == (3, 4, 4)
    assert quat_log(q).shape == (3, 4, 3)


# ----------------------------- kvaterniony --------------------------------- #
def test_quat_exp_matches_scipy():
    phi = random_phi(100, max_angle=3.0)
    expected = to_wxyz(ScipyR.from_rotvec(phi).as_quat())
    for a, b in zip(quat_exp(phi), expected):
        assert same_rotation(a, b)


def test_quat_to_R_matches_scipy_and_exp_so3():
    phi = random_phi(100)
    q = quat_exp(phi)
    np.testing.assert_allclose(quat_to_R(q), exp_so3(phi), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-14)


def test_R_to_quat_roundtrip_all_branches():
    # náhodné rotace + rotace o π kolem os pokrývají všechny větve Shepperdovy metody
    Rs = [ScipyR.random(200, random_state=2).as_matrix()]
    for axis in np.eye(3):
        Rs.append(exp_so3(np.pi * axis)[None])
    R = np.concatenate(Rs)
    q = R_to_quat(R)
    assert np.all(q[:, 0] >= 0)
    np.testing.assert_allclose(quat_to_R(q), R, atol=1e-12)


def test_quat_mult_composes_rotations():
    p, q = quat_exp(random_phi(50)), quat_exp(random_phi(50))
    np.testing.assert_allclose(quat_to_R(quat_mult(p, q)), quat_to_R(p) @ quat_to_R(q), atol=1e-12)


def test_quat_conj_is_inverse():
    q = quat_exp(random_phi(50))
    ident = quat_mult(q, quat_conj(q))
    np.testing.assert_allclose(ident, quat_identity((50,)), atol=1e-14)


def test_quat_log_exp_roundtrip():
    phi = random_phi(300)
    np.testing.assert_allclose(quat_log(quat_exp(phi)), phi, atol=1e-9)
    # i znaménkově převrácený kvaternion (stejná rotace) dá stejný log
    np.testing.assert_allclose(quat_log(-quat_exp(phi)), phi, atol=1e-9)


def test_quat_rotate_matches_matrix():
    q = quat_exp(random_phi(50))
    v = rng.normal(size=(50, 3))
    np.testing.assert_allclose(quat_rotate(q, v), np.einsum("nij,nj->ni", quat_to_R(q), v), atol=1e-12)
    # rotace zachovává normu
    np.testing.assert_allclose(np.linalg.norm(quat_rotate(q, v), axis=-1), np.linalg.norm(v, axis=-1), atol=1e-12)


# ------------------- pravá perturbace (konvence ESKF) ----------------------- #
def test_boxplus_boxminus_roundtrip():
    q = quat_exp(random_phi(100))
    d = random_phi(100, max_angle=1.0)
    q2 = quat_boxplus(q, d)
    np.testing.assert_allclose(quat_boxminus(q2, q), d, atol=1e-9)


def test_boxplus_is_right_multiplication():
    """R(q ⊕ dθ) = R(q) Exp(dθ): perturbace v tělesném rámci."""
    q = quat_exp(random_phi(50))
    d = random_phi(50, max_angle=0.5)
    np.testing.assert_allclose(quat_to_R(quat_boxplus(q, d)), quat_to_R(q) @ exp_so3(d), atol=1e-12)


def test_body_velocity_jacobian_wrt_right_perturbation():
    """h(R, v_w) = Rᵀ v_w; dh/dδθ = [R̂ᵀ v_w]x = [v_b]x pro R = R̂ Exp(δθ)."""
    R_hat = exp_so3(random_phi(1)[0])
    v_w = rng.normal(size=3)
    eps = 1e-6
    J = np.empty((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        h_p = (R_hat @ exp_so3(d)).T @ v_w
        h_m = (R_hat @ exp_so3(-d)).T @ v_w
        J[:, i] = (h_p - h_m) / (2 * eps)
    v_b = R_hat.T @ v_w
    # Rᵀ v = Exp(-δθ) v_b ≈ v_b - δθ × v_b = v_b + [v_b]x δθ
    np.testing.assert_allclose(J, skew(v_b), atol=1e-8)
