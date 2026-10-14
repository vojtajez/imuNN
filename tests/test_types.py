import numpy as np
import pytest

from imuNN.core.rotations import quat_exp
from imuNN.core.types import (
    ERR_DIM,
    BA_SL,
    BG_SL,
    P_SL,
    TH_SL,
    V_SL,
    FilterState,
    ImuData,
    ImuSample,
    Measurement,
    NominalState,
    NoiseParams,
    Sequence,
    SensorCalib,
    Trajectory,
)

rng = np.random.default_rng(0)


def make_calib(T_BS=None):
    return SensorCalib(
        T_BS=np.eye(4) if T_BS is None else T_BS,
        rate_hz=200,
        gyro_noise_density=1.6968e-4,
        gyro_random_walk=1.9393e-5,
        acc_noise_density=2e-3,
        acc_random_walk=3e-3,
    )


def make_traj(n=10, t0=0.0):
    t = t0 + 0.005 * np.arange(n)
    q = quat_exp(rng.normal(size=(n, 3)) * 0.3)
    return Trajectory(t, rng.normal(size=(n, 3)), q, rng.normal(size=(n, 3)), np.zeros((n, 3)), np.zeros((n, 3)))


def make_imu(n=12):
    return ImuData(0.005 * np.arange(n), rng.normal(size=(n, 3)), rng.normal(size=(n, 3)))


def test_error_state_slices_cover_15():
    covered = np.zeros(ERR_DIM, int)
    for s in (P_SL, V_SL, TH_SL, BA_SL, BG_SL):
        covered[s] += 1
    assert np.all(covered == 1)


def test_imu_shape_checks():
    with pytest.raises(ValueError):
        ImuData(np.arange(3.0), np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        ImuData(np.arange(3.0), np.zeros((4, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        ImuData(np.array([0.0, 1.0, 1.0]), np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        ImuSample(0.0, np.zeros(2), np.zeros(3))


def test_imu_sample_and_slice():
    imu = make_imu(12)
    assert len(imu) == 12
    s = imu.sample(3)
    assert s.t == imu.t[3]
    np.testing.assert_array_equal(s.acc, imu.acc[3])
    sub = imu.slice(2, 7)
    assert len(sub) == 5
    np.testing.assert_array_equal(sub.gyr, imu.gyr[2:7])


def test_trajectory_checks_and_v_body():
    tr = make_traj()
    for i in range(len(tr)):
        np.testing.assert_allclose(tr.v_body()[i], tr.state(i).R.T @ tr.v[i], atol=1e-12)
    bad_q = tr.q.copy()
    bad_q[0] *= 2
    with pytest.raises(ValueError):
        Trajectory(tr.t, tr.p, bad_q, tr.v, tr.ba, tr.bg)
    with pytest.raises(ValueError):
        Trajectory(tr.t[:-1], tr.p, tr.q, tr.v, tr.ba, tr.bg)
    assert len(tr.slice(2, 5)) == 3


def test_types_are_immutable():
    ns = NominalState.identity()
    with pytest.raises(AttributeError):
        ns.p = np.ones(3)


def test_nominal_state_and_filter_state():
    ns = NominalState.identity()
    np.testing.assert_allclose(ns.R, np.eye(3))
    with pytest.raises(ValueError):
        NominalState(np.zeros(3), np.zeros(3), np.array([2.0, 0, 0, 0]), np.zeros(3), np.zeros(3))
    fs = FilterState(0.0, ns, np.eye(ERR_DIM))
    assert fs.P.shape == (15, 15)
    with pytest.raises(ValueError):
        FilterState(0.0, ns, np.eye(9))


def test_measurement_shapes():
    m = Measurement(1.0, np.zeros(3), np.diag([1.0, 2.0, 3.0]))
    assert m.z.shape == (3,)
    with pytest.raises(ValueError):
        Measurement(1.0, np.zeros(3), np.eye(2))


def test_calib_properties():
    c = make_calib()
    assert c.is_identity_extrinsics
    assert c.dt == pytest.approx(0.005)
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.0, 0.0]
    c2 = make_calib(T)
    assert not c2.is_identity_extrinsics
    np.testing.assert_allclose(c2.p_BS, [0.1, 0, 0])
    np.testing.assert_allclose(c2.R_BS, np.eye(3))


def test_noise_params_from_calib_and_P0():
    c = make_calib()
    n = NoiseParams.from_calib(c, scale=2.0, init_std_p=0.5)
    assert n.acc_noise_density == pytest.approx(4e-3)
    assert n.gyro_random_walk == pytest.approx(2 * 1.9393e-5)
    assert n.meas_var_min == 1e-4
    P0 = n.P0()
    assert P0.shape == (15, 15)
    np.testing.assert_allclose(np.diag(P0)[P_SL], 0.25)
    assert np.all(np.linalg.eigvalsh(P0) > 0)


def test_sequence_validates_ranges():
    imu = make_imu(100)
    gt = make_traj(50, t0=0.1)
    seq = Sequence("s", imu, gt, make_calib(), t0_ns=1403636579758555392)
    assert isinstance(seq.t0_ns, int)
    with pytest.raises(ValueError):
        Sequence("bad", imu, make_traj(50, t0=-1.0), make_calib(), t0_ns=0)
    with pytest.raises(ValueError):
        Sequence("bad", imu, make_traj(200, t0=0.0), make_calib(), t0_ns=0)
