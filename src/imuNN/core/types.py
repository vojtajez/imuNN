from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from imuNN.core.rotations import quat_to_R

# constants

GRAVITY = 9.81
G_W = np.array([0.0, 0.0, -GRAVITY])

ERR_DIM = 15
P_SL = slice(0, 3)  # dp
V_SL = slice(3, 6)  # dv
TH_SL = slice(6, 9)  # dtheta (right perturbation, in the body frame)
BA_SL = slice(9, 12)  # db_a
BG_SL = slice(12, 15)  # db_g

_QUAT_TOL = 1e-3


# helpers

def _arr(name: str, x, *shape: int | None) -> np.ndarray:
    """Convert to float64 and check the shape. None in shape means any length."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != len(shape) or any(s is not None and s != d for s, d in zip(shape, a.shape)):
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    return a


def _set(obj, **fields) -> None:
    for k, v in fields.items():
        object.__setattr__(obj, k, v)


def _check_unit_quats(name: str, q: np.ndarray) -> None:
    if not np.all(np.abs(np.linalg.norm(q, axis=-1) - 1.0) < _QUAT_TOL):
        raise ValueError(f"{name}: quaternions are not unit norm (tolerance {_QUAT_TOL})")


def _check_same_length(**arrays: np.ndarray) -> None:
    lengths = {k: len(v) for k, v in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"length mismatch: {lengths}")


def _check_increasing(name: str, t: np.ndarray) -> None:
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError(f"{name}: time must be strictly increasing")



# IMU

@dataclass(frozen=True, slots=True, eq=False)
class ImuSample:
    """A single IMU sample (raw, in the sensor frame). acc [m/s^2], gyr [rad/s]."""

    t: float
    acc: np.ndarray  # (3,)
    gyr: np.ndarray  # (3,)

    def __post_init__(self):
        _set(self, t=float(self.t), acc=_arr("acc", self.acc, 3), gyr=_arr("gyr", self.gyr, 3))


@dataclass(frozen=True, slots=True, eq=False)
class ImuData:
    """A sequence of IMU measurements (raw, bias not subtracted)."""

    t: np.ndarray  # (N,)
    acc: np.ndarray  # (N, 3)
    gyr: np.ndarray  # (N, 3)

    def __post_init__(self):
        _set(
            self,
            t=_arr("t", self.t, None),
            acc=_arr("acc", self.acc, None, 3),
            gyr=_arr("gyr", self.gyr, None, 3),
        )
        _check_same_length(t=self.t, acc=self.acc, gyr=self.gyr)
        _check_increasing("ImuData", self.t)

    def __len__(self) -> int:
        return len(self.t)

    def sample(self, i: int) -> ImuSample:
        return ImuSample(self.t[i], self.acc[i], self.gyr[i])

    def slice(self, start: int, stop: int) -> ImuData:
        return ImuData(self.t[start:stop], self.acc[start:stop], self.gyr[start:stop])


# Trajectory (ground truth and filter output)

@dataclass(frozen=True, slots=True, eq=False)
class Trajectory:
    """State over time: position and velocity in the world, body->world orientation, biases in the sensor frame."""

    t: np.ndarray  # (N,)
    p: np.ndarray  # (N, 3)
    q: np.ndarray  # (N, 4) wxyz
    v: np.ndarray  # (N, 3) in the world frame
    ba: np.ndarray  # (N, 3)
    bg: np.ndarray  # (N, 3)

    def __post_init__(self):
        _set(
            self,
            t=_arr("t", self.t, None),
            p=_arr("p", self.p, None, 3),
            q=_arr("q", self.q, None, 4),
            v=_arr("v", self.v, None, 3),
            ba=_arr("ba", self.ba, None, 3),
            bg=_arr("bg", self.bg, None, 3),
        )
        _check_same_length(t=self.t, p=self.p, q=self.q, v=self.v, ba=self.ba, bg=self.bg)
        _check_increasing("Trajectory", self.t)
        _check_unit_quats("Trajectory.q", self.q)

    def __len__(self) -> int:
        return len(self.t)

    def state(self, i: int) -> NominalState:
        return NominalState(self.p[i], self.v[i], self.q[i], self.ba[i], self.bg[i])

    def slice(self, start: int, stop: int) -> Trajectory:
        s = slice(start, stop)
        return Trajectory(self.t[s], self.p[s], self.q[s], self.v[s], self.ba[s], self.bg[s])

    def v_body(self) -> np.ndarray:
        """Velocity in the body frame, v_b = R_wb^T v_w, shape (N, 3) (the network target)."""
        return np.einsum("nji,nj->ni", quat_to_R(self.q), self.v)


# Filter state and measurement

@dataclass(frozen=True, slots=True, eq=False)
class NominalState:
    """ESKF nominal state [p, v, q, b_a, b_g]."""

    p: np.ndarray  # (3,) world
    v: np.ndarray  # (3,) world
    q: np.ndarray  # (4,) wxyz, body->world
    ba: np.ndarray  # (3,)
    bg: np.ndarray  # (3,)

    def __post_init__(self):
        _set(
            self,
            p=_arr("p", self.p, 3),
            v=_arr("v", self.v, 3),
            q=_arr("q", self.q, 4),
            ba=_arr("ba", self.ba, 3),
            bg=_arr("bg", self.bg, 3),
        )
        _check_unit_quats("NominalState.q", self.q)

    @classmethod
    def identity(cls) -> NominalState:
        return cls(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))

    @property
    def R(self) -> np.ndarray:
        """R_wb (3, 3)."""
        return quat_to_R(self.q)


@dataclass(frozen=True, slots=True, eq=False)
class FilterState:
    """Nominal state + error-state covariance (15x15) at time t."""

    t: float
    nominal: NominalState
    P: np.ndarray  # (15, 15)

    def __post_init__(self):
        _set(self, t=float(self.t), P=_arr("P", self.P, ERR_DIM, ERR_DIM))


@dataclass(frozen=True, slots=True, eq=False)
class Measurement:
    """Generic measurement: value z and covariance R at time t (independent of the measurement source).

    For the network: z = v_b (3,), R = diag(sigma^2) (3x3).
    """

    t: float
    z: np.ndarray  # (m,)
    R: np.ndarray  # (m, m)

    def __post_init__(self):
        z = _arr("z", self.z, None)
        m = len(z)
        _set(self, t=float(self.t), z=z, R=_arr("R", self.R, m, m))


# Calibration and noise parameters

@dataclass(frozen=True, slots=True, eq=False)
class SensorCalib:
    """Values from ``imu0/sensor.yaml`` (continuous noise densities in dataset units)."""

    T_BS: np.ndarray  # (4, 4) sensor pose in the body frame
    rate_hz: float
    gyro_noise_density: float  # rad / s / sqrt(Hz)
    gyro_random_walk: float  # rad / s^2 / sqrt(Hz)
    acc_noise_density: float  # m / s^2 / sqrt(Hz)
    acc_random_walk: float  # m / s^3 / sqrt(Hz)

    def __post_init__(self):
        _set(self, T_BS=_arr("T_BS", self.T_BS, 4, 4), rate_hz=float(self.rate_hz))

    @property
    def R_BS(self) -> np.ndarray:
        return self.T_BS[:3, :3]

    @property
    def p_BS(self) -> np.ndarray:
        return self.T_BS[:3, 3]

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def is_identity_extrinsics(self) -> bool:
        return bool(np.allclose(self.T_BS, np.eye(4), atol=1e-9))


@dataclass(frozen=True, slots=True, eq=False)
class NoiseParams:
    """Filter tuning: continuous noise densities for Q, initial uncertainties P0 and a clamp for R.

    No default values on purpose: all numbers come from the config (``filter`` section),
    see ``from_config``. Kept separate from ``SensorCalib``: calibration is a property of
    the sensor, this is a filter choice.
    """

    acc_noise_density: float
    gyro_noise_density: float
    acc_random_walk: float
    gyro_random_walk: float
    init_std_p: float  # m
    init_std_v: float  # m/s
    init_std_theta: float  # rad
    init_std_ba: float  # m/s^2
    init_std_bg: float  # rad/s
    meas_var_min: float  # (m/s)^2, lower clamp of the velocity-measurement variance

    @classmethod
    def from_config(cls, calib: SensorCalib, cfg: Mapping) -> NoiseParams:
        """Build from the ``filter`` section of the config (a plain dict) and the sensor calibration.

        Noise densities come from ``calib`` multiplied by ``cfg["noise_scale"]``;
        initial uncertainties come directly from ``cfg["init_std"]``.
        """
        scale = cfg["noise_scale"]
        init_std = cfg["init_std"]
        return cls(
            acc_noise_density=scale["acc_noise"] * calib.acc_noise_density,
            gyro_noise_density=scale["gyro_noise"] * calib.gyro_noise_density,
            acc_random_walk=scale["acc_rw"] * calib.acc_random_walk,
            gyro_random_walk=scale["gyro_rw"] * calib.gyro_random_walk,
            init_std_p=init_std["p"],
            init_std_v=init_std["v"],
            init_std_theta=init_std["theta"],
            init_std_ba=init_std["ba"],
            init_std_bg=init_std["bg"],
            meas_var_min=cfg["meas_var_min"],
        )

    def P0(self) -> np.ndarray:
        """Initial error-state covariance (15x15, diagonal)."""
        std = np.concatenate(
            [
                np.full(3, self.init_std_p),
                np.full(3, self.init_std_v),
                np.full(3, self.init_std_theta),
                np.full(3, self.init_std_ba),
                np.full(3, self.init_std_bg),
            ]
        )
        return np.diag(std**2)


# Sequence

@dataclass(frozen=True, slots=True, eq=False)
class Sequence:
    """One dataset sequence: IMU and ground truth in the IMU frame on a common time axis."""

    name: str
    imu: ImuData
    gt: Trajectory
    calib: SensorCalib
    t0_ns: int  # absolute time of t = 0 [ns]

    def __post_init__(self):
        _set(self, t0_ns=int(self.t0_ns))
        if len(self.imu) == 0 or len(self.gt) == 0:
            raise ValueError(f"{self.name}: empty data")
        if self.gt.t[0] < self.imu.t[0] - 1e-9 or self.gt.t[-1] > self.imu.t[-1] + 1e-9:
            raise ValueError(f"{self.name}: ground truth extends beyond the IMU range")
