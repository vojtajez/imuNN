# Core modules: `rotations.py` and `types.py`

Short reference of what each piece does and where it will be used. Conventions apply to the whole project.

## Conventions

| Item | Convention |
|---|---|
| Quaternion | `q = [w, x, y, z]`, Hamilton, unit norm |
| Rotation | `R_wb` / `q_wb`: body -> world, `v_w = R_wb @ v_b` |
| Perturbation | right (local, body frame): `R = R_hat · Exp(δθ)`, `q = q_hat ⊗ Exp(δθ)` |
| World frame | z-up, gravity `G_W = [0, 0, -9.81]`; the accelerometer measures specific force |
| Time | `float64` seconds relative to `Sequence.t0_ns` |
| ESKF error state | 15D `[δp, δv, δθ, δb_a, δb_g]` (slices `P_SL, V_SL, TH_SL, BA_SL, BG_SL`) |
| Batches | all rotation functions accept leading batch dims: `(..., 3)`, `(..., 3, 3)`, `(..., 4)` |

## `core/rotations.py`

Pure numpy, no state. Reference: Solà, *Quaternion kinematics for the error-state Kalman filter* (arXiv 1711.02508).

| Function | What it does | Where it is used |
|---|---|---|
| `skew(v)` / `vee(M)` | cross-product matrix `[v]×` and its inverse | ESKF Jacobians (`H_θ = [v_b]×`, `F` blocks), inside `exp_so3` |
| `exp_so3(φ)` | rotation vector -> rotation matrix (Rodrigues, Taylor for small angles) | right perturbation of `R`, tests against finite differences |
| `log_so3(R)` | rotation matrix -> rotation vector, `|φ| ∈ [0, π]`, stable near π | orientation error, evaluation |
| `quat_identity`, `quat_normalize`, `quat_conj` | basic quaternion helpers | state initialization, re-normalization after integration, inverse |
| `quat_mult(p, q)` | Hamilton product `p ⊗ q` (`R(p⊗q) = R(p)R(q)`) | composing orientations |
| `quat_exp(φ)` / `quat_log(q)` | rotation vector <-> quaternion | gyro integration in strapdown (`q ← q ⊗ Exp(ω̂ dt)`), conversion of error states |
| `quat_to_R(q)` / `R_to_quat(R)` | quaternion <-> matrix (`R_to_quat` returns `w ≥ 0`, Shepperd) | `h(X) = R̂ᵀ v` in the measurement model, parsing ground truth, window preprocessing |
| `quat_rotate(q, v)` | rotate a vector by `q` | body <-> world vector conversion (e.g. gravity in the body frame `R_wbᵀ g_w` as network input) |
| `quat_boxplus(q, δθ)` | `q ⊕ δθ = q ⊗ Exp(δθ)`, re-normalized | ESKF error injection after an update |
| `quat_boxminus(q1, q2)` | `Log(q2⁻¹ ⊗ q1)` so that `q2 ⊕ (q1 ⊖ q2) = q1` | orientation error metric, NEES |

## `core/types.py`

Immutable dataclasses (`frozen=True, slots=True, eq=False`) over numpy arrays. Shapes are checked in `__post_init__`; an invalid object cannot be created. Why dataclasses and not pydantic: no overhead in the 200 Hz filter loop and natural numpy support.

| Type | Content | Where it is used |
|---|---|---|
| `GRAVITY`, `G_W` | gravity magnitude and vector in the world frame | strapdown prediction, gravity input of the network |
| `ERR_DIM`, `*_SL` | size and slices of the 15D error state | indexing `P`, `F`, `H`, error injection |
| `ImuSample` | one sample `(t, acc, gyr)` | streaming into the filter at runtime |
| `ImuData` | arrays `t (N)`, `acc (N,3)`, `gyr (N,3)`; raw, bias not subtracted | parser output, strapdown, windows. `.sample(i)`, `.slice(a, b)` |
| `Trajectory` | `t, p, q, v, ba, bg` as `(N, ...)` arrays | ground truth and filter output. `.v_body()` = network target, `.state(i)`, `.slice(a, b)` |
| `NominalState` | `p, v, q, ba, bg` | state propagated by the ESKF; `.R` gives `R_wb`, `.identity()` |
| `FilterState` | `t`, `NominalState`, `P (15x15)` | what the estimator holds between steps |
| `Measurement` | `t, z, R` (any dimension) | the filter's only input besides IMU: network velocity (`z = v_b`, `R = diag(σ²)`), oracle ground-truth velocity (B3 baseline), future sources |
| `SensorCalib` | `T_BS`, `rate_hz`, noise densities from `imu0/sensor.yaml` | filled by the parser; source for `Q` |
| `NoiseParams` | noise densities, initial stds, `meas_var_min` | filter tuning: `from_calib(calib, scale)` builds it, `P0()` gives the initial covariance |
| `Sequence` | `name, imu, gt, calib, t0_ns` | one dataset sequence, the unit consumed by the window dataset, runner and evaluation. Validates that ground truth lies within the IMU range |

### Notes
- `SensorCalib` (property of the sensor) and `NoiseParams` (choice of the filter) are separate on purpose: tuning `Q` must not change the calibration.
- `Trajectory.q` is checked to be unit norm with tolerance `1e-3`; the parser re-normalizes ground truth on load (EuRoC quaternions deviate up to ~3e-4).
- In EuRoC the ground truth (`state_groundtruth_estimate0`) is already in the IMU frame and `T_BS` is the identity for all sequences, so no extrinsic transform is applied (`SensorCalib.is_identity_extrinsics` can assert it).
