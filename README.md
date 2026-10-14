# imuNN: Learned Inertial Odometry for Micro Aerial Vehicles

> **Note:** This project is in active production development. Core modules, filters, and neural architectures are undergoing continuous refinement.

## Overview
**imuNN** is an IMU-only state estimation framework designed to estimate the 6-DOF trajectory (velocity, orientation) and sensor biases of a micro aerial vehicle (MAV) purely from inertial measurements, without relying on external references such as GPS, vision, or LiDAR.

Standard dead-reckoning from consumer-grade IMU integration suffers from unbounded super-linear position drift due to uncompensated biases and sensor noise. **imuNN** addresses this by coupling high-rate classical inertial mechanization with deep learning:

- **Error-State Kalman Filter (ESKF):** Propagates full 15-D nominal state kinematics at sensor frequency (200 Hz) using a right-perturbation local Lie group formulation ($SO(3)$).
- **Learned Motion Priors:** A neural network ingests temporal windows of body-frame IMU signals and encoded orientation to periodically infer body-frame velocity pseudo-measurements along with heteroscedastic uncertainty ($\hat{v}_b, \operatorname{diag}(\sigma^2)$).
- **Statistical Sensor Fusion:** The filter fuses the network's learned predictions to constrain velocity drift and estimate IMU biases online, retaining physical kinematic consistency while mitigating dead-reckoning divergence.

```
       +------------------+
       | Raw IMU (200 Hz) |
       +--------+---------+
                |
       +--------v--------+         +----------------------------+
       | ESKF Prediction |-------->| Window Preprocessor        |
       |  (15-D State)   |         | (Raw IMU + Body Gravity)   |
       +--------+--------+         +--------------+-------------+
                |                                 | (20 Hz)
                |                          +------v-----+
                |                          | Neural Net |
                |                          +------+-----+
                |                                 |
                |   Body Velocity & Variance      |
                |<--------------------------------+
       +--------v--------+
       |   ESKF Update   |
       | (Bias & States) |
       +-----------------+
```

## Key Architectural Principles
- **No Absolute Measurement Leaks:** The network strictly estimates relative body velocity, preserving real-world physical invariants without over-promising absolute unobservable degrees of freedom (absolute position, global yaw).
- **Decoupled Filter & Learned Models:** Clean abstraction between estimator kinematics, measurement models, and network inference engines.
- **Formulation for High-Dynamic General Bodies:** Tailored for agile 6-DOF aerial platforms without non-holonomic wheeled constraints.
