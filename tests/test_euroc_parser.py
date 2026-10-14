from pathlib import Path

import numpy as np
import pytest

from imuNN.data.euroc_parser import load_euroc_sequence, parse_sensor_yaml

DATA_DIR = Path("data/raw/machine_hall/MH_01_easy")


@pytest.mark.skipif(not DATA_DIR.exists(), reason="MH_01_easy data directory not found")
def test_parse_sensor_yaml():
    yaml_path = DATA_DIR / "mav0/imu0/sensor.yaml"
    calib = parse_sensor_yaml(yaml_path)

    assert calib.rate_hz == pytest.approx(200.0)
    assert calib.gyro_noise_density == pytest.approx(1.6968e-04)
    assert calib.gyro_random_walk == pytest.approx(1.9393e-05)
    assert calib.acc_noise_density == pytest.approx(2.0e-03)
    assert calib.acc_random_walk == pytest.approx(3.0e-03)
    assert calib.is_identity_extrinsics


@pytest.mark.skipif(not DATA_DIR.exists(), reason="MH_01_easy data directory not found")
def test_load_euroc_sequence():
    seq = load_euroc_sequence(DATA_DIR)

    assert seq.name == "MH_01_easy"
    assert len(seq.imu) > 0
    assert len(seq.gt) > 0

    # Ensure start is at t=0
    assert seq.gt.t[0] == pytest.approx(0.0)
    assert seq.imu.t[0] == pytest.approx(0.0, abs=1e-3)

    # Sequence lengths should be practically identical after trimming to GT interval
    assert abs(len(seq.imu) - len(seq.gt)) <= 1

    # Check timestamps strictly increasing
    assert np.all(np.diff(seq.imu.t) > 0)
    assert np.all(np.diff(seq.gt.t) > 0)

    # Check that quaternions are normalized and w >= 0
    assert np.allclose(np.linalg.norm(seq.gt.q, axis=1), 1.0, atol=1e-6)
    assert np.all(seq.gt.q[:, 0] >= 0.0)
