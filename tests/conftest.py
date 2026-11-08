"""Shared fixtures for SPACEtomo tests."""

import json
import os
import shutil
import configparser
from pathlib import Path

import numpy as np
import pytest

# Install mock serialem at import time (before pytest collects test modules),
# so that any module-level `import serialem` in tested code sees the mock.
from SPACEtomo.modules.dummy import install_mock_serialem
install_mock_serialem()

import SPACEtomo.config as config
config.DUMMY = True


# Every test runs once per emulated SerialEM version (restrict with --sem-version 4.2)
def pytest_addoption(parser):
    import serialem
    parser.addoption("--sem-version", action="append", choices=serialem.versions(),
                     help="SerialEM version(s) the mock emulates (default: all)")


def pytest_generate_tests(metafunc):
    import serialem
    if "sem_version" in metafunc.fixturenames:
        chosen = metafunc.config.getoption("sem_version") or serialem.versions()
        metafunc.parametrize("sem_version", chosen, indirect=True, ids=lambda v: f"sem{v}")


@pytest.fixture(autouse=True)
def sem_version(request):
    """Emulated SerialEM version for the test (each test starts with a fresh mock state)."""
    import serialem
    from SPACEtomo.modules.buf import Buffer
    previous = serialem.get_version()
    serialem.set_version(request.param)
    serialem.reset_mock_state()

    # Reproducible synthetic grids (SPACETOMO_TEST_SEED to vary) and no buffer state from previous tests
    np.random.seed(int(os.environ.get("SPACETOMO_TEST_SEED", 0)))
    Buffer.instances, Buffer.nav = [], None
    yield request.param
    serialem.set_version(previous)


@pytest.fixture()
def test_dir(tmp_path):
    """Create a temporary session directory with mic_params and settings."""
    return tmp_path


def make_mic_params(dest: Path):
    """Write a mic_params.json file to dest.

    Matrices and tilt axis rotations are from a real Krios/K3 session (SerialEM calibrations); pixel sizes give realistic
    fields of view on the non-square K3 camera: Record 1.4 x 1.0 µm, Focus one mag higher, View 22 x 16 µm, Search 69 x 49 µm.
    """
    from SPACEtomo.modules import utils

    def rotM(angle):
        angle = np.radians(angle)
        return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    # Real calibrations (c2ss per pixel size, scaled to the pixel sizes used here)
    c2ss = {"view": (np.array([[-0.00194144481793046, 0.0002074808144243434], [-0.0002074808144243434, -0.00194144481793046]]), 3.905),
            "rec": (np.array([[-6.891629891470075e-05, 6.259731890168041e-06], [-6.259731890168041e-06, -6.891629891470075e-05]]), 0.1384),
            "search": (np.array([[-0.0005187348579056561, 4.283008092897944e-05], [-4.283008092897944e-05, -0.0005187348579056561]]), 1.041)}
    ss2is = {"view": np.array([[-0.8903269171714783, -0.42751845717430115], [-0.46758294105529785, 0.8783678412437439]]),
             "rec": np.array([[-0.8787484169006348, -0.4537494480609894], [-0.4608369469642639, 0.8715442419052124]]),
             "search": np.array([[-0.8833439946174622, -0.4521932899951935], [-0.4644295573234558, 0.8690666556358337]])}
    areas = {   # pix size [nm], tilt axis rotation [deg], beam diameter [µm], calibration
        "view": (3.905, 96.10000633360785, 38.2, "view"),
        "rec": (0.25, 95.18998759663057, 1.8, "rec"),
        "focus": (0.20, 95.18998759663057, 1.5, "rec"),
        "search": (12.0, 94.72000115861476, 90.0, "search"),
    }

    mic_params = {
        "file_name": "mic_params.json",
        "WG_image_state": "WG",
        "IM_image_state": "IM",
        "MM_image_state": ["MM1"],
        "cam_dims": np.array([5760, 4092]),
        "IS_limit": 20.0,
        "s2ss_matrix": np.array([[-1.0323213338851929, -0.005192025564610958], [-0.005323351360857487, -1.0114827156066895]]),
        "WG_pix_size": None,
        "IM_pix_size": None,
    }
    for area, (pix_size, rotation, beam_diameter, calibration) in areas.items():
        matrix, calibration_pix_size = c2ss[calibration]
        mic_params[f"{area}_pix_size"] = pix_size
        mic_params[f"{area}_c2ss_matrix"] = matrix * pix_size / calibration_pix_size
        mic_params[f"{area}_ss2is_matrix"] = ss2is[calibration]
        mic_params[f"{area}_ta_rotation"] = rotation
        mic_params[f"{area}_rotM"] = rotM(rotation)
        mic_params[f"{area}_beam_diameter"] = beam_diameter

    with open(dest, "w") as f:
        json.dump(mic_params, f, indent=4, default=utils.convertToTaggedString)


def make_settings(dest: Path, session_dir: Path, automation_level: int = 4, manual_selection: bool = False):
    """Write a SPACEtomo_settings.ini file (same settings as a standalone DUMMY run)."""
    from SPACEtomo.modules.dummy.session import writeSettings
    writeSettings(dest, session_dir, automation_level, manual_selection)


@pytest.fixture()
def session_dir(test_dir):
    """Set up a complete session directory ready for a SPACEtomo run."""
    import serialem as sem

    ses_dir = test_dir
    grid_dir = ses_dir / "G01"
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / "SPACE_maps").mkdir(exist_ok=True)

    # Write mic_params
    for dest in [ses_dir / "mic_params.json", grid_dir / "mic_params.json"]:
        make_mic_params(dest)

    # Write settings
    make_settings(ses_dir / "SPACEtomo_settings.ini", ses_dir)
    shutil.copy(ses_dir / "SPACEtomo_settings.ini", grid_dir / "SPACEtomo_settings.ini")

    # Set session dir in mock persistent vars
    sem.reset_mock_state()
    sem.SetPersistentVar("ses_dir", str(ses_dir))

    return ses_dir
