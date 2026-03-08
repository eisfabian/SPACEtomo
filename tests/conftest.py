"""Shared fixtures for SPACEtomo tests."""

import json
import shutil
import configparser
from pathlib import Path

import numpy as np
import pytest

from SPACEtomo.modules.dummy import install_mock_serialem


@pytest.fixture(scope="session", autouse=True)
def mock_serialem():
    """Install mock serialem module once for the entire test session."""
    install_mock_serialem()

    import SPACEtomo.config as config
    config.DUMMY = True

    yield

    # Reset mock state
    import serialem as sem
    sem.reset_mock_state()


@pytest.fixture()
def test_dir(tmp_path):
    """Create a temporary session directory with mic_params and settings."""
    return tmp_path


def make_mic_params(dest: Path):
    """Write a mic_params.json file to dest."""
    from SPACEtomo.modules import utils

    rot_angle = np.radians(-84.0)
    rotM = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle),  np.cos(rot_angle)],
    ])
    mic_params = {
        "file_name": "mic_params.json",
        "WG_image_state": "WG",
        "IM_image_state": "IM",
        "MM_image_state": ["MM1"],
        "cam_dims": np.array([4096, 4096]),
        "rec_cam_dims": np.array([4096, 4096]),
        "rec_pix_size": 1.5,
        "rec_ta_rotation": -84.0,
        "rec_rotM": rotM,
        "rec_beam_diameter": 800.0,
        "view_pix_size": 6.0,
        "view_ta_rotation": -84.0,
        "view_rotM": rotM,
        "view_beam_diameter": 3200.0,
        "view_offset": np.array([0.0, 0.0]),
        "focus_offset": np.array([5.0, 0.0]),
        "focus_pix_size": 6.0,
        "search_pix_size": 12.0,
        "search_ta_rotation": -84.0,
        "search_rotM": rotM,
        "search_beam_diameter": 6400.0,
        "is_limit": 15.0,
        "IS_limit": 15.0,
        "s2ss_matrix": np.array([[0.98, -0.17], [0.17, 0.98]]),
        "ss2s": np.array([[0.98, -0.17], [0.17, 0.98]]),
        "WG_pix_size": None,
    }
    with open(dest, "w") as f:
        json.dump(mic_params, f, indent=4, default=utils.convertToTaggedString)


def make_settings(dest: Path, session_dir: Path, automation_level: int = 4, manual_selection: bool = False):
    """Write a SPACEtomo_settings.ini file."""
    settings = configparser.ConfigParser()
    settings["INFO"] = {"SPACEtomo version": "1.4.0b5"}
    settings["SETTINGS"] = {
        "automation_level": str(automation_level),
        "grid_list": "[1]",
        "lamella": "True",
        "exclude_lamella_classes": "['broken', 'gone']",
        "WG_distance_threshold": "5",
        "WG_image_state": "WG",
        "IM_image_state": "IM",
        "MM_image_state": "['MM1']",
        "WG_wait_for_inspection": "False",
        "manual_selection": str(manual_selection),
        "MM_wait_for_inspection": "False",
        "target_list": "['coating', 'mitos']",
        "avoid_list": "['white', 'black', 'crack', 'ice']",
        "target_score_threshold": "0.095",
        "penalty_weight": "0.3",
        "sparse_targets": "True",
        "target_edge": "True",
        "extra_tracking": "False",
        "max_tilt": "50",
        "beam_margin": "5.0",
        "IS_limit": "15.0",
        "script_numbers": "[10, 11, 5]",
        "session_dir": str(session_dir),
        "external_map_dir": "",
    }
    with open(dest, "w") as f:
        settings.write(f)


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
