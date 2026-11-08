#!/usr/bin/env python
# ===================================================================
# ScriptName:   session
# Purpose:      Standalone DUMMY run: SPACEtomo on a synthetic grid with the simulated microscope (no SerialEM).
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein / Claude
# Created:      2026/11/08
# Revision:     v1.4
# Last Change:  2026/11/08: created
# ===================================================================

import configparser
import importlib
from pathlib import Path

import numpy as np


def writeSettings(dest, session_dir, automation_level=4, manual_selection=False):
    """Writes SPACEtomo_settings.ini for a run with the simulated microscope (imaging states WG, IM, MM1)."""

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
        "target_list": "['mitos', 'nucleus']",
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


def prepareSession(session_dir, automation_level=4, manual_selection=False, grid_name="G01"):
    """Creates session and grid folders with settings and tells the mock serialem module about the session."""

    import serialem as sem

    session_dir = Path(session_dir)
    grid_dir = session_dir / grid_name
    (grid_dir / "SPACE_maps").mkdir(parents=True, exist_ok=True)
    for dest in (session_dir / "SPACEtomo_settings.ini", grid_dir / "SPACEtomo_settings.ini"):
        writeSettings(dest, session_dir, automation_level, manual_selection)
    sem.SetPersistentVar("ses_dir", str(session_dir))
    return session_dir


def run(session_dir, automation_level=4, seed=0):
    """Runs SPACEtomo (levels 1-4) with the simulated microscope; the synthetic grid is seeded."""

    import SPACEtomo.config as config
    config.DUMMY = True
    from SPACEtomo.modules.dummy import install_mock_serialem, reset_mock_state
    install_mock_serialem()
    import serialem as sem

    if int(automation_level) > 4:
        raise ValueError("Automation level 5 (PACEtomo acquisition) cannot be simulated.")

    reset_mock_state()
    np.random.seed(int(seed))
    session_dir = prepareSession(session_dir, int(automation_level))

    import SPACEtomo.run as run_module
    importlib.reload(run_module)        # picks up the session dir from the mock
    try:
        run_module.main()
    except sem.SEMexited:
        pass                            # a finished run ends the script with Exit()
    print(f"Dummy run finished. Ground truth of the synthetic grid: {session_dir / 'G01' / 'dummy_ground_truth.json'}")
