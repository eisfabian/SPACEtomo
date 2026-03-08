#!/usr/bin/env python
# ===================================================================
# Purpose:      Dummy functions for SerialEM utilities needed by other packages and scripts.
# Author:       Fabian Eisenstein
# Created:      2024/08/26
# Revision:     v1.4
# Last Change:  2026/03/06: added missing functions to match real utils_sem.py
#               2024/08/26: initial version
# ===================================================================

import sys
import time
import mrcfile
import numpy as np
from pathlib import Path
from SPACEtomo.modules.utils import log
from SPACEtomo import __version__, version_SerialEM

def connectSerialEM(ip="127.0.0.1", port=48888):
    """Establishes connection to SerialEM."""

    log(f"#DUMMY: Connected to SerialEM IP: {ip}, Port: {port}")

def checkVersion():
    """Check for minimal version of SerialEM."""

    log(f"#DUMMY: Checked SerialEM version.")

def checkImagingStates(states=[], low_dose_expected=[]):
    """Checks if user provided imaging states are sensible."""

    log(f"Checking imaging states...")
    log(f"#DUMMY: Checked imaging states.")

def getImagingStates():
    """Returns a list of imaging states in SerialEM."""

    # Return some plausible dummy states: (index, name, low_dose, camera, mag_index, pixel_size)
    imaging_states = [
        (1, "LM", -1, 0, 5, 400.0),
        (2, "IM", -1, 0, 15, 10.0),
        (3, "MM_View", 0, 0, 25, 1.0),
        (4, "MM_Record", 3, 0, 30, 0.3),
    ]
    log(f"#DUMMY: Retrieved {len(imaging_states)} imaging states.")
    return imaging_states

def getLowDoseParams(area: str):
    """Gets low dose area parameters from SerialEM."""

    ld_areas = {"V": 0, "F": 1, "T": 2, "R": 3, "S": 4}

    param_dict = {
        "low_dose_area": ld_areas.get(area, 0),
        "mag_index": 25,
        "spot_size": 6,
        "intensity": 0.12,
        "axis_offset_microns": 0.0,
        "mode": 0,
        "filter_slit_in": 1,
        "filter_slit_width": 20.0,
        "energy_loss": 0.0,
        "zero_loss_flag": 0,
        "beam_x_offset": 0.0,
        "beam_y_offset": 0.0,
        "alpha": -999.0,
        "diffraction_focus": -999.0,
        "beam_tilt_x": 0.0,
        "beam_tilt_y": 0.0,
        "probe_mode": 0,
        "dark_field_mode_flag": 0,
        "dark_field_tilt_x": 0.0,
        "dark_field_tilt_y": 0.0,
        "dose_modulation_percent": 100.0,
    }

    log(f"#DUMMY: Retrieved low dose parameters for area {area}.")
    return param_dict

def getSessionDir():
    """Sets up top level dir for SPACEtomo session."""

    return Path.cwd()

def setSessionDir(ses_dir):
    """Sets session directory in SerialEM and logs it."""

    ses_dir = Path(ses_dir)
    if not ses_dir.exists():
        log(f"ERROR: Session directory [{ses_dir}] does not exist!")
    log(f"DEBUG: Session directory set to {ses_dir}")

def getGridList(grid_list, automation_level):
    """Checks which grids need to be processed."""

    if not isinstance(grid_list, list):
        grid_list = [grid_list]

    remaining_grid_list = grid_list

    # Save rest of list in persistent var for later calls of script (only necessary when acquisition is automated)
    if automation_level >= 5:
        # Only run all steps on current grid, then run PACEtomo before running SPACEtomo again
        remaining_grid_list = remaining_grid_list[:1]

    return remaining_grid_list

def prepareEnvironment(cur_dir, external_dir):
    """Sets up dirs and logs."""

    # Check if external processing directory is valid
    external = False
    if external_dir == "":
        map_dir = cur_dir / "SPACE_maps"
        map_dir.mkdir(exist_ok=True)
    else:
        map_dir = Path(external_dir)
        if map_dir.exists():
            external = True
        else:
            log(f"ERROR: External map [{map_dir}] directory does not exist!")
            sys.exit()

    return map_dir, external

def setDirectory(directory):
    """Sets directory in SerialEM and logs it."""

    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    log(f"DEBUG: Directory set to {directory}")

def openOldFile(file_path, max_attempts=5, delay=5):
    """Allows for multiple attempts to open file to prevent crash when file is being copied during access."""

    log(f"#DUMMY: File [{file_path}] was opened.")

def switchToFile(file_path):
    """Switches to file in SerialEM."""

    log(f"#DUMMY: Switched to file [{file_path}] in SerialEM.")

def closeFile(file_path, save_mdoc=False):
    """Closes file in SerialEM."""

    switchToFile(file_path)

    if save_mdoc:
        log(f"#DUMMY: Mdoc for file [{file_path}] was saved.")
    log(f"#DUMMY: File [{file_path}] was closed.")

def getSectionNumber(file_path: Path):
    """Gets section number from open file or mrc file header."""

    if file_path.exists():
        with mrcfile.open(file_path) as mrc:
            return int(mrc.header["nz"])
    else:
        switchToFile(file_path)
        log(f"#DUMMY: Determined number of sections of file [{file_path}].")
        return 1

def addMdocData(file_path, key, value):
    """Adds key-value pair to mdoc file."""

    if isinstance(value, (list, tuple, np.ndarray)):
        value = " ".join(str(v) for v in value)

    switchToFile(file_path)
    log(f"#DUMMY: Added {key} = {value} to mdoc of file [{file_path}].")

def openNewLog(name=None, force=False):

    # Start new log with version numbers
    if name:
        log(f"SPACEtomo Version {__version__}")
        log(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

def checkWarnings():
    log(f"#DUMMY: Checked for warnings.")

def setupAcquisition(spacetomo_script_id, postaction_script_id, pacetomo_script_id):
    log(f"Starting PACEtomo acquisition of X areas!")

def confirmationBox(text):
    """Opens a popup box for user confirmation and script cancel option."""

    log(text)

def exitSPACEtomo():
    """Successfully exit session."""

    log("##### SPACEtomo run completed! #####")
    log(time.strftime("%d.%m.%Y %H:%M:%S", time.localtime()))

def exitPACEtomo():
    """Successfully exit session."""

    log("##### PACEtomo run completed! #####")
    log(time.strftime("%d.%m.%Y %H:%M:%S", time.localtime()))
