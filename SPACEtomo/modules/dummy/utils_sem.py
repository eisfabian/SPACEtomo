#!/usr/bin/env python
# ===================================================================
# Purpose:      Dummy functions for SerialEM utilities needed by other packages and scripts.
# Author:       Fabian Eisenstein
# Created:      2024/08/26
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

def getSessionDir():
    """Sets up top level dir for SPACEtomo session."""

    return Path.cwd()

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
            sys.Exit()

    return map_dir, external

def setDirectory(directory):
    """Sets directory in SerialEM and logs it."""

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