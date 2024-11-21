#!/usr/bin/env python
# ===================================================================
# Purpose:      Dummy functions for SerialEM utilities needed by other packages and scripts.
# Author:       Fabian Eisenstein
# Created:      2024/08/26
# ===================================================================

import sys
import time
from pathlib import Path
from SPACEtomo.modules.utils import log
from SPACEtomo import __version__, version_SerialEM

def connectSerialEM(ip="127.0.0.1", port=48888):
    """Establishes connection to SerialEM."""
    
    log(f"#DUMMY: Connected to SerialEM IP: {ip}, Port: {port}")

def checkVersion():
    """Check for minimal version of SerialEM."""

    log(f"#DUMMY: Checked SerialEM version.")

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

def prepareEnvironment(session_dir, grid_name, external_dir):
    """Sets up dirs and logs."""

    # Set subfolder for grid
    cur_dir = session_dir / grid_name
    cur_dir.mkdir(exist_ok=True)
    
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

    return cur_dir, map_dir, external


def openNewLog(name=None, force=False):
   
    # Start new log with version numbers
    if name:
        log(f"SPACEtomo Version {__version__}")
        log(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

def setupAcquisition(spacetomo_script_id, postaction_script_id, pacetomo_script_id):
    log(f"Starting PACEtomo acquisition of X areas!")

def confirmationBox(text):
    """Opens a popup box for user confirmation and script cancel option."""

    log(text)

def exitSPACEtomo():
    """Successfully exit session."""

    log("##### SPACEtomo run completed! #####")
    log(time.strftime("%d.%m.%Y %H:%M:%S", time.localtime()))
