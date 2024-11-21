#!/usr/bin/env python
# ===================================================================
# Purpose:      Functions for SerialEM utilities needed by other packages and scripts.
# Author:       Fabian Eisenstein
# Created:      2024/08/13
# ===================================================================

# Check SerialEM availability
try:
    import serialem as sem
    # Check if PySEMSocket is connected
    try:
        sem.Delay(0, "s")
        SERIALEM = True
    except sem.SEMmoduleError:
        SERIALEM = False
except ModuleNotFoundError:
    SERIALEM = False

import sys
import time
from pathlib import Path
from SPACEtomo.modules.utils import log
from SPACEtomo import __version__, version_SerialEM

def connectSerialEM(ip="127.0.0.1", port=48888):
    """Establishes connection to SerialEM."""
    
    sem.ConnectToSEM(port, ip)

    sem.SuppressReports()
    sem.ErrorsToLog()
    sem.SetNewFileType(0)           # set file type to mrc in case user changed default file type

def checkVersion():
    """Check for minimal version of SerialEM."""

    version, date_string = version_SerialEM
    versionCheck = sem.IsVersionAtLeast(version, date_string)
    if not versionCheck and sem.IsVariableDefined("warning_version") == 0:
        runScript = sem.YesNoBox("\n".join(["WARNING: You are using a version of SerialEM that does not support all SPACEtomo features. It is recommended to update to the latest SerialEM beta version!", "", "Do you want to attempt to run SPACEtomo regardless?"]))
        if not runScript:
            sem.Exit()
        else:
            sem.SetPersistentVar("warning_version", "")

def getSessionDir():
    """Sets up top level dir for SPACEtomo session."""

    if sem.IsVariableDefined("ses_dir") == 0:
        choice = sem.UserSetDirectory("Please choose a directory for this session!")
        if choice == "Cancel":
            log("ERROR: No session directory was selected. Aborting run!")
            sem.Exit()
        ses_dir = sem.ReportDirectory()
        sem.SetPersistentVar("ses_dir", ses_dir)
    else:
        ses_dir = sem.GetVariable("ses_dir")
        sem.SetDirectory(str(ses_dir))

    return Path(ses_dir)

def getGridList(grid_list, automation_level):
    """Checks which grids need to be processed."""

    if not isinstance(grid_list, list):
        grid_list = [grid_list]

    # Check if this is not first grid in list
    if sem.IsVariableDefined("grid_list") == 1:
        remaining_grid_list = sem.GetVariable("grid_list").split(",")

        # If grid list is empty, finish run
        if len(remaining_grid_list) == 1 and remaining_grid_list[0] == "":
            exitSPACEtomo()

        else:
            # Convert to int
            remaining_grid_list = [int(no) for no in remaining_grid_list]

            # Check if grid list was changed since the persistent var was created
            if not all(grid in grid_list for grid in remaining_grid_list):
                log("WARNING: Grid list was updated and previously saved grid list was discarded!")
                remaining_grid_list = grid_list

    else:
        remaining_grid_list = grid_list

    # Save rest of list in persistent var for later calls of script (only necessary when acquisition is automated)
    if automation_level >= 5:
        sem.SetPersistentVar("grid_list", ",".join(str(no) for no in remaining_grid_list[1:]))
        sem.SetStatusLine(5, "Grid: " + str(len(grid_list) - len(remaining_grid_list) + 1)  + " / " + str(len(grid_list)))

        # Only run all steps on current grid, then run PACEtomo before running SPACEtomo again
        remaining_grid_list = remaining_grid_list[:1]

    return remaining_grid_list

def prepareEnvironment(session_dir, grid_name, external_dir):
    """Sets up dirs and logs."""

    # Save old log before switching dir
    openNewLog()

    # Set subfolder for grid
    cur_dir = session_dir / grid_name
    cur_dir.mkdir(exist_ok=True)
    sem.SetDirectory(str(cur_dir))

    # Opens log in new dir
    openNewLog(grid_name, force=True)

    # Check if external processing directory is valid
    external = False
    if external_dir == "":
        try:
            import ultralytics
        except ModuleNotFoundError:
            log(f"ERROR: Packages for local processing could not be found. Please set up external processing!")
            sem.exit()
        map_dir = cur_dir / "SPACE_maps"
        map_dir.mkdir(exist_ok=True)
    else:
        map_dir = Path(external_dir)
        if map_dir.exists():
            external = True
        else:
            log(f"ERROR: External map [{map_dir}] directory does not exist!")
            sem.Exit()

    # Close all files
    while sem.ReportFileNumber() > 0:
        sem.CloseFile()

    return cur_dir, map_dir, external

def openNewLog(name=None, force=False):
    if not force:
        if name:
            sem.SaveLogOpenNew(name)
        else:
            sem.SaveLogOpenNew()
    else:
        sem.CloseLogOpenNew(1)
        if name:
            sem.SaveLog(1, name)
    
    # Start new log with version numbers
    if name:
        log(f"SPACEtomo Version {__version__}")
        sem.ProgramTimeStamps()
        log(f"DEBUG: Python {sys.version}")

def setupAcquisition(spacetomo_script_id, postaction_script_id, pacetomo_script_id):
    num_acq, *_ = sem.ReportNumNavAcquire()
    log(f"Starting PACEtomo acquisition of {int(num_acq)} areas!")

    # Setting Acquire at Items scripts
    sem.NavAcqAtEndUseParams("F")
    sem.SetNavAcqAtEndParams("prim", 2)
    sem.SetNavAcqAtEndParams("scrp-p", pacetomo_script_id)      # PACEtomo script
    sem.SetNavAcqAtEndParams("scrp-b", 0)
    sem.SetNavAcqAtEndParams("scrp-a", postaction_script_id)    # SPACEtomo_postAction script
    sem.RunScriptAfterNavAcquire(spacetomo_script_id)           # SPACEtomo script

    sem.StartNavAcquireAtEnd()

def confirmationBox(text):
    """Opens a popup box for user confirmation and script cancel option."""

    sem.Pause(text)

def exitSPACEtomo():
    """Successfully exit session."""

    log("##### SPACEtomo run completed! #####")
    log(time.strftime("%d.%m.%Y %H:%M:%S", time.localtime()))
    sem.ClearStatusLine(0)
    sem.ClearPersistentVars()
    sem.Exit()