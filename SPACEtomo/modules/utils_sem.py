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
import mrcfile
import numpy as np
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

def checkImagingStates(states=[], low_dose_expected=[]):
    """Checks if user provided imaging states are sensible."""

    log(f"Checking imaging states...")

    if len(states) != len(low_dose_expected):
        log("ERROR: Number of imaging states and low dose requirements do not match!")
        sem.Exit()

    mag_list = []
    for s, state in enumerate(states):
        state_props = sem.ImagingStateProperties(str(state))

        if isinstance(state_props, (list, tuple)):
            error, index, low_dose, camera, mag_index, name = state_props
        else:
            error = state_props

        # Check if state exists
        if error > 0:
            if error == 1 or error == 3:
                log(f"ERROR: Imaging state {state} does not exist!")
            elif error == 2:
                log(f"ERROR: Imaging state {state} is ambiguous!")
            else:
                log(f"ERROR: Imaging state {state} caused unspecified error!")
            sem.Exit()

        # Check if state meets low dose requirement
        if low_dose_expected[s] and low_dose < 0:
            log(f"ERROR: Imaging state {state} is not a low dose state but should be!")
            sem.Exit()
        elif not low_dose_expected[s] and low_dose > 0:
            log(f"ERROR: Imaging state {state} is a low dose state but should not be!")
            sem.Exit()

        if low_dose < 0: # All states in low dose seem to have mag_index = 0 and should not be compared
            mag_list.append(mag_index)

    # Check if mags are duplicate
    if len(mag_list) != len(set(mag_list)):
        log("ERROR: Imaging states have duplicate magnifications!")
        sem.Exit()

def getSessionDir():
    """Sets up top level dir for SPACEtomo session."""

    if sem.IsVariableDefined("ses_dir") == 0:
        choice = sem.UserSetDirectory("Please choose a directory for this session!")
        if choice == "Cancel":
            log("ERROR: No session directory was selected. Aborting run!")
            sem.Exit()
        ses_dir = Path(sem.ReportDirectory())
        sem.SetPersistentVar("ses_dir", str(ses_dir))
    else:
        ses_dir = Path(sem.GetVariable("ses_dir"))
        if not ses_dir.exists():
            log(f"ERROR: Session directory [{ses_dir}] does not exist! Please run SPACEtomo again to choose a new directory.")
            sem.CLearPersistentVars()
            sem.Exit()
        setDirectory(ses_dir)

    # Check if grid dir was chosen instead
    if list(ses_dir.glob("*.nav")) or list(ses_dir.glob("SPACE_maps")):
        sem.ClearPersistentVars() # Clear persistent vars to allow for new directory selection in case user does not continue
        confirmationBox("WARNING: It seems like you chose a grid directory instead of a session directory. A new folder for each grid will be created in the chosen directory.")
        sem.SetPersistentVar("ses_dir", str(ses_dir)) # Save grid dir as session dir in case user continues

    return ses_dir

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

def prepareEnvironment(cur_dir, external_dir):
    """Sets up dirs and logs."""

    # Save SerialEM settings
    sem.SaveSettings()

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

    return map_dir, external

def setDirectory(directory: Path):
    """Sets directory in SerialEM and logs it."""

    directory.mkdir(exist_ok=True)
    sem.SetDirectory(str(directory))
    log(f"DEBUG: Directory set to {directory}")

def openOldFile(file_path, max_attempts=5, delay=5):
    """Allows for multiple attempts to open file to prevent crash when file is being copied during access."""

    attempts = 0
    while attempts < max_attempts:
        try:
            sem.StartTry(1)
            sem.OpenOldFile(str(file_path))
            # Break on success
            break
        except Exception as e:
            attempts += 1
            if attempts == max_attempts:
                raise e
            log(f"WARNING: File [{file_path}] could not be opened. Trying again... [{attempts}]")
            time.sleep(delay)
        finally:
            sem.EndTry()

def switchToFile(file_path):
    """Switches to file in SerialEM."""

    # Search for file number and switch to matching file
    success = False
    for f in range(1, int(sem.ReportNumOpenFiles()) + 1):
        if sem.ReportOpenImageFile(f) == str(file_path):
            sem.SwitchToFile(f)
            success = True
            break

    if not success:
        log(f"ERROR: File [{file_path}] is not open in SerialEM!")

def closeFile(file_path, save_mdoc=False):
    """Closes file in SerialEM."""

    if sem.IsImageFileOpen(str(file_path)):
        switchToFile(file_path)
    else:
        log(f"WARNING: File [{file_path}] is already closed!")
        return

    if save_mdoc:
        sem.WriteAutodoc()
    sem.CloseFile()

def getSectionNumber(file_path: Path):
    """Gets section number from open file or mrc file header."""

    if sem.IsImageFileOpen(str(file_path)):
        switchToFile(file_path)
        return int(sem.ReportFileZsize())
    
    elif file_path.exists():
        with mrcfile.open(file_path) as mrc:
            return int(mrc.header["nz"])
        
    else:
        log(f"ERROR: Could not determine section number! File might not exist.")
        return 0
    
def addMdocData(file_path, key, value):
    """Adds key-value pair to mdoc file."""

    if isinstance(value, (list, tuple, np.ndarray)):
        value = " ".join(str(v) for v in value)

    switchToFile(file_path)
    sem.AddToAutodoc(key, str(value))

def openNewLog(name=None, force=False):
    if not force:
        if name:
            sem.SaveLogOpenNew(str(name))
        else:
            sem.SaveLogOpenNew()
    else:
        sem.CloseLogOpenNew(1)
        if name:
            sem.SaveLog(1, str(name))
    
    # Start new log with version numbers
    if name:
        log(f"SPACEtomo Version {__version__}")
        sem.ProgramTimeStamps()
        log(f"DEBUG: Python {sys.version}")

def checkWarnings():
    # Warnings
    # Focus area offset
    if int(sem.ReportLowDose()[0]) and int(sem.ReportAxisPosition("F")[0]) != 0 and sem.IsVariableDefined("warningFocusArea") == 0:
        confirmationBox("WARNING: Position of Focus area is not 0! Please set it to 0 to autofocus on the tracking target!")
        sem.SetPersistentVar("warningFocusArea", "")

    # Tilt axis offset
    tiltAxisOffset = sem.ReportTiltAxisOffset()[0]
    if float(tiltAxisOffset) == 0 and sem.IsVariableDefined("warningTAOffset") == 0:
        confirmationBox("WARNING: No tilt axis offset was set! Please run the PACEtomo_measureOffset script to determine appropiate tilt axis offset.")
        sem.SetPersistentVar("warningTAOffset", "")

    # Coma vs image shift calibrations
    try:
        sem.StartTry(1)
        sem.ReportComaVsISmatrix()
    except sem.SEMerror:
        confirmationBox("WARNING: Coma vs image shift is not calibrated! This might result in excessive beam tilts during acquisition.")
        log(f"NOTE: Continuing without coma vs image shift calibration...")
    finally:
        sem.EndTry()

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
    log(text)

def exitSPACEtomo():
    """Successfully exit session."""

    log("##### SPACEtomo run completed! #####")
    log(time.strftime("%d.%m.%Y %H:%M:%S", time.localtime()))
    sem.ClearStatusLine(0)
    sem.ClearPersistentVars()
    sem.Exit()

def exitPACEtomo():
    """Successfully exit session."""

    log("##### PACEtomo run completed! #####")
    log(time.strftime("%d.%m.%Y %H:%M:%S", time.localtime()))
    sem.ClearStatusLine(0)
    sem.ClearPersistentVars()
    sem.Exit()