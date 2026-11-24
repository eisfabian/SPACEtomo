#!Python
# ===================================================================
#ScriptName SPACEtomo_postAction
# Purpose:      Post action script to setup targets from SPACEtomo runs.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/07/13
# Revision:     v1.2
# Last Change:  2026/11/24: external processing without local MM model (#10)
#               2026/11/19: no targets file when target setup failed, failed areas are marked and not retried
#               2026/11/14: fixed endless wait for predictions and repeated setup of maps with multiple target areas
#               2026/11/14: wait for inspection instead of ending acquisition, maps without targets count as processed
#               2026/11/06: fixed crash when collecting new point files
#               2024/09/04: converted settings to configparser
#               2024/08/21: made callable
#               2024/08/20: adjusted to refactor
#               2024/04/10: fixed typo
#               2024/03/25: added start of acquisition if not already running
#               2024_03_18: added wait_for_inspection
#               2024/03/08: outsourced target selection to updateQueue, adjusted check for point files instead of segmentations
#               2024/03/07: read in mic_params file instead of instantiating it
#               2023/11/12: fixes after Krios test
#               2023/10/31: fixes after Krios test
#               2023/10/16: fixes after Krios test
#               2023/10/05: outsourced functions
#               2023/10/03: fixes after Krios test
#               2023/09/27: clean up
#               2023/09/26: added queue for SPACEruns
#               2023/09/22: switched to nnUnet
#               2023/07/13: copied and crudely adjusted target setup from SPACEtomo_WG
# ===================================================================

import time
import json
from SPACEtomo.modules.utils import log
import SPACEtomo.modules.ext as space_ext
import SPACEtomo.modules.utils as utils
import SPACEtomo.modules.utils_sem as usem
from SPACEtomo.modules.scope import ImagingParams
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.mod_wg import WGModel
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.tgt import PACEArea

def getAreaName(point_file):
    """Returns the target area name of a point file (map name, with _A# suffix if the map has several target areas)."""

    map_name = point_file.name.split("_points")[0]
    map_areas = sorted(point_file.parent.glob(map_name + "_points*.json"))
    if len(map_areas) > 1:
        return map_name + f"_A{map_areas.index(point_file) + 1}"
    return map_name

def hasTargets(point_file):
    """Checks if point file contains targets (maps without targets get an empty point file)."""

    try:
        with open(point_file, "r", encoding="utf-8") as f:
            area = json.load(f, object_hook=utils.revertTaggedString)
    except (OSError, ValueError):
        return True     # Unreadable or not completely written yet: leave it to the target setup
    return len(area.get("points", [])) > 0

def isInspected(point_file, wait_for_inspection):
    """Checks if targets of the point file's map were inspected (if inspection is required)."""

    if not wait_for_inspection:
        return True
    map_name = point_file.name.split("_points")[0]
    return (point_file.parent / (map_name + "_inspected.txt")).exists()

def getFailedFile(cur_dir, area_name):
    """Returns file marking a target area whose setup failed (not retried by the post action until the file is deleted)."""

    return cur_dir / (area_name + "_setup_failed.txt")

def findUnprocessedPointFiles(map_dir, cur_dir, wait_for_inspection=False):
    """Returns point files whose target area has no targets file yet and whose setup did not fail before.

    Empty point files (map without targets) only count as processed once inspected, as manual selection starts with empty point files.
    """

    unprocessed = []
    for point_file in sorted(map_dir.glob("*_points*.json")):
        area_name = getAreaName(point_file)
        if (cur_dir / (area_name + "_tgts.txt")).exists() or getFailedFile(cur_dir, area_name).exists():
            continue
        if hasTargets(point_file) or not isInspected(point_file, wait_for_inspection):
            unprocessed.append(point_file)
    return unprocessed

def waitForPointFiles(map_dir, cur_dir, wait_for_inspection=False, update_function=None, interval=60):
    """Waits until point files are ready for target setup or no more maps are processed or inspected. Returns the ready point files."""

    while True:
        # Check pending maps before looking for point files, so a point file written in between is not missed
        mm_list, seg_list, _ = space_ext.monitorFiles(map_dir)
        unprocessed_point_files = findUnprocessedPointFiles(map_dir, cur_dir, wait_for_inspection)
        ready_point_files = [point_file for point_file in unprocessed_point_files if isInspected(point_file, wait_for_inspection)]
        if ready_point_files or not (mm_list or seg_list or unprocessed_point_files):
            return ready_point_files

        if unprocessed_point_files:
            log("Waiting for targets to be inspected by operator...")
        else:
            log("Waiting for next prediction before setting up targets...")
        time.sleep(interval)
        if update_function:
            update_function()

def main():
    CUR_DIR = utils.getCurDir()

    # Read settings written by main SPACE script
    settings = utils.loadSettings(CUR_DIR / "SPACEtomo_settings.ini")

    # Check if external processing directory is valid
    MAP_DIR, EXTERNAL_RUN = usem.prepareEnvironment(CUR_DIR, settings["external_map_dir"])

    # Instantiate mic params from settings
    imaging_params = ImagingParams(None, None, None, CUR_DIR)

    # Instantiate nav
    nav = Navigator(is_open=True)

    # Load models
    WG_model = WGModel(MAP_DIR, EXTERNAL_RUN)
    MM_model = space_ext.MMModel(info_dir=MAP_DIR if EXTERNAL_RUN else None)
    MM_model.setDimensions(imaging_params)

    # Instantiate tgt params from settings
    tgt_params = space_ext.TgtParams(file_dir=CUR_DIR, MM_model=MM_model)

    # Update SPACE runs and queue
    if not EXTERNAL_RUN:
        space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params)

    # Check for unprocessed point files (targets file is named by target area)
    unprocessed_point_files = findUnprocessedPointFiles(MAP_DIR, CUR_DIR, settings["MM_wait_for_inspection"])

    # Check for areas still to be acquired
    num_acq = len(nav.searchByEntry("Acquire", ["1"]))

    # If collection would stop, wait for maps still being processed or inspected
    if num_acq == 0 and not any(isInspected(point_file, settings["MM_wait_for_inspection"]) for point_file in unprocessed_point_files):
        unprocessed_point_files = waitForPointFiles(MAP_DIR, CUR_DIR, settings["MM_wait_for_inspection"],
                                                    update_function=(lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params)) if not EXTERNAL_RUN else None)

    # Set up targets for unprocessed point_files
    for point_file in unprocessed_point_files:
        map_name = point_file.name.split("_points")[0]
        area_name = getAreaName(point_file)

        # Check if targets were inspected
        if not isInspected(point_file, settings["MM_wait_for_inspection"]):
            log("")
            log(f"WARNING: {map_name} targets are still waiting for inspection! Skipping...")
            continue

        log(f"\nSetting up targets for {map_name}...")

        # Load map from nav
        map_id = nav.getIDfromNote(map_name + ".mrc")
        map_img = Buffer(nav_id=map_id)

        # Instantiate target area
        target_area = PACEArea(point_file, MM_model, imaging_params, map_img)
        if len(target_area) == 0: 
            log(f"WARNING: No targets selected on {area_name}. Skipping...")
            continue
        target_area.checkISLimit()

        # Set up navigator and create virtual maps
        if not target_area.prepareForAcquisition(CUR_DIR / (area_name + ".mrc"), nav):
            failed_file = getFailedFile(CUR_DIR, area_name)
            failed_file.write_text("Target setup failed, see SerialEM log. Delete this file to retry the target setup in the next post action.\n")
            log(f"ERROR: Targets on {area_name} could not be set up for acquisition! Skipping... (Delete {failed_file.name} to retry.)")
            continue

        # Save target file
        targets, geo_points, tgt_settings = target_area.getTargetsInfo(area_name)
        utils.writeTargets(CUR_DIR / (area_name + "_tgts.txt"), targets, geo_points, settings=tgt_settings)

    num_acq = len(nav.searchByEntry("Acquire", ["1"]))
    log(f"Calling PACEtomo acquisition of {int(num_acq)} areas!")

    # Start acquisition if not already running
    #if not sem.ReportIfNavAcquiring()[0]:
    #    sem.StartNavAcquireAtEnd()