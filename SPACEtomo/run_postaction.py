#!Python
# ===================================================================
#ScriptName SPACEtomo_postAction
# Purpose:      Post action script to setup targets from SPACEtomo runs.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/07/13
# Revision:     v1.2
# Last Change:  2024/09/04: converted settings to configparser
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

import sys
import time
from pathlib import Path
from SPACEtomo.modules.utils import log
import SPACEtomo.modules.ext as space_ext
import SPACEtomo.modules.utils as utils
from SPACEtomo.modules.scope import ImagingParams
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.mod_wg import WGModel
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.tgt import PACEArea

def main():
    CUR_DIR = utils.getCurDir()

    # Read settings written by main SPACE script
    settings = utils.loadSettings(CUR_DIR / "SPACEtomo_settings.ini")

    # Check if external processing directory is valid
    MM_external = False
    if settings["external_map_dir"] == "":
        MAP_DIR = CUR_DIR / "SPACE_maps"
    else:
        MAP_DIR = Path(settings["external_map_dir"])
        if MAP_DIR.exists():
            MM_external = True
        else:
            log(f"ERROR: External map directory [{MAP_DIR}] does not exist!")
            sys.exit()

    # Instantiate mic params from settings
    imaging_params = ImagingParams(None, None, None, CUR_DIR)

    # Instantiate nav
    nav = Navigator(is_open=True)

    # Load models
    WG_model = WGModel()
    MM_model = space_ext.MMModel()
    MM_model.setDimensions(imaging_params)

    # Instantiate tgt params from settings
    tgt_params = space_ext.TgtParams(file_dir=CUR_DIR, MM_model=MM_model)

    # Update SPACE runs and queue
    if not MM_external:
        space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params)

    # Check for unprocessed point files
    point_files = sorted(MAP_DIR.glob("*_points*.json"))
    tgt_files = sorted(CUR_DIR.glob("*tgts.txt"))

    unprocessed_point_files = []
    for point_file in point_files:
        map_name = point_file.name.split("_points")[0]
        if map_name not in [tgt_file.name.split("_tgts")[0] for tgt_file in tgt_files]:
            unprocessed_point_files.append(point_file)

    # Check for unprocessed maps
    mm_list, seg_list, wg_list  = space_ext.monitorFiles(MAP_DIR)

    # Check for areas still to be acquired
    num_acq = len(nav.searchByEntry("Acquire", ["1"]))

    # Check if collection would stop
    if num_acq == 0 and len(unprocessed_point_files) == 0:
        # Check if there will be further point files generated
        if len(mm_list) > 0 or len(seg_list) > 0:
            # Wait for another segmentation to be processed
            start_len_seg = len(seg_list)
            while len(seg_list) >= start_len_seg:
                log("Waiting for next prediction before setting up targets...")
                time.sleep(60)
                if not MM_external:
                    space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params)
                mm_list, seg_list, wg_list  = space_ext.monitorFiles(MAP_DIR)

            # Get point files that were not found previously
            point_files_new = sorted(MAP_DIR.glob("*_points*.json"))
            unprocessed_point_files = point_files_new - point_files

    # Set up targets for unprocessed point_files
    for point_file in unprocessed_point_files:
        map_name = point_file.name.split("_points")[0]

        # Check for map name areas to determine area_name
        map_areas = sorted(MAP_DIR.glob(map_name + "_points*.json"))
        if len(map_areas > 1):
            area_name = map_name + f"_A{map_areas.index(point_file) + 1}"
        else:
            area_name = map_name

        # Check if targets were inspected
        if settings["MM_wait_for_inspection"]:
            inspected = (MAP_DIR / (map_name + "_inspected.txt")).exists()
        else:
            inspected = True

        if not inspected:
            log("")
            log(f"WARNING: {map_name} targets are still waiting for inspection! Skipping...")
            continue

        log(f"\nSetting up targets for {map_name}...")

        # Load map from nav
        map_id = nav.getIDfromNote(map_name + ".mrc")
        map_img = Buffer(nav_id=map_id)

        # Instantiate target area
        target_area = PACEArea(point_file, MM_model, imaging_params, map_img)
        target_area.checkISLimit()

        # Set up navigator and create virtual maps
        target_area.prepareForAcquisition(CUR_DIR / (area_name + ".mrc"), nav)

        # Save target file
        targets, geo_points, tgt_settings = target_area.getTargetsInfo()
        utils.writeTargets(CUR_DIR / (area_name + "_tgts.txt"), targets, geo_points, settings=tgt_settings)

    num_acq = len(nav.searchByEntry("Acquire", ["1"]))
    log(f"Calling PACEtomo acquisition of {int(num_acq)} areas!")

    # Start acquisition if not already running
    #if not sem.ReportIfNavAcquiring()[0]:
    #    sem.StartNavAcquireAtEnd()