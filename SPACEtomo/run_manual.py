#!Python
# ===================================================================
# Purpose:      Converts SerialEM montage map to SPACE map and facilitates SPACEtomo target selection.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2025/01/09
# Revision:     v1.3
# Last Change:  2025/03/21: fixed map naming
#               2025/02/17: added handling of multiple maps
#               2025/01/09: created
# ===================================================================

import SPACEtomo.config as config

# Put SerialEM python modules on PATH
import sys
if not config.DUMMY:
    sys.path.insert(len(sys.path), config.SERIALEM_PYTHON_PATH)
    try:
        import serialem as sem
        SERIALEM = True

        # Connect to SEM
        from SPACEtomo.modules.utils_sem import connectSerialEM
        connectSerialEM(config.SERIALEM_IP, config.SERIALEM_PORT)

    except ModuleNotFoundError:
        SERIALEM = False
else:
    SERIALEM = False

import mrcfile
from pathlib import Path
import SPACEtomo.modules.ext as space_ext
from SPACEtomo.modules.utils import log
from SPACEtomo.modules.scope import Microscope, ImagingParams
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.mod_wg import WGModel
from SPACEtomo.modules.tgt import PACEArea
import SPACEtomo.modules.utils as utils
import SPACEtomo.modules.utils_sem as usem

if config.DUMMY:    # Shadow classes with dummy classes
    log(f"WARNING: Running SPACEtomo in DUMMY mode! Run SPACEconfig dummy command to toggle it off!")
    from SPACEtomo.modules.dummy.scope import MicroscopeDummy as Microscope
    from SPACEtomo.modules.dummy.buf import BufferDummy as Buffer
    import SPACEtomo.modules.dummy.utils_sem as usem

### Run on import: Connect to SerialEM, get session directory

# Connect SerialEM
usem.connectSerialEM(config.SERIALEM_IP, config.SERIALEM_PORT)

# Check SerialEM version
usem.checkVersion()

# Check if session dir has been chosen previously or ask for it
#SES_DIR = usem.getSessionDir()
choice = sem.UserSetDirectory("Please choose a directory for this grid!")
if choice == "Cancel":
    log("ERROR: No directory was selected. Aborting run!")
    sem.Exit()
SES_DIR = Path(sem.ReportDirectory())

###

def main():
    # Read settings file
    settings = utils.loadSettings(SES_DIR / "SPACEtomo_settings.ini")
    if settings["manual_selection"] and not settings["MM_wait_for_inspection"]: # Force inspection in case of manual selection
        settings["MM_wait_for_inspection"] = True

    # Check model
    if config.NO_MM_MODEL:
        log(f"ERROR: Lamella segmentation model could not be found! Please import it using the SPACEmodel command!")
        return

    # Setup dir
    CUR_DIR = SES_DIR
    MAP_DIR, EXTERNAL_RUN = usem.prepareEnvironment(CUR_DIR, settings["external_map_dir"])

    # Open log
    usem.openNewLog("SPACEtomo_target_selection")

    # Initialize microscope control
    microscope = Microscope()
    imaging_params = ImagingParams(None, None, settings["MM_image_state"], file_dir=SES_DIR)
    imaging_params.IS_limit = microscope.is_limit

    # Check if imaging states are valid
    usem.checkImagingStates(settings["MM_image_state"], low_dose_expected=[True] * len(settings["MM_image_state"]))

    # Load nav
    nav = Navigator(is_open=True)
    Buffer.nav = nav            # Provide reference for all buffers

    if config.DUMMY:            # Provide additional references to dummy classes
        microscope.cur_dir = CUR_DIR
        microscope.map_dir = MAP_DIR
        microscope.nav = nav
        Buffer.imaging_params = imaging_params
        if imaging_params.rec_ta_rotation is None:
            log(f"ERROR: Cannot run in DUMMY mode without mic_params.json!")
            return
        
    #########################################
    ############ STEP 1: MM map  ############
    #########################################

    # Get list of map items marked for Acquisition
    acquire_list = nav.searchByEntry("Acquire", "1")
    map_list = nav.searchByEntry("Type", "2", subset=acquire_list) if acquire_list else []
    if map_list:
        log(f"Found {len(map_list)} maps marked for acquisition.", style=1)
    else:
        # Use selected nav item
        if not nav.selected_item:
            log(f"ERROR: No item selected in Navigator! Please select a map or mark multiple maps for acquisition!")
            return
        else:
            if nav.selected_item.item_type != 2:
                log(f"ERROR: Selected item is not a map! Please select a map or mark multiple maps for acquisition!")
                return
        map_list = [nav.selected_item.nav_index - 1]

    # Save settings for SPACEtomo_postAction.py
    utils.saveSettings(MAP_DIR / "SPACEtomo_settings.ini", settings, list(settings.keys())[0])
    utils.saveSettings(CUR_DIR / "SPACEtomo_settings.ini", settings, list(settings.keys())[0])

    # Load models
    WG_model = WGModel(MAP_DIR, EXTERNAL_RUN)
    MM_model = space_ext.MMModel()

    # Get imaging parameters and save
    log("Getting imaging parameters...")
    microscope.changeImagingState(imaging_params.MM_image_state, low_dose_expected=True)
    microscope.changeC2Aperture(config.c2_apertures[2])
    microscope.changeLowDoseArea("V")
    imaging_params.getViewParams(microscope)
    imaging_params.getSearchParams(microscope)
    microscope.changeLowDoseArea("R")
    imaging_params.getRecParams(microscope)
    imaging_params.getFocusParams(microscope)

    MM_model.setDimensions(imaging_params)

    # Set up common parameters
    tgt_params = space_ext.TgtParams(settings["target_list"], settings["avoid_list"], MM_model, settings["sparse_targets"], settings["target_edge"], settings["penalty_weight"], settings["target_score_threshold"], settings["max_tilt"], imaging_params, settings["extra_tracking"], config.max_iterations)

    # Export params for postAction script
    imaging_params.export(CUR_DIR)
    tgt_params.export(CUR_DIR)
    imaging_params.export(MAP_DIR)
    tgt_params.export(MAP_DIR)

    first_map = True
    map_names = []
    for map_id in map_list:
        map_file = nav.items[map_id].map_file
        # Check if map file contains more than one section
        with mrcfile.open(map_file, permissive=True) as mrc:
            num_sections = mrc.header.nz
        # Add section number to map name if necessary
        if int(nav.items[map_id].entries["MapSection"][0]) == 0 and num_sections == 1:
            map_name = map_file.stem
        else:
            map_name = map_file.stem + f"_{nav.items[map_id].entries['MapSection'][0].zfill(2)}"
        map_names.append(map_name)

        # Save montage as rescaled input image
        log(f"Saving map image for {map_name}...")
        map_img = Buffer(nav_id=map_id)
        map_img.findGrid()
        save_future = map_img.saveImg(MAP_DIR / (map_name + ".png"), target_pix_size=MM_model.pix_size if settings["rescale_map"] else None)
        if save_future is not None:
            save_future.result() # Temporary, because montage collection is not threaded, TODO: multiprocessing instead
            save_future = None

        if settings["manual_selection"] or config.DUMMY:
            MM_model.saveEmptySeg(MAP_DIR, map_name)
        else:
            if not EXTERNAL_RUN:
                # Queue SPACEtomo run for new montage
                space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params)

        if first_map:
            # Open tgt selection GUI if manual_selection is selected and maps are not external (and only after first map finished collection)
            if settings["manual_selection"] and not EXTERNAL_RUN:
                utils.guiProcess("targets", MAP_DIR, auto_close=True)

            first_map = False

    ##############################################
    ############ STEP 2: Target setup ############
    ##############################################

    total_target_num = 0
    for map_id, map_name in zip(map_list, map_names):
        map_file = nav.items[map_id].map_file
        map_img = Buffer(nav_id=map_id)

        # Make sure target selection is ready for setup
        nav.pull()
        utils.waitForFile(MAP_DIR / (map_name + "_points*.json"), 
                        "Waiting for first lamella to be processed before setting up targets...", 
                        function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else lambda: utils.monitorExternal(MAP_DIR)
                        )
        # Make sure targets were inspected
        if settings["MM_wait_for_inspection"]:
            # Open GUI if not already opened for manual selection
            if not settings["manual_selection"] and not EXTERNAL_RUN:
                utils.guiProcess("targets", MAP_DIR, auto_close=True)
            # Wait for inspection
            utils.waitForFile(MAP_DIR / (map_name + "_inspected.txt"), 
                            "Waiting for targets to be inspected by operator...", 
                            function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else lambda: utils.monitorExternal(MAP_DIR)
                            )

        log(f"\nSetting up targets for {map_name}...")

        # Save targets for PACEtomo
        if list(CUR_DIR.glob(map_name + "*_tgts.txt")):
            log(f"WARNING: Targets file for {map_name} already exists! Skipping target setup!")
        else:
            target_num = 0

            # Go over all target areas on lamella
            point_files = sorted(MAP_DIR.glob(map_name + "_points*.json"))
            for ta, point_file in enumerate(point_files):
                if len(point_files) > 1:
                    area_name = map_name + f"_A{ta + 1}"
                else:
                    area_name = map_name

                # Instantiate target area
                target_area = PACEArea(point_file, MM_model, imaging_params, map_img)
                if len(target_area) == 0: 
                    log(f"WARNING: No targets selected on {area_name}. Skipping...")
                    continue
                
                target_area.checkISLimit()

                # Set up navigator and create virtual maps
                target_area.prepareForAcquisition(CUR_DIR / (area_name + ".mrc"), nav)

                # Save target file
                targets, geo_points, tgt_settings = target_area.getTargetsInfo(area_name)
                utils.writeTargets(CUR_DIR / (area_name + "_tgts.txt"), targets, geo_points, settings=tgt_settings)
                target_num += len(targets)

            log(f"NOTE: Targets selected on {map_name}: {target_num}")
            total_target_num += target_num

        nav.items[map_id].changeAcquire(0)

    log(f"\nTotal targets selected: {total_target_num}", style=1)
    usem.exitSPACEtomo()

if __name__ == "__main__":
    main()