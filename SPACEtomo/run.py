#!Python
# ===================================================================
#ScriptName SPACEtomo
# Purpose:      Finds lamellae in grid map, collects medium mag montages and finds targets using deep learning models.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/31
# Revision:     v1.2
# Last Change:  2025/02/06: outsourced MM map acquisition to mma.py
#               2025/01/10: outsourced IM align to ima.py
#               2024/12/23: added alignTo IM image during grid realign
#               2024/11/20: switched to nav based item shift, added wait for threaded saving (temporary)
#               2024/10/21: Added retention of high confidence lamellae
#               2024/09/04: converted settings to configparser
#               2024/08/19: finished refactor (except ext.py)
#               2024/08/16: completed level 1 refactor, completed level 2 refactor, started refactor of target setup
#               2024/08/13: started switching to Microscope and Navigator classes, also started switch to Pathlib
#               2024/04/12: fixed grid_list conversion
#               2024/04/09: fixed GUI opening multiple times, rename preexisting run file, moved import functions past SetDirectory
#               2024/04/02: moved Rec param saving before MM montages to allow earlier target picking, added setting scripts for Acquire at Items, added status line for multi grid
#               2024/03/26: added check for GPU
#               2024/03/25: fixes after Krios 3 test, added opening of SPACEtomo_tgt, added manual_selection by creating empty segmentation, added more text outputs
#               2024/03/18: added multi-grid support, added waiting for manual inspection (x_inspected.txt file)
#               2024/03/08: outsourced target selection to updateQueue, introduced TgtParams
#               2024/03/07: adjusted for external target selection
#               2024/02/14: adjustments to run lamella detection externally
#               2024/02/13: split collectWGMap from lamella detection (1.1)
#               2023/12/12: added version check
#               2023/11/06: used IM lamella detection as confirmation
#               2023/10/31: fixes after Krios test
#               2023/10/27: adjusted plotting option
#               2023/10/25: added open nav file
#               2023/10/23: added extra tracking
#               2023/10/16: fixes after Krios test
#               2023/10/12: fixes after Krios test
#               2023/10/05: added option for inference on external machine
#               2023/10/04: outsourced functions
#               2023/10/03: added auto identification of geo points, added geo points to tgts fie
#               2023/10/02: fixes after Krios test, added intermediate mag lamella detection for offset refinement
#               2023/09/27: clean up, added manual intervention, montage fixes
#               2023/09/26: fixes after Krios test, added queue for SPACEruns
#               2023/09/22: switched to nnUNet model and adjusted script accordingly including target selection from segmentation
#               2023/07/13: added writing of SPACE_run file to be read by post script
#               2023/07/10: binned heat maps, fixed target specimen shifts, added score of targets based on camera dimensions, cleanup
#               2023/07/07: added target setup, tested and fixed on Krios
#               2023/07/05: fixes and cleanup, added per lamella montage size
#               2023/07/04: first Krios test
#               2023/06/19: added montage loop
#               2023/05/31: first test in SerialEM
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
    
import time
import numpy as np
from pathlib import Path
import SPACEtomo.modules.ext as space_ext
from SPACEtomo.modules.utils import log
from SPACEtomo.modules.scope import Microscope, ImagingParams
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.mod_wg import WGModel, Boxes
from SPACEtomo.modules.ima import IMAlignment
from SPACEtomo.modules.mma import MMMAcquisition
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
SES_DIR = usem.getSessionDir()

###

def main():
    # Read settings file
    settings = utils.loadSettings(SES_DIR / "SPACEtomo_settings.ini")
    if not settings["lamella"]:
        settings["WG_wait_for_inspection"] = True
        settings["manual_selection"] = True
    if settings["manual_selection"] and not settings["MM_wait_for_inspection"]: # Force inspection in case of manual selection
        settings["MM_wait_for_inspection"] = True

    # Check model
    if config.NO_MM_MODEL:
        log(f"ERROR: Lamella segmentation model could not be found! Please import it using the SPACEmodel command!")
        return
    
    # Check multi-grid status
    remaining_grid_list = usem.getGridList(settings["grid_list"], settings["automation_level"])

    # Show warning if grid will be changed after target setup
    if settings["automation_level"] == 4 and len(remaining_grid_list) > 1:
        log(f"WARNING: You selected multiple grids in combination with target selection. Reloading grids might cause problems with realignment to target areas.")
        usem.confirmationBox(f"WARNING: You selected {len(remaining_grid_list)} grids in combination with target selection. Reloading grids might cause problems with realignment to target areas!")

    # Initialize microscope control
    microscope = Microscope()
    microscope.checkAutoloader()
    imaging_params = ImagingParams(settings["WG_image_state"], settings["IM_image_state"], settings["MM_image_state"], file_dir=SES_DIR)
    imaging_params.IS_limit = microscope.is_limit

    # Check if imaging states are valid
    #log(f"Checking imaging states...")
    #if not microscope.changeImagingState(imaging_params.MM_image_state, low_dose_expected=True):
    #    log(f"ERROR: The provided medium mag imaging state [MM_image_state] is not in Low Dose mode!")
    #    return
    if settings["automation_level"] >= 2:
        usem.checkImagingStates(states=[settings["WG_image_state"], settings["IM_image_state"]] + settings["MM_image_state"], low_dose_expected=[False, False] + [True] * len(settings["MM_image_state"]))
    else:
        usem.checkImagingStates(states=[settings["WG_image_state"]], low_dose_expected=[False])

    # Run on every grid
    for grid_slot in remaining_grid_list:

        # Get grid name
        grid_name = microscope.autoloader.get(grid_slot)
        if grid_name is None:
            log(f"ERROR: The grid slot {grid_slot} is empty! If this grid is currently on the stage, please give it a name in the Autoloader panel!")
            continue

        # Setup dirs
        CUR_DIR = SES_DIR / grid_name
        usem.setDirectory(CUR_DIR)
        usem.openNewLog(grid_name)
        MAP_DIR, EXTERNAL_RUN = usem.prepareEnvironment(CUR_DIR, settings["external_map_dir"])

        # Save settings for SPACEtomo_postAction.py
        utils.saveSettings(MAP_DIR / "SPACEtomo_settings.ini", settings)
        utils.saveSettings(CUR_DIR / "SPACEtomo_settings.ini", settings)

    ########################################
    ############ STEP 1: WG map ############
    ########################################

        if settings["automation_level"] >= 1:

            # Load model
            WG_model = WGModel(MAP_DIR, EXTERNAL_RUN)  # load a custom model

            # Load grid
            nav = Navigator(CUR_DIR / (grid_name + ".nav"))
            Buffer.nav = nav            # Provide reference for all buffers

            if config.DUMMY:            # Provide additional references to dummy classes
                microscope.cur_dir = CUR_DIR
                microscope.map_dir = MAP_DIR
                microscope.nav = nav
                microscope.imaging_params = imaging_params
                Buffer.microscope = microscope
                Buffer.imaging_params = imaging_params
                if imaging_params.rec_ta_rotation is None:
                    log(f"ERROR: Cannot run in DUMMY mode without mic_params.json!")
                    return

            if microscope.loaded_grid == grid_slot:
                already_loaded = True
            else:
                already_loaded = False
                microscope.loadGrid(grid_slot)
            if config.aperture_control:
                microscope.changeObjAperture(0)
            microscope.openValves()

            # Check for WG montage
            wg_nav_id = nav.getIDfromNote(grid_name + ".mrc")
            if wg_nav_id is None:
                # Make WG montage
                wg_nav_id = microscope.collectFullMontage(imaging_params, WG_model, overlap=config.WG_montage_overlap)
                nav.pull()
            else:
                if not already_loaded:
                    # Run realign routine on whole grid map
                    microscope.changeImagingState(settings["WG_image_state"], low_dose_expected=False)
                    microscope.changeC2Aperture(config.c2_apertures[0]) # Insert C2 aperture for WG state
                    microscope.realignGrid(nav, wg_nav_id)
                    # Use image alignment on IM image
                    microscope.changeImagingState(settings["IM_image_state"], low_dose_expected=False)
                    microscope.changeC2Aperture(config.c2_apertures[1]) # Insert C2 aperture for IM state
                    im_nav_ids = nav.searchByEntry("label", "PP", partial=True)     # Preliminary position/polygon (PP)
                    for nav_id in im_nav_ids:
                        log (f"Realigning IM image {nav.items[nav_id].label}...")
                        microscope.moveStage(nav.items[nav_id].stage)
                        im_buf = Buffer("O", nav_id=nav_id)
                        microscope.record()
                        new_im_buf = Buffer("A")
                        _, ss_shift = new_im_buf.alignTo(im_buf, avoid_image_shift=True)
                        stage_shift = microscope.getMatrices()["ss2s"] @ ss_shift
                        break # TODO: possibly add check for success
                    nav.shiftItems(-stage_shift, skip_item_ids=[wg_nav_id])

            # Save WG map
            wg_map = Buffer(nav_id=wg_nav_id)
            if not settings["lamella"]:
                wg_map.findGrid(spacing_nm=50000) # 300 mesh should have ~50 micron spacing
            imaging_params.WG_pix_size = wg_map.pix_size
            save_future = wg_map.saveImg(MAP_DIR / (grid_name + "_wg.png"), WG_model.pix_size)

            # Find lamellae on grid
            box_file = MAP_DIR / (grid_name + "_wg_boxes.json")
            if settings["lamella"]:
                roi_boxes = WG_model.findLamellae(grid_name, save_future=save_future, label_prefix="PP", exclude_cats=settings["exclude_lamella_classes"])

            # Open ROI selection GUI if wait_for_inspection is selected and maps are not external and not already inspected
            if settings["WG_wait_for_inspection"] and not (MAP_DIR / (grid_name + "_wg_inspected.txt")).exists():
                utils.guiProcess("grid", MAP_DIR / (grid_name + "_wg.png"), auto_close=True)

            # Wait for ROI boxes to be inspected
            if settings["WG_wait_for_inspection"]:
                log("Waiting for WG map to be inspected by operator...")
                inspect_file = MAP_DIR / (grid_name + "_wg_inspected.txt")
                utils.waitForFile(inspect_file, "Still waiting for inspection by operator...", msg_interval=180)
                # Load boxes again after inspection
                roi_boxes = Boxes(box_file, label_prefix="PP", exclude_cats=settings["exclude_lamella_classes"])

            # Check for previously determined ROI positions
            roi_PP_ids = nav.searchByEntry("label", "PP", partial=True)     # Preliminary position (PP)
            roi_FP_ids = nav.searchByEntry("label", "FP", partial=True)     # Final position (FP)
            log(f"DEBUG: PP entries: {len(roi_PP_ids)}")
            log(f"DEBUG: FP entries: {len(roi_FP_ids)}")

            # Add boxes to nav via map buffer (if no polygons in nav yet)
            if len(roi_PP_ids) == 0:
                wg_map.addNavBoxes(roi_boxes, padding_factor=config.MM_padding_factor if settings["lamella"] else 1.0)
                nav.pull()

                # Create virtual maps for IM alignment
                for b, box in enumerate(roi_boxes.boxes):
                    # Make virtual map from montage in buffer
                    virt_map_file = CUR_DIR / (grid_name + f"_IM{b + 1}_ref.mrc")
                    virt_map = np.flip(wg_map.getCropImage(np.flip(box.center) * roi_boxes.pix_size / wg_map.pix_size, np.flip(microscope.camera_dims)), axis=0)       # crop image and flip y-axis
                    log(f"DEBUG: Virtual map dimenstions: {virt_map.shape}")
                    utils.writeMrc(virt_map_file, virt_map, wg_map.pix_size)

            if settings["automation_level"] >= 2:
                # Check if intermediate mag step was done (if there are as many images as preliminary points and final points)
                log(f"DEBUG: Test if IM has to be done:\nIM pngs: {len(list(MAP_DIR.glob(grid_name + '_IM*_wg.png')))}\nPP entries: {len(roi_PP_ids)}\nFP entries: {len(roi_FP_ids)}\n\n")
                alignment = IMAlignment(CUR_DIR, MAP_DIR, microscope, nav, imaging_params, label_prefix="PP", final_prefix="FP", WG_model=WG_model if settings["lamella"] else None)
                alignment.align(settings)
                alignment.saveAlignment()

                roi_nav_ids = nav.searchByEntry("label", "FP", partial=True)     # Final position (FP)

                if not roi_nav_ids:
                    log(f"WARNING: No regions of interest found on grid [{grid_name}]. If regions are visually identifiable, please check your settings or continue manually.")
                    continue    # to next grid

            log("##### Completed WG map acquisition and MM map setup! [Level 1] #####")

    #########################################
    ############ STEP 2: MM maps ############
    #########################################

        if settings["automation_level"] >= 2:

            # Load model
            MM_model = space_ext.MMModel()

            # Insert objective aperture
            if config.aperture_control:
                microscope.changeObjAperture(config.objective_aperture)

            # Enter Low Dose Mode
            microscope.changeImagingState(settings["MM_image_state"], low_dose_expected=True)
            microscope.changeC2Aperture(config.c2_apertures[2]) # Insert C2 aperture for MM state
            if not microscope.low_dose:
                log("WARNING: Medium mag montage imaging state is NOT in Low Dose! This will cause issues with target selection.")

            # Get imaging parameters and save
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

            # Remove preexisting SPACE_runs
            if not EXTERNAL_RUN and (MAP_DIR / "SPACE_runs.json").exists():
                (MAP_DIR / "SPACE_runs.json").replace(MAP_DIR / "SPACE_runs.json~")

            # Collect MM montages at all ROI positions
            MM_map_ids = []
            for i, nav_id in enumerate(roi_nav_ids):
                # Make sure no MM map is waiting in queue before starting next montage. TODO: better queue management
                if settings["automation_level"] >= 3 and not EXTERNAL_RUN and i > 0:
                    space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params)

                # Get MM map name
                map_name = f"{grid_name}_L{str(i + 1).zfill(2)}"
                map_file = CUR_DIR / (map_name + ".mrc")

                # Collect medium mag montage
                mm_acquisition = MMMAcquisition(MAP_DIR, map_file, nav_id, microscope, nav)
                mm_acquisition.realign(roi_nav_ids[i:])
                map_id = mm_acquisition.collectMap()
                save_future = mm_acquisition.saveMap(map_id, MM_model.pix_size, find_grid=not settings["lamella"], save_future=save_future)
                MM_map_ids.append(map_id)

                # Trigger segmentation
                if settings["automation_level"] >= 3:
                    if settings["manual_selection"]:
                        MM_model.saveEmptySeg(MAP_DIR, map_name)
                    else:
                        if not EXTERNAL_RUN:
                            # Queue SPACEtomo run for new montage
                            space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params)

                # Open tgt selection GUI if manual_selection is selected and maps are not external (and only after first map finished collection)
                if settings["automation_level"] >= 4 and i == 0 and settings["manual_selection"]:
                    utils.guiProcess("targets", MAP_DIR, auto_close=True)
            log("##### Completed collection of MM maps! [Level 2] #####")

    ##############################################
    ############ STEP 3: Target setup ############
    ##############################################

        total_targets = 0
        if settings["automation_level"] >= 4:
            # Make sure at least the first target selection is ready for setup
            nav.pull()
            map_file = Path(nav.items[MM_map_ids[0]].note)
            utils.waitForFile(MAP_DIR / (map_file.stem + "_points*.json"), 
                            "Waiting for first MM map to be processed before setting up targets...", 
                            function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else lambda: utils.monitorExternal(MAP_DIR)
                            )
            # Make sure targets were inspected
            if settings["MM_wait_for_inspection"]:
                # Open GUI if not already opened for manual selection
                if not settings["manual_selection"]:
                    utils.guiProcess("targets", MAP_DIR, auto_close=True)
                # Wait for inspection
                utils.waitForFile(MAP_DIR / (map_file.stem + "_inspected.txt"), 
                                "Waiting for targets to be inspected by operator...", 
                                function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else lambda: utils.monitorExternal(MAP_DIR)
                                )

            # Loop over all MM maps
            for m, map_id in enumerate(MM_map_ids):
                map_name = Path(nav.items[map_id].note).stem
                map_file = MAP_DIR / (map_name + ".png")
                map_img = Buffer(nav_id=map_id)

                if not EXTERNAL_RUN:
                    space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params)

                # Check if point file exists
                if not list(MAP_DIR.glob(map_name + "_points*.json")) or (settings["MM_wait_for_inspection"] and not (MAP_DIR / (map_name + "_inspected.txt")).exists()):

                    # Check if there are already items to be acquired
                    nav.pull()
                    num_acq = len(nav.searchByEntry("Acquire", ["1"]))

                    # If acquisition cannot start yet or automation is 4, wait for point files
                    if settings["automation_level"] < 5 or (settings["automation_level"] >= 5 and num_acq == 0):
                        # Wait for points files
                        utils.waitForFile(MAP_DIR / (map_file.stem + "_points*.json"), 
                                        "Waiting for first MM map to be processed before setting up targets...", 
                                        function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else None
                                        )
                        # Make sure targets were inspected
                        if settings["MM_wait_for_inspection"]:
                            utils.waitForFile(MAP_DIR / (map_file.stem + "_inspected.txt"), 
                                            "Waiting for targets to be inspected by operator...", 
                                            function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else None
                                            )
                    else:
                        log("Target setup for remaining MM maps postponed until predictions are available.")
                        break           # rely on SPACEtomo_postAction script to setup more targets between PACEtomo runs


                log(f"\nSetting up targets for {map_name}...")

                # Save targets for PACEtomo
                if list(CUR_DIR.glob(map_name + "*_tgts.txt")):
                    log(f"WARNING: Targets file for {map_name} already exists! Skipping target setup!")
                else:
                    target_num = 0

                    # Go over all target areas on MM map
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
                        target_area.prepareForAcquisition(CUR_DIR / (area_name + ".mrc"), nav, grid_name)

                        # Save target file
                        targets, geo_points, tgt_settings = target_area.getTargetsInfo(area_name)
                        utils.writeTargets(CUR_DIR / (area_name + "_tgts.txt"), targets, geo_points, settings=tgt_settings)
                        target_num += len(targets)

                    log(f"NOTE: Targets selected on {map_name}: {target_num}")
                    total_targets += target_num

            log(f"NOTE: Total targets selected: {total_targets}")
            log("##### Completed target selection step! [Level 4] #####")

    # Check if any maps are marked for reacquisition
    reacq_files = list(MAP_DIR.glob("*_reacquire.json"))
    if reacq_files:
        log(f"WARNING: The {len(reacq_files)} maps are marked for reacquisition. Please rerun the SPACEtomo script!")
        usem.exitSPACEtomo()

    #############################################
    ############ STEP 4: Acquisition ############
    #############################################

    # Start acquire at items
    if settings["automation_level"] >= 5:
        log("The SPACEtomo setup script completed!")
        log(time.strftime("%d.%m.%Y %H:%M:%S", time.localtime()))

        usem.setupAcquisition(*settings["script_numbers"])
    else:
        usem.exitSPACEtomo()

if __name__ == "__main__":
    main()