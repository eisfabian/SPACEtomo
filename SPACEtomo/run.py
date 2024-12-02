#!Python
# ===================================================================
#ScriptName SPACEtomo
# Purpose:      Finds lamellae in grid map, collects lamella montages and finds targets using deep learning models.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/31
# Revision:     v1.2
# Last Change:  2024/11/20: switched to nav bases item shift, added wait for threaded saving (temporary)
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
from copy import deepcopy
from pathlib import Path
import SPACEtomo.modules.ext as space_ext
from SPACEtomo.modules.utils import log
from SPACEtomo.modules.scope import Microscope, ImagingParams
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.mod_wg import WGModel, Boxes
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
    imaging_params = ImagingParams(settings["WG_image_state"], settings["IM_mag_index"], settings["MM_image_state"], file_dir=SES_DIR)
    microscope = Microscope(imaging_params)

    # Check if imaging states are valid
    log(f"Checking imaging states...")
    if not microscope.changeImagingState(imaging_params.MM_image_state, low_dose_expected=True):
        log(f"ERROR: The provided medium mag imaging state [MM_image_state] is not in Low Dose mode!")
        return

    # Run on every grid
    for grid_slot in remaining_grid_list:

        # Get grid name
        grid_name = microscope.autoloader[grid_slot]

        # Setup dir
        CUR_DIR, MAP_DIR, EXTERNAL_RUN = usem.prepareEnvironment(SES_DIR, grid_name, settings["external_map_dir"])

        # Save settings for SPACEtomo_postAction.py
        utils.saveSettings(MAP_DIR / "SPACEtomo_settings.ini", settings)
        utils.saveSettings(CUR_DIR / "SPACEtomo_settings.ini", settings)

    ########################################
    ############ STEP 1: WG map ############
    ########################################

        if settings["automation_level"] >= 1:

            # Load model
            WG_model = WGModel(EXTERNAL_RUN)  # load a custom model

            # Load grid
            nav = Navigator(CUR_DIR / (grid_name + ".nav"))
            Buffer.nav = nav            # Provide reference for all buffers

            if config.DUMMY:            # Provide additional references to dummy classes
                microscope.cur_dir = CUR_DIR
                microscope.map_dir = MAP_DIR
                microscope.nav = nav
                Buffer.imaging_params = imaging_params
                if imaging_params.rec_ta_rotation is None:
                    log(f"ERROR: Cannot run in DUMMY mode without mic_params.json!")
                    return

            microscope.loadGrid(grid_slot)
            if config.aperture_control:
                microscope.changeObjAperture(0)
            microscope.openValves()

            # Check for WG montage
            wg_nav_id = nav.getIDfromNote(grid_name + ".mrc")
            if wg_nav_id is None:
                # Make WG montage
                wg_nav_id = microscope.collectFullMontage(WG_model, overlap=config.WG_montage_overlap)
                nav.pull()

            # Save WG map
            wg_map = Buffer(nav_id=wg_nav_id)
            save_future = wg_map.saveImg(MAP_DIR / (grid_name + "_wg.png"), WG_model.pix_size)

            # Find lamellae on grid
            box_file = MAP_DIR / (grid_name + "_wg_boxes.json")
            if not box_file.exists():
                if not EXTERNAL_RUN:
                    if save_future is not None:
                        save_future.result()
                        save_future = None
                    WG_model.findLamellae(MAP_DIR, grid_name, threshold=config.WG_detection_threshold, device=utils.DEVICE)
                else:
                    # Wait for boxes file to be written
                    log("Waiting for external lamella detection...")
                    utils.waitForFile(box_file, "WARNING: Still waiting for lamella detection. Check if SPACEtomo_monitor.py is running!")
            else:
                log(f"WARNING: Previously detected lamellae were found. Skipping lamella detection!")

            # Open lamella selection GUI if wait_for_inspection is selected and maps are not external and not already inspected
            if settings["WG_wait_for_inspection"] and not EXTERNAL_RUN and not (MAP_DIR / (grid_name + "_wg_inspected.txt")).exists():
                utils.guiProcess("lamella", MAP_DIR / (grid_name + "_wg.png"), auto_close=True)

            # Wait for lamella boxes to be inspected
            if settings["WG_wait_for_inspection"]:
                log("Waiting for lamella detection to be inspected by operator...")
                inspect_file = MAP_DIR / (grid_name + "_wg_inspected.txt")
                utils.waitForFile(inspect_file, "Still waiting for inspection by operator...", msg_interval=180)

            # Load boxes from file and remove undesired classes
            lamella_boxes = Boxes(box_file, label_prefix="PL", exclude_cats=settings["exclude_lamella_classes"])

            # Check for previously determined lamella positions
            lamella_PL_ids = nav.searchByEntry("label", "PL", partial=True)     # Preliminary lamellae (PL)
            lamella_FL_ids = nav.searchByEntry("label", "FL", partial=True)     # Final lamellae (FL)
            log(f"DEBUG: PL entries: {len(lamella_PL_ids)}")
            log(f"DEBUG: FL entries: {len(lamella_FL_ids)}")

            # Add boxes to nav via map buffer (if no lamella in nav yet)
            if len(lamella_PL_ids) == 0:
                wg_map.addNavBoxes(lamella_boxes, padding_factor=config.MM_padding_factor)
                nav.pull()

            # Check if intermediate mag step was done (if there are as many images as preliminary lamella points and final lamella points)
            log(f"DEBUG: Test if IM has to be done:\nIM pngs: {len(list(MAP_DIR.glob(grid_name + '_IM*_wg.png')))}\nPL entries: {len(lamella_PL_ids)}\nFL entries: {len(lamella_FL_ids)}\n\n")
            if not len(list(MAP_DIR.glob(grid_name + "_IM*_wg.png"))) == len(lamella_PL_ids) or not lamella_FL_ids:
                log("DEBUG: Collecting IM images...\n")
                microscope.changeImagingState(settings["WG_image_state"], low_dose_expected=False)
                microscope.setMag(settings["IM_mag_index"])
                lamella_nav_ids = nav.searchByEntry("label", "PL", partial=True)     # Preliminary lamellae (PL)
                shifted = False
                eucentricity = False
                for n, nav_id in enumerate(lamella_nav_ids):

                    # If lamella nav item was already converted from polygon to map, skip it
                    if nav.items[nav_id].entries["Type"] == ["2"]:
                        log(f"WARNING: Lamella {n + 1} was already imaged at intermediate mag and will be skipped.")
                        continue

                    log(f"Moving to lamella {n + 1}...")
                    microscope.moveStage(nav.items[nav_id].stage)

                    # Go to eucentricity roughly at IM mag
                    if not eucentricity:
                        microscope.eucentricity()
                        eucentricity = True
                        # Update z for all lamella items
                        for nid in lamella_nav_ids:
                            nav.items[nid].changeZ(microscope.stage[2])
                        nav.push()

                    # Collect IM image and save as map
                    log(f"Collecting intermediate mag image...")
                    im_file = CUR_DIR / (grid_name + f"_IM{n + 1}.mrc")
                    microscope.record(save=im_file)
                    im_img = Buffer(buffer="A")
                    im_nav_id = nav.newMap(buffer=im_img, img_file=im_file, label=f"PL{n + 1}", note=nav.items[nav_id].note, coords=nav.items[nav_id].stage) # Coords are only needed for DUMMY and ignored for normal map

                    # Update IM pix_size
                    if not imaging_params.IM_pix_size:
                        imaging_params.IM_pix_size = im_img.pix_size

                    # Save IM image
                    if save_future is not None:
                        save_future.result()
                        save_future = None
                    save_future = im_img.saveImg(MAP_DIR / (grid_name + f"_IM{n + 1}_wg.png"), WG_model.pix_size)

                    # Retain preliminary polygon nav item in case no lamella is detected at IM
                    pre_item = deepcopy(nav.items[nav_id])

                    # Overwrite polygon with map
                    nav.replaceItem(nav_id, im_nav_id)
                    im_img.nav_id = nav_id # Update nav id of buffer

                    # Find lamellae on image
                    log(f"Finding lamella...")
                    box_file = MAP_DIR / (grid_name + f"_IM{n + 1}_wg_boxes.json")
                    if not EXTERNAL_RUN:
                        if save_future is not None:
                            save_future.result()
                            save_future = None
                        WG_model.findLamellae(MAP_DIR, grid_name + f"_IM{n + 1}", threshold=config.WG_detection_threshold, device=utils.DEVICE)
                    else:
                        # Wait for boxes file to be written
                        log("Waiting for external lamella detection...")
                        utils.waitForFile(box_file, "WARNING: Still waiting for lamella detection. Check if SPACEmonitor is running!")

                    im_boxes = Boxes(box_file)

                    if im_boxes:
                        if len(im_boxes) > 1:
                            log(f"WARNING: Found more than one lamella. Using lamella closest to image center.")

                        # Only keep box closest to center
                        im_boxes.sortBy("center_dist")
                        im_boxes.boxes = im_boxes.boxes[:1]

                        log(f"NOTE: Detected lamella of class: {config.WG_model_categories[im_boxes.boxes[0].cat]} [{round(im_boxes.boxes[0].prob * 100)} %]")

                        # Check if too close to previously detected lamella
                        lamella_FL_ids = nav.searchByEntry("label", "FL", partial=True)     # Final lamellae (FL)
                        log(f"DEBUG: FL ids: {lamella_FL_ids}")
                        if nav.searchByCoords(im_img.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_img.pix_size)), margin=settings["WG_distance_threshold"], subset=lamella_FL_ids):
                            log(f"DEBUG: Distance: {im_img.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_img.pix_size))}")
                            log(f"DEBUG: Search result: {nav.searchByCoords(im_img.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_img.pix_size)), margin=settings['WG_distance_threshold'], subset=lamella_FL_ids)}")
                            log(f"WARNING: Lamella seems to be among already detected lamellae and will be skipped.")
                            continue

                        # Add box to nav via map buffer
                        im_img.addNavBoxes(im_boxes, labels=[f"FL{n + 1}"], padding_factor=config.MM_padding_factor)                 # Add final lamella
                        nav.pull()

                        # Shift items by vector between new box center stage coords and nav item stage coords
                        if not shifted:
                            #nav.shiftItems(nav.items[nav_id].getVector(im_img.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_img.pix_size))))
                            shift = nav.items[nav_id].getVector(im_img.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_img.pix_size)))

                            for remaining_nav_id in lamella_nav_ids[n + 1:]:
                                log(f"DEBUG: Shifted item {remaining_nav_id} by {shift}")
                                nav.items[remaining_nav_id].changeStage(shift, relative=True)

                            shifted = True
                            nav.push()

                    else:
                        # Retain preliminary lamella if initial confidence was high (includes manually selected lamella)
                        if float(pre_item.entries["UserValue1"][0]) >= 0.9:
                            nav.items.append(pre_item) # Add initial polygon back to nav
                            pre_item.nav_index = len(nav) # Adjust nav_index
                            nav.push()
                            # Only rename label and note after nav was pushed as it will update SerialEM directly
                            pre_item.changeLabel(f"FL{n + 1}")
                            pre_item.changeNote(f"FL{n + 1}: " + pre_item.note.split(":")[1])
                            log(f"WARNING: No lamella detected! Initially detected lamella was retained due to high confidence [{round(float(pre_item.entries['UserValue1'][0]) * 100)} %].")
                        else:
                            log(f"WARNING: No lamella detected. If you want to add lamellae manually, please select wait_for_inspection in the settings.")

            lamella_nav_ids = nav.searchByEntry("label", "FL", partial=True)     # Final lamellae (FL)

            if not lamella_nav_ids:
                log(f"WARNING: No lamellae found on grid [{grid_name}]. If lamellae are visually identifiable, please check your settings or continue manually.")
                continue    # to next grid

            log("##### Completed lamella detection step! [Level 1] #####")

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
            if not microscope.low_dose:
                log("WARNING: Lamella montage imaging state is NOT in Low Dose! This will cause issues with target selection.")

            # Get imaging parameters and save
            microscope.changeLowDoseArea("V")
            imaging_params.getViewParams()
            microscope.changeLowDoseArea("R")
            imaging_params.getRecParams()

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

            # Collect MM montages at all lamella positions
            MM_map_ids = []
            for i, nav_id in enumerate(lamella_nav_ids):
                # Make sure no MM map is waiting in queue before starting next montage. TODO: better queue management
                if settings["automation_level"] >= 3 and not EXTERNAL_RUN and i > 0:
                    space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params)

                # Get lamella map name
                map_name = f"{grid_name}_L{str(i + 1).zfill(2)}"
                map_file = CUR_DIR / (map_name + ".mrc")

                # Collect montage
                log(f"Collecting map for lamella {map_name}...")

                # Check if montage already exists in nav
                map_id = nav.getIDfromNote(map_file.name, warn=False)

                # Check if file also exists
                if not map_file.exists():
                    # Check if it was marked for reacquisition by renaming and update nav note accordingly
                    if old_map_files := list(map_file.parent.glob(map_file.stem + "_old*.mrc")):
                        nav.items[map_id].changeNote(old_map_files[-1].name)
                        map_id = None
                        # Reset inspection status
                        if (MAP_DIR / (map_file.stem + "_inspected.txt")).exists():
                            (MAP_DIR / (map_file.stem + "_inspected.txt")).unlink()

                # Collect map if no nav item found
                if map_id is None:
                    # Delete potentially aborted montage file
                    if map_file.exists():
                        map_file.unlink()
                    map_id = microscope.collectPolygonMontage(nav_id, map_file, config.MM_montage_overlap)
                    nav.pull()
                    nav.items[map_id].changeLabel(f"L{str(i + 1).zfill(2)}")
                MM_map_ids.append(map_id)

                # Save montage as rescaled input image
                log(f"Saving map for lamella {map_name}...")
                map_img = Buffer(nav_id=map_id)
                save_future = map_img.saveImg(MAP_DIR / (map_file.stem + ".png"), MM_model.pix_size)
                if save_future is not None:
                    save_future.result() # Temporary, because montage collection is not threaded, TODO: multiprocessing instead
                    save_future = None

                if settings["automation_level"] >= 3:
                    if settings["manual_selection"]:
                        MM_model.saveEmptySeg(MAP_DIR, map_name)
                    else:
                        if not EXTERNAL_RUN:
                            # Queue SPACEtomo run for new montage
                            space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params)

                # Open tgt selection GUI if manual_selection is selected and maps are not external (and only after first map finished collection)
                if settings["automation_level"] >= 4 and i == 0 and settings["manual_selection"] and not EXTERNAL_RUN:
                    utils.guiProcess("targets", MAP_DIR, auto_close=True)
            log("##### Completed collection of lamella maps! [Level 2] #####")

    ##############################################
    ############ STEP 3: Target setup ############
    ##############################################

        total_targets = 0
        if settings["automation_level"] >= 4:
            # Make sure at least the first target selection is ready for setup
            nav.pull()
            map_file = Path(nav.items[MM_map_ids[0]].note)
            utils.waitForFile(MAP_DIR / (map_file.stem + "_points*.json"), 
                            "Waiting for first lamella to be processed before setting up targets...", 
                            function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else lambda: utils.monitorExternal(MAP_DIR)
                            )
            # Make sure targets were inspected
            if settings["MM_wait_for_inspection"]:
                # Open GUI if not already opened for manual selection
                if not settings["manual_selection"] and not EXTERNAL_RUN:
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

                # Check if file exists
                if not (CUR_DIR / (map_name + ".mrc")).exists():
                    # Check if it was marked for reacquisition by renaming and update nav note accordingly
                    if old_map_files := list(CUR_DIR.glob(map_name + "_old*.mrc")):
                        nav.items[map_id].changeNote(old_map_files[-1].name)
                        log(f"WARNING: {map_name} was marked for reacquisition and target setup will be skipped.")
                        continue

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
                                        "Waiting for first lamella to be processed before setting up targets...", 
                                        function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else None
                                        )
                        # Make sure targets were inspected
                        if settings["MM_wait_for_inspection"]:
                            utils.waitForFile(MAP_DIR / (map_file.stem + "_inspected.txt"), 
                                            "Waiting for targets to be inspected by operator...", 
                                            function_call=lambda: space_ext.updateQueue(MAP_DIR, WG_model, MM_model, imaging_params, tgt_params) if not EXTERNAL_RUN else None
                                            )
                    else:
                        log("Target setup for remaining lamellae postponed until predictions are available.")
                        break           # rely on SPACEtomo_postAction script to setup more targets between PACEtomo runs


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

            log("##### Completed target selection step! [Level 4] #####")

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