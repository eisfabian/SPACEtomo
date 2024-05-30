#!Python
# ===================================================================
#ScriptName SPACEtomo
# Purpose:      Finds lamellae in grid map, collects lamella montages and finds targets using deep learning models.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/31
# Revision:     v1.1
# Last Change:  2024/04/12: fixed grid_list conversion
# ===================================================================

### TODO before running:
# - Microscope alignments
# - Run inventory and name grids in autoloader
# - Low Dose mode setup (ensure that both WG image state and MM (Low Dose) image states produce good images)
# - Set Shift offset between Record and View mag and check if "High Def IS" calibrations are necessary (Apply large IS in View and check if Record acquires at same position)
# - PACEtomo preparation and script settings (for automation level 5)
# - Setup "Acquire at Items" dialogue to run PACEtomo as main action and SPACEtomo_postAction as "Run after primary action" (for automation level 5)
# - In case of inference on an external GPU workstation, start SPACEtomo_monitor.py script and set external_map_dir accordingly
###

############ SETTINGS ############ 

automation_level        = 4                     # = 1: collect WG map and find lamellae
                                                # = 2: collect MM maps for each lamella
                                                # = 3: segment lamella MM maps
                                                # = 4: setup targets based on segmentation for each lamella
                                                # = 5: start PACEtomo batch acquisition of all targets

SPACE_DIR               = "D:\SPACEtomo"        # path to SPACEtomo files
script_numbers          = [10, 11, 13]          # Numbers of [SPACEtomo.py, SPACEtomo_postAction.py, PACEtomo.py] scripts in SerialEM script editor
external_map_dir        = ""                    # dir where maps are saved and segmentations are expected (run SPACEtomo_monitor.py externally to manage runs and queue)

# Grid settings [Level 1]

grid_list               = 2                     # number of target grid in autoloader, set to 0 if grid is already loaded
include_broken          = False                 # include lamella classified as broken
grid_default_name       = ""                    # only used when grid_list = 0 (grid is already loaded)

WG_image_state          = 1                     # imaging state name or index for whole grid montage (set up before!)
WG_montage_overlap      = 0.15                  # overlap between pieces of whole grid montage
WG_detection_threshold  = 0.1                   # confidence threshold for lamella acceptance

WG_offset_via_IM        = True                  # if offset between WG map and MM is too large, use intermediate mag to find lamella again and shift items to marker
IM_mag_index            = 10                    # intermediate mag index to use (10: 580x) 
WG_distance_threshold   = 5                     # minimum distance [microns] between lamellae to not be considered duplicate detection (only used when WG_offset_via_IM = True)

# Lamella montage settings [Level 2]

MM_image_state          = 13                    # imaging state name or index for Low Dose Mode (set up before!)
MM_montage_overlap      = 0.25                  # overlap between pieces of medium mag montage
MM_padding_factor       = 1.2                   # padding of estimated lamella size for medium mag montage
MM_mean_threshold       = 0                     # enable manual targeting if View image is too dark (mean count below this threshold)

aperture_control        = True                  # set to True if SerialEM can control apertures (https://bio3d.colorado.edu/SerialEM/hlp/html/setting_up_serialem.htm#apertures)
objective_aperture      = 70                    # diameter of objective aperture

# Targeting settings [Level 4]

wait_for_inspection     = False                 # = True: wait for inspection of targets using SPACEtomo_tgt.py [Target selection GUI]
manual_selection        = False                 # = True: creates empty segmentation, allows for manual target setup in combination with wait_for_inspection and the SPACEtomo_tgt.py script

target_list             = ["mitos"]             # list of target classes, additive, possible classes: ["background", "white", "black", "crack", "coating", "cell", "cellwall", "nucleus", "vacuole", "mitos", "lipiddroplets", "vesicles", "multivesicles", "membranes", "dynabeads", "ice", "cryst", "lamella"]
avoid_list              = ["black", "white", "ice", "crack"]    # list of classes to avoid, additive
target_score_threshold  = 0.01                  # weighted fraction of FOV containing the target, keep at 0 for any structures smaller than the FOV
sparse_targets          = True                  # use for sparse targets on the sample (e.g. mitos, vesicles)
target_edge             = False                 # use for edge targets (e.g. nuclear envelope, cell periphery)
penalty_weight          = 0.3                   # factor to downweigh overlap with classes to avoid
max_iterations          = 10                    # maximum number of iterations for target placement optimization
extra_tracking          = False                 # add an extra center target for the tracking tilt series (not working properly yet)

max_tilt                = 60                    # maximum tilt angle [degrees] of tilt series to be acquired 
save_plot               = False                 # saves all plots of all substeps (slow)

########## END SETTINGS ########## 

import serialem as sem
import os
import sys
sys.path.insert(len(sys.path), SPACE_DIR)
import glob
import subprocess
import time
from datetime import datetime
import SPACEtomo_functions as space
import SPACEtomo_functions_ext as space_ext

versionCheck = sem.IsVersionAtLeast("40100", "20240317")
if not versionCheck and sem.IsVariableDefined("warningVersion") == 0:
	runScript = sem.YesNoBox("\n".join(["WARNING: You are using a version of SerialEM that does not support all SPACEtomo features. It is recommended to update to the latest SerialEM beta version!", "", "Do you want to run SPACEtomo regardless?"]))
	if not runScript:
		sem.Exit()
	else:
		sem.SetPersistentVar("warningVersion", "")

# Check if session dir has been chosen previously
if sem.IsVariableDefined("sesDir") == 0:
    sem.UserSetDirectory("Please choose a directory for this session!")
    SES_DIR = sem.ReportDirectory()
    sem.SetPersistentVar("sesDir", SES_DIR)
else:
    SES_DIR = sem.GetVariable("sesDir")

# Check multigrid status
if not isinstance(grid_list, list):
    grid_list = [grid_list]

# Check if this is not first grid in list
if sem.IsVariableDefined("gridList") == 1:
    remaining_grid_list = sem.GetVariable("gridList").split(",")
    if len(remaining_grid_list) == 1 and remaining_grid_list[0] == "":
        sem.Echo("SPACEtomo run completed!")
        sem.ClearStatusLine(0)
        sem.ClearPersistentVars()
        sem.Exit()
    else:
        # Convert to int
        remaining_grid_list = [int(no) for no in remaining_grid_list]
else:
    remaining_grid_list = grid_list

# Save rest of list in persistent var for later calls of script (only necessary when acquisition is automated)
if automation_level >= 5:
    sem.SetPersistentVar("gridList", ",".join(str(no) for no in remaining_grid_list[1:]))
    sem.SetStatusLine(5, "Grid: " + str(len(grid_list) - len(remaining_grid_list) + 1)  + " / " + str(len(grid_list)))
    # Only run all steps on current grid, then run PACEtomo before running SPACEtomo again
    remaining_grid_list = remaining_grid_list[:1]

# Run on every grid
for grid_slot in remaining_grid_list:

    # Get name grid
    grid_name, load_grid = space.getGridName(grid_slot, grid_default_name)

    # Set subfolder for grid
    CUR_DIR = os.path.join(SES_DIR, grid_name)
    if not os.path.exists(CUR_DIR):
        os.makedirs(CUR_DIR)
    sem.SetDirectory(CUR_DIR)
    # Change constant in functions script
    space.CUR_DIR = CUR_DIR

    # Check if external processing directory is valid
    external = False
    if external_map_dir == "":
        MAP_DIR = os.path.join(CUR_DIR, "SPACE_maps")
        if not os.path.exists(MAP_DIR):
            os.makedirs(MAP_DIR)
    else:
        if os.path.exists(external_map_dir):
            MAP_DIR = external_map_dir
            space.saveSettings(os.path.join(MAP_DIR, "SPACEtargets"), globals())
            external = True
        else:
            sem.Echo("ERROR: External map directory does not exist!")
            sem.Exit()

    # Save settings for SPACEtomo_postAction.py
    space.saveSettings(os.path.join(CUR_DIR, "SPACEtargets"), globals())

    IS_limit = sem.ReportProperty("ImageShiftLimit")
    mic_params = space.MicParams(WG_image_state, IM_mag_index, MM_image_state, IS_limit)

########################################
############ STEP 1: WG map ############
########################################

    if automation_level >= 1:

        # Load model
        WG_model = space_ext.WGModel(external)  # load a custom model

        # Load grid
        space.loadGrid(grid_slot, grid_name, load_grid)

        # Retract objective aperture
        if aperture_control:
            sem.SetApertureSize(2, 0)

        # Open column valves
        sem.SetColumnOrGunValve(1)

        # Make WG montage
        space.collectWGMap(grid_name, MAP_DIR, mic_params, WG_montage_overlap, WG_model)

        # Find lamellae on grid
        if not external:
            space_ext.findLamellae(MAP_DIR, grid_name, WG_model, WG_detection_threshold, device=space.DEVICE, plot=save_plot)
        else:
            # Wait for boxes file to be written
            next_update = time.time() + 60
            sem.Echo("Waiting for external lamella detection...")
            while not os.path.exists(os.path.join(MAP_DIR, grid_name + "_boxes.json")):
                if time.time() > next_update:
                    sem.Echo("WARNING: Still waiting for lamella detection. Check if SPACEtomo_monitor.py is running!")
                    next_update = time.time() + 60
                sem.Delay(1, "s")

        # Draw nav points
        lamella_nav_ids, lamella_bboxes = space.drawNavPoints(grid_name, MAP_DIR, mic_params, WG_model, include_broken, WG_distance_threshold)

        if len(lamella_nav_ids) == 0:
            sem.Echo("WARNING: No lamellae found on grid [" + grid_name + "]. If lamellae are visually identifiable, please check your settings or continue manually.")
            continue    # to next grid

        # Find mag offset via intermediate mag
        if WG_offset_via_IM:
            lamella_nav_ids, lamella_bboxes = space.findOffset(grid_name, MAP_DIR, lamella_nav_ids, lamella_bboxes, mic_params, WG_model, distance_threshold=WG_distance_threshold, external=external)

        sem.Echo("Completed lamella detection step! [Level 1]")

#########################################
############ STEP 2: MM maps ############
#########################################

    if automation_level >= 2:

        # Load model
        MM_model = space_ext.MMModel()

        # Insert objective aperture
        if aperture_control:
            sem.SetApertureSize(2, objective_aperture)

        # Enter Low Dose Mode
        sem.GoToImagingState(str(mic_params.MM_image_state))

        # Get View mic parameters
        sem.GoToLowDoseArea("V")
        mic_params.getViewParams()

        # Get Rec mic parameters
        sem.GoToLowDoseArea("R")
        mic_params.getRecParams()
        MM_model.setDimensions(mic_params)

        # Set up common parameters
        tgt_params = space_ext.TgtParams(target_list, avoid_list, MM_model, sparse_targets, target_edge, penalty_weight, target_score_threshold, max_tilt, mic_params, extra_tracking, max_iterations)

        # Export params for postAction script
        mic_params.export(CUR_DIR)
        tgt_params.export(CUR_DIR)
        mic_params.export(MAP_DIR)
        tgt_params.export(MAP_DIR)

        # Remove preexisting SPACE_runs
        if not external and os.path.exists(os.path.join(MAP_DIR, "SPACE_runs.json")):
            os.rename(os.path.join(MAP_DIR, "SPACE_runs.json"), os.path.join(MAP_DIR, "SPACE_runs.json~"))

        # Collect MM montages at all lamella positions
        MM_map_ids = []
        for i, nav_id in enumerate(lamella_nav_ids):
            map_name = grid_name + "_L" + str(i + 1).zfill(2)

            # Collect montage
            sem.Echo("Collecting map for lamella " + map_name + "...")
            map_id = space.collectMMMap(nav_id, map_name, MM_mean_threshold, lamella_bboxes[i], MM_padding_factor, MM_montage_overlap, mic_params, WG_model)
            MM_map_ids.append(map_id)

            # Update z height of remaining lamellae
            for j in range(i + 1, len(lamella_nav_ids)):
                sem.UpdateItemZ(lamella_nav_ids[j])

            # Save montage as rescaled input image
            sem.Echo("Saving map for lamella " + map_name + "...")
            space.saveMMMap(map_id, MAP_DIR, map_name, mic_params, MM_model)

            if automation_level >= 3:
                if manual_selection:
                    space_ext.saveEmptySeg(MAP_DIR, map_name)
                else:
                    if not external:
                        # Queue SPACEtomo run for new montage
                        space_ext.updateQueue(MAP_DIR, WG_model, MM_model, mic_params, save_plot=save_plot)

        sem.Echo("Completed collection of lamella maps! [Level 2]")

##############################################
############ STEP 3: Target setup ############
##############################################

    total_targets = 0
    if automation_level >= 4:

        # Make sure at least the first target selection is ready
        sem.ReportOtherItem(MM_map_ids[0])
        map_name = os.path.splitext(sem.GetVariable("navNote"))[0]

        point_files = sorted(glob.glob(os.path.join(MAP_DIR, map_name + "_points*.json")))
        inspected = not wait_for_inspection

        # Open tgt selection GUI if wait_for_inspection is selected and maps are not external
        if ((len(point_files) > 0 and wait_for_inspection) or manual_selection) and not external:
            DETACHED_PROCESS = 0x00000008           # From here: https://stackoverflow.com/questions/89228/calling-an-external-command-in-python#2251026
            subprocess.Popen(["python", os.path.join(SPACE_DIR, "SPACEtomo_tgt.py"), MAP_DIR], creationflags=DETACHED_PROCESS)

        while len(point_files) == 0 or not inspected:
            sem.Echo("Waiting for first lamella to be processed before setting up targets...")
            sem.Delay(60, "s")
            point_files = sorted(glob.glob(os.path.join(MAP_DIR, map_name + "_points*.json")))
            if not external:
                space_ext.updateQueue(MAP_DIR, WG_model, MM_model, mic_params, tgt_params, save_plot=save_plot)
            if wait_for_inspection:
                inspected = os.path.exists(os.path.join(MAP_DIR, map_name + "_inspected.txt"))

        # Loop over all MM maps
        for m, map_id in enumerate(MM_map_ids):
            sem.LoadOtherMap(map_id)
            buffer, *_ = sem.ReportCurrentBuffer()
            sem.ReportOtherItem(map_id)
            map_name = os.path.splitext(sem.GetVariable("navNote"))[0]
            map_seg = os.path.join(MAP_DIR, map_name + "_seg.png")


            if not external:
                space_ext.updateQueue(MAP_DIR, WG_model, MM_model, mic_params, tgt_params, save_plot=save_plot)

            # Check if point file exists
            point_files = sorted(glob.glob(os.path.join(MAP_DIR, map_name + "_points*.json")))
            if wait_for_inspection:
                inspected = os.path.exists(os.path.join(MAP_DIR, map_name + "_inspected.txt"))
            else:
                inspected = True
            if len(point_files) == 0 or not inspected:
                num_acq, *_ = sem.ReportNumNavAcquire()
                if automation_level < 5 or (automation_level >= 5 and num_acq == 0):
                    sem.Echo("Waiting for next prediction before setting up targets...")
                    while len(point_files) == 0 or not inspected:
                        sem.Delay(60, "s")
                        sem.Echo(".")
                        point_files = sorted(glob.glob(os.path.join(MAP_DIR, map_name + "_points*.json")))
                        if not external:
                            space_ext.updateQueue(MAP_DIR, WG_model, MM_model, mic_params, tgt_params, save_plot=save_plot)
                        if wait_for_inspection:
                            inspected = os.path.exists(os.path.join(MAP_DIR, map_name + "_inspected.txt"))
                else:
                    sem.Echo("Target setup for remaining lamellae postponed until predictions are available.")
                    break       # rely on SPACEtomo_postAction script to setup more targets between PACEtomo runs

            sem.Echo("")
            sem.Echo("Setting up targets for " + map_name + "...")

            # Save targets for PACEtomo
            space.saveAsTargets(buffer, MAP_DIR, map_name, MM_model, mic_params)

        sem.Echo("Completed target selection step! [Level 4]")

#############################################
############ STEP 4: Acquisition ############
#############################################

# Start acquire at items
if automation_level >= 5:
    num_acq, *_ = sem.ReportNumNavAcquire()
    sem.Echo("Starting PACEtomo acquisition of " + str(int(num_acq)) + " areas!")

    # Setting Acquite at Items scripts
    sem.NavAcqAtEndUseParams("F")
    sem.SetNavAcqAtEndParams("prim", 2)
    sem.SetNavAcqAtEndParams("scrp-p", script_numbers[2])   # PACEtomo script
    sem.SetNavAcqAtEndParams("scrp-b", 0)
    sem.SetNavAcqAtEndParams("scrp-a", script_numbers[1])   # SPACEtomo_postAction script
    sem.RunScriptAfterNavAcquire(script_numbers[0])         # SPACEtomo script

    sem.StartNavAcquireAtEnd()

sem.Echo("The SPACEtomo setup script completed!")
sem.Echo(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))