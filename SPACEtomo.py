#!Python
# ===================================================================
#ScriptName SPACEtomo
# Purpose:      Finds lamellae in grid map, collects lamella montages and finds targets using deep learning models.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/31
# Revision:     v1.0beta
# Last Change:  2023/12/12: added version check
# ===================================================================

### TODO before running:
# - Microscope alignments
# - Run inventory and name grids in autoloader
# - Low Dose mode setup (ensure that both WG image state and MM (Low Dose) image states produce good images)
# - Set SerialEM directory
# - PACEtomo preparation and script settings (for automation level 5)
# - Setup "Acquire at Items" dialogue to run PACEtomo as main action and SPACEtomo_postAction as "Run after primary action" (for automation level 5)
# - In case of inference on an external GPU workstation, start SPACEtomo_monitor script and set external_map_dir accordingly
###

############ SETTINGS ############ 

grid_slot = 2                   # number of target grid in autoloader, set to 0 if grid is already loaded
include_broken = False          # include lamella classified as broken
grid_default_name = ""          # only used when grid_slot = 0 (grid is already loaded)

# Targeting settings

target_list = ["mitos"]         # list of target classes, additive, possible classes: ["background", "white", "black", "crack", "coating", "cell", "cellwall", "nucleus", "vacuole", "mitos", "lipiddroplets", "vesicles", "multivesicles", "membranes", "dynabeads", "ice", "cryst", "lamella"]
avoid_list = ["black", "white", "ice", "crack"]   # list of classes to avoid, additive
target_score_threshold = 0.01   # weighted fraction of FOV containing the target, keep at 0 for any structures smaller than the FOV
sparse_targets = True           # use for sparse targets on the sample (e.g. mitos, vesicles)
target_edge = False             # use for edge targets (e.g. nuclear envelope, cell periphery)
penalty_weight = 0.3            # factor to downweigh overlap with classes to avoid
max_iterations = 10             # maximum number of iterations for target placement optimization
extra_tracking = False          # add an extra center target for the tracking tilt series (not working properly yet)

# Workflow settings
SPACE_DIR = "D:\SPACEtomo"      # path to SPACEtomo files
automation_level = 4            # = 1: collect WG map and find lamellae
                                # = 2: collect MM maps for each lamella
                                # = 3: segment lamella MM maps
                                # = 4: setup targets based on segmentation for each lamella
                                # = 5: start PACEtomo batch acquisition of all targets

save_plot = False               # saves all plots of all substeps (slow)
aperture_control = True         # set to True if SerialEM can control apertures (https://bio3d.colorado.edu/SerialEM/hlp/html/setting_up_serialem.htm#apertures)
objective_aperture = 70         # diameter of objective aperture

WG_image_state = 1              # imaging state name or index for whole grid montage (set up before!)
MM_image_state = 13             # imaging state name or index for Low Dose Mode (set up before!)
WG_offset_via_IM = True         # if offset between WG map and MM is too large, use intermediate mag to find lamella again and shift items to marker
IM_mag_index = 10               # intermediate mag index to use (10: 580x) 

MM_padding_factor = 1.2         # padding of estimated lamella size for medium mag montage
MM_mean_threshold = 0           # enable manual targeting if View image is too dark (mean count below this threshold)

WG_montage_overlap = 0.15       # overlap between pieces of whole grid montage
MM_montage_overlap = 0.25       # overlap between pieces of medium mag montage

WG_detection_threshold = 0.1    # confidence threshold for lamella acceptance

max_tilt = 60                   # maximum tilt angle [degrees] of tilt series to be acquired 
tolerance = 0.1                 # if two targets overlap, move them up to this fraction of the beam diameter to allow fitting of more targets

# External processing settings
external_map_dir = ""           # dir where maps are saved and segmentations are expected (run SPACEtomo_monitor.py externally to manage runs and queue)

########## END SETTINGS ########## 

import serialem as sem
import os
import sys
sys.path.insert(len(sys.path), SPACE_DIR)
import SPACEtomo_functions as space

versionCheck = sem.IsVersionAtLeast("40100", "20231001")
if not versionCheck and sem.IsVariableDefined("warningVersion") == 0:
	runScript = sem.YesNoBox("\n".join(["WARNING: You are using a version of SerialEM that does not support all SPACEtomo features. It is recommended to update to the latest SerialEM beta version!", "", "Do you want to run SPACEtomo regardless?"]))
	if not runScript:
		sem.Exit()
	else:
		sem.SetPersistentVar("warningVersion", "")

CUR_DIR = sem.ReportDirectory()

if "SerialEM" in CUR_DIR:
    sem.UserSetDirectory("Please choose a directory for saving montages, targets and tilt series!")
    CUR_DIR = sem.ReportDirectory()

# Check if external processing directory is valid
MM_external = False
if external_map_dir == "":
    MAP_DIR = CUR_DIR
else:
    if os.path.exists(external_map_dir):
        MAP_DIR = external_map_dir
        #space.saveSettings(os.path.join(MAP_DIR, "SPACEtargets"), globals())
        MM_external = True
    else:
        sem.Echo("ERROR: External map directory does not exist!")
        sem.Exit()

# Save settings for SPACEtomo_postAction.py
space.saveSettings(os.path.join(CUR_DIR, "SPACEtargets"), globals())

mic_params = space.MicParams(WG_image_state, IM_mag_index, MM_image_state)

########################################
############ STEP 1: WG map ############
########################################

if automation_level >= 1:

    # Load model
    WG_model = space.WGModel()  # load a custom model

    # Load grid
    grid_name = space.loadGrid(grid_slot, grid_default_name)

    # Retract objective aperture
    if aperture_control:
        sem.SetApertureSize(2, 0)

    # Open column valves
    sem.SetColumnOrGunValve(1)

    # Find lamellae on grid
    lamella_nav_ids, lamella_bboxes = space.findLamellae(grid_name, mic_params, WG_montage_overlap, WG_model, WG_detection_threshold, include_broken, save_plot)

    # Find mag offset via intermediate mag
    if WG_offset_via_IM:
        lamella_nav_ids, lamella_bboxes = space.findOffset(lamella_nav_ids, lamella_bboxes, mic_params, WG_model)


#########################################
############ STEP 2: MM maps ############
#########################################

if automation_level >= 2:

    # Load model
    MM_model = space.MMModel()

    # Insert objective aperture
    if aperture_control:
        sem.SetApertureSize(2, objective_aperture)

    # Prepare MM montages
    sem.GoToImagingState(str(mic_params.MM_image_state))
    sem.GoToLowDoseArea("V")
    mic_params.getViewParams()

    # Collect MM montages at all lamella positions
    MM_map_ids = []
    for i, nav_id in enumerate(lamella_nav_ids):
        map_name = grid_name + "_L" + str(i + 1).zfill(2)

        # Collect montage
        map_id = space.collectMMMap(nav_id, map_name, MM_mean_threshold, lamella_bboxes[i], MM_padding_factor, MM_montage_overlap, mic_params, WG_model)
        MM_map_ids.append(map_id)

        # Update z height of remaining lamellae
        for j in range(i + 1, len(lamella_nav_ids)):
            sem.UpdateItemZ(lamella_nav_ids[j])

        # Save montage as rescaled input image
        space.saveMMMap(map_id, MAP_DIR, map_name, mic_params, MM_model)

        if automation_level >= 3:
            if not MM_external:
                # Queue SPACEtomo run for new montage
                space.queueSpaceRun(MM_model, map_name)


##############################################
############ STEP 3: Target setup ############
##############################################

total_targets = 0
if automation_level >= 4:

    # Make sure at least the first segmentation is ready
    sem.ReportOtherItem(MM_map_ids[0])
    map_name = os.path.splitext(sem.GetVariable("navNote"))[0]
    map_seg = os.path.join(MAP_DIR, map_name + "_seg.png")
    while not os.path.exists(map_seg):
        sem.Echo("Waiting for first prediction before setting up targets...")
        sem.Delay(60)
        if not MM_external:
            space.queueSpaceRun(MM_model)

    sem.GoToLowDoseArea("R")
    mic_params.getRecParams()
    MM_model.setDimensions(mic_params)

    # Set up common parameters
    weight_mask, edge_weight_masks = space.makeScoreWeights(MM_model, target_edge)
    grid_vecs = space.findGridVecs(MM_model, max_tilt, mic_params)

    # Loop over all MM maps
    for m, map_id in enumerate(MM_map_ids):
        sem.LoadOtherMap(map_id)
        buffer, *_ = sem.ReportCurrentBuffer()
        sem.ReportOtherItem(map_id)
        map_name = os.path.splitext(sem.GetVariable("navNote"))[0]
        map_seg = os.path.join(MAP_DIR, map_name + "_seg.png")

        # Check if segmentation exists
        if not MM_external:
            space.queueSpaceRun(MM_model)
        if not os.path.exists(map_seg):
            num_acq, *_ = sem.ReportNumNavAcquire()
            if automation_level >= 5 and num_acq == 0:
                sem.Echo("Waiting for next prediction before setting up targets...")
                while not os.path.exists(map_seg):
                    sem.Delay(60, "s")
                    sem.Echo(".")
            else:
                sem.Echo("Target setup for remaining lamellae postponed until predictions are available.")
                break       # rely on SPACEtomo_post script to setup more targets between PACEtomo runs

        sem.Echo("")
        sem.Echo("Setting up targets for " + map_name + "...")

        # Instantiate lamella and find points
        lamella = space.Lamella(map_name, MAP_DIR, target_list, avoid_list, MM_model, weight_mask, edge_weight_masks, grid_vecs, mic_params, max_tilt, save_plot)
        lamella.findPoints(sparse_targets, penalty_weight, target_score_threshold, max_iterations, extra_tracking)

        if len(lamella.points) == 0:
            sem.Echo("WARNING: No targets found!")
            sem.Echo("If you visually identified targets, please adjust your settings or add them manually!")
            # Write empty PACE target file
            lamella.saveAsTargets(buffer, penalty_weight)
            continue
        
        sem.Echo("Final targets: " + str(len(lamella.points)))
        sem.Echo("Saving overview image...")
        lamella.plotTargets(tracking_id=0, overlay=lamella.target_mask, save=os.path.join(CUR_DIR, map_name + "_" + target_list[0] +"_targets.png"))
        sem.Echo("Saved at " + os.path.join(CUR_DIR, map_name + "_targets_" + target_list[0] +".png"))

        # Find geo points for sample geometry measurement
        lamella.findGeoPoints()

        # Save targets for PACEtomo
        lamella.saveAsTargets(buffer, penalty_weight)

        total_targets += len(lamella.points)

    sem.Echo("Total identified targets: " + str(total_targets))

#############################################
############ STEP 4: Acquisition ############
#############################################

# Start acquire at items
if automation_level >= 5:
    num_acq, *_ = sem.ReportNumNavAcquire()
    sem.Echo("Starting PACEtomo acquisition of " + str(int(num_acq)) + " areas!")
    sem.StartNavAcquireAtEnd()