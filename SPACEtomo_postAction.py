#!Python
# ===================================================================
#ScriptName SPACEtomo_postAction
# Purpose:      Post action script to setup targets from SPACEtomo runs.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/07/13
# Revision:     v1.0beta
# Last Change:  2023/11/12: fixes after Krios test
# ===================================================================

import serialem as sem
import os
import sys

CUR_DIR = sem.ReportDirectory()

# Read settings written by main SPACE script
if os.path.exists(os.path.join(CUR_DIR, "SPACEtargets_settings.txt")):
    exec(open(os.path.join(CUR_DIR, "SPACEtargets_settings.txt")).read())
else:
    sem.Echo("ERROR: No SPACE settings file was found. No targets were set up.")
    sem.Exit()

# Check if external processing directory is valid
MM_external = False
if external_map_dir == "":
    MAP_DIR = CUR_DIR
else:
    if os.path.exists(external_map_dir):
        MAP_DIR = external_map_dir
        MM_external = True
    else:
        sem.Echo("ERROR: External map directory does not exist!")
        sem.Exit()

# Import SPACE functions (needs SPACE folder from settings file)
sys.path.insert(len(sys.path), SPACE_DIR)
import SPACEtomo_functions as space

# Instantiate mic params from settings
mic_params = space.MicParams(WG_image_state, IM_mag_index, MM_image_state)

# Load model
MM_model = space.MMModel()

# Update SPACE runs and queue
if not MM_external:
    space.queueSpaceRun(MM_model)

# Make list of finished SPACE runs
if os.path.exists(os.path.join(MAP_DIR, "SPACE_runs.txt")):
    with open(os.path.join(MAP_DIR, "SPACE_runs.txt"), "r") as f:
        space_lines = f.readlines()
else:
    sem.Echo("ERROR: No SPACE runs file was found. No targets were set up.")
    sem.Exit()

space_maps = []
active_runs = []
for line in space_lines:
    map_name = os.path.splitext(os.path.basename(line))[0]
    map_seg = os.path.join(MAP_DIR, map_name + "_seg.png")
    map_tgt = os.path.join(CUR_DIR, map_name + "_tgts.txt")
    if not os.path.exists(map_seg):
        active_runs.append(map_name)
    elif not os.path.exists(map_tgt):
        sem.Echo("SPACEtomo [" + line + "] run finished!")
        space_maps.append(map_name)
    else:
        sem.Echo("Targets file for " + line + " already exists. Skipping...")

# Check if any targets are remaining and wait for next segmentation if not
if len(space_maps) > 0:
    sem.Echo("Setting up targets for " + str(len(space_maps)) + " lamellae...")
else:
    num_acq, *_ = sem.ReportNumNavAcquire()
    while num_acq == 0 and len(space_maps) == 0 and len(active_runs) > 0:
        sem.Echo("Waiting for next prediction before setting up targets...")
        sem.Delay(60, "s")
        if not MM_external:
            space.queueSpaceRun(MM_model)
        for map_name in active_runs:
            map_seg = os.path.join(MAP_DIR, map_name + "_seg.png")
            if os.path.exists(map_seg):
                space_maps.append(map_name)

    # Check again in case results were added
    if len(space_maps) == 0:
        if len(active_runs) > 0:
            sem.Echo("Target setup for remaining " + str(len(active_runs)) + " lamellae postponed until predictions are available.")
        else:
            sem.Echo("All lamellae have been set up and no more predictions are running.")
        sem.Exit()

# Get microscope parameters
sem.GoToLowDoseArea("V")
mic_params.getViewParams()
sem.GoToLowDoseArea("R")
mic_params.getRecParams()
MM_model.setDimensions(mic_params)

# Set up common parameters
weight_mask, edge_weight_masks = space.makeScoreWeights(MM_model, target_edge)
grid_vecs = space.findGridVecs(MM_model, max_tilt, mic_params)

# Loop over all MM maps
for m, map_name in enumerate(space_maps):           # adjusted from main script
    map_id = int(sem.NavIndexWithNote(map_name + ".mrc"))
    sem.LoadOtherMap(map_id)
    buffer, *_ = sem.ReportCurrentBuffer()

    sem.Echo("")
    sem.Echo("Setting up targets for " + map_name + "...")
    
    # Instantiate lamella and find points
    lamella = space.Lamella(map_name, MAP_DIR, target_list, avoid_list, MM_model, weight_mask, edge_weight_masks, grid_vecs, mic_params, max_tilt)
    lamella.findPoints(sparse_targets, penalty_weight, target_score_threshold, max_iterations)

    if len(lamella.points) == 0:
        sem.Echo("WARNING: No targets found!")
        sem.Echo("If you visually identified targets, please adjust your settings or add them manually!")
        # Write empty PACE target file
        lamella.saveAsTargets(buffer, penalty_weight)
        continue
    
    sem.Echo("Final targets: " + str(len(lamella.points)))
    sem.Echo("Saving overview image...")
    lamella.plotTargets(tracking_id=0, overlay=lamella.target_mask, save=os.path.join(CUR_DIR, map_name + "_" + target_list[0] +"_targets.png"))
    sem.Echo("Saved at " + os.path.join(CUR_DIR, map_name + "_targets_" + target_list[0] + ".png"))

    # Find geo points for sample geometry measurement
    lamella.findGeoPoints()

    # Save targets for PACEtomo
    lamella.saveAsTargets(buffer, penalty_weight)

num_acq, *_ = sem.ReportNumNavAcquire()
sem.Echo("Continuing PACEtomo acquisition of " + str(int(num_acq)) + " areas!")