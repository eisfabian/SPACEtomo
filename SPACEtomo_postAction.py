#!Python
# ===================================================================
#ScriptName SPACEtomo_postAction
# Purpose:      Post action script to setup targets from SPACEtomo runs.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/07/13
# Revision:     v1.1
# Last Change:  2024/04/10: fixed typo
# ===================================================================

import serialem as sem
import os
import sys
import glob

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
    MAP_DIR = os.path.join(CUR_DIR, "SPACE_maps")
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
import SPACEtomo_functions_ext as space_ext

# Instantiate mic params from settings
mic_params = space_ext.MicParams_ext(CUR_DIR)

# Load models
WG_model = space_ext.WGModel()
MM_model = space_ext.MMModel()
MM_model.setDimensions(mic_params)

# Instantiate tgt params from settings
tgt_params = space_ext.TgtParams(file_dir=CUR_DIR, MM_model=MM_model)

# Update SPACE runs and queue
if not MM_external:
    space_ext.updateQueue(MAP_DIR, WG_model, MM_model, mic_params, tgt_params, save_plot=save_plot)

# Check for unprocessed point files
point_files = sorted(glob.glob(os.path.join(MAP_DIR, "*_points*.json")))
tgt_files = sorted(glob.glob(os.path.join(CUR_DIR, "*tgts.txt")))

unprocessed_point_files = []
for point_file in point_files:
    map_name = os.path.basename(point_file).split("_points")[0]
    if map_name not in [os.path.basename(tgt_file).split("_tgts")[0] for tgt_file in tgt_files]:
        unprocessed_point_files.append(point_file)

# Check for unprocessed maps
mm_list, seg_list, wg_list  = space_ext.monitorFiles(MAP_DIR)

# Check for areas still to be acquired
num_acq, *_ = sem.ReportNumNavAcquire()

# Check if collection would stop
if num_acq == 0 and len(unprocessed_point_files) == 0:
    # Check if there will be further point files generated
    if len(mm_list) > 0 or len(seg_list) > 0:
        # Wait for another segmentation to be processed
        start_len_seg = len(seg_list)
        while len(seg_list) >= start_len_seg:
            sem.Echo("Waiting for next prediction before setting up targets...")
            sem.Delay(60, "s")
            if not MM_external:
                space_ext.updateQueue(MAP_DIR, WG_model, MM_model, mic_params, tgt_params, save_plot=save_plot)
            mm_list, seg_list, wg_list  = space_ext.monitorFiles(MAP_DIR)

        # Get point files that were not found previously
        point_files_new = sorted(glob.glob(os.path.join(MAP_DIR, "*_points*.json")))
        unprocessed_point_files = point_files_new - point_files

# Set up targets for unprocessed point_files
for point_file in unprocessed_point_files:
    map_name = os.path.basename(point_file).split("_points")[0]

    if wait_for_inspection:
        inspected = os.path.exists(os.path.join(MAP_DIR, map_name + "_inspected.txt"))
    else:
        inspected = True

    if not inspected:
        sem.Echo("")
        sem.Echo("WARNING: " + map_name + " targets are still waiting for inspection! Skipping...")
        continue

    sem.Echo("")
    sem.Echo("Setting up targets for " + map_name + "...")

    # Load map from nav
    map_id = int(sem.NavIndexWithNote(map_name + ".mrc"))
    sem.LoadOtherMap(map_id)
    buffer, *_ = sem.ReportCurrentBuffer()

    # Save targets for PACEtomo
    space.saveAsTargets(buffer, MAP_DIR, map_name, MM_model, mic_params)

num_acq, *_ = sem.ReportNumNavAcquire()
sem.Echo("Continuing PACEtomo acquisition of " + str(int(num_acq)) + " areas!")

# Start acquisition if not already running
if not sem.ReportIfNavAcquiring()[0]:
    sem.StartNavAcquireAtEnd()