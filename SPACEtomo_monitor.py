#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_monitor
# Purpose:      Monitors an external directory and runs a target selection deep learning model on medium mag montages of lamella and generates a segmentation that can be used for PACEtomo target selection.
#               More information at http://github.com/eisfabian/PACEtomo
# Usage:        python SPACEtomo_monitor.py [--dir MAP_DIR] [--gpu 0,1,2,3] [--plot]
# Author:       Fabian Eisenstein
# Created:      2023/10/05
# Revision:     v1.1
# Last Change:  2024/04/09: deleted run file upon start to avoid problems upon restart
# ===================================================================

import os
import sys
import time
import argparse
import subprocess
import SPACEtomo_functions_ext as space_ext

SPACE_DIR = os.path.dirname(__file__)

# Process arguments
parser = argparse.ArgumentParser(description='Monitors an external directory and runs lamella detection, lamella segmentation and target selection on appropiate files.')
parser.add_argument('--dir', dest='map_dir', type=str, default=None, help='Absolute path to folder to be monitored. This should be the same directory that was set in SPACEtomo on the SerialEM PC. (Default: Folder this script is run from)')
parser.add_argument('--gpu', dest='gpu', type=str, default="0", help='Comma-separated IDs of GPUs to use. (Default: 0)')
parser.add_argument('--plot', dest='save_plot', action='store_true', help='Create plots of intermediate target selection steps (slow).')
args = parser.parse_args()

if args.map_dir is not None: 
    if os.path.exists(args.map_dir):
        MAP_DIR = args.map_dir
    else:
        print("ERROR: Folder does not exist!")
        sys.exit()
else:
    MAP_DIR = os.getcwd()

gpus = args.gpu.split(",")
gpu_list = [int(gpu) for gpu in gpus]

save_plot = args.save_plot

# Load model
WG_model = space_ext.WGModel()

MM_model = None
mic_params = None
tgt_params = None

# Move previous run file
if os.path.exists(os.path.join(MAP_DIR, "SPACE_runs.json")):
    os.rename(os.path.join(MAP_DIR, "SPACE_runs.json"), os.path.join(MAP_DIR, "SPACE_runs.json~"))

# Indicator for monitors running (lamella detection, lamella segmentation, target selection)
status = "(oxx)"

# Set start time
start_time = time.time()
next_time = start_time + 60

print("SPACEtomo Version " + space_ext.versionSPACE)
print("")
print("Start monitoring " + MAP_DIR + " for all maps... " + status)
print("Using GPUs:", gpu_list)
print("")

while True:
    now = time.time()
    if now > next_time:
        next_time = now + 60
        print("##### Running for " + str(int(round((now - start_time) / 60))) + " min... " + status + " #####")
    
    # Check if mic_params exist
    if mic_params is None and os.path.exists(os.path.join(MAP_DIR, "mic_params.json")):
        status = "(oox)"
        print("NOTE: Microscope parameters found. Including lamella segmentation. " + status)
        # Instantiate mic params from settings
        mic_params = space_ext.MicParams_ext(MAP_DIR)        
        # Load model
        MM_model = space_ext.MMModel()

    # Check if tgt_params exist
    if tgt_params is None and os.path.exists(os.path.join(MAP_DIR, "tgt_params.json")):
        status = "(ooo)"
        print("NOTE: Target parameters found. Including target setup. " + status)
        # Reimport mic params to get Rec params
        mic_params = space_ext.MicParams_ext(MAP_DIR) 
        MM_model.setDimensions(mic_params)
        # Instantiate tgt params from settings
        tgt_params = space_ext.TgtParams(file_dir=MAP_DIR, MM_model=MM_model)

        print("Opening target selection GUI for inspection...")
        subprocess.Popen(["python", os.path.join(SPACE_DIR, "SPACEtomo_tgt.py"), MAP_DIR])

    # Run queue
    space_ext.updateQueue(MAP_DIR, WG_model, MM_model, mic_params, tgt_params, gpu_list, save_plot)
    time.sleep(5)
