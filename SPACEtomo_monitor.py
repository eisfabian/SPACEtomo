#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_monitor
# Purpose:      Monitors an external directory and runs a target selection deep learning model on medium mag montages of lamella and generates a segmentation that can be used for SPACEtomo target selection.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/10/05
# Revision:     v1.0beta
# Last Change:  2023/12/04: added check for memory error
# ===================================================================

############ SETTINGS ############ 

external_map_dir = r"X:\SPACEtomo_maps"

########## END SETTINGS ########## 

import os
import glob
import time
import subprocess
import SPACEtomo_config as config

########### FUNCTIONS ############

# Make list of map files in inventory
def monitorFiles():
    file_list = sorted(glob.glob(os.path.join(MAP_DIR, "*.png")))
    map_list = []
    seg_list = []
    for file in file_list:
        if os.path.splitext(file)[0].split("_")[-1] == "seg":
            seg_list.append(file.split("_seg")[0])
        elif os.path.splitext(file)[0].split("_")[-1] == "segtemp":
            continue
        else:
            map_list.append(os.path.splitext(file)[0])
    unprocessed_map_list = []
    for map_name in map_list:
        if map_name not in seg_list:
            unprocessed_map_list.append(map_name)
    return unprocessed_map_list

def readMapList(file):
    map_list = []
    if os.path.exists(file):
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            map_name = os.path.splitext(line)[0]
            map_list.append(map_name)
    return map_list    

def writeMapList(file, map_list):   
    output = ""
    for map_name in map_list:
        output += map_name + ".png" + "\n"
    with open(file, "w") as f:
        f.write(output)

def checkMemErr(map_name):
    error_file = os.path.join(MAP_DIR, map_name + "_SPACE.err")
    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            if "not enough memory" in f.read():
                return True
            else:
                return False
    else:
        return False

# Control queue for MM montage inference
def updateQueue():
    # Check runs files
    space_runs = readMapList(os.path.join(MAP_DIR, "SPACE_runs.txt"))
    space_queue = readMapList(os.path.join(MAP_DIR, "SPACE_queue.txt"))
    unprocessed_maps = monitorFiles()

    active_runs = []
    add_to_queue = []   # add to queue after submitting new runs to give file time to finish copying
    for map_name in unprocessed_maps:
        if map_name in space_queue:
            continue
        if map_name in space_runs and not checkMemErr(map_name):
            active_runs.append(map_name)
        else:
            add_to_queue.append(map_name)

    # Submit new run
    while len(active_runs) < config.MM_model_max_runs and len(space_queue) > 0:
        map_name = space_queue.pop(0)
        out_file = open(os.path.join(MAP_DIR, map_name + "_SPACE.err"), "w")
        subprocess.Popen(["python", os.path.join(SPACE_DIR, config.MM_model_script), os.path.join(MAP_DIR, map_name + ".png")], stdout=out_file, stderr=subprocess.STDOUT, text=True)
        print("Starting inference for " + os.path.basename(map_name) + ".png...")
        space_runs.append(map_name)
        active_runs.append(map_name)

    # add to queue after submitting new runs to give file time to finish copying
    for map_name in add_to_queue:
        space_queue.append(map_name)

    if len(active_runs) > 0:
        print("# Processing:")
        for map_name in active_runs:
            print(os.path.basename(map_name))

    if len(space_queue) > 0:
        print("# Queue:")
        for map_name in space_queue:
            print(os.path.basename(map_name))

    # Write runs files
    writeMapList(os.path.join(MAP_DIR, "SPACE_runs.txt"), space_runs)
    writeMapList(os.path.join(MAP_DIR, "SPACE_queue.txt"), space_queue)

######### END FUNCTIONS ##########

MAP_DIR = external_map_dir
"""
print("Waiting for start of SPACEtomo session...")
# Read settings written by main SPACE script
while "SPACE_DIR" not in globals():
    if os.path.exists(os.path.join(MAP_DIR, "SPACEtargets_settings.txt")):
        exec(open(os.path.join(MAP_DIR, "SPACEtargets_settings.txt")).read())
    else:
        time.sleep(30)
print("Successfully read SPACEtomo settings.")
"""
SPACE_DIR = os.path.dirname(__file__)

start_time = time.time()
while True:
    print("##### Running for " + str(int((time.time() - start_time) / 60)) + " min... #####")
    updateQueue()
    time.sleep(60)
