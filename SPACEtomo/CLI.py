#!/usr/bin/env python
# ===================================================================
# ScriptName:   CLI
# Purpose:      User interface for inspecting and labeling lamella grid atlases collected with SPACEtomo.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/04/26
# Revision:     v1.2
# Last Change:  2024/07/18: reworked GUI to class
# #             2024/06/07: added lamella table row highlighting, added save coords in YOLO format
#               2024/06/05: introduced Menu class, added shortcuts for most buttons
#               2024/06/04: added pix size check for export, added rescaling of mrc file
#               2024/06/03: added include empty tiles option, added save json option
#               2024/05/31: added status line, fixed tiles bounds, added next button, added hide/showElements
#               2024/05/30: added rescaling, saving as png, fixed export tiles
#               2024/05/29: outsourced plot clearing
#               2024/05/23: added lamella labels on plot, added padding when exporting edge tiles, fixed plotTiles
#               2024/05/10: added reading and writing of YOLO format labels, added saving of tiles with labels, added map class
#               2024/05/02: added lamella color from config, added reset zoom, delegated plot generation, call lamella detection
#               2024/04/30: added lamella focus buttons, outline scaling, list sorting
#               2024/04/26: Copy most of SPACEtomo_tgt
# ===================================================================

import sys
try:
    import dearpygui.dearpygui as dpg
except ModuleNotFoundError:
    print("WARNING: GUI library is not installed.")
import argparse
import shutil
from pathlib import Path

from SPACEtomo.modules.utils import log
from SPACEtomo.modules.gui.lam_sel import LamellaGUI
from SPACEtomo.modules.gui.tgt_sel import TargetGUI

# Attempt SerialEM import
import SPACEtomo.config as config
sys.path.insert(len(sys.path), config.SERIALEM_PYTHON_PATH)
try:
    import serialem as sem
    SERIALEM = True
except ModuleNotFoundError:
    SERIALEM = False

def show(gui, path=""):
    dpg.create_context()
    main = gui(path)
    main.show()
    dpg.destroy_context()
    
def run_process(process):
    log(f"Starting {process}...")
    process()
    
def main():
    # Process arguments
    parser = argparse.ArgumentParser(description="Command line interface for different SPACEtomo functions (e.g. GUIs).")
    parser.add_argument("task", type=str, default="targets", help="Name of the SPACEtomo task to run. Options: targets -> Open target selection GUI | lamella -> Open lamella selection GUI | run -> Start SPACEtomo run | scripts -> Get copy of SerialEM scripts")
    parser.add_argument("map_file", nargs="?", type=str, default="", help="Absolute path to a map file to load in GUI [.png].")
    args = parser.parse_args()

    # Get GUI
    gui = None
    process = None
    if args.task.lower() == "targets" or args.task.lower() == "target":
        gui = TargetGUI
    elif args.task.lower() == "lamella" or args.task.lower() == "lamellae":
        gui = LamellaGUI
    elif args.task.lower() == "run":
        if SERIALEM:
            from SPACEtomo import run
            process = run.main
        else:
            log(f"ERROR: SPACEtomo {args.task} requires SerialEM!")
    elif args.task.lower() == "scripts":
        serialem_script_dir = Path(__file__).parent / "SerialEM_scripts"
        target_path = Path.cwd() / "SerialEM_scripts"
        if not target_path.exists():
            shutil.copytree(serialem_script_dir, target_path)
            (target_path / "__init__.py").unlink()
            log(f"SerialEM scripts were successfully copied to: {target_path}")
        else:
            log(f"ERROR: A folder named SerialEM_scripts already exists!")
    else:
        raise ValueError(f"No task found with name {args.task}!")

    # Check map file
    if gui:
        map_file = ""
        if args.map_file != "": 
            map_file = Path(args.map_file)
            if not map_file.exists():
                map_file = ""
                log("WARNING: Map file does not exist! Attempting to start task without map...")            
    
        # Run GUI
        show(gui, map_file)
        
    if process:
        run_process(process)


if __name__ == "__main__":
    main()