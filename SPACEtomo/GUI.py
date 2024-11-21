#!/usr/bin/env python
# ===================================================================
# ScriptName:   GUI
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
from SPACEtomo.modules.utils import log
try:
    import dearpygui.dearpygui as dpg
except:
    log("ERROR: DearPyGUI module not installed! If you cannot install it, please run the target selection GUI from an external machine.")
    sys.exit()
import argparse
from pathlib import Path
from SPACEtomo.modules.gui.lam_sel import LamellaGUI
from SPACEtomo.modules.gui.tgt_sel import TargetGUI

def run_process(gui, path="", auto_close=False):
    dpg.create_context()
    main = gui(path, auto_close=auto_close)
    main.show()
    dpg.destroy_context()
    
def main():
    # Process arguments
    parser = argparse.ArgumentParser(description="Calls GUI to inspect whole grid maps and detected lamellae.")
    parser.add_argument("gui", type=str, default="targets", help="Name of GUI to be opened. Options: targets, lamella")
    parser.add_argument("map_file", nargs="?", type=str, default="", help="Absolute path to a map [.png].")
    parser.add_argument("--auto_close", action="store_true", help="Auto-close GUI after inspection.")
    args = parser.parse_args()

    # Get GUI
    if args.gui == "targets":
        gui = TargetGUI
    elif args.gui == "lamella":
        gui = LamellaGUI
    else:
        raise ValueError(f"No GUI found with name {args.gui}!")

    # Check map file
    map_file = ""
    if args.map_file != "": 
        map_file = Path(args.map_file)
        if not map_file.exists():
            map_file = ""
            log("WARNING: Map file does not exist! Attempting to open GUI without map...")

    # Run GUI
    run_process(gui, map_file, args.auto_close)
    

if __name__ == "__main__":
    main()