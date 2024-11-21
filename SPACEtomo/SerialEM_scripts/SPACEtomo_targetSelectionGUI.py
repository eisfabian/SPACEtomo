#!Python
# ===================================================================
#ScriptName SPACEtomo_targetSelectionGUI
# Purpose:      Script to start SPACEtomo target selection GUI.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# ===================================================================

import serialem as sem
import sys
from SPACEtomo.modules.utils import log
try:
    import dearpygui.dearpygui as dpg
except:
    log("ERROR: DearPyGUI module not installed! If you cannot install it, please run the target selection GUI from an external machine.")
    sys.exit()
from pathlib import Path

from SPACEtomo.modules import utils
from SPACEtomo.modules.gui.tgt_sel import TargetGUI

def run(path=""):
    dpg.create_context()
    main = TargetGUI(path)
    main.show()
    dpg.destroy_context()

def main():
    # Look for SPACE_maps
    cur_dir = utils.getCurDir()
    space_maps = sorted(cur_dir.glob("**/SPACE_maps"))
    print(space_maps)
    if len(space_maps) > 0:
        run(space_maps[0])
    else:
        utils.log(f"ERROR: No SPACE_maps directory found within current directory [{cur_dir}]!")

if __name__ == "__main__":
    main()