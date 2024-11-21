#!Python
# ===================================================================
#ScriptName SPACEtomo_lamellaDetectionGUI
# Purpose:      Script to start SPACEtomo lamella detection GUI.
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

from SPACEtomo.modules.gui.lam_sel import LamellaGUI

def main():
    dpg.create_context()
    main = LamellaGUI()
    main.show()
    dpg.destroy_context()

if __name__ == "__main__":
    main()