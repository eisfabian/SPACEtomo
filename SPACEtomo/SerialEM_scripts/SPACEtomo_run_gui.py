#!Python
# ===================================================================
#ScriptName SPACEtomo
# Purpose:      Finds lamellae in grid map, collects lamella montages and finds targets using deep learning models.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/31
# ===================================================================

### TODO before running:
# - Microscope alignments
# - Run inventory and name grids in autoloader
# - Low Dose mode setup (ensure that both WG image state and MM (Low Dose) image states produce good images)
# - Set Shift offset between Record and View mag and check if "High Def IS" calibrations are necessary (Apply large IS in View and check if Record acquires at same position)
# - PACEtomo preparation and script settings (for automation level 5)
# - Setup "Acquire at Items" dialogue to run PACEtomo as main action and SPACEtomo_prepareTargets as "Run after primary action" (for automation level 5)
# - In case of inference on an external GPU workstation, run SPACEmonitor command on that machine and set external_map_dir accordingly
###

try:
    import serialem as sem
except ModuleNotFoundError:
    print("WARNING: Trying to run SerialEM externally!")

import json
from pathlib import Path

import SPACEtomo.modules.utils as utils
import SPACEtomo.modules.utils_sem as usem
from SPACEtomo.modules.scope import Microscope

# Check if session dir has been chosen previously or ask for it
SES_DIR = usem.getSessionDir()

# Fetch microscope data for GUI
utils.log("Fetching microscope data...")

# Save autoloader data
microscope = Microscope()
microscope.checkAutoloader()
autoloader_file = Path(SES_DIR) / "autoloader.json"
with open(autoloader_file, "w") as f:
    json.dump(microscope.autoloader, f)

# Save imaging states
imaging_states = usem.getImagingStates()
imaging_states_file = Path(SES_DIR) / "imaging_states.json"
with open(imaging_states_file, "w") as f:
    json.dump(imaging_states, f)

# Run GUI as subprocess
utils.log("Starting SPACEtomo GUI...")
utils.guiProcess("settings", str(SES_DIR), blocking=True, extra_args=["--run_mode"])

from SPACEtomo.run import main
main()
