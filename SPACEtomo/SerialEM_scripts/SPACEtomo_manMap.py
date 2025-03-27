#!Python
# ===================================================================
#ScriptName SPACEtomo_manMap
# Purpose:      Finds targets on a manually collected map using manual selection or deep learning models.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2025/01/09
# ===================================================================

### TODO before running:
# - Microscope alignments
# - Low Dose mode setup (ensure that MM (Low Dose) image state produces good images)
# - Set Shift offset between Record and View mag and check if "High Def IS" calibrations are necessary (Apply large IS in View and check if Record acquires at same position)
# - PACEtomo preparation and script settings
# - In case of inference on an external GPU workstation, start SPACEtomo_monitor.py script and set external_map_dir accordingly
###

############ SETTINGS ############ 

MM_image_state          = [3]                  # Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (set up before!)
rescale_map             = False                 # Rescale map to MM_model pixel size (can be skipped when using manual_selection)

# Targeting settings

manual_selection        = True                  # = True: creates empty segmentation, allows for manual target setup by operator using the GUI
MM_wait_for_inspection  = True                  # = True: wait for inspection of targets using the GUI

target_list             = ["mitos"]             # List of target classes, additive, possible classes: ["background", "white", "black", "crack", "coating", "cell", "cellwall", "nucleus", "vacuole", "mitos", "lipiddroplets", "vesicles", "multivesicles", "membranes", "dynabeads", "ice", "cryst", "lamella"]
avoid_list              = ["black", "white", "ice", "crack", "dynabeads"]    # list of classes to avoid, additive
target_score_threshold  = 0.01                  # Weighted fraction of FOV containing the target, keep at 0 for any structures smaller than the FOV
sparse_targets          = True                  # Use for sparse targets on the sample (e.g. mitos, vesicles)
target_edge             = False                 # Use for edge targets (e.g. nuclear envelope, cell periphery)
penalty_weight          = 0.3                   # Factor to downweight overlap with classes to avoid

extra_tracking          = False                 # Add an extra center target for the tracking tilt series (not working properly yet)
max_tilt                = 60                    # Maximum tilt angle [degrees] of tilt series to be acquired 

# Needed for external processing:
external_map_dir        = ""                    # Path to directory where maps are saved and segmentations are expected (run "SPACEmonitor" on external machine to manage runs and queue)

########## END SETTINGS ########## 

try:
    import serialem as sem
except ModuleNotFoundError:
    print("WARNING: Trying to run SerialEM externally!")

from SPACEtomo.run_manual import main
import SPACEtomo.modules.utils as utils

cur_dir = utils.getCurDir()
utils.saveSettings(cur_dir / "SPACEtomo_settings.ini", globals(), start="MM_image_state")
main()
