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

############ SETTINGS ############ 

automation_level        = 4                     # = 1: collect WG map (and find lamellae)
                                                # = 2: collect MM maps for each area/lamella
                                                # = 3: segment lamella MM maps (if manual_selection is not selected)
                                                # = 4: setup targets manually or based on segmentation for each area/lamella
                                                # = 5: start PACEtomo batch acquisition of all targets

# Grid and lamella settings

grid_list               = [1]                   # Number or list of numbers of target grid(s) in autoloader
lamella                 = True                  # = True: find lamellae in grid map, turn off for non-lamella samples
exclude_lamella_classes = ["gone"]              # Lamella classes to exclude, possible classes: ["broken", "gone", "thick", "wedge", "contaminated", "good"]
WG_distance_threshold   = 5                     # Minimum distance [microns] between lamellae to not be considered duplicate detection

# Imaging states

WG_image_state          = 1                     # Imaging state name or index for whole grid montage (set up before!)
IM_image_state          = 2                     # Imaging state name or index for intermediate mag (set up before!) (usually ~580x) 
MM_image_state          = [3]                   # Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (set up before!)

# User inspection settings

WG_wait_for_inspection  = True                  # = True: wait for inspection of area/lamellae detection by operator using the GUI
manual_selection        = True                  # = True: creates empty segmentation, allows for manual target setup by operator using the GUI
MM_wait_for_inspection  = True                  # = True: wait for inspection of targets using the GUI

# Auto-Targeting settings [Only needed on Level 4+ if not using manual_selection]

target_list             = ["mitos"]             # List of target classes, additive, possible classes: ["background", "white", "black", "crack", "coating", "cell", "cellwall", "nucleus", "vacuole", "mitos", "lipiddroplets", "vesicles", "multivesicles", "membranes", "dynabeads", "ice", "cryst", "lamella"]
avoid_list              = ["black", "white", "ice", "crack", "dynabeads"]    # list of classes to avoid, additive
target_score_threshold  = 0.01                  # Weighted fraction of FOV containing the target, keep at 0 for any structures smaller than the FOV
sparse_targets          = True                  # Use for sparse targets on the sample (e.g. mitos, vesicles)
target_edge             = False                 # Use for edge targets (e.g. nuclear envelope, cell periphery)
penalty_weight          = 0.3                   # Factor to downweight overlap with classes to avoid

extra_tracking          = False                 # Add an extra center target for the tracking tilt series (not working properly yet)
max_tilt                = 60                    # Maximum tilt angle [degrees] of tilt series to be acquired 

# Needed for automation level 5
script_numbers          = [10, 11, 13]          # Numbers of [SPACEtomo_run.py, SPACEtomo_prepareTargets.py, PACEtomo.py] scripts in SerialEM script editor

# Needed for external processing
external_map_dir        = ""                    # Path to directory where maps are saved and segmentations are expected (run "SPACEmonitor" on external machine to manage runs and queue)

########## END SETTINGS ########## 

try:
    import serialem as sem
except ModuleNotFoundError:
    print("WARNING: Trying to run SerialEM externally!")

from SPACEtomo.run import main
import SPACEtomo.modules.utils as utils

cur_dir = utils.getCurDir()
utils.saveSettings(cur_dir / "SPACEtomo_settings.ini", globals())
main()
