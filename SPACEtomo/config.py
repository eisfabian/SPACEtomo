#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_config
# Purpose:      Configurations for deep learning models used in SPACEtomo.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/10/04
# Revision:     v1.2
# Last Change:  2024/12/13: added check for backup config
#               2024/09/26: outsourcing of more settings from SPACEtomo_run scripts
#               2024/08/20: added SerialEM connection setting
#               2024/05/02: changed model colors to RGBA
#               2024/03/12: removed max_runs
#               2023/10/04: outsourcing of settings from main SPACEtomo script
# ===================================================================

# SerialEM specific settings
SERIALEM_IP = "127.0.0.1"
SERIALEM_PORT = 48888
SERIALEM_PYTHON_PATH = "C:/Program Files/SerialEM/PythonModules"

DUMMY = False
DEBUG = False
BREAKPOINTS = False

# Acquisition settings
WG_montage_overlap      = 0.15                  # overlap between pieces of whole grid montage
WG_detection_threshold  = 0.1                   # confidence threshold for lamella acceptance

MM_montage_overlap      = 0.25                  # overlap between pieces of medium mag montage
MM_padding_factor       = 1.5                   # padding of estimated lamella size for medium mag montage

aperture_control        = True                  # set to True if SerialEM can control apertures (https://bio3d.colorado.edu/SerialEM/hlp/html/setting_up_serialem.htm#apertures)
c2_apertures            = [0, 0, 0]             # diameter of C2 apertures for [WG, IM, MM] imaging states, keep 0 to avoid changing apertures
objective_aperture      = 100                   # diameter of objective aperture

# Microscope specific settings
# Glacios (2 condenser lens system)
smallest_c2_aperture    = 20                    # smallest C2 aperture size
beam_sizes              = [0.6, 14]             # beam sizes [microns] in nano probe and micro probe using smallest C2 aperture

# Target selection settings
max_iterations          = 10                    # maximum number of iterations for target placement optimization

# Model specific settings (depend on how the model was trained)

# WG model (YOLOv8)
WG_model_file = '2024_07_26_lamella_detect_400nm_yolo8.pt'
WG_model_pix_size = 400.0 # nm/px
WG_model_sidelen = 1024 # px
WG_model_categories = ["broken",           "contaminated",     "good",             "thick",            "wedge",             "gone"          ] 
WG_model_gui_colors = [(255, 125, 0, 255), (255, 215, 0, 255), (50, 150, 50, 255), (59, 92, 128, 255), (87, 138, 191, 255), (200, 0, 0, 255)]
WG_model_nav_colors = [0,                  3,                  1,                  2,                  2,                   0               ]

# Category name     Description                                                         Color
# ----------------- ------------------------------------------------------------------- ----------
# good:             intact lamellae with good thickness                                 green (1)
# thick:            dark lamellae                                                       blue (2)
# wedge/unfinished: lamellae that are good on one end but thick on other end            blue (2)
# contaminated:     lamellae with visible ice blobs                                     yellow (3)
# broken:           lamella with crack, only parts remaining, only one side attached    orange (0)
# gone:             trench without lamella                                              red (0)

# MM model (nnUNet)
MM_model_script = "run_nnUNet.py"
MM_model_folder = 'model'
MM_model_folds = [0, 1, 2, 3, 4]
MM_model_pix_size = 22.83 / 10  # nm/px



######################################
# Self-check (NO CHANGES BELOW HERE) #
######################################
#               #####                #
#             ##     ##              #
#            ## #   # ##             #
#            ##   #   ##             #
#            ## #   # ##             #
#             ##     ##              #
#               #####                #
######################################
import sys
import shutil
from pathlib import Path

config_backup = Path(__file__).parent / "models" / "config.py"
if config_backup.exists():
    from SPACEtomo.models import config as backup
    if WG_model_file != backup.WG_model_file or MM_model_folder != backup.MM_model_folder:
        print(f"WARNING: Loading backup config file from previous SPACEtomo version. Please run the command/script again!")

        # Add any newly added settings to backup config
        # Identify missing variables
        missing_vars = {var: globals()[var] for var in globals() if not var.startswith(("__", "sys", "shutil", "Path", "backup")) and not hasattr(backup, var)}

        # Read the content of the backup file
        with open(config_backup, "r") as f:
            content = f.readlines()

        # Find the index of the "Self-check" section
        self_check_index = next((i for i, line in enumerate(content) if "Self-check" in line), len(content)) - 1

        # Insert the missing variable definitions before the "Self-check" section
        new_vars_content = [f"{var} = {repr(value)} # New setting was added automatically by update\n" for var, value in missing_vars.items()]
        updated_content = content[:self_check_index] + new_vars_content + content[self_check_index:]

        # Write the updated content back to the backup file
        with open(config_backup, "w") as f:
            f.writelines(updated_content)

        # Overwrite the current file with the updated backup file
        shutil.copy(config_backup, __file__)
        sys.exit()

if Path(WG_model_file).exists():
    NO_WG_MODEL = False
else:
    NO_WG_MODEL = True

if Path(MM_model_folder).exists():
    NO_MM_MODEL = False
else:
    NO_MM_MODEL = True
