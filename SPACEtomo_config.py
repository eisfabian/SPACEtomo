#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_config
# Purpose:      Configurations for deep learning models used in SPACEtomo.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/10/04
# Revision:     v1.1
# Last Change:  2024/03/12: removed max_runs
# ===================================================================

# Model specific settings (depend on how the model was trained)
# WG model (YOLOv8)
WG_model_file = "2023_11_16_lamella_detect_400nm_yolo8.pt"
WG_model_pix_size = 400         # nm / px
WG_model_sidelen = 1024
WG_model_categories = ["broken", "contaminated", "good", "thick"]
WG_model_colors = ["red", "yellow", "green", "orange"]
WG_model_nav_colors = [0, 2, 1, 3]

# MM model (nnUNet)
MM_model_script = "SPACEtomo_nnUNet.py"
MM_model_folder = "model"
MM_model_folds = [0, 1, 2, 3, 4]
MM_model_pix_size = 22.83 / 10  # nm / px