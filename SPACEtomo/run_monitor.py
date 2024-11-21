#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_monitor
# Purpose:      Monitors an external directory and runs a target selection deep learning model on medium mag montages of lamella and generates a segmentation that can be used for PACEtomo target selection.
#               More information at http://github.com/eisfabian/PACEtomo
# Usage:        python SPACEtomo_monitor.py [--dir MAP_DIR] [--gpu 0,1,2,3] [--plot]
# Author:       Fabian Eisenstein
# Created:      2023/10/05
# Revision:     v1.2
# Last Change:  2024/08/20: adjusted to refactor
#               2024/04/09: deleted run file upon start to avoid problems upon restart
#               2024/04/04: added call to tgt GUI
#               2024/03/25: fixes after Krios 3 test
#               2024/03/12: replaced script settings with arguments, added stepwise inclusion of params as they appear
#               2024/03/11: simplification and fixes
#               2024/03/07: rewrote updateQueue function, changed runs and queue to single json file, moved functions to functions_ext
#               2024/02/14: added WG map monitoring and lamella detection, set timer step down to 1 s, added multi GPU usage
#               2023/12/04: added check for memory error
#               2023/11/15: added file being added to queue before submitting run to give time to finish copying
#               2023/10/31: removed wait for settings
#               2023/10/18: fixes after local test
#               2023/10/05: 
# ===================================================================

import sys
import time
import argparse
import subprocess
from pathlib import Path
import torch

from SPACEtomo.modules.mod_wg import WGModel
from SPACEtomo.modules.scope import ImagingParams
import SPACEtomo.modules.ext as space_ext
from SPACEtomo.modules.utils import log
from SPACEtomo import __version__

def main():
    # Process arguments
    parser = argparse.ArgumentParser(description='Monitors an external directory and runs lamella detection, lamella segmentation and target selection on appropiate files.')
    parser.add_argument('--dir', dest='map_dir', type=str, default=None, help='Absolute path to folder to be monitored. This should be the same directory that was set in SPACEtomo on the SerialEM PC. (Default: Folder this script is run from)')
    parser.add_argument('--gpu', dest='gpu', type=str, default="0", help='Comma-separated IDs of GPUs to use. (Default: 0)')
    parser.add_argument('--plot', dest='save_plot', action='store_true', help='Create plots of intermediate target selection steps (slow).')
    args = parser.parse_args()

    if args.map_dir is not None: 
        if Path(args.map_dir).exists():
            MAP_DIR = Path(args.map_dir)
        else:
            log("ERROR: Folder does not exist!")
            sys.exit()
    else:
        MAP_DIR = Path.cwd()

    gpus = args.gpu.split(",")
    gpu_list = [int(gpu) for gpu in gpus]
    gpu_list = gpu_list if torch.cuda.is_available() else []        # Check GPU availability

    save_plot = args.save_plot

    # Load model
    WG_model = WGModel()

    MM_model = None
    mic_params = None
    tgt_params = None

    # Move previous run file
    if (MAP_DIR / "SPACE_runs.json").exists():
        (MAP_DIR / "SPACE_runs.json").replace(MAP_DIR / "SPACE_runs.json~")

    # Indicator for monitors running (lamella detection, lamella segmentation, target selection)
    status = "(oxx)"

    # Set start time
    start_time = time.time()
    next_time = start_time + 60

    log(f"SPACEtomo Version {__version__}\n")
    log(f"Start monitoring {MAP_DIR} for all maps... {status}")
    log(f"Using GPUs: {gpu_list}\n")

    while True:
        now = time.time()
        if now > next_time:
            next_time = now + 60
            log(f"##### Running for {int(round((now - start_time) / 60))} min... {status} #####")
        
        # Check if mic_params exist
        if mic_params is None and (MAP_DIR / "mic_params.json").exists():
            status = "(oox)"
            log(f"NOTE: Microscope parameters found. Including lamella segmentation. {status}")
            # Instantiate mic params from settings
            mic_params = ImagingParams(None, None, None, file_dir=MAP_DIR)        
            # Load model
            MM_model = space_ext.MMModel()

        # Check if tgt_params exist
        if tgt_params is None and (MAP_DIR / "tgt_params.json").exists():
            status = "(ooo)"
            log(f"NOTE: Target parameters found. Including target setup. {status}")
            # Reimport mic params to get Rec params
            mic_params = space_ext.MicParams_ext(MAP_DIR) 
            MM_model.setDimensions(mic_params)
            # Instantiate tgt params from settings
            tgt_params = space_ext.TgtParams(file_dir=MAP_DIR, MM_model=MM_model)

            log("Opening target selection GUI for inspection...")
            subprocess.Popen([sys.executable, Path(__file__).parent / "GUI.py", "targets", MAP_DIR])

        # Run queue
        space_ext.updateQueue(MAP_DIR, WG_model, MM_model, mic_params, tgt_params, gpu_list, save_plot)
        time.sleep(5)

if __name__ == "__main__":
    main()