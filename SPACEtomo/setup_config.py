#!/usr/bin/env python
# ===================================================================
# ScriptName:   setup_model
# Purpose:      Get copy of config file or set file as config file.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/21
# Revision:     v1.2
# Last Change:  
# ===================================================================

import shutil
import argparse
from pathlib import Path

from SPACEtomo.modules.utils import log
import SPACEtomo.config as config

def main():
    # Process arguments
    parser = argparse.ArgumentParser(description='Makes copy of config file to be edited.')
    parser.add_argument('task', type=str, help="Use 'get' to get a copy of the config file and 'set' to use a file as config file.")
    parser.add_argument('--file', type=str, default="config.py", help="Select a file to use as SPACEtomo config file.")
    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file)
        config_path = Path(__file__).parent / "config.py"

        # Make copy of config file
        if args.task == "get":
            if file_path.exists():
                log(f"File [{file_path.name}] already exists! Please choose another file name!")
                return
            
            shutil.copy(config_path, file_path)

            log(f"Config file was successfully copied to {file_path}!")   

        # Set file as config file and check if all values are present
        elif args.task == "set":
            if not file_path.exists():
                log(f"File [{file_path.name}] does not exist! Please provide a valid config file!")
                return

            # Get old config entries
            config_vars = [var for var in dir(config) if not var.startswith("__")]
            
            # Get new config entries
            with open(file_path, "r") as f:
                config_content = f.read()

            # Compare vars
            missing_vars = []
            for var in config_vars:
                if not var in config_content:
                    missing_vars.append(var)

            if missing_vars:
                log("ERROR: New config file is missing variables:\n" + "\n".join(missing_vars))
                return
            
            # Check if model files exist
            for line in config_content.splitlines():
                if ("WG_model_file" in line or "MM_model_folder" in line) and not "exists" in line:
                    model_file = Path(line.split("=")[-1].strip().strip("r").replace("'", "").replace('"', ""))
                    if not model_file.exists():
                        log(f"ERROR: New config file contains invalid model file: {model_file}")
                        return
            
            # Overwrite config file
            shutil.copy(file_path, config_path)

            log(f"Config file was successfully updated!")       

        # Toggle dummy mode
        elif args.task == "dummy":
            # Adjust config
            with open(config_path, "r") as f:
                config_lines = f.readlines()

            new_config = []
            for line in config_lines:
                if "DUMMY" in line:
                    new_config.append(f"DUMMY = {not config.DUMMY}\n")
                else:
                    new_config.append(line)

            with open(config_path, "w") as f:
                f.writelines(new_config)

            log(f"NOTE: DUMMY mode was set to {not config.DUMMY}!")

        # Toggle debug mode
        elif args.task == "debug":
            # Adjust config
            with open(config_path, "r") as f:
                config_lines = f.readlines()

            new_config = []
            for line in config_lines:
                if "DEBUG" in line:
                    new_config.append(f"DEBUG = {not config.DEBUG}\n")
                else:
                    new_config.append(line)

            with open(config_path, "w") as f:
                f.writelines(new_config)

            log(f"NOTE: DEBUG mode was set to {not config.DEBUG}!")

        else:
            log(f"ERROR: Unknown task {args.task}!")

if __name__ == "__main__":
    main()
