#!/usr/bin/env python
# ===================================================================
# ScriptName:   setup_model
# Purpose:      Download model and adjust config file.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/21
# Revision:     v1.2
# Last Change:  2024/09/24: added more model management (list, activate, remove, clear)
#               2024/09/04: fixed literal strings for model paths
# ===================================================================

import argparse
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse
from zipfile import ZipFile

from SPACEtomo.modules.utils import log, loadDatasetJson
from SPACEtomo import config

SPACE_DIR = Path(__file__).parent

ACCEPTED_EXT = [".pt", ".zip"]

def checkName(dir, name):
    """Checks if name already exists as model."""

    models = list(dir.glob(name + "*"))
    if models:
        log(f"ERROR: Model with name {models[0].stem} already exists! Please choose another name!")
        return False
    return True

def checkType(model_file):
    """Checks file extension to determine type of model."""

    if model_file.suffix == ".zip" or model_file.is_dir():
        model_type = "unet"
    elif model_file.suffix == ".pt":
        model_type = "yolo"
    else:
        log(f"ERROR: Currently models are only recognized as .pt or .zip files.")
        return False
    return model_type

def checkValidityUnet(model_dir):
    """Looks for essential files for model."""

    dataset_file = sorted(model_dir.glob("**/dataset.json"))
    folds = sorted(model_dir.glob("**/fold_*"))
    if not dataset_file or not folds:
        log(f"ERROR: {model_dir.name} is not a valid model and will be removed!")
        shutil.rmtree(model_dir)
        return False
    return dataset_file[0]

def cleanModelDir(model_dir):
    models = list(model_dir.glob("*"))
    for model in models:
        if model.name.endswith(".zip") or model.name == "unzip":
            model.unlink()
            log(f"WARNING: Removed failed model import [{model.name}]!")

def downloadModel(url, model_dir, model_name):
    """Downloads and saves model file."""

    url_path = Path(urlparse(url).path)

    # Check extension
    if url_path.suffix in ACCEPTED_EXT:

        # Check name
        if not model_name:
            model_name = url_path.stem
            if not checkName(model_dir, model_name):
                return False

        # Save model
        model_file = model_dir / (model_name + url_path.suffix)
        log(f"Downloading...")
        urlretrieve(url, model_file)
        return model_file
    else:
        log(f"ERROR: File type not known or not supported. Please provide the direct link to the model file {ACCEPTED_EXT}!")
        return False

def collectMetaDataYolo(file_name):
    """Collects meta data."""

    pix_size = None
    sidelen = None
    # Try to extract meta_data from filename
    cols = file_name.split("_")
    for col in cols:
        if "nm" in col:
            val = col.split("nm")[0]
            try:
                pix_size = float(val)
            except:
                pix_size = None
        elif "sl" in col:
            val = col.split("sl")[0]
            try:
                sidelen = float(val)
            except:
                sidelen = None

    meta_data = {}
    meta_data["WG_model_pix_size"] = pix_size
    meta_data["WG_model_sidelen"] = sidelen
    
    return meta_data

def collectMetaDataUnet(model_path):
    """Collects meta data."""

    meta_data = {}
    # Try to extract meta_data from dataset.json
    dataset_file = sorted(model_path.glob("**/dataset.json"))
    dataset_file = dataset_file[0]
    cats, pix_size, img_num = loadDatasetJson(dataset_file)
    folds = sorted(model_path.glob("**/fold_*"))
    meta_data["MM_model_folds"] = [fold.name.split("_")[-1] for fold in folds]
    meta_data["MM_model_pix_size"] = pix_size

    return meta_data

def listModels(model_dir):
    """Lists all models in models dir."""

    model_list = sorted(model_dir.glob("*"))
    log(f"\n############################################\n Registered SPACEtomo models: (* = active)")
    for model_file in model_list:
        if model_file.name.startswith("."):
            continue
        # Mark model as selected if config model path exists and is same as model_file
        if (not config.NO_WG_MODEL and Path(config.WG_model_file).samefile(model_file)) or (not config.NO_MM_MODEL and Path(config.MM_model_folder).samefile(model_file)):
            selected = "   *"
        else:
            selected = "    "
        log(f"{selected} {model_file.name}")
    if not model_list: 
        log(f"-")
    log("############################################\n")
    return

def activateModel(model_file, alternative_name=""):
    """Activates model by supplying path and meta_data to config."""
    
    # Check model type and validity
    model_type = checkType(model_file)
    if model_type == "unet" and not checkValidityUnet(model_file):
        return

    # Extract meta data
    if model_type == "yolo":
        name = alternative_name if alternative_name else model_file.name
        meta_data = collectMetaDataYolo(name)
    elif model_type == "unet":
        meta_data = collectMetaDataUnet(model_file)
    else:
        return

    # Adjust config
    with open(SPACE_DIR / "config.py", "r") as f:
        config_lines = f.readlines()

    new_config = []
    for line in config_lines:
        # YOLO
        if model_type == "yolo":
            # Model file name
            if line.strip().startswith("WG_model_file"):
                new_config.append(f"WG_model_file = \"{model_file.as_posix()}\"\n")
            # Pix size
            elif line.strip().startswith("WG_model_pix_size") and meta_data["WG_model_pix_size"]:
                new_config.append(f"WG_model_pix_size = {meta_data['WG_model_pix_size']} # nm/px\n")
            # Side length
            elif line.strip().startswith("WG_model_sidelen") and meta_data["WG_model_sidelen"]:
                new_config.append(f"WG_model_sidelen = '{meta_data['WG_model_sidelen']}'\n")
            else:
                new_config.append(line)

        # nnU-Net
        elif model_type == "unet":
            # Model folder
            if line.strip().startswith("MM_model_folder"):
                new_config.append(f"MM_model_folder = \"{model_file.as_posix()}\"\n")
            # Folds MM_model_folds = [0, 1, 2, 3, 4]
            elif line.strip().startswith("MM_model_folds") and meta_data["MM_model_folds"]:
                new_config.append(f"MM_model_folds = [{', '.join(meta_data['MM_model_folds'])}]\n")
            # Pix size
            elif line.strip().startswith("MM_model_pix_size") and meta_data["MM_model_pix_size"]:
                new_config.append(f"MM_model_pix_size = {meta_data['MM_model_pix_size']} # nm/px\n")
            else:
                new_config.append(line)

    with open(SPACE_DIR / "config.py", "w") as f:
        f.writelines(new_config)

    # Import new config values
    from SPACEtomo import config as config_new

    if model_type == "yolo":
        log(f"NOTE: YOLO model [{model_file.name}] was successfully activated.\nNOTE: Please double check the validity of the config values for:\nWG_model_pix_size = {config_new.WG_model_pix_size}   # nm/px\nWG_model_sidelen = {config_new.WG_model_sidelen}\nWG_model_categories = {config_new.WG_model_categories}")
    elif model_type == "unet":
        log(f"NOTE: Segmentation model [{model_file.name}] was successfully activated.\nNOTE: Please double check the validity of the config value for\nMM_model_pix_size = {config_new.MM_model_pix_size}   # nm/px")

def removeModel(model_file):
    """Deletes model and activates alternative if possible."""

    model_type = checkType(model_file)

    if model_file.is_dir():
        shutil.rmtree(model_file)
    else:
        model_file.unlink()
    log(f"NOTE: Model [{model_file.name}] was successfully removed!")

    model_list = sorted(model_file.parent.glob("*"))
    for model in model_list:
        if (model.is_dir() and model_type == "unet") or (model.suffix == ".pt" and model_type == "yolo"):
            activateModel(model)
            return

def addModel(model_dir, model_url, model_path, model_name=None):
    """Imports model file and unpacks it if necessary."""

    # Check if name exists
    if model_name:
        if not checkName(model_dir, model_name): 
            return

    # Check URL and download file
    if model_url:
        model_file = downloadModel(model_url, model_dir, model_name)
        if not model_file:
            return
        original_file_name = Path(urlparse(model_url).path).stem

        # Check name
        if not model_name:
            model_name = original_file_name

    # Check path and copy file
    elif model_path:
        file_path = Path(model_path)
        original_file_name = file_path.stem

        # Check extension
        if file_path.suffix in ACCEPTED_EXT:

            # Check name
            if not model_name:
                model_name = file_path.stem
                if not checkName(model_dir, model_name):
                    return

            # Copy model
            log(f"Importing...")
            model_file = model_dir / (model_name + file_path.suffix)
            shutil.copy(file_path, model_file)
        else:
            log(f"ERROR: File type not known or not supported. Please provide a path to the model file {ACCEPTED_EXT}!")
            return
    else:
        log("ERROR: Need either the URL of a model file to download or the path to a downloaded model file.")
        return

    if model_file.suffix == ".zip":
        # Create temp dir
        temp_dir = model_file.parent / "unzip"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        # Unzip model
        log(f"Extracting...")
        with ZipFile(model_file, "r") as zip_file:
            zip_file.extractall(temp_dir)
        
        # Delete zip file
        model_file.unlink()

        dataset_file = checkValidityUnet(temp_dir)
        if not dataset_file:
            return
        
        # Move to new model dir
        model_file = model_file.parent / model_name
        shutil.move(dataset_file.parent, model_file)
        temp_dir.rmdir()

    log(f"NOTE: New model {model_name} was imported!")

    activateModel(model_file, original_file_name)

def clearModels(model_dir):
    #TODO
    pass

def main():
    # Process arguments
    parser = argparse.ArgumentParser(description='Downloads or imports model and adjusts config file accordingly.')
    parser.add_argument('task', type=str, nargs="?", default='list', help='Task to be run: list (list models), activate (activate model), add (add a model), remove (remove a model by name) or clear (remove all models).')
    parser.add_argument('--url', dest='url', type=str, default=None, help='URL to trained YOLO model file [.pt] or zipped nnUnet model folder [.zip].')
    parser.add_argument('--path', dest='path', type=str, default=None, help='Path to downloaded trained YOLO model file [.pt] or zipped nnUnet model folder [.zip].')
    parser.add_argument('--name', dest='name', type=str, default=None, help='A unique name for the model.')
    args = parser.parse_args()

    # Make model folder
    model_dir = SPACE_DIR / "models"
    if not model_dir.exists():
        model_dir.mkdir()

    # Remove any non-model files from failed imports
    cleanModelDir(model_dir)

    # List models
    if args.task == "list":
        listModels(model_dir)
        return
    
    # Activate model
    if args.task == "activate":
        if args.name:
            if (model_dir / args.name).exists():
                activateModel(model_dir / args.name)
                return
            else:
                log(f"ERROR: Model with name [{args.name}] does not exist!")
                return
        else:
            log(f"ERROR: Please provide model name using --name!")
        
    # Add/import model and activate
    if args.task == "add":
        addModel(model_dir, args.url, args.path, args.name)
        return
    
    # Remove model
    if args.task == "remove":
        if args.name:
            if (model_dir / args.name).exists():
                removeModel(model_dir / args.name)
                return
            else:
                log(f"ERROR: Model with name [{args.name}] does not exist!")
        else:
            log(f"ERROR: Please provide model name using --name!")

    # Clear models
    if args.task == "clear":
        shutil.rmtree(model_dir)
        log(f"NOTE: All existing models were deleted. Please import new models to use SPACEtomo!")

if __name__ == "__main__":
    main()