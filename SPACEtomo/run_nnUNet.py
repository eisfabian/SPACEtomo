#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_nnUnet
# Purpose:      Runs a target selection deep learning model on medium mag montages of lamella and generates a segmentation that can be used for PACEtomo target selection.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/19
# Revision:     v1.2
# Last Change:  2024/08/21: adjusted to refactor, made callable
#               2024/03/27: fixed device for lamella detection (again??)
#               2024/03/25: fixes after Krios 3 test
#               2024/03/11: Name fixes, added checkpoint file check
#               2024/02/15: added cpu inference if cuda not available (Patrick Cleeve)
#               2024/02/14: added import of SPACE ext functions
#               2023/11/06: changed to log output
#               2023/10/31: removed loading of settings and assume script is located inside SPACE_DIR
#               2023/10/18: fixed paths for external runs
#               2023/10/05: load config to get model data
#               2023/10/03: fixes after Krios test, padding to YOLO model side length	
#               2023/09/26: added YOLO bbox for lamella, read settings from settings file
#               2023/09/22: switched to nnUnet and simplified script
#               2023/07/10: more robust masking, binned heat maps, cleanup
#               2023/07/07: tested and fixes on Krios
#               2023/07/05: conversion to background process, implemented masked prediction speedup
#               2023/05/19: first test in SerialEM
# ===================================================================

import os
import sys
import time
import json
import logging
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pathlib import Path
import torch

import SPACEtomo.config as config
from SPACEtomo.modules.mod_wg import WGModel

def main():
    # Check if filename was given
    if len(sys.argv) == 2:
        montage_file = Path(sys.argv[1])
        if not montage_file.exists():
            print("ERROR: Map file does not exist!")
            sys.exit()
    else:
        print (f"Usage: python {sys.argv[0]} [input]")
        sys.exit("Missing arguments!")

    # Set directory to dir of given filename
    MAP_DIR = montage_file.parent

    # Set vars for nnUNet (not really used) before import to avoid errors
    os.environ["nnUNet_raw"] = str(MAP_DIR)
    os.environ["nnUNet_preprocessed"] = str(MAP_DIR)
    os.environ["nnUNet_results"] = str(MAP_DIR)

    from nnunetv2.inference import predict_from_raw_data as predict

    # Start log file
    logging.basicConfig(filename=MAP_DIR / (montage_file.stem + "_SPACE.log"), level=logging.INFO, format='')
    logging.info(f"Processing {montage_file}...")

    # Read segmentation classes
    with open(Path(config.MM_model_folder) / "dataset.json", "r") as f:
        dataset_json = json.load(f)
    classes = dataset_json["labels"]
    logging.info("Loaded class labels.")

    # Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   # dynamically set device
    is_cuda_device = True if device.type == "cuda" else False
    logging.info(f"Cuda available: {is_cuda_device}")

    # Load YOLO model
    WG_model = WGModel()

    # Load map
    start_time = time.time()
    montage_map = np.array(Image.open(montage_file))
    time_point1 = time.time()
    logging.info(f"Map was loaded in {int(time_point1 - start_time)} s.")

    # Rescale map for YOLO lamella bbox detection
    full_shape = np.array(montage_map.shape)
    division = np.round(full_shape * config.MM_model_pix_size / config.WG_model_pix_size).astype(int)
    pixel_shape = full_shape // division
    pixel_map = np.zeros(division)
    for i in range(division[0]):
        for j in range(division[1]):
            pixel = montage_map[i * pixel_shape[0]:(i + 1) * pixel_shape[0], j * pixel_shape[1]:(j + 1) * pixel_shape[1]]
            pixel_map[i, j] = np.mean(pixel)

    # Save rescaled map
    montage_img = Image.fromarray(pixel_map.astype(np.uint8))
    montage_img.save(MAP_DIR / (montage_file.stem + "_wg.png"))

    # Detect lamella
    bboxes = WG_model.findLamellae(MAP_DIR, montage_file.stem, save_boxes=True, device=device)

    # Check and upscale resulting box
    if len(bboxes) > 0:
        #bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)     # sort according to probability
        bboxes.sortBy("prob", reverse=True)                 # sort according to probability
        bbox = bboxes.boxes[0]

        # Scale coords up to full MMM size
        #bbox[0] = (bbox[0]) * pixel_shape[0]
        #bbox[2] = (bbox[2]) * pixel_shape[0]
        #bbox[1] = (bbox[1]) * pixel_shape[1]
        #bbox[3] = (bbox[3]) * pixel_shape[1]
        bbox = bbox * pixel_shape[0]        # pixel shape should always be square

        logging.info(f"Lamella bounding box: {bbox.xyxycc[:4]}")
        logging.info(f"Lamella was categorized as: {config.WG_model_categories[bbox.cat]} ({round(bbox.prob * 100, 1)} %)")

        bounds = np.round(bbox.xyxycc[:4]).astype(int)
        crop = montage_map[bounds[1]:bounds[3], bounds[0]:bounds[2]]

        time_point2 = time.time()
        logging.info(f"Bounding box was detected in {int(time_point2 - time_point1)} s.")
    else:
        bounds = None
        crop = montage_map

        time_point2 = time.time()
        logging.info(f"WARNING: No bounding box was detected in {int(time_point2 - time_point1)} s.")
        logging.info("Using whole montage map...")

    # Setup input and output for nnUNet
    dtype=np.float16 if is_cuda_device else np.float32                                      # half precision for cuda only
    input_img = np.array(crop, dtype=dtype)[np.newaxis, np.newaxis, :, :]

    # Check checkpoint files
    if (Path(config.MM_model_folder) / f"fold_{config.MM_model_folds[0]}" / "checkpoint_best.pth").exists():
        checkpoint_file = "checkpoint_best.pth"
    else:
        checkpoint_file = "checkpoint_final.pth"

    # Use temp name to pad later
    if bounds is None:
        out_name = MAP_DIR / (montage_file.stem + "_seg")
    else:
        out_name = MAP_DIR / (montage_file.stem + "_segtemp")

    # Do nnUNet inference
    predictor = predict.nnUNetPredictor(
        tile_step_size=0.5,
        perform_everything_on_device=is_cuda_device,
        device=device,
        allow_tqdm=False
    )

    predictor.initialize_from_trained_model_folder(
        config.MM_model_folder,
        config.MM_model_folds,
        checkpoint_name=checkpoint_file
    )

    logging.info("Model loaded.")
    logging.info("Starting prediction...")

    predictor.predict_single_npy_array(
        input_image=input_img, 
        image_properties={"spacing": [999, 1, 1]},
        output_file_truncated=str(out_name),
    )

    logging.info("Postprocessing...")

    # Pad segmentation to original map size
    if bounds is not None:
        segmentation = np.array(Image.open(str(out_name) + ".png"))
        padding = ((max(0, bounds[1]), max(0, full_shape[0] - bounds[3])), (max(0, bounds[0]), max(0, full_shape[1] - bounds[2])))
        segmentation = np.pad(segmentation, padding, constant_values=classes["black"])
        seg_out = Image.fromarray(np.uint8(segmentation))
        seg_out.save(MAP_DIR / (montage_file.stem + "_seg.png"))
        (out_name.parent / (out_name.stem + ".png")).unlink()

    logging.info(f"Prediction was completed in {round((time.time() - time_point2) / 60, 1)} min.")
    logging.info(f"Finished processing {montage_file.name}.")

if __name__ == "__main__":
    main()