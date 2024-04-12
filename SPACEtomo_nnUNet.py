#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_nnUnet
# Purpose:      Runs a target selection deep learning model on medium mag montages of lamella and generates a segmentation that can be used for PACEtomo target selection.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/19
# Revision:     v1.1
# Last Change:  2024/03/27: fixed device for lamella detection (again??)
# ===================================================================

import os
import sys
import glob
import time
import json
import logging
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch

# Check if filename was given
if len(sys.argv) == 2:
    montage_file = sorted(glob.glob(sys.argv[1]))[0]
else:
    print ("Usage: python " + sys.argv[0] + " [input]")
    sys.exit("Missing arguments!")

# Set directory to dir of given filename
MAP_DIR = os.path.dirname(montage_file)
SPACE_DIR = os.path.dirname(__file__)

# Set vars for nnUNet (not really used) before import to avoid errors
os.environ["nnUNet_raw"] = MAP_DIR
os.environ["nnUNet_preprocessed"] = MAP_DIR
os.environ["nnUNet_results"] = MAP_DIR

from nnunetv2.inference import predict_from_raw_data as predict

# Start log file
logging.basicConfig(filename=os.path.splitext(montage_file)[0] + "_SPACE.log", level=logging.INFO, format='')
logging.info("Processing " + montage_file)

# Import model configs (needs SPACE folder from settings file)
sys.path.insert(len(sys.path), SPACE_DIR)
import SPACEtomo_config as config
import SPACEtomo_functions_ext as space_ext
logging.info("Loaded config.")

# Read segmentation classes
with open(os.path.join(SPACE_DIR, config.MM_model_folder, "dataset.json"), "r") as f:
    dataset_json = json.load(f)
classes = dataset_json["labels"]
logging.info("Loaded class labels.")

# Check device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   # dynamically set device
is_cuda_device = True if device.type == "cuda" else False 

# Load YOLO model
WG_model = space_ext.WGModel()

# Load map
start_time = time.time()
montage_map = np.array(Image.open(montage_file))
time_point1 = time.time()
logging.info("Map was loaded in " + str(int(time_point1 - start_time)) + " s.")

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
montage_img.save(os.path.splitext(montage_file)[0] + "_wg.png")

# Detect lamella
bboxes = space_ext.findLamellae(MAP_DIR, os.path.splitext(os.path.basename(montage_file))[0], WG_model, save_boxes=True, device=device, plot=False)

# Check and upscale resulting box
if len(bboxes) > 0:
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)     # sort according to probability
    bbox = bboxes[0]

    # Scale coords up to full MMM size
    bbox[0] = (bbox[0]) * pixel_shape[0]
    bbox[2] = (bbox[2]) * pixel_shape[0]
    bbox[1] = (bbox[1]) * pixel_shape[1]
    bbox[3] = (bbox[3]) * pixel_shape[1]

    logging.info("Lamella bounding box: " + str(bbox))
    logging.info("Lamella was categorized as: " + str(config.WG_model_categories[int(bbox[4])]) + " (" + str(round(bbox[5] * 100, 1)) + " %)")

    bounds = np.round(bbox[0:4]).astype(int)
    crop = montage_map[bounds[1]:bounds[3], bounds[0]:bounds[2]]

    time_point2 = time.time()
    logging.info("Bounding box was detected in " + str(int(time_point2 - time_point1)) + " s.")
else:
    bounds = None
    crop = montage_map

    time_point2 = time.time()
    logging.info("WARNING: No bounding box was detected in " + str(int(time_point2 - time_point1)) + " s.")
    logging.info("Using whole montage map...")

# Setup input and output for nnUNet
dtype=np.float16 if is_cuda_device else np.float32                                      # half precision for cuda only
input_img = np.array(crop, dtype=dtype)[np.newaxis, np.newaxis, :, :]

# Check checkpoint files
if os.path.exists(os.path.join(SPACE_DIR, config.MM_model_folder, "fold_0", "checkpoint_best.pth")):
    checkpoint_file = "checkpoint_best.pth"
else:
    checkpoint_file = "checkpoint_final.pth"

# Use temp name to pad later
if bounds is None:
    out_name = os.path.splitext(montage_file)[0] + "_seg"
else:
    out_name = os.path.splitext(montage_file)[0] + "_segtemp"

# Do nnUNet inference
predictor = predict.nnUNetPredictor(
    tile_step_size=0.5,
    perform_everything_on_device=is_cuda_device,
    device=device,
    allow_tqdm=False
)

predictor.initialize_from_trained_model_folder(
    os.path.join(SPACE_DIR, config.MM_model_folder),
    config.MM_model_folds,
    checkpoint_name=checkpoint_file
)

logging.info("Model loaded.")
logging.info("Starting prediction...")

predictor.predict_single_npy_array(
    input_image=input_img, 
    image_properties={"spacing": [999, 1, 1]},
    output_file_truncated=out_name,
)

logging.info("Postprocessing...")

# Pad segmentation to original map size
if bounds is not None:
    segmentation = np.array(Image.open(out_name + ".png"))
    padding = ((max(0, bounds[1]), max(0, full_shape[0] - bounds[3])), (max(0, bounds[0]), max(0, full_shape[1] - bounds[2])))
    segmentation = np.pad(segmentation, padding, constant_values=classes["black"])
    seg_out = Image.fromarray(np.uint8(segmentation))
    seg_out.save(os.path.splitext(montage_file)[0] + "_seg.png")
    os.remove(out_name + ".png")

logging.info("Prediction was completed in " + str(round((time.time() - time_point2) / 60, 1)) + " min.")
logging.info("Finished processing " + os.path.basename(montage_file) + ".")
