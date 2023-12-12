#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_nnUnet
# Purpose:      Runs a target selection deep learning model on medium mag montages of lamella and generates a segmentation that can be used for PACEtomo target selection.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/19
# Revision:     v1.0beta
# Last Change:  2023/11/06: changed to log output
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
from ultralytics import YOLO, settings

# Check if filename was given
if len(sys.argv) == 2:
    montage_file = sorted(glob.glob(sys.argv[1]))[0]
else:
    print ("Usage: python " + sys.argv[0] + " [input]")
    sys.exit("Missing arguments!")

# Set directory to dir of given filename
CUR_DIR = os.path.dirname(montage_file)
SPACE_DIR = os.path.dirname(__file__)

# Set vars for nnUNet (not really used) before import to avoid errors
os.environ["nnUNet_raw"] = CUR_DIR
os.environ["nnUNet_preprocessed"] = CUR_DIR
os.environ["nnUNet_results"] = CUR_DIR

device = torch.device('cuda')
from nnunetv2.inference import predict_from_raw_data as predict

# Start log file
logging.basicConfig(filename=os.path.splitext(montage_file)[0] + "_SPACE.log", level=logging.INFO, format='')
logging.info("Processing " + montage_file)

# Import model configs (needs SPACE folder from settings file)
sys.path.insert(len(sys.path), SPACE_DIR)
import SPACEtomo_config as config
logging.info("Loaded config.")

# Read segmentation classes
with open(os.path.join(SPACE_DIR, config.MM_model_folder, "dataset.json"), "r") as f:
    dataset_json = json.load(f)
classes = dataset_json["labels"]
logging.info("Loaded class labels.")

# Load YOLO model
WG_model = YOLO(os.path.join(SPACE_DIR, config.WG_model_file))  # load a custom model
settings.update({'sync': False})    # no tracking by google analytics

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
# Add padding to emulate grid map
padded_map = np.zeros((config.WG_model_sidelen, config.WG_model_sidelen))
padded_map[0: pixel_map.shape[0], 0: pixel_map.shape[1]] = pixel_map

# Run YOLO model to detect bbox
yolo_input = np.dstack([padded_map, padded_map, padded_map])
results = WG_model(yolo_input)

# Check and upscale resulting box
if len(results[0].boxes) > 0:
    bbox = np.array(results[0].boxes.xyxy.to("cpu"))

    bbox[:, 0] = (bbox[:, 0]) * pixel_shape[0]
    bbox[:, 2] = (bbox[:, 2]) * pixel_shape[0]
    bbox[:, 1] = (bbox[:, 1]) * pixel_shape[1]
    bbox[:, 3] = (bbox[:, 3]) * pixel_shape[1]

    cat = np.reshape(results[0].boxes.cls.to("cpu"), (bbox.shape[0], 1))
    conf = np.reshape(results[0].boxes.conf.to("cpu"), (bbox.shape[0], 1))
    bbox = np.hstack([bbox, cat, conf])[0]

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
input_img = np.array(crop, dtype=np.float16)[np.newaxis, np.newaxis, :, :]

# Use temp name to pad later
if bounds is None:
    out_name = os.path.splitext(montage_file)[0] + "_seg"
else:
    out_name = os.path.splitext(montage_file)[0] + "_segtemp"

# Do nnUNet inference
predictor = predict.nnUNetPredictor(
    tile_step_size=0.5,
    perform_everything_on_gpu=True,
    device=device,
    allow_tqdm=False
)

predictor.initialize_from_trained_model_folder(
    os.path.join(SPACE_DIR, config.MM_model_folder),
    config.MM_model_folds,
    checkpoint_name="checkpoint_final.pth"
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
