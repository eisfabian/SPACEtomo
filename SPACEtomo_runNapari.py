#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_runNapari
# Purpose:      Loads map and segmentation or layer folder into Napari for manual labeling. Saves layers when closing, which can be imported into SPACEtomo_TI.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/03/14
# Revision:     v1.1
# Last Change:  2024/03/22: fixed classes loaded in separate layers with according color, fixed exporting of layers
# ===================================================================

import os
import sys
import json
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images
import napari

##### SETTINGS #####

color = True                            # use colors from Yeast model (False will let napari determine the colors)

### END SETTINGS ###

# Process arguments
parser = argparse.ArgumentParser(description="Runs Napari and loads segmentation.")
parser.add_argument("--file_path", default="", type=str, help="Absolute path to MM map [png].")
parser.add_argument("--seg_path", default="", type=str, help="Absolute path to segmentation [png].")
parser.add_argument("--folder", default="", type=str, help='Alternatively: Path to folder with image layers.')
args = parser.parse_args()

file_path = args.file_path
seg_path = args.seg_path
layer_folder = args.folder

### FUNCTIONS ###

def loadDatasetJson(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            dataset_json = json.load(f)
        classes = dataset_json["labels"]
        img_num = dataset_json["numTraining"]
        if "pixel_size" in dataset_json.keys():
            pix_size = dataset_json["pixel_size"]
        else:
            pix_size = None

        return classes, pix_size, img_num
    else:
        return None, None, None
    
def loadClasses():
    global CLASSES
    json_file = os.path.join(os.path.dirname(seg_path), "dataset.json")
    if not os.path.exists(json_file):
        json_file = os.path.join(os.path.dirname(file_path), "dataset.json")
        if not os.path.exists(json_file):
            json_file = os.path.join(os.path.dirname(seg_path), os.pardir, "dataset.json")
            if not os.path.exists(json_file):
                json_file = "dataset.json"
                if not os.path.exists(json_file):
                    print("ERROR: No dataset.json file was found!")
                    sys.exit()
            
    CLASSES, *_ = loadDatasetJson(json_file)

### END FUNCTIONS ###

# Colors for figure
class_colors = {"background": (204, 204, 204, 255), "white": (255, 255, 255, 255), "black": (0, 0, 0, 255), "crack": (238, 238, 238, 255), "coating": (85, 85, 85, 255), "cell": (255, 247, 231, 255), "cellwall": (255, 213, 128, 255), "nucleus": (246, 168, 25, 255), "vacuole": (231, 127, 36, 255), "mitos": (201, 47, 40, 255), "lipiddroplets": (216, 88, 40, 255), "vesicles": (216, 88, 40, 255), "multivesicles": (216, 88, 40, 255), "membranes": (216, 88, 40, 255), "dynabeads": (87, 138, 191, 255), "ice": (87, 138, 191, 255), "cryst": (59, 92, 128, 255)}

if (file_path != "" and seg_path == "") or ((file_path == "" and seg_path != "")) or (file_path == "" and seg_path == "" and layer_folder == ""):
    print("ERROR: Check input arguments!")
    sys.exit()

if file_path != "" and not os.path.exists(file_path):
    print("ERROR: " + file_path + " does not exist!")
    sys.exit()

if seg_path != "" and not os.path.exists(seg_path):
    print("ERROR: " + seg_path + " does not exist!")
    sys.exit()

if layer_folder != "" and not os.path.exists(layer_folder):
    print("ERROR: " + layer_folder + " does not exist!")
    sys.exit()


# Start Napari
napari_viewer = napari.Viewer()

# Set color scheme
if color:
    color_map = napari.utils.CyclicLabelColormap(np.array(list(class_colors.values())) / 255)
else:
    color_map = None

# Load from segmentation
if layer_folder == "":
    # Load image
    img_name = os.path.splitext(os.path.basename(file_path))[0].split("_0000")[0]
    img = np.array(Image.open(file_path), dtype=np.uint8)

    # Load segmentation
    seg = np.array(Image.open(seg_path))

    # Load classes
    loadClasses()


    # Add layers
    napari_viewer.add_image(img, name=img_name)


    for i, l in enumerate(tqdm(CLASSES.keys())):
        if CLASSES[l] == 0:
            continue
        label = np.zeros(seg.shape, dtype=np.uint8)
        label[seg == CLASSES[l]] = CLASSES[l]

        napari_viewer.add_labels(label, name=str(len(CLASSES) - CLASSES[l]).zfill(2) + "_" + l, colormap=color_map)

# Load from layers
else:
    layer_list = sorted(glob.glob(os.path.join(layer_folder, "*.png")), reverse=True)

    # Get img name from layer folder since it should be named accordingly
    img_name = os.path.basename(layer_folder)

    layers = []
    layer_names = []
    for l, layer_file in enumerate(tqdm(layer_list)):
        if img_name in os.path.basename(layer_file):
            img = np.array(Image.open(layer_file), dtype=np.uint8)
            if img.ndim > 2:
                img = img[:, :, 0]
            continue
        layer_raw = np.sum(np.array(Image.open(layer_file).convert("RGB")), axis=-1)    # sum up all colors to make sure semitransparent layers are considered properly

        layer = np.zeros(layer_raw.shape, dtype=bool)
        layer[layer_raw < 3 * 255] = True            # anything not white
        layer[layer_raw == 3 * 255] = False          # anything white
        layer[layer_raw == 0] = False                # anything black

        layers.append(layer)

        # Figure out class name from file name
        layer_name = os.path.splitext(os.path.basename(layer_file))[0].split("_")[-1]
        # Remove second .png in case of photoshop layer files
        if layer_name.endswith(".png"): layer_name = os.path.splitext(layer_name)[0]
        layer_names.append(layer_name)

    napari_viewer.add_image(img, name=img_name)

    for l, layer in enumerate(layers):
        label = (layer * (l + 1)).astype(np.uint8)
        napari_viewer.add_labels(label, name=str(len(layers) - l).zfill(2) + "_" + layer_names[l], colormap=color_map)
        #napari_viewer.add_image(label, rgb=True, name=str(len(layers) + 1).zfill(2) + "_" + layer_name)

napari.run()

# Make new folder to save layers to
if layer_folder != "":
    layer_path = layer_folder
else:
    base_name = os.path.splitext(os.path.basename(seg_path))[0].split("_seg")[0]
    base_path = os.path.dirname(seg_path)
    layer_path = os.path.join(base_path, base_name)

if not os.path.exists(layer_path):
    os.makedirs(layer_path)

# Export layers
print("Exporting " + str(len(napari_viewer.layers)) + " layers...")
for l, layer in enumerate(tqdm(napari_viewer.layers)):
    if layer.name == img_name:
        # If lamella map, just save as png
        Image.fromarray(img).save(os.path.join(layer_path, layer.name + ".png"))        
        #layer.save(os.path.join(layer_path, layer.name + ".png"))
    else:
        # If label layer, make values uniform and save as color pngs
        label = np.zeros(layer.data.shape, dtype=bool)
        label[layer.data > 0] = True
        if not color:
            label = np.dstack([label * 255 // 1.5, label * 255 // 8, label * 255 // 8, label * 255]).astype(np.uint8)
        else:
            if l in class_colors.keys():
                layer_color = class_colors[l]
            else:
                layer_color = (255 // 1.5, 255 // 8, 255 // 8, 255)     # dark red
            label = (np.dstack([label, label, label, label]) * layer_color).astype(np.uint8)

        Image.fromarray(label).save(os.path.join(layer_path, layer.name + ".png"))

print("NOTE: Layers were saved to " + layer_path)