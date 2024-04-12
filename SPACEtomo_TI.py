#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_TI
# Purpose:      User interface for training SPACEtomo segmentation models using nnU-Netv2
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/02/15
# Revision:     v1.1
# Last Change:  2024/03/22: fixed runNapari, disabled targetselection by default (made independent GUI for it)
# ===================================================================

##### SETTINGS #####

dynamic_textures = False                # slower performance, but fix some Segmentation fault crashes
show_tgt_selection = False              # show options for target selection in inspection tab (it's recommended to do target selection in SPACEtomo_tgt.py)

### END SETTINGS ###

import os
os.environ["__GLVND_DISALLOW_PATCHING"] = "1"           # helps to minimize Segmentation fault crashes on Linux when deleting textures
import sys
import copy
import shutil
import glob
import json
import dearpygui.dearpygui as dpg
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import time
import datetime
import mrcfile
import subprocess
from skimage import exposure, transform, draw
import torch

versionSPACE = "1.1"
CUR_DIR = os.getcwd()
SPACE_DIR = os.path.dirname(__file__)

# Check if functions file exists
if os.path.exists(os.path.join(SPACE_DIR, "SPACEtomo_functions_ext.py")):
    sys.path.insert(len(sys.path), SPACE_DIR)
    import SPACEtomo_functions_ext as space_ext
    FUNC_IMPORT = True
    versionSPACE = space_ext.versionSPACE
else:
    FUNC_IMPORT = False

# Check if napari is installed
try:
    import napari
    napari_installed = True
except ModuleNotFoundError:
    napari_installed = False

# Find available GPUs
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("NOTE: Found CUDA device.")
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("NOTE: Found MPS device.")
else:
    DEVICE = "cpu"
    print("NOTE: Found no GPU, using CPU.")


dpg.create_context()

##### Utilities

def openTxt(filename, app_data=None, user_data=None):
    if os.path.exists(filename):
        subprocess.Popen(["open", filename])
    elif os.path.exists(user_data):
        subprocess.Popen(["open", user_data])

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
    
def window_size_change():
    # Update items anchored to side of window
    dpg.set_item_pos("logo_img", pos=(10, dpg.get_viewport_height() - 40 - logo_dims[0]))
    dpg.set_item_pos("logo_text", pos=(10 + logo_dims[1] / 2 - (40), dpg.get_viewport_height() - 40 - logo_dims[0] / 2))
    dpg.set_item_pos("version_text", pos=(dpg.get_viewport_width() - 100, 10))

def cancel_callback():
    pass

def mouse_click(sender, app_data):
    mouse_coords = np.array(dpg.get_plot_mouse_pos())
    mouse_coords_global = dpg.get_mouse_pos(local=False)    # need global coords, because plot coords give last value at edge of plot when clicking outside of plot

    # Get active tab
    tab_label = dpg.get_item_label(dpg.get_value("tabbar"))

    # Left mouse button functions
    if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        # Send to tab specific function
        if tab_label == "Inference":
            inf_mouse_click_left(mouse_coords, mouse_coords_global)
        elif tab_label == "Inspection":
            ins_mouse_click_left(mouse_coords, mouse_coords_global)

    # Right mouse button functions
    elif dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
        if tab_label == "Inspection":
            ins_mouse_click_right(mouse_coords, mouse_coords_global)        

##### Inference functions

def inf_loadMap(sender, app_data):
    global file_path, data, meta_data, mont_shape, pix_size, pix_size_model, inf_flip_map
    file_path = sorted(glob.glob(app_data["file_path_name"]))[0]

    # Get list of mrc and map files for next map selection
    map_list = glob.glob(os.path.join(os.path.dirname(file_path), "*.mrc"))
    map_list.extend(glob.glob(os.path.join(os.path.dirname(file_path), "*.map")))
    map_list = sorted(map_list)

    # Delete buttons if present and reorganize menu
    if dpg.does_item_exist("inf_mont"): dpg.delete_item("inf_mont")
    if dpg.does_item_exist("inf_mont_but"): dpg.delete_item("inf_mont_but")

    dpg.set_item_label("inf_plot", os.path.basename(file_path) + " loading...")
    dpg.set_value("inf_2", "Loading...\n\n\n\n\n")
    dpg.set_value("inf_tileid", " \n")
    dpg.set_value("inf_3", " \n ")
    if dpg.does_item_exist("inf_pix"): dpg.delete_item("inf_pix")
    dpg.add_text(default_value="", tag="inf_pix", parent="inf_left", before="inf_numimg")
    if dpg.does_item_exist("inf_butexp"): dpg.delete_item("inf_butexp")
    dpg.add_text(default_value="", tag="inf_butexp", parent="inf_left", before="inf_numimg")

    # Delete any previous maps from texture registry
    if "data" in globals():
        if dpg.does_item_exist("inf_borderplot_yel"): dpg.delete_item("inf_borderplot_yel")
        for i in range(data.shape[2]):
            if dpg.does_item_exist("inf_imgplot" + str(i)): dpg.delete_item("inf_imgplot" + str(i)) 
            #if dpg.does_item_exist("inf_img" + str(i)): dpg.delete_item("inf_img" + str(i))
            if dpg.does_item_exist("inf_borderplot_whi" + str(i)): dpg.delete_item("inf_borderplot_whi" + str(i)) 
        if dpg.does_item_exist("inf_tex") and not dynamic_textures: 
            print("NOTE: Deleting texture registry. If GUI crashes here with Segmentation Fault, try setting dynamic_textures = True!")
            dpg.delete_item("inf_tex")
            time.sleep(0.1)             # helps to reduce Segmentation fault crashes

    # Load mrc file
    with mrcfile.open(file_path) as mrc:
        vals = np.array((np.min(mrc.data), np.max(mrc.data), round(np.mean(mrc.data), 2)), dtype=float)
        print("Map statistics:")
        print("Min, max, mean:", vals)
        print("Cutoff:", np.quantile(mrc.data, 0.99))
        data = exposure.rescale_intensity(mrc.data, in_range=(0, np.quantile(mrc.data, 0.99)), out_range=(0, 255)).astype(np.uint8)
        if data.ndim < 3:
            data = np.expand_dims(data, 0)

        header = mrc.header
        pix_size = float(mrc.voxel_size.x)

        # Save metadata
        meta_data = {"original_map": {"path": file_path, "name": os.path.splitext(os.path.basename(file_path))[0], "pixel_size": round(pix_size, 2), "min_val": vals[0], "max_val": vals[1], "mean_val": vals[2], "tiles": data.shape[0], "tile_dimensions": data[0].shape}}

    print("Pixel size:", pix_size)

    if data.shape[0] == 1: mont_shape = [1, 1]
    elif data.shape[0] == 4: mont_shape = [2, 2]
    elif data.shape[0] == 6: mont_shape = [2, 3]
    elif data.shape[0] == 8: mont_shape = [2, 4]
    elif data.shape[0] == 9: mont_shape = [3, 3]
    elif data.shape[0] == 10: mont_shape = [2, 5]
    elif data.shape[0] == 12: mont_shape = [3, 4]
    elif data.shape[0] == 14: mont_shape = [2, 7]
    elif data.shape[0] == 15: mont_shape = [3, 5]
    elif data.shape[0] == 16: mont_shape = [4, 4]
    elif data.shape[0] == 18: mont_shape = [3, 6]
    elif data.shape[0] == 24: mont_shape = [4, 6]
    elif data.shape[0] == 25: mont_shape = [5, 5]
    elif data.shape[0] == 28: mont_shape = [4, 7]
    elif data.shape[0] == 30: mont_shape = [5, 6]
    elif data.shape[0] == 32: mont_shape = [4, 8]
    elif data.shape[0] == 36: mont_shape = [4, 9]
    elif data.shape[0] == 40: mont_shape = [4, 10]        
    else:
        print("WARNING: " + str(data.shape[0]) + " is not covered! Please enter montage dimensions manually!")
        mont_shape = [1, data.shape[0]]      

    if "inf_flip_map" not in globals():
        inf_flip_map = False
    if inf_flip_map:
        mont_shape = mont_shape[::-1]

    meta_data["original_map"]["montage_shape"] = mont_shape

    # Check pixel size of previous exports and model
    _, pix_size_input, pix_size_model = inf_checkPixelSize(model_list)
    if pix_size_input is not None:
        pix_size_model = meta_data["pixel_size"] = pix_size_input
    if pix_size_model is None:
        pix_size_model = pix_size

    # Generate new map
    if not dpg.does_item_exist("inf_tex"):
        dpg.add_texture_registry(tag="inf_tex")
    for i in range(mont_shape[1]):
        for j in range(mont_shape[0]):
            tile = i * mont_shape[0] + j
            if tile >= len(data): break
            bounds = np.array([i * data.shape[2], (mont_shape[0] - (j + 1)) * data.shape[1], (i + 1) * data.shape[2], (mont_shape[0] - j) * data.shape[1]]) * pix_size / 10000
            image = np.ravel(np.dstack([data[tile], data[tile], data[tile], np.full(data[tile].shape, 255)])) / 255

            if not dynamic_textures:
                dpg.add_static_texture(width=data.shape[2], height=data.shape[1], default_value=image, tag="inf_img" + str(tile), parent="inf_tex")
            else:
                if dpg.does_item_exist("inf_img" + str(tile)):
                    dpg.set_value("inf_img" + str(tile), image)
                else:
                    dpg.add_dynamic_texture(width=data.shape[2], height=data.shape[1], default_value=image, tag="inf_img" + str(tile), parent="inf_tex")
            dpg.add_image_series("inf_img" + str(tile), bounds_min=bounds[:2], bounds_max=bounds[2:], parent="inf_x_axis", tag="inf_imgplot" + str(tile))
            dpg.fit_axis_data("inf_x_axis")
            dpg.fit_axis_data("inf_y_axis")

    # Make highlight border
    stroke = int(np.max(data.shape) * 0.01)
    border = np.zeros([data.shape[1], data.shape[2]])
    border[1:stroke, :] = 1
    border[-stroke:-1, :] = 1
    border[:, 1:stroke] = 1
    border[:, -stroke:-1] = 1
    border_yel = np.ravel(np.dstack([border, 0.84 * border, 0 * border, border]))
    border_whi = np.ravel(np.dstack([0.75 * border, 0.75 * border, 0.75 * border, border]))

    if not dpg.does_item_exist("inf_border_yel"):
        with dpg.texture_registry():
            dpg.add_static_texture(width=data.shape[2], height=data.shape[1], default_value=border_yel, tag="inf_border_yel")
            dpg.add_static_texture(width=data.shape[2], height=data.shape[1], default_value=border_whi, tag="inf_border_whi")

    inf_outlineExportedTiles()

    dpg.set_item_label("inf_plot", os.path.basename(file_path) + " [" + str(round(pix_size, 1)) + " Å/px]")

    if len(map_list) > 1:
        next_map_id = map_list.index(file_path) + 1
        if next_map_id >= len(map_list): 
            next_map_id = 0
            print("WARNING: Reached end of folder. Next map will start from beginning.")
        if dpg.does_item_exist("inf_butnext"): dpg.delete_item("inf_butnext")
        dpg.add_button(label="Load next", callback=lambda: inf_loadMap("_",{"file_path_name": map_list[next_map_id]}), tag="inf_butnext", parent="inf_load")

    with dpg.group(tag="inf_mont", horizontal=True, parent="inf_left", before="inf_2"):
        dpg.add_text("x")
        dpg.add_input_int(tag="inf_mont_x", default_value=mont_shape[1], step=0, width=50)
        dpg.add_text("y")
        dpg.add_input_int(tag="inf_mont_y", default_value=mont_shape[0], step=0, width=50)
    with dpg.group(tag="inf_mont_but", horizontal=True, parent="inf_left", before="inf_2"):
        dpg.add_button(label="Reorder", callback=inf_reorderMap)
        dpg.add_button(label="Flip", callback=inf_flipMap)
    dpg.set_value("inf_2", "\n2. Select a tile")

def inf_outlineExportedTiles():
    # Find any files exported from the loaded map in datasets and input
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_in_datasets = glob.glob(os.path.join(NN_RAW, "**", "imagesTr", file_name + "*"))
    file_in_datasets.extend(glob.glob(os.path.join(CUR_DIR, "input*", file_name + "*")))

    # Highlight the tiles that were exported previously
    tile_list = []
    for tile_file in file_in_datasets:
        tile_id = int(tile_file.split("_")[-2])
        if not tile_id in tile_list:        # no double highlighting
            tile_list.append(tile_id)
            i = tile_id  // mont_shape[0]
            j = mont_shape[0] - tile_id % mont_shape[0] - 1
            bounds = np.array([i * data.shape[2], (j + 1) * data.shape[1], (i + 1) * data.shape[2], j * data.shape[1]]) * pix_size / 10000
            if not dpg.does_item_exist("inf_borderplot_whi" + str(tile_id)):
                dpg.add_image_series("inf_border_whi", bounds_min=bounds[:2], bounds_max=bounds[2:], parent="inf_x_axis", tag="inf_borderplot_whi" + str(tile_id))            

    """
    for file in inf_input_files:
        if os.path.splitext(os.path.basename(file_path))[0] in file:
            tile_id = int(file.split("_")[-2])
            i = tile_id  // mont_shape[0]
            j = mont_shape[0] - tile_id % mont_shape[0] - 1
            bounds = np.array([i * data.shape[2], (j + 1) * data.shape[1], (i + 1) * data.shape[2], j * data.shape[1]]) * pix_size / 10000
            if not dpg.does_item_exist("inf_borderplot_whi" + str(tile_id)):
                dpg.add_image_series("inf_border_whi", bounds_min=bounds[:2], bounds_max=bounds[2:], parent="inf_x_axis", tag="inf_borderplot_whi" + str(tile_id))
    """

def inf_reorderMap():
    global meta_data
    # Delete any previous maps from plot
    if dpg.does_item_exist("inf_borderplot_yel"): dpg.delete_item("inf_borderplot_yel")
    for i in range(100):
        if dpg.does_item_exist("inf_imgplot" + str(i)): dpg.delete_item("inf_imgplot" + str(i)) 
        if dpg.does_item_exist("inf_borderplot_whi" + str(i)): dpg.delete_item("inf_borderplot_whi" + str(i)) 

    # Get mont_shape from text input
    mont_shape[1] = dpg.get_value("inf_mont_x")
    mont_shape[0] = dpg.get_value("inf_mont_y")

    # Set to maximum of tile size
    if mont_shape[1] >= data.shape[0]:
        mont_shape[1] = data.shape[0]
        dpg.set_value("inf_mont_x", mont_shape[1])
    if mont_shape[0] >= data.shape[0]:
        mont_shape[0] = data.shape[0]
        dpg.set_value("inf_mont_y", mont_shape[0])

    # Update meta data
    meta_data["original_map"]["montage_shape"] = mont_shape

    # Generate new map
    for i in range(mont_shape[1]):
        for j in range(mont_shape[0]):
            tile = i * mont_shape[0] + j
            bounds = np.array([i * data.shape[2], (mont_shape[0] - (j + 1)) * data.shape[1], (i + 1) * data.shape[2], (mont_shape[0] - j) * data.shape[1]]) * pix_size / 10000

            if not dpg.does_item_exist("inf_img" + str(tile)): break
            dpg.add_image_series("inf_img" + str(tile), bounds_min=bounds[:2], bounds_max=bounds[2:], parent="inf_x_axis", tag="inf_imgplot" + str(tile))
            dpg.fit_axis_data("inf_x_axis")
            dpg.fit_axis_data("inf_y_axis")

    inf_outlineExportedTiles()

def inf_flipMap():
    global inf_flip_map
    inf_flip_map = not inf_flip_map
    dpg.set_value("inf_mont_x", mont_shape[0])
    dpg.set_value("inf_mont_y", mont_shape[1])
    inf_reorderMap()

def inf_mouse_click_left(mouse_coords, mouse_coords_global):
    global tile_id
    #mouse_coords = np.array(dpg.get_plot_mouse_pos())
    #mouse_coords_global = dpg.get_mouse_pos(local=False)    # need global coords, because plot coords give last value at edge of plot when clicking outside of plot

    if "data" in globals() and np.all(mouse_coords > 0) and mouse_coords_global[0] > 200:
        mouse_coords = mouse_coords * 10000 / pix_size

        x = int(mouse_coords[0] / data.shape[2])
        y = int(mouse_coords[1] / data.shape[1])

        if x > mont_shape[1] - 1 or y > mont_shape[0] - 1:
            return

        tile_id = x * mont_shape[0] + (mont_shape[0] - y - 1)

        dpg.set_value("inf_tileid", "Tile: (" + str(x) + "," + str(y) + ") [" + str(tile_id) + "]")

        bounds = np.array([x * data.shape[2], (y + 1) * data.shape[1], (x + 1) * data.shape[2], y * data.shape[1]]) * pix_size / 10000
        if dpg.does_item_exist("inf_borderplot_yel"): dpg.delete_item("inf_borderplot_yel")
        dpg.add_image_series("inf_border_yel", bounds_min=bounds[:2], bounds_max=bounds[2:], parent="inf_x_axis", tag="inf_borderplot_yel")

        # Pixel size input and export button
        dpg.set_value("inf_3", "\n3. Export as png")
        if dpg.does_item_exist("inf_pix"): dpg.delete_item("inf_pix")
        with dpg.group(tag="inf_pix", horizontal=True, parent="inf_left", before="inf_numimg"):
            dpg.add_text("Pixel size:")
            # Make editable only if no images have been exported
            if "pixel_size" in meta_data.keys():
                dpg.add_text(tag="inf_pixsize", default_value=round(pix_size_model, 2))
                dpg.add_text(" [Å/px]")
            else:
                dpg.add_input_float(tag="inf_pixsize", default_value=round(pix_size_model, 2), min_value=pix_size, format="%.2f", step=0, width=50, label="[Å/px]")
        if dpg.does_item_exist("inf_butexp"): dpg.delete_item("inf_butexp")
        dpg.add_button(label="Export tile", callback=inf_exportAsPng, tag="inf_butexp", parent="inf_left", before="inf_numimg")

def inf_exportAsPng():
    global inf_input_files, pix_size_model, meta_data
    if not os.path.exists(os.path.join(CUR_DIR, "input")): 
        os.makedirs(os.path.join(CUR_DIR, "input"))
        inf_input_files = []

    # Check if file was already exported
    file_name = os.path.join(CUR_DIR, "input", os.path.splitext(os.path.basename(file_path))[0] + "_" + str(tile_id).zfill(2) + "_0000.png")
    if os.path.exists(file_name):
        print("ERROR: This image was already exported.")
        return

    # Check if file was already used in previous dataset and same pixel size
    file_in_datasets = glob.glob(os.path.join(NN_RAW, "**", "imagesTr", os.path.basename(file_name)))
    if len(file_in_datasets) > 0:
        for file in file_in_datasets:
            dataset_path = file.split("imagesTr")[0]
            _, pix_size_previous, _ = loadDatasetJson(os.path.join(dataset_path, "dataset.json"))
            if pix_size_previous is not None and pix_size_previous == pix_size_model:
                print("ERROR: This image was already used for training.")
                return
            print("WARNING: This image was already used for training, but is exported now at different pixel size.")
            
    dpg.set_value("inf_expstatus", "Saving...")
    
    # Rescale image to model pixel size
    pix_size_model = round(float(dpg.get_value("inf_pixsize")), 2)
    if pix_size != pix_size_model:
        if pix_size_model < pix_size:
            print("WARNING: Upscaling images is not recommended!")
        image = Image.fromarray(np.uint8(transform.rescale(data[tile_id], pix_size / pix_size_model) * 255))
    else:
        image = Image.fromarray(data[tile_id])

    # Save image
    image.save(file_name)
    print("Image saved: " + file_name)

    # Save meta data
    meta_data.update({"pixel_size": round(pix_size_model, 2), "dimensions": image.size[::-1], "tile_id": tile_id})
    save_path = os.path.splitext(file_name)[0] + ".json"
    with open(save_path, "w+") as f:
        json.dump(meta_data, f, indent=4)
    print("Meta data saved: " + save_path)

    inf_input_files.append(file_name)
    inf_outlineExportedTiles()

    dpg.set_value("inf_expstatus", "")
    dpg.set_value("inf_numimg", "Total images: " + str(len(inf_input_files)))

    dpg.set_value("inf_4", "\n4. Choose model")
    if dpg.does_item_exist("inf_selmod"): dpg.delete_item("inf_selmod")
    if dpg.does_item_exist("inf_butmod"): dpg.delete_item("inf_butmod")
    if len(model_list) > 0:
        dpg.add_combo(model_list, default_value=model_list[-1], callback=inf_checkModel, tag="inf_selmod", parent="inf_left", before="inf_5")
        dpg.set_value("inf_5", "\n5. Segment images")
        if dpg.does_item_exist("inf_butinf"): dpg.delete_item("inf_butinf")
        dpg.add_button(label="Run inference", callback=inf_inference, tag="inf_butinf", parent="inf_left", before="inf_left_final")
    else:
        dpg.add_button(label="Find model", callback=lambda: dpg.show_item("inf_file2"), tag="inf_butmod", parent="inf_left", before="inf_5")


def inf_importModel(sender, app_data):
    model_path = app_data["file_path_name"]
    shutil.copytree(model_path, "model_0")

    if dpg.does_item_exist("inf_butmod"): dpg.delete_item("inf_butmod")
    dpg.set_value("inf_5", "Importing model...")

    # Read dataset.json and look for pixel size
    check_pix_size, pix_size_input, pix_size_model = inf_checkPixelSize("model_0")
    if not check_pix_size:
        print("WARNING: Model pixel size is not the same as export pixel size. Please export images at model pixel size (" + str(round(pix_size_model, 2)) + " Å/px).")

    # Wait for checkpoint files to exist
    timeout = 0
    while len(glob.glob(os.path.join(CUR_DIR, "model_0", "**", "checkpoint_final.pth"))) < 5 and len(glob.glob(os.path.join(CUR_DIR, "model_0", "**", "checkpoint_best.pth"))) < 5 and timeout < 100:
        time.sleep(1)
        timeout += 1
    if timeout >= 100:
        print("ERROR: Model could not be imported.")
        return

    model_list.append("model_0")

    dpg.add_combo(model_list, default_value=model_list[-1], callback=inf_checkModel, tag="inf_selmod", parent="inf_left", before="inf_5")

    dpg.set_value("inf_5", "\n5. Segment images")
    if dpg.does_item_exist("inf_butinf"): dpg.delete_item("inf_butinf")
    if not check_pix_size:
        dpg.add_text("Model pixel size differs \nfrom export pixel size. \nPlease reexport images at \nthe proper pixel size.", tag="inf_butinf", color=error_color, parent="inf_left", before="inf_left_final")
    else:
        dpg.add_button(label="Run inference", callback=inf_inference, tag="inf_butinf", parent="inf_left", before="inf_left_final")


def inf_inference():
    model_name = dpg.get_value("inf_selmod")
    model_no = model_name.split("_")[-1]

    # Check if checkpoint files exist
    if os.path.exists(os.path.join(model_name, "fold_0", "checkpoint_final.pth")):
        chk_name = "checkpoint_final.pth"
    elif os.path.exists(os.path.join(model_name, "fold_0", "checkpoint_best.pth")):
        chk_name = "checkpoint_best.pth"
    else:
        print("ERROR: Checkpoint file not found. Try using a different model!")
        return
    
    # Check if pixel sizes are consistent:
    check_pix_size, *_ = inf_checkPixelSize(model_name)
    if not check_pix_size:
        print("ERROR: Model pixel size differs from export pixel size. Please reexport images at the proper pixel size.")
        return
    
    # Delete textures to free up GPU memory for inference
    if "data" in globals():
        if dpg.does_item_exist("inf_borderplot_yel"): dpg.delete_item("inf_borderplot_yel")
        for i in range(data.shape[2]):
            if dpg.does_item_exist("inf_imgplot" + str(i)): dpg.delete_item("inf_imgplot" + str(i)) 
            if dpg.does_item_exist("inf_borderplot_whi" + str(i)): dpg.delete_item("inf_borderplot_whi" + str(i)) 
        if dpg.does_item_exist("inf_tex") and not dynamic_textures: 
            print("NOTE: Deleting texture registry. If GUI crashes here with Segmentation Fault, try setting dynamic_textures = True!")
            dpg.delete_item("inf_tex")
            time.sleep(0.1)             # helps to reduce Segmentation fault crashes

    # Define dirs
    input_dir = os.path.join(CUR_DIR, "input_" + str(model_no))
    output_dir = os.path.join(CUR_DIR, "output_" + str(model_no))
    model_dir = os.path.join(CUR_DIR, model_name)

    # Lock input folder by renaming it
    os.rename(os.path.join(CUR_DIR, "input"), input_dir)

    if model_name != "":
        dpg.set_value("inf_left_final", "Running inference...\nThis might take \nseveral minutes.")
        subprocess.run(["nnUNetv2_predict_from_modelfolder", "-i", input_dir, "-o", output_dir, "-m", model_dir + "/", "-chk", chk_name, "-device", DEVICE])
        dpg.set_value("inf_left_final", "Segmentation finished.")
    else:
        print("WARNING: No model selected.")

def inf_checkPixelSize(model_name):
    # Get pixel size for previous exports
    input_json_list = glob.glob(os.path.join(CUR_DIR, "input", "*.json"))
    if len(input_json_list) > 0:
        with open(input_json_list[0], "r") as f:
            pix_size_input = json.load(f)["pixel_size"]
    else:
        pix_size_input = None

    # Check if argument is model list or name
    if isinstance(model_name, list):
        if len(model_name) > 0:
            model_name = model_list[-1]
        else:
            return True, pix_size_input, None
        
    # Get model pixel size
    _, pix_size_model, _ = loadDatasetJson(os.path.join(CUR_DIR, model_name, "dataset.json"))

    # Check compatibility
    if pix_size_input is not None and pix_size_model is not None and pix_size_input != pix_size_model:
        print("Model pixel size: " + str(pix_size_model))
        print("Input pixel size: " + str(pix_size_input))
        return False, pix_size_input, pix_size_model
    else:
        return True, pix_size_input, pix_size_model
    
def inf_checkModel():
    model_name = dpg.get_value("inf_selmod")
    check_pix_size, *_ = inf_checkPixelSize(model_name)

    if dpg.does_item_exist("inf_butinf"): dpg.delete_item("inf_butinf")
    if not check_pix_size:
        dpg.add_text("Model pixel size differs \nfrom export pixel size. \nPlease reexport images at \nthe proper pixel size.", tag="inf_butinf", color=error_color, parent="inf_left", before="inf_left_final")
    else:
        dpg.add_button(label="Run inference", callback=inf_inference, tag="inf_butinf", parent="inf_left", before="inf_left_final")
    

##### Inspection functions

def ins_loadMap(sender, app_data):
    global dims, binning, file_path, seg_path, image_orig, seg_folder, pix_size_png
    file_path = sorted(glob.glob(os.path.splitext(app_data["file_path_name"])[0] + ".png"))[0]

    dpg.set_item_label("ins_plot", os.path.basename(file_path) + " loading...")

    # Check for metadata file to get pixel size
    if os.path.exists(os.path.splitext(file_path)[0] + ".json"):
        with open(os.path.splitext(file_path)[0] + ".json", "r") as f:
            meta_data = json.load(f)
        pix_size_png = float(meta_data["pixel_size"])
        dpg.set_item_label("ins_x_axis", "x [µm]")
        dpg.set_item_label("ins_y_axis", "y [µm]")
    else:
        pix_size_png = 10000
        dpg.set_item_label("ins_x_axis", "x [px]")
        dpg.set_item_label("ins_y_axis", "y [px]")

    image = np.array(Image.open(file_path)).astype(float) / 255
    image_orig = copy.deepcopy(image) # make copy for export
    dims = [image.shape[1], image.shape[0]]
    binning = 1

    if np.max(dims) > 16384:  # hard limit for texture sizes on apple GPU
        print("WARNING: Map is too large and will be binned by 2 (for display only)! Export will be unbinned.")
        image = image[::2, ::2]
        dims = [image.shape[1], image.shape[0]]
        binning = 2

    image = np.ravel(np.dstack([image, image, image, np.ones(image.shape)]))

    if dpg.does_item_exist("ins_img"): dpg.delete_item("ins_img")
    if dpg.does_item_exist("ins_imgplot"): dpg.delete_item("ins_imgplot")
    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[0], height=dims[1], default_value=image, tag="ins_img")
    dpg.add_image_series("ins_img", bounds_min=(0, 0), bounds_max=np.array(dims)*pix_size_png/10000*binning, parent="ins_x_axis", tag="ins_imgplot")
    dpg.fit_axis_data("ins_x_axis")
    dpg.fit_axis_data("ins_y_axis")

    seg_folder = False
    if "_0000.png" in file_path:
        seg_path = file_path.split("_0000.png")[0] + ".png"
        if not os.path.exists(seg_path) and "input" in seg_path:
            seg_path = seg_path.split("input")[0] + "output" + seg_path.split("input")[1]
            seg_folder = True
        if not os.path.exists(seg_path) and "imagesTr" in seg_path:
            seg_path = seg_path.split("imagesTr")[0] + "labelsTr" + seg_path.split("imagesTr")[1]
            seg_folder = True
    else:
        seg_path =  os.path.splitext(file_path)[0] + "_seg.png"
    if os.path.exists(seg_path):
        ins_loadSeg(seg_path)
    else:
        seg_folder = False
        print("WARNING: Segmentation was not found.")

    dpg.set_item_label("ins_plot", os.path.basename(file_path))

def ins_loadSeg(filename):
    global seg, seg_orig
    seg = np.array(Image.open(filename))
    dims = seg.shape
    if binning == 2:  # hard limit for texture sizes on apple GPU
        seg_orig = copy.deepcopy(seg)
        dims = (dims[0] // 2, dims[1] // 2)
        seg = seg[::2, ::2]
    ins_loadClasses()
    ins_loadMask()

# Create mask from segmentation and selected classes
def ins_makeMask(seg, class_names):
    mask = np.zeros(seg.shape)
    for name in class_names:
        mask[seg == CLASSES[name]] = 1

    return mask

def ins_loadMask(sender=None, class_list=[]):
    if not isinstance(class_list, list) or len(class_list) == 0:
        class_list = [dpg.get_value("ins_class")]

    mask = ins_makeMask(seg, class_list)
    mask = np.ravel(np.dstack([mask, np.zeros(mask.shape), np.zeros(mask.shape), mask * np.full(mask.shape, 0.25)]))
    dpg.delete_item("seg")
    dpg.delete_item("segplot")
    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[0], height=dims[1], default_value=mask, tag="seg")
    dpg.add_image_series("seg", bounds_min=(0, 0), bounds_max=np.array(dims)*pix_size_png/10000*binning, parent="ins_x_axis", tag="segplot")

def ins_loadClasses():
    global CLASSES
    json_file = os.path.join(os.path.dirname(seg_path), "dataset.json")
    if not os.path.exists(json_file):
        json_file = os.path.join(os.path.dirname(file_path), "dataset.json")
        if not os.path.exists(json_file):
            json_file = os.path.join(os.path.dirname(seg_path), os.pardir, "dataset.json")
            if not os.path.exists(json_file):
                json_file = "dataset.json"
                if not os.path.exists(json_file):
                    print("WARNING: No dataset.json file was found!")
                    return
            
    CLASSES, *_ = loadDatasetJson(json_file)

    dpg.set_value("ins_2", "\n2. Inspect segmentation")
    dpg.set_value("ins_cls", "Classes:")
    if dpg.does_item_exist("ins_class"): dpg.delete_item("ins_class")
    #dpg.add_radio_button([key for key in CLASSES.keys()], horizontal=False, default_value=list(CLASSES.keys())[0], callback=loadMask, tag="class", parent="ins_left", before="ins_3")
    dpg.add_combo([key for key in CLASSES.keys()], default_value=list(CLASSES.keys())[0], callback=ins_loadMask, tag="ins_class", parent="ins_left", before="ins_3")

    dpg.set_value("ins_3", "\n3. Export classes as layers")
    if dpg.does_item_exist("ins_butexp"): dpg.delete_item("ins_butexp")
    dpg.add_button(label="Export map", callback=ins_exportAsLayers, tag="ins_butexp", parent="ins_left", before="ins_4")

    if dpg.does_item_exist("ins_butexpfol"): dpg.delete_item("ins_butexpfol")
    if seg_folder:
        dpg.add_button(label="Export folder", callback=ins_exportFolderAsLayers, tag="ins_butexpfol", parent="ins_left", before="ins_4")

    if dpg.does_item_exist("exportbar"): dpg.delete_item("exportbar")

    if not dpg.does_item_exist("ins_butnap") and napari_installed:
        dpg.add_button(label="Open in Napari", callback=openNapari, tag="ins_butnap", parent="ins_left", before="ins_final")


    if show_tgt_selection:
        if FUNC_IMPORT and mic_params is not None and tgt_params is not None:
            # Show targets if point file exists
            ins_showTargets(load_from_file=True)

            if not dpg.does_item_exist("ins_tsmenu"):
                with dpg.collapsing_header(label="Target selection", tag="ins_tsmenu", parent="ins_left", before="ins_final"):
                    # Targeting settings
                    dpg.add_input_text(label="Target classes", tag="target_list", default_value=",".join(tgt_params.target_list), width=100)
                    dpg.add_input_text(label="Avoid classes", tag="avoid_list", default_value=",".join(tgt_params.penalty_list), width=100)
                    dpg.add_input_float(label="Score threshold", tag="target_score_threshold", default_value=tgt_params.threshold, format="%.2f", step=0, width=50)
                    dpg.add_input_float(label="Penalty weight", tag="penalty_weight", default_value=tgt_params.penalty, format="%.2f", step=0, width=50)
                    dpg.add_input_int(label="Max. tilt angle", tag="max_tilt", default_value=tgt_params.max_tilt, step=0, width=50)
                    dpg.add_input_float(label="Image shift limit", tag="IS_limit", default_value=mic_params.IS_limit, format="%.2f", step=0, width=50)
                    dpg.add_checkbox(label="Sparse targets", tag="sparse_targets", default_value=tgt_params.sparse)
                    dpg.add_checkbox(label="Target edge", tag="target_edge", default_value=tgt_params.edge)
                    dpg.add_checkbox(label="Extra tracking", tag="extra_tracking", default_value=tgt_params.extra_track)

                with dpg.group(horizontal=True, tag="ins_butgrp"):
                    dpg.add_button(label="Select targets", callback=ins_runTargetSelection, tag="ins_butts")
        else:
            dpg.set_value("ins_final", "Target selection not possible.")


def ins_exportAsLayers():
    ins_exportFolderAsLayers(folder=False)

def ins_exportFolderAsLayers(folder=True, color=True):
    if folder:
        seg_list = sorted(glob.glob(os.path.join(os.path.dirname(seg_path), "*.png")))
    else:
        seg_list = [seg_path]

    if dpg.does_item_exist("exportbar"): dpg.delete_item("exportbar")
    dpg.add_progress_bar(default_value=0, width=-1, overlay="0%", tag="exportbar", parent="ins_left", before="ins_4")

    for s, seg_file in enumerate(seg_list):
        if folder:
            img_file = os.path.splitext(seg_file.split("output")[0] + "input" + seg_file.split("output")[1])[0] + "_0000.png"
        else:
            img_file = file_path

        if os.path.exists(img_file):
            img = np.array(Image.open(img_file))
        else:
            print("ERROR: MM map (" + os.path.basename(img_file) + ") not found.")
            return
        seg = np.array(Image.open(seg_file))

        base_name = os.path.splitext(os.path.basename(seg_file))[0].split("_seg")[0]
        base_path = os.path.dirname(seg_file)
        layer_path = os.path.join(base_path, base_name)

        if not os.path.exists(layer_path):
            os.makedirs(layer_path)

        for i, l in enumerate(CLASSES.keys()):
            progress = (s * (len(CLASSES) + 1) + i) / (len(seg_list) * (len(CLASSES) + 1))
            dpg.set_value("exportbar", progress)
            dpg.configure_item("exportbar", overlay=f"{round(progress * 100)}%")
            if CLASSES[l] == 0:
                continue
            label = np.zeros(seg.shape)
            label[seg == CLASSES[l]] = 255
            if not color:
                label = np.dstack([label // 1.5, label // 8, label // 8, label])
            else:
                label = label / 255
                if l in class_colors.keys():
                    layer_color = class_colors[l]
                else:
                    layer_color = (255 // 1.5, 255 // 8, 255 // 8, 255)     # dark red
                label = np.dstack([label, label, label, label]) * layer_color

            save_path = os.path.join(layer_path, str(len(CLASSES) - CLASSES[l]).zfill(2) + "_" + l + ".png")
            Image.fromarray(np.uint8(label)).save(save_path)

        progress = ((s * (len(CLASSES) + 1)) + len(CLASSES)) / (len(seg_list) * (len(CLASSES) + 1))
        dpg.set_value("exportbar", progress)
        dpg.configure_item("exportbar", overlay=f"{round(progress * 100)}%")     

        img = Image.fromarray(np.uint8(img))
        save_path = os.path.join(layer_path, base_name + ".png")
        img.convert("RGB").save(save_path)
        print("Layers saved: " + layer_path)

    dpg.set_value("exportbar", 1)
    dpg.configure_item("exportbar", overlay=f"{100}%")

    dpg.delete_item("exportbar")
    dpg.set_value("ins_4", "\n4. Edit layers externally")

def ins_runTargetSelection():
    global tgt_params, mic_params, MM_model
    if FUNC_IMPORT:
        map_dir = os.path.dirname(file_path)
        map_name = os.path.splitext(os.path.basename(file_path))[0]

        # Check for existing point files and delete them
        point_files = sorted(glob.glob(os.path.join(map_dir, map_name + "_points*.json")))
        for file in point_files:
            os.remove(file)

        # Update tgt params
        tgt_params.target_list = [cat.strip() for cat in dpg.get_value("target_list").split(",")]
        tgt_params.penalty_list = [cat.strip() for cat in dpg.get_value("avoid_list").split(",")]
        tgt_params.parseLists(MM_model)
        tgt_params.checkLists(MM_model)

        tgt_params.sparse = dpg.get_value("sparse_targets")
        tgt_params.edge = dpg.get_value("target_edge")
        tgt_params.penalty = dpg.get_value("penalty_weight")
        tgt_params.threshold = dpg.get_value("target_score_threshold")
        tgt_params.max_tilt = dpg.get_value("max_tilt")
        tgt_params.extra_track = dpg.get_value("extra_tracking")

        mic_params.IS_limit = dpg.get_value("IS_limit")

        # Load overlay
        ins_loadMask(None, tgt_params.target_list)

        # Run target selection
        space_ext.runTargetSelection(map_dir, map_name, tgt_params, mic_params, MM_model, alt_seg_path=seg_path, save_final_plot=False)

        ins_showTargets(load_from_file=True)
        # Delete save button (only activated when point was dragged)
        if dpg.does_item_exist("ins_buttsexp"): dpg.delete_item("ins_buttsexp")


def ins_showTargets(load_from_file=False):
    global tgt_overlay_dims, target_areas
    map_dir = os.path.dirname(file_path)
    map_name = os.path.splitext(os.path.basename(file_path))[0]

    if load_from_file:
        # Load json data for all point files
        point_files = sorted(glob.glob(os.path.join(map_dir, map_name + "_points*.json")))
        if len(point_files) > 0:
            target_areas = []
            for file in point_files:
                # Load json data
                with open(file, "r") as f:
                    target_areas.append(json.load(f, object_hook=space_ext.revertArray))
        else:
            return

    # Delete previous target plots
    for i in range(1000):
        if dpg.does_item_exist("ins_tgtplot" + str(i)): dpg.delete_item("ins_tgtplot" + str(i))   
        if dpg.does_item_exist("ins_tgtdrag" + str(i)): dpg.delete_item("ins_tgtdrag" + str(i)) 
        if dpg.does_item_exist("ins_tgtoverlayplot" + str(i)): 
            dpg.delete_item("ins_tgtoverlayplot" + str(i))
        else:
            break

    if not dpg.does_item_exist("ins_tgtoverlay"):
        # Generate target overlay texture
        rec_dims = np.array(tgt_params.weight.shape)
        tgt_overlay = np.zeros([int(MM_model.beam_diameter), int(MM_model.beam_diameter / np.cos(np.radians(tgt_params.max_tilt)))])
        canvas = Image.fromarray(tgt_overlay).convert('RGB')
        draw = ImageDraw.Draw(canvas)
        draw.ellipse((0, 0, tgt_overlay.shape[1] - 1, tgt_overlay.shape[0] - 1), outline="#ffd700", width=10)
        canvas = canvas.rotate(-mic_params.view_ta_rotation, expand=True)
        rect = ((canvas.width - rec_dims[1]) // 2, (canvas.height - rec_dims[0]) // 2, (canvas.width + rec_dims[1]) // 2, (canvas.height + rec_dims[0]) // 2)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(rect, outline="#578abf", width=10)
        tgt_overlay = np.array(canvas).astype(float) / 255

        draw.rectangle(rect, outline="#c92b27", width=10)
        trk_overlay = np.array(canvas).astype(float) / 255


        tgt_overlay_dims = np.array(tgt_overlay.shape)[:2]
        alpha = np.zeros(tgt_overlay.shape[:2])
        alpha[np.sum(tgt_overlay, axis=-1) > 0] = 1
        tgt_overlay_image = np.ravel(np.dstack([tgt_overlay, alpha]))
        trk_overlay_image = np.ravel(np.dstack([trk_overlay, alpha]))

        with dpg.texture_registry():
            dpg.add_static_texture(width=int(tgt_overlay_dims[1]), height=int(tgt_overlay_dims[0]), default_value=tgt_overlay_image, tag="ins_tgtoverlay")
            dpg.add_static_texture(width=int(tgt_overlay_dims[1]), height=int(tgt_overlay_dims[0]), default_value=trk_overlay_image, tag="ins_trkoverlay")

    tgt_counter = 0
    for t, target_area in enumerate(target_areas):
        if len(target_area["points"]) == 0: continue
        # Transform coords to plot
        x_vals = target_area["points"][:, 1] * pix_size_png / 10000
        y_vals = dims[1] * binning - target_area["points"][:, 0] * pix_size_png / 10000

        dpg.add_scatter_series(x_vals, y_vals, tag="ins_tgtplot" + str(t), parent="ins_x_axis")
        # Load color if not out of bounds of prepared themes
        if dpg.does_item_exist("scatter_theme" + str(t)):
            dpg.bind_item_theme("ins_tgtplot" + str(t), "scatter_theme" + str(t))

        for p in range(len(x_vals)):
            # add draggable point
            dpg.add_drag_point(label="tgt_" + str(p + 1).zfill(3), user_data="pt_" + str(t) + "_" + str(p), tag="ins_tgtdrag" + str(tgt_counter), color=cluster_colors[t % len(cluster_colors)], default_value=(x_vals[p], y_vals[p]), callback=ins_dragPointUpdate, parent="ins_plot")

            scaled_overlay_dims = tgt_overlay_dims * pix_size_png / 10000
            bounds_min = (x_vals[p] - scaled_overlay_dims[1] // 2, y_vals[p] - scaled_overlay_dims[0] // 2)
            bounds_max = (x_vals[p] + scaled_overlay_dims[1] // 2, y_vals[p] + scaled_overlay_dims[0] // 2)
            if p == 0:
                dpg.add_image_series("ins_trkoverlay", bounds_min=bounds_min, bounds_max=bounds_max, parent="ins_x_axis", tag="ins_tgtoverlayplot" + str(tgt_counter))
            else:
                dpg.add_image_series("ins_tgtoverlay", bounds_min=bounds_min, bounds_max=bounds_max, parent="ins_x_axis", tag="ins_tgtoverlayplot" + str(tgt_counter))

            tgt_counter += 1


def ins_dragPointUpdate(sender, app_data, user_data):
    coords = dpg.get_value(sender)[:2]
    if dpg.does_item_exist("ins_tempplot"): dpg.delete_item("ins_tempplot")
    dpg.add_scatter_series([coords[0]], [coords[1]], tag="ins_tempplot", parent="ins_x_axis")
    dpg.bind_item_theme("ins_tempplot", "scatter_theme3")   # red theme

def ins_tgtUpdate():
    # Only execute when targes are loaded
    if "target_areas" not in globals():
        return
    
    # Go through all points
    update = False
    for i in range(1000):
        # Check if point exists
        if dpg.does_item_exist("ins_tgtdrag" + str(i)):
            # Get coords from drag point value
            coords = np.array(dpg.get_value("ins_tgtdrag" + str(i))[:2])
            # Get area and point IDs from user data embedded in drag point
            point_id = np.array(dpg.get_item_user_data(("ins_tgtdrag" + str(i))).split("_")[1:], dtype=int)
            # Transform points to plot points for comparison
            old_coords = np.array([target_areas[point_id[0]]["points"][point_id[1]][1] * pix_size_png / 10000, dims[1] * binning - target_areas[point_id[0]]["points"][point_id[1]][0] * pix_size_png / 10000])
            # Go to next points if coords have not changed
            if np.all(coords == old_coords):
                continue
            else:
                # Update coords if they have changed
                target_areas[point_id[0]]["points"][point_id[1]][1] = coords[0] / pix_size_png * 10000
                target_areas[point_id[0]]["points"][point_id[1]][0] = (dims[1] * binning - coords[1]) / pix_size_png * 10000
                update = True
        else:
            break
    # Replot targets if any coords have changed
    if update:
        ins_showTargets()
        if not dpg.does_item_exist("ins_buttsexp"):
            dpg.add_button(label="Save", callback=ins_exportPoints, tag="ins_buttsexp", parent="ins_butgrp")

# Export points
def ins_exportPoints():
    map_dir = os.path.dirname(file_path)
    map_name = os.path.splitext(os.path.basename(file_path))[0]
    if len(target_areas) > 0:
        for t, target_area in enumerate(target_areas):
            with open(os.path.join(map_dir, map_name + "_points" + str(t) + ".json"), "w+") as f:
                json.dump(target_area, f, indent=4, default=space_ext.convertArray)
    else:
        # Write empty points file to ensure empty targets file is written and map is considered processed
        with open(os.path.join(map_dir, map_name + "_points.json"), "w+") as f:
            json.dump({"points": []}, f)    

    # Delete save button (only activated when point was dragged)
    if dpg.does_item_exist("ins_buttsexp"): dpg.delete_item("ins_buttsexp")

# Add points by shift + left clicking
def ins_mouse_click_left(mouse_coords, mouse_coords_global):
    # Check if mouse click was in plot range and if Shift is pressed (to not double signal when dragging)
    if dpg.is_key_down(dpg.mvKey_Shift) and "target_areas" in globals() and np.all(mouse_coords > 0) and mouse_coords_global[0] > 200:
        # Transform mouse coords to px coords
        img_coords = np.array([(dims[1] * binning - mouse_coords[1]) / pix_size_png * 10000, mouse_coords[0] / pix_size_png * 10000])

        # Get camera dims
        rec_dims = np.array(tgt_params.weight.shape)

        # Check if coords are out of bounds
        if not rec_dims[0] <= img_coords[0] < dims[1] * binning - rec_dims[0] or not rec_dims[1] <= img_coords[1] < dims[0] * binning - rec_dims[1]:
            return
        
        if len(target_areas[0]["points"]) > 0:
            # Check if coords are too close to existing point (also allows for dragging to work without creating new point)
            for target_area in target_areas:
                for point in target_area["points"]:
                    if np.linalg.norm(point - img_coords) < np.min(rec_dims):
                        print("WARNING: Target is too close to existing target! It will not be added.")
                        return

            # Figure out which target area tracking targets is closest
            track_points = [target_area["points"][0] for target_area in target_areas]
            closest_area = np.argmin(np.linalg.norm(track_points - img_coords, axis=1))

            # Add point
            target_areas[closest_area]["points"] = np.vstack([target_areas[closest_area]["points"], img_coords])
            target_areas[closest_area]["scores"] = np.append(target_areas[closest_area]["scores"], [1])
        else:
            target_areas[0]["points"] = img_coords[np.newaxis, :]
            target_areas[0]["scores"] = np.array([1])

        print("NOTE: Added new target!")

        ins_showTargets()
        if not dpg.does_item_exist("ins_buttsexp"):
            dpg.add_button(label="Save", callback=ins_exportPoints, tag="ins_buttsexp", parent="ins_butgrp")

#Call SPACEtomo_runNapari script
def openNapari():
    # If layer folder already exists, open folder
    map_name = os.path.splitext(os.path.basename(file_path))[0].split("_0000")[0]
    folder_name = os.path.join(os.path.dirname(seg_path), map_name)

    if os.path.exists(os.path.join(folder_name, map_name + ".png")):
        print("NOTE: Opening exported layer folder in Napari. Closing Napari will overwrite layers.")
        subprocess.Popen(["python", os.path.join(SPACE_DIR, "SPACEtomo_runNapari.py"), "--folder", folder_name])

    # if not: open segmentation
    else:
        print("NOTE: Opening segmentation in Napari. Closing Napari will export layers.")
        subprocess.Popen(["python", os.path.join(SPACE_DIR, "SPACEtomo_runNapari.py"), "--file_path", file_path, "--seg_path", seg_path])


# Delete points by right clicking
def ins_mouse_click_right(mouse_coords, mouse_coords_global):
    # Check if mouse click was in plot range
    if "target_areas" in globals() and np.all(mouse_coords > 0) and mouse_coords_global[0] > 200:
        # Transform mouse coords to px coords
        img_coords = np.array([(dims[1] * binning - mouse_coords[1]) / pix_size_png * 10000, mouse_coords[0] / pix_size_png * 10000])

        # Get camera dims
        rec_dims = np.array(tgt_params.weight.shape)

        # Check if coords are out of bounds
        if not 0 <= img_coords[0] < dims[1] * binning or not 0 <= img_coords[1] < dims[0] * binning:
            return

        # Check for point within range
        if len(target_areas[0]["points"]) > 0:
            # Check if coords are too close to existing point (also allows for dragging to work without creating new point)
            closest_point_id = np.zeros(2)
            closest_point_dist = 1e9
            for t, target_area in enumerate(target_areas):
                for p, point in enumerate(target_area["points"]):
                    dist = np.linalg.norm(point - img_coords)
                    if dist < closest_point_dist:
                        closest_point_dist = dist
                        closest_point_id = np.array([t, p])

            # If closest point is in range
            if closest_point_dist < np.min(rec_dims):
                print(target_areas)
                target_areas[closest_point_id[0]]["points"] = np.delete(target_areas[closest_point_id[0]]["points"], closest_point_id[1], axis=0)
                target_areas[closest_point_id[0]]["scores"] = np.delete(target_areas[closest_point_id[0]]["scores"], closest_point_id[1], axis=0)
                print(target_areas)
                print("NOTE: Target was deleted!")

                ins_showTargets()
                if not dpg.does_item_exist("ins_buttsexp"):
                    dpg.add_button(label="Save", callback=ins_exportPoints, tag="ins_buttsexp", parent="ins_butgrp")            
            

##### Training functions
    
def tra_newDataset():
    dpg.delete_item("tra_new")
    dpg.delete_item("tra_old")
    dpg.set_value("tra_1", "1. Load segmentations")

    dataset_list = sorted(glob.glob(os.path.join(NN_RAW, "Dataset*", "")))

    if len(dataset_list) > 0:
        combo_datasets = []
        for old_dataset in dataset_list:
            img_path = os.path.join(old_dataset, "imagesTr")
            num_images = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
            combo_datasets.append(os.path.basename(os.path.dirname(old_dataset)) + " (" + str(num_images) + " images)")

        dpg.add_combo(combo_datasets, default_value=combo_datasets[0], tag="tra_dataset", parent="tra_left", before="tra_2")
        dpg.add_button(label="Import from dataset", callback=tra_loadSegmentations, parent="tra_left", before="tra_2")
        dpg.add_text("", parent="tra_left", before="tra_2")

    dpg.add_button(label="Add layer folder", callback=lambda: dpg.show_item("tra_file1"), parent="tra_left", before="tra_2")

# Load old dataset for training UI
def tra_loadDataset():
    global dataset, tra_classes, class_number, stats

    results_list = sorted(glob.glob(os.path.join(NN_RAW, "Dataset*", "")))

    # Check if raw AND preprocessed dataset exist
    if len(results_list) > 0 and os.path.exists(os.path.basename(os.path.dirname(NN_PRE)).join(results_list[-1].split(os.path.basename(os.path.dirname(NN_RAW))))):
        dpg.delete_item("tra_new")
        dpg.delete_item("tra_old")

        tra_classes, _, num_img = loadDatasetJson(os.path.join(results_list[-1], "dataset.json"))
        tra_classes.pop("background")

        class_number = len(tra_classes)

        dataset = [{} for i in range(num_img)]

        dpg.set_value("tra_head", "Training dataset statistics:")
        dpg.set_value("tra_img","Images: " + str(len(dataset)))
        dpg.set_value("tra_cls","Classes: " + str(class_number))

        if os.path.exists(os.path.join(results_list[-1], "dataset_stats.json")):
            with open(os.path.join(results_list[-1], "dataset_stats.json"), "r") as f:
                stats = json.load(f)
            tra_makeStatTable()

        training_num = int(os.path.normpath(results_list[-1]).split("set")[-1].split("_")[0])

        tra_generateTrainUI(training_num)

        tra_makeModelTable(training_num)

    else:
        print("WARNING: No previous dataset found!")

#def tra_loadOutput():
#    output_list = sorted(glob.glob(os.path.join(CUR_DIR, "output*/")))
#    for output_dir in output_list:
#        tra_loadLayers(output_dir)

# Load segmentations from old dataset for new dataset (not additive to previously loaded data)
def tra_loadSegmentations():
    global dataset, stats, tra_pix_size_input, tra_classes, class_number

    if "dataset" not in globals():
        dataset = []
    if "tra_classes" not in globals():
        tra_classes = {}
    if "stats" not in globals():
        stats = {}

    training_num = int(dpg.get_value("tra_dataset").split()[0].split("set")[-1])

    # Read classes from dataset.json
    tra_classes, tra_pix_size_input, _ = loadDatasetJson(os.path.join(NN_RAW, "Dataset" + str(training_num).zfill(3), "dataset.json"))
    tra_classes.pop("background")

    # Set image and label folders
    imagesTr = os.path.join(NN_RAW, "Dataset" + str(training_num).zfill(3), "imagesTr")
    labelsTr = os.path.join(NN_RAW, "Dataset" + str(training_num).zfill(3), "labelsTr")

    # Get list of images from labels folder
    img_list = sorted(glob.glob(os.path.join(labelsTr, "*.png")))
    img_list = [os.path.basename(os.path.splitext(path)[0]) for path in img_list]

    # Show progress bar
    dpg.add_progress_bar(default_value=0, width=-1, overlay="0%", tag="tra_importbar", parent="tra_left", before="tra_2")

    # Go through images and import img and segmentation
    for i, img_name in enumerate(img_list):
        # Update progress bar
        progress = i / len(img_list)
        dpg.set_value("tra_importbar", progress)
        dpg.configure_item("tra_importbar", overlay=f"{round(progress * 100)}%")
    
        # Load
        seg = np.array(Image.open(os.path.join(labelsTr, img_name + ".png")), dtype=np.uint8)
        img = np.array(Image.open(os.path.join(imagesTr, img_name + "_0000.png")), dtype=np.uint8)

        # Generate layers from segmentation
        layers = []
        layer_names = []
        for layer_name in tra_classes.keys():
            layer = np.zeros(seg.shape, dtype=bool)
            layer[seg == tra_classes[layer_name]] = True

            layers.append(layer)
            layer_names.append(layer_name)

            # Stats
            tra_addStats(layer, layer_name)

        dataset.append({"name": img_name, "image": img, "layers": np.array(layers), "layer_names": layer_names})

    # Delete progress bar
    dpg.delete_item("tra_importbar")

    # Show dataset info
    tra_showInfo()

def tra_addFolder(sender, app_data):
    layer_path = app_data["file_path_name"]
    tra_loadLayers(layer_path)

# Load load layers of manual segmentations for new dataset (additive)
def tra_loadLayers(layer_path):
    global dataset, stats, tra_pix_size_input, tra_classes

    if "dataset" not in globals():
        dataset = []
    if "tra_classes" not in globals():
        tra_classes = {}
    if "stats" not in globals():
        stats = {}

    # Check if pixel size is compatible
    _, pix_size, _ = loadDatasetJson(os.path.join(layer_path, "dataset.json"))
    if "tra_pix_size_input" in globals() and tra_pix_size_input is not None:
        if tra_pix_size_input != pix_size:
            print("ERROR: Pixel size is not compatible with already loaded data.")
            return
    else:
        tra_pix_size_input = pix_size

    # Find all subdirs
    img_list = sorted(glob.glob(os.path.join(layer_path, "*", "")))
    img_list = [os.path.basename(os.path.dirname(path)) for path in img_list]
    print("Found images: " + str(len(img_list)))

    dpg.add_progress_bar(default_value=0, width=-1, overlay="0%", tag="tra_importbar", parent="tra_left", before="tra_2")
    
    for i, img_name in enumerate(img_list):
        # Check if image is already in training data
        if any(data.get("name") == img_name for data in dataset):
            print("WARNING: Image is already part of dataset. Skipping " + img_name + "...")
            continue

        # Search first for photoshop outputs (saves as .png.png because of imported layer names)
        layer_list = sorted(glob.glob(os.path.join(layer_path, img_name, "*.png.png")), reverse=True)
        # If no photoshop output, search for normal .png endings
        if len(layer_list) == 0:
            layer_list = sorted(glob.glob(os.path.join(layer_path, img_name, "*.png")), reverse=True)

        layers = []
        layer_names = []
        for l, layer_file in enumerate(layer_list):
            progress = (i * (len(layer_list)) + l) / (len(img_list) * len(layer_list))
            dpg.set_value("tra_importbar", progress)
            dpg.configure_item("tra_importbar", overlay=f"{round(progress * 100)}%")

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
            if layer_name not in tra_classes.keys():
                tra_classes[layer_name] = len(tra_classes) + 1

            #print(layer_names[-1], np.min(layer), np.max(layer), np.mean(layer))

            # Stats
            tra_addStats(layer, layer_name)

        dataset.append({"name": img_name, "image": img, "layers": np.array(layers), "layer_names": layer_names})

    # Delete progress bar
    dpg.delete_item("tra_importbar")

    # Show dataset info
    tra_showInfo()

def tra_showInfo():
    global class_number
    if len(dataset) > 0:
        class_number = len(tra_classes)

        dpg.set_value("tra_head", "Training dataset statistics:")
        dpg.set_value("tra_img","Images: " + str(len(dataset)))
        dpg.set_value("tra_cls","Classes: " + str(class_number))
        tra_makeStatTable()

        dpg.set_value("tra_2", "\n2. Select classes")
        dpg.set_value("tra_3", "\n3. Select pixel size")

        # Get pixel sizes from latest model
        _, pix_size_model, _ = loadDatasetJson(os.path.join(CUR_DIR, model_list[-1], "dataset.json"))
        if pix_size_model is None:
            pix_size_model = -1

        if dpg.does_item_exist("tra_pix1"): dpg.delete_item("tra_pix1")
        if dpg.does_item_exist("tra_pix2"): dpg.delete_item("tra_pix2")
        with dpg.group(tag="tra_pix1", horizontal=True, parent="tra_left", before="tra_4"):
            dpg.add_text("Input:")
            # Make editable only if pixel size is unknown
            if "tra_pix_size_input" in globals() and tra_pix_size_input is not None:
                dpg.add_text(tag="tra_pixsize_in", default_value=round(tra_pix_size_input, 2))
                dpg.add_text(" [Å/px]")
            else:
                dpg.add_input_float(tag="tra_pixsize_in", default_value=-1, format="%.2f", step=0, width=50, label="[Å/px]")
        with dpg.group(tag="tra_pix2", horizontal=True, parent="tra_left", before="tra_4"):
            dpg.add_text("Training:")
            dpg.add_input_float(tag="tra_pixsize_out", default_value=round(pix_size_model, 2), format="%.2f", step=0, width=50, label="[Å/px]")
        dpg.set_value("tra_4", "\n4. Create training data")
        if dpg.does_item_exist("tra_butsav"): dpg.delete_item("tra_butsav")
        dpg.add_button(label="Save dataset", callback=tra_saveDataset, tag="tra_butsav", parent="tra_left", before="tra_5")

def tra_addStats(layer, layer_name):
    global stats

    if np.max(layer) > 0:
        if layer_name in stats.keys():
            stats[layer_name]["occurrence"] += 1
            stats[layer_name]["pixels"] += np.sum(layer) / 1000
        else:
            stats[layer_name] = {"occurrence": 1, "pixels": np.sum(layer) / 1000}
    else:       # Also add classes that are named but empty
        if layer_name not in stats.keys():
            stats[layer_name] = {"occurrence": 0, "pixels": 0}    

def tra_makeStatTable():
    if dpg.does_item_exist("tbl_model"): dpg.delete_item("tbl_model")
    if dpg.does_item_exist("tbl_stats"): dpg.delete_item("tbl_stats")
    dpg.add_table(parent="tra_right", tag="tbl_stats")

    dpg.add_table_column(label="Select", parent="tbl_stats", init_width_or_weight=50, width_fixed=True)
    dpg.add_table_column(label="Class", parent="tbl_stats")
    dpg.add_table_column(label="Occurrence", parent="tbl_stats")
    dpg.add_table_column(label="[%]", parent="tbl_stats")
    dpg.add_table_column(label="Pixels [k]", parent="tbl_stats")
    dpg.add_table_column(label="[%]", parent="tbl_stats")

    for class_name in tra_classes.keys():
        with dpg.table_row(parent="tbl_stats"):
            with dpg.table_cell():
                dpg.add_checkbox(tag="check_cls_" + class_name, callback=tra_updateClassNumber, default_value=True)
            with dpg.table_cell():
                dpg.add_text(class_name)
            with dpg.table_cell():
                dpg.add_text(str(round(stats[class_name]["occurrence"])))
            with dpg.table_cell():
                dpg.add_text(str(round(stats[class_name]["occurrence"] / len(dataset) * 100, 1)))
            with dpg.table_cell():
                dpg.add_text(str(round(stats[class_name]["pixels"]))) 
            with dpg.table_cell():
                dpg.add_text(str(round(stats[class_name]["pixels"] / np.sum([x["pixels"] for x in stats.values()]) * 100, 1)))  

    with dpg.table_row(parent="tbl_stats"):
        with dpg.table_cell():
            dpg.add_text("")


def tra_updateClassNumber(sender, app_data):
    global class_number

    if app_data:
        class_number += 1
    else:
        class_number -= 1
    dpg.set_value("tra_cls", "Classes: " + str(class_number))
    
def tra_saveDataset():
    dpg.set_value("tra_5", "Saving data...")
    dpg.add_progress_bar(default_value=0, width=-1, overlay="0%", tag="tra_exportbar", parent="tra_left", before="tra_proc")

    # Find latest training folder
    training_list = sorted(glob.glob(os.path.join(NN_RAW, "Dataset*/")))
    if len(training_list) > 0:
        training_num = int(os.path.normpath(training_list[-1]).split("set")[-1]) + 1
    else:
        training_num = 1

    imagesTr = os.path.join(NN_RAW, "Dataset" + str(training_num).zfill(3), "imagesTr")
    labelsTr = os.path.join(NN_RAW, "Dataset" + str(training_num).zfill(3), "labelsTr")
    os.makedirs(imagesTr)
    os.makedirs(labelsTr)

    # Check class selection and assign proper segmentation value
    class_value = 0
    for class_name in tra_classes:
        if dpg.get_value("check_cls_" + class_name):
            class_value += 1
            tra_classes[class_name] = class_value
        else:
            tra_classes[class_name] = 0

    # Get pixel sizes
    pix_size_in = round(float(dpg.get_value("tra_pixsize_in")), 2)
    pix_size_out = round(float(dpg.get_value("tra_pixsize_out")), 2)

    # Save images and segmentations
    for d, data in enumerate(dataset):
        # Update progress bar
        progress = d / len(dataset)
        dpg.set_value("tra_exportbar", progress)
        dpg.configure_item("tra_exportbar", overlay=f"{round(progress * 100)}%")

        # Rescale image first to get shape
        if pix_size_in != pix_size_out:
            img_out = np.uint8(transform.rescale(data["image"], pix_size_in / pix_size_out) * 255)
        else:
            img_out = data["image"]

        # Rescale boolean layers individually to avoid scaling artifacts and ensure proper pixel values
        segmentation = np.zeros(img_out.shape, dtype=np.uint8)
        for l, layer in enumerate(data["layers"]):
            if pix_size_in != pix_size_out:
                layer_scaled = transform.rescale(layer, pix_size_in / pix_size_out)
            else:
                layer_scaled = layer

            segmentation[layer_scaled == True] = tra_classes[data["layer_names"][l]]

        Image.fromarray(img_out).save(os.path.join(imagesTr, data["name"] + "_0000.png"))
        Image.fromarray(segmentation).save(os.path.join(labelsTr, data["name"] + ".png"))
    print("Images saved: " + imagesTr)
    print("Segmentations saved: " + labelsTr)

    dpg.set_value("tra_5", "Saved.")
    # Delete progress bar
    dpg.delete_item("tra_exportbar")

    # Make dataset.json
    tra_makeDatasetJson(training_num, pix_size_out)
    tra_makeStatsJson(training_num)

    # Run nnU-Net preprocessing
    tra_preprocess(training_num)

    tra_generateTrainUI(training_num)

        
def tra_makeDatasetJson(training_num, pix_size):
    dataset_json = {"channel_names": {"0": "map"}, "labels": {"background": 0}, "numTraining": len(dataset), "file_ending": ".png", "pixel_size": pix_size}
    for class_name in tra_classes.keys():
        if tra_classes[class_name] > 0:
            dataset_json["labels"][class_name] = tra_classes[class_name]

    save_path = os.path.join(NN_RAW, "Dataset" + str(training_num).zfill(3), "dataset.json")
    with open(save_path, "w+") as f:
        json.dump(dataset_json, f, indent=4)
    print("Dataset meta data saved: " + save_path)

def tra_makeStatsJson(training_num):
    save_path = os.path.join(NN_RAW, "Dataset" + str(training_num).zfill(3), "dataset_stats.json")
    with open(save_path, "w+") as f:
        json.dump(stats, f, indent=4)
    print("Dataset statistics saved: " + save_path)

def tra_preprocess(training_num):
    dpg.set_value("tra_5", "Preprocessing data...")
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", str(training_num), "--verify_dataset_integrity"])
    dpg.set_value("tra_5", "\n5. Train model")

def tra_generateTrainUI(training_num):
    dpg.set_value("tra_5", "\n5. Train model")
    # Check if more than 1 device are available
    if DEVICE == "cuda":
        device_counter = torch.cuda.device_count()
        device_list = [DEVICE + ":" + str(i) for i in range(device_counter)]
    else:
        device_counter = 1
        device_list = [DEVICE]

    if device_counter > 0:
        if not dpg.does_item_exist("tra_gpu"): 
            dpg.add_combo(device_list, default_value=device_list[0], label="GPU", tag="tra_gpu", parent="tra_left", before="tra_proc")
    if not dpg.does_item_exist("tra_fold"): 
        dpg.add_combo(["fold " + str(i) for i in range(5)], default_value="fold 0", label="Model", tag="tra_fold", parent="tra_left", before="tra_proc")

    if not dpg.does_item_exist("tra_buttra"): 
        dpg.add_button(label="Start training", callback=lambda: tra_train(training_num), tag="tra_buttra", parent="tra_left", before="tra_proc")

def tra_train(training_num):
    if dpg.does_item_exist("tra_gpu"):
        train_device = dpg.get_value("tra_gpu")
    else:
        train_device = DEVICE
    fold_number = dpg.get_value("tra_fold").split("fold ")[-1]

    print("NOTE: Training of fold_" + str(fold_number) + " started on device: " + train_device)

    # Copy environment variable to change cuda devices if necessary
    environ = os.environ.copy()
    if DEVICE == "cuda":
        environ["CUDA_VISIBLE_DEVICES"] = train_device.split(":")[-1]

    subprocess.Popen(["nnUNetv2_train", str(training_num), "2d", str(fold_number), "-device", DEVICE, "--npz"], env=environ)#, shell=True, stdout=out_file, stderr=subprocess.STDOUT, text=True)

    dpg.set_value("tra_proc", "Training in progress...\nCheck nnUNet_results folder \nfor updates!")

    tra_makeModelTable(training_num)

def tra_makeModelTable(training_num):
    # Read log files
    log_data = []
    for f in range(5):
        text_logs = sorted(glob.glob(os.path.join(NN_RES, "Dataset" + str(training_num).zfill(3) + "*", "nnUNetTrainer__nnUNetPlans__2d", "fold_" + str(f), "training_log*.txt")))
        if len(text_logs) > 0:
            with open(text_logs[-1], "r") as f:
                line = f.readline()                 # read one line at the time until start time
                timeout = 0
                while not line.startswith("20") and timeout < 100:
                    line = f.readline()
                    timeout += 1
                start_time = line.split()[:2]
                if len(start_time) == 2:
                    start = datetime.datetime.strptime(start_time[0] + " " + start_time[1][:8], "%Y-%m-%d %H:%M:%S")
                    start_time = start_time[0] + "\n" + start_time[1][:8]
                else:
                    start_time = start = None

                f.seek(0, 2)                        # go to end of file
                fsize = f.tell()
                f.seek(max(fsize - 2048, 0), 0)     # go to end of file - 2048 characters
                lines = f.readlines()[-30:]         # read last 30 lines
            
            epoch = 0
            for line in lines:
                if "Epoch" in line and not "time" in line:
                    epoch = int(line.strip().split()[-1]) + 1
            update_time = lines[-1].split()[:2]
            if len(update_time) == 2:
                update = datetime.datetime.strptime(update_time[0] + " " + update_time[1][:8], "%Y-%m-%d %H:%M:%S")
                update_time = update_time[0] + "\n" + update_time[1][:8]
            else:
                update_time = update = None

            if start is not None and update is not None:
                threshold = datetime.timedelta(minutes=5)
                now = datetime.datetime.now()
                if now - update > threshold and update - start < threshold:
                    error = True
                else:
                    error = False

            log_data.append({"logfile": text_logs[-1], "epoch": epoch, "start": start_time, "update": update_time, "error": error})
        else:
            log_data.append({"logfile": None, "epoch": 0, "start": None, "update": None, "error": False})

    # Make table
    if dpg.does_item_exist("tbl_model"): dpg.delete_item("tbl_model")
    dpg.add_table(parent="tra_right", tag="tbl_model")

    dpg.add_table_column(label="Training", parent="tbl_model")
    for f in range(5):
        dpg.add_table_column(label="Fold " + str(f), parent="tbl_model")


    with dpg.table_row(parent="tbl_model"):
        with dpg.table_cell():
            dpg.add_text("Started")
        
        for f in range(5):
            with dpg.table_cell():
                dpg.add_checkbox(default_value=os.path.exists(os.path.join(NN_RES, "Dataset" + str(training_num).zfill(3), "nnUNetTrainer__nnUNetPlans__2d", "fold_" + str(f))), enabled=False)


    with dpg.table_row(parent="tbl_model"):
        with dpg.table_cell():
            dpg.add_text("Epoch")
        
        for f in range(5):
            with dpg.table_cell():
                    dpg.add_text(str(log_data[f]["epoch"]))


    with dpg.table_row(parent="tbl_model"):
        with dpg.table_cell():
            dpg.add_text("Finished")
        
        for f in range(5):
            with dpg.table_cell():
                dpg.add_checkbox(default_value=os.path.exists(os.path.join(NN_RES, "Dataset" + str(training_num).zfill(3), "nnUNetTrainer__nnUNetPlans__2d", "fold_" + str(f), "checkpoint_final.pth")), enabled=False)


    with dpg.table_row(parent="tbl_model"):
        with dpg.table_cell():
            dpg.add_text("Start time")
        
        for f in range(5):
            with dpg.table_cell():
                    dpg.add_text(log_data[f]["start"])

    with dpg.table_row(parent="tbl_model"):
        with dpg.table_cell():
            dpg.add_text("Last update")
        
        for f in range(5):
            with dpg.table_cell():
                    dpg.add_text(log_data[f]["update"])


    with dpg.table_row(parent="tbl_model"):
        with dpg.table_cell():
            dpg.add_button(label="Refresh", callback=lambda: tra_makeModelTable(training_num))

        for f in range(5):
            with dpg.table_cell():
                if log_data[f]["error"]:
                    dpg.add_text("Possible error\nor slow training.\nCheck the log!", color=error_color)
                if log_data[f]["logfile"] is not None:
                    dpg.add_button(label="Open log", callback=openTxt, user_data=log_data[f]["logfile"])
        
    if all([os.path.exists(os.path.join(NN_RES, "Dataset" + str(training_num).zfill(3), "nnUNetTrainer__nnUNetPlans__2d", "fold_" + str(f), "checkpoint_final.pth")) for f in range(5)]):
        with dpg.table_row(parent="tbl_model"):
            with dpg.table_cell():
                dpg.add_button(label="Export Model", callback=lambda: dpg.show_item("tra_file2"))

def tra_exportModel(sender, app_data):
    model_path = app_data["file_path_name"]
    if os.path.exists(model_path):
        print("ERROR: Path already exists! Please enter a new name!")
        return
    
    results_list = sorted(glob.glob(os.path.join(NN_RES, "Dataset*", "")))
    raw_path = os.path.basename(os.path.dirname(NN_RAW)).join(results_list[-1].split(os.path.basename(os.path.dirname(NN_RES))))
    dataset_path = os.path.join(results_list[-1], "nnUNetTrainer__nnUNetPlans__2d")
    
    for i in range(5):
        os.makedirs(os.path.join(model_path, "fold_" + str(i)))
        shutil.copy(os.path.join(dataset_path, "fold_" + str(i), "checkpoint_best.pth"), os.path.join(model_path, "fold_" + str(i), "checkpoint_best.pth"))

    shutil.copy(os.path.join(dataset_path, "dataset.json"), os.path.join(model_path, "dataset.json"))
    shutil.copy(os.path.join(dataset_path, "plans.json"), os.path.join(model_path, "plans.json"))

    if os.path.exists(os.path.join(raw_path, "dataset_stats.json")):
        shutil.copy(os.path.join(raw_path, "dataset_stats.json"), os.path.join(model_path, "dataset_stats.json"))
    else:
        print("WARNING: Dataset statistics file is missing!")

    print("NOTE: Model was successfuly exported: " + model_path)

def makeLogo(radius=100, stroke=3, oversampling=2):
    radius = radius * oversampling
    stroke = stroke * oversampling

    img = np.zeros([2 * radius, 2 * radius, 4])
    i, j = draw.disk((radius, radius), radius=radius, shape=img.shape[:2])
    img[i, j] = [0, 0, 0, 1]

    i, j = draw.disk((radius, radius), radius=radius - stroke, shape=img.shape[:2])
    gradient = np.array([(x - radius) / radius * np.array([0.965, 0.659, 0.098, 1])  + (1 - (x - radius) / radius) * np.array([0.788, 0.184, 0.157, 1]) for x in i])
    img[i, j] = gradient

    i, j = draw.disk((radius, radius + stroke - 1), radius=radius - stroke * 2, shape=img.shape[:2])
    img[i, j] = [0, 0, 0, 1]

    i, j = draw.disk((radius, radius + stroke - 1), radius=radius - stroke * 3, shape=img.shape[:2])
    img[i, j, :] = 0 

    img = img[radius:2 * radius, :, :]

    img = transform.rescale(img, (1/oversampling, 1/oversampling, 1), anti_aliasing=True)

    return np.ravel(img[::-1]), img.shape


################
    
# Setup nnU-Net folders
if CUR_DIR != SPACE_DIR:
    NN_RAW = os.path.join(CUR_DIR, "nnUNet_raw")
    NN_PRE = os.path.join(CUR_DIR, "nnUNet_preprocessed")
    NN_RES = os.path.join(CUR_DIR, "nnUNet_results")

    if not os.path.exists(NN_RAW): os.makedirs(NN_RAW)
    if not os.path.exists(NN_PRE): os.makedirs(NN_PRE)
    if not os.path.exists(NN_RES): os.makedirs(NN_RES)

    os.environ["nnUNet_raw"] = NN_RAW
    os.environ["nnUNet_preprocessed"] = NN_PRE
    os.environ["nnUNet_results"] = NN_RES
else:
    print("ERROR: Please run SPACEtomo TI in a new folder!")
    sys.exit()

heading_color = (255, 200, 0, 255)
error_color = (200, 0, 0, 255)
subtle_color = (255, 255, 255, 64)

# Colors for figure
class_colors = {"background": (204, 204, 204, 255), "white": (255, 255, 255, 255), "black": (0, 0, 0, 255), "crack": (238, 238, 238, 255), "coating": (85, 85, 85, 255), "cell": (255, 247, 231, 255), "cellwall": (255, 213, 128, 255), "nucleus": (246, 168, 25, 255), "vacuole": (231, 127, 36, 255), "mitos": (201, 47, 40, 255), "lipiddroplets": (216, 88, 40, 255), "vesicles": (216, 88, 40, 255), "multivesicles": (216, 88, 40, 255), "membranes": (216, 88, 40, 255), "dynabeads": (87, 138, 191, 255), "ice": (87, 138, 191, 255), "cryst": (59, 92, 128, 255)}

# Get files in input folder
inf_input_files = []
if os.path.exists(os.path.join(CUR_DIR, "input")):
    inf_input_files = sorted(glob.glob(os.path.join(CUR_DIR, "input", "*.png")))

# Find imported models and symlinks to trained datasets in main folder
model_list = sorted(glob.glob(os.path.join(CUR_DIR, "model*", "")))
model_list = [os.path.basename(os.path.normpath(model_name)) for model_name in model_list]

# Check for datasets finished training and make symlink to use for inference
results_list = sorted(glob.glob(os.path.join(NN_RES, "Dataset*", "")))
for result in results_list:
    if all([os.path.exists(os.path.join(result, "nnUNetTrainer__nnUNetPlans__2d", "fold_" + str(f), "checkpoint_final.pth")) for f in range(5)]):
        result_num = int(os.path.basename(os.path.normpath(result)).split("set")[-1].split("_")[0])
        if not os.path.exists("model_" + str(result_num)):
            os.symlink(os.path.join(result, "nnUNetTrainer__nnUNetPlans__2d"), os.path.join(CUR_DIR, "model_" + str(result_num)))
            model_list.append("model_" + str(result_num))

# Look for mic_params and tgt_params
if FUNC_IMPORT:
    # Check if mic_params exist
    if os.path.exists(os.path.join(CUR_DIR, "mic_params.json")):
        print("NOTE: Microscope parameters found.")
        # Instantiate mic params from settings
        mic_params = space_ext.MicParams_ext(CUR_DIR)        
        # Load model
        MM_model = space_ext.MMModel()
        MM_model.setDimensions(mic_params)

        # Check if tgt_params exist
        if os.path.exists(os.path.join(CUR_DIR, "tgt_params.json")):
            print("NOTE: Target parameters found.")
            # Instantiate tgt params from settings
            tgt_params = space_ext.TgtParams(file_dir=CUR_DIR, MM_model=MM_model)
        else:
            # Use defaults
            tgt_params = space_ext.TgtParams(["lamella"], ["black", "white", "crack", "ice"], MM_model, False, False, 0.3, 0.01, 60, mic_params, False, 10)
    else:
        mic_params = None
        tgt_params = None

# Set some default vars
binning = 1
dims = (1, 1)
image = np.zeros(4)
mask = np.zeros(4)

# Make logo
logo, logo_dims = makeLogo()

with dpg.texture_registry():
    dpg.add_static_texture(width=logo_dims[1], height=logo_dims[0], default_value=logo, tag="logo")

# Create themes for plots
cluster_colors = [(87, 138, 191, 255), (229, 242, 255, 255), (59, 92, 128, 255), (200, 0, 0, 255)]
for c, color in enumerate(cluster_colors):
    with dpg.theme(tag="scatter_theme" + str(c)):
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)


# Create file dialogues

with dpg.file_dialog(directory_selector=False, show=False, callback=ins_loadMap, tag="ins_file1", cancel_callback=cancel_callback, width=700 ,height=400): 
    dpg.add_file_extension(".*") 
    dpg.add_file_extension(".png", color=(100, 255, 100, 255))     

with dpg.file_dialog(directory_selector=False, show=False, callback=inf_loadMap, tag="inf_file1", cancel_callback=cancel_callback, width=700 ,height=400): 
    dpg.add_file_extension(".*") 
    dpg.add_file_extension(".mrc", color=(100, 255, 100, 255))   
    dpg.add_file_extension(".map", color=(100, 255, 100, 255))

with dpg.file_dialog(directory_selector=True, show=False, callback=inf_importModel, tag="inf_file2", cancel_callback=cancel_callback, width=700 ,height=400): 
    dpg.add_file_extension(".*") 

with dpg.file_dialog(directory_selector=True, show=False, callback=tra_addFolder, tag="tra_file1", cancel_callback=cancel_callback, width=700 ,height=400): 
    dpg.add_file_extension(".*") 

with dpg.file_dialog(directory_selector=True, show=False, callback=tra_exportModel, tag="tra_file2", cancel_callback=cancel_callback, width=700 ,height=400): 
    dpg.add_file_extension(".*") 

# Create event handlers

with dpg.handler_registry() as mouse_handler:
    m_click = dpg.add_mouse_click_handler(callback=mouse_click)
    m_release_left = dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=ins_tgtUpdate)

dpg.set_viewport_resize_callback(callback=window_size_change)

dpg.create_viewport(title="SPACEtomo TI")
dpg.setup_dearpygui()

with dpg.window(label="GUI", tag="GUI", no_scrollbar=True, no_scroll_with_mouse=True):

    with dpg.tab_bar(label="tabbar", tag="tabbar"):

##### Inference
        with dpg.tab(label="Inference", tag="inf_tab"):
            dpg.add_text("Run inference on MM map tiles using an nnU-Net model:")

            with dpg.table(header_row=False):
                dpg.add_table_column(init_width_or_weight=200, width_fixed=True)
                dpg.add_table_column()

                with dpg.table_row():
                    with dpg.table_cell(tag="inf_left"):
                        dpg.add_text(default_value="1. Load your MM map", tag="inf_1", color=heading_color)
                        with dpg.group(tag="inf_load", horizontal=True):
                            dpg.add_button(label="Find map", callback=lambda: dpg.show_item("inf_file1"))

                        dpg.add_text(default_value="\n\n\n\n\n", tag="inf_2", color=heading_color)
                        dpg.add_text(default_value="", tag="inf_tileid")

                        dpg.add_text(default_value="\n ", tag="inf_3", color=heading_color)
                        dpg.add_text(default_value="", tag="inf_pix")
                        dpg.add_text(default_value="", tag="inf_butexp")
                        dpg.add_text(default_value="", tag="inf_expstatus")
                        if len(inf_input_files) > 0:
                            dpg.add_text(default_value="Total images: " + str(len(inf_input_files)), tag="inf_numimg")
                            dpg.add_text(default_value="\n4. Choose model", tag="inf_4", color=heading_color)
                            if len(model_list) > 0:
                                dpg.add_combo(model_list, tag="inf_selmod", default_value=model_list[-1], callback=inf_checkModel)
                            else:
                                dpg.add_button(label="Find model", callback=lambda: dpg.show_item("inf_file2"), tag="inf_butmod")
                        else: 
                            dpg.add_text(default_value="", tag="inf_numimg")
                            dpg.add_text(default_value="\n", tag="inf_4", color=heading_color)

                        if len(inf_input_files) > 0 and len(model_list) > 0:
                            dpg.add_text(default_value="\n5. Segment images", tag="inf_5", color=heading_color)
                            check_pix_size, *_ = inf_checkPixelSize(model_list)
                            if not check_pix_size:
                                dpg.add_text("Model pixel size differs \nfrom export pixel size. \nPlease reexport images at \nthe proper pixel size.", tag="inf_butinf", color=error_color)
                            else:
                                dpg.add_button(label="Run inference", callback=inf_inference, tag="inf_butinf")
                        else:
                            dpg.add_text(default_value="\n", tag="inf_5", color=heading_color)
                            dpg.add_text(default_value="", tag="inf_butinf")
                        dpg.add_text(default_value="", tag="inf_left_final")

                    with dpg.table_cell():
                        with dpg.plot(label="Map", width=-1, height=-1, equal_aspects=True, tag="inf_plot"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="x [µm]", tag="inf_x_axis")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y [µm]", tag="inf_y_axis")

            # Create tooltips
            with dpg.tooltip("inf_1", delay=0.5):
                dpg.add_text("Select an .mrc or .map lamella montage.\nIf the map looks scrambled, manually adjust the montage dimensions and reorder!")
            with dpg.tooltip("inf_2", delay=0.5):
                dpg.add_text("Left click on map to select a tile")
            with dpg.tooltip("inf_3", delay=0.5):
                dpg.add_text("Choose a pixel size matching the segmentation model and export tile.\nRepeat steps 1-3 until you have >5-10 images.")
            with dpg.tooltip("inf_4", delay=0.5):
                dpg.add_text("Choose or import a segmentation model. Start with a published SPACEtomo model.")
            with dpg.tooltip("inf_5", delay=0.5):
                dpg.add_text("Segment all exported tiles. Make sure the exported pixel size matches the model!")


##### Inspection
        with dpg.tab(label="Inspection", tag="ins_tab"):

            dpg.add_text("Inspect MM map segmentations and export as layers:")

            with dpg.table(header_row=False):
                dpg.add_table_column(init_width_or_weight=200, width_fixed=True)
                dpg.add_table_column()

                with dpg.table_row():
                    with dpg.table_cell(tag="ins_left"):
                        dpg.add_text(default_value="1. Load your MM map", tag="ins_1", color=heading_color)
                        dpg.add_button(label="Find map", callback=lambda: dpg.show_item("ins_file1"))
                        dpg.add_text(default_value="\n", tag="ins_2", color=heading_color)
                        dpg.add_text(default_value="", tag="ins_cls")

                        dpg.add_text(default_value="\n", tag="ins_3", color=heading_color)
                        dpg.add_text(default_value="\n", tag="ins_4", color=heading_color)
                        dpg.add_text(default_value="", tag="ins_final", color=subtle_color)

                    with dpg.table_cell():
                        with dpg.plot(label="Map", width=-1, height=-1, equal_aspects=True, no_menus=True, tag="ins_plot"):
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="x [µm]", tag="ins_x_axis")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y [µm]", tag="ins_y_axis")

            # Create tooltips
            with dpg.tooltip("ins_1", delay=0.5):
                dpg.add_text("Select a map from an input folder. (You can also open a map generated by SPACEtomo for inspection if its segmentation is in the same folder.)")
            with dpg.tooltip("ins_2", delay=0.5):
                dpg.add_text("Select the class to be shown as overlay.")
            with dpg.tooltip("ins_3", delay=0.5):
                dpg.add_text("Export the segmentation (or all segmentations in the same folder) as separate image for each class.")
            with dpg.tooltip("ins_4", delay=0.5):
                dpg.add_text("Open the exported images as layers in your graphics editor, correct them, and export them again in the same folder.")

            if dpg.does_item_exist("ins_butnap"):
                with dpg.tooltip("ins_butnap", delay=0.5):
                    dpg.add_text("Open segmentation as layers in Napari. Layers will be automatically saved when Napari is closed.")

##### Training
        with dpg.tab(label="Training", tag="tra_tab"):
            dpg.add_text("Import segmentation layers and create training dataset:")

            with dpg.table(header_row=False):
                dpg.add_table_column(init_width_or_weight=200, width_fixed=True)
                dpg.add_table_column()

                with dpg.table_row():
                    with dpg.table_cell(tag="tra_left"):
                        dpg.add_button(label="Create new dataset", callback=tra_newDataset, tag="tra_new")
                        dpg.add_button(label="Load previous dataset", callback=tra_loadDataset, tag="tra_old")

                        dpg.add_text(default_value="", tag="tra_1", color=heading_color)
                        
                        dpg.add_text(default_value="", tag="tra_2", color=heading_color)

                        dpg.add_text(default_value="", tag="tra_3", color=heading_color)

                        dpg.add_text(default_value="", tag="tra_pix1")
                        dpg.add_text(default_value="", tag="tra_pix2")

                        dpg.add_text(default_value="", tag="tra_4", color=heading_color)

                        dpg.add_text(default_value="", tag="tra_5", color=heading_color)

                        dpg.add_text(default_value="", tag="tra_proc")


                    with dpg.table_cell(tag="tra_right"):
                        dpg.add_text(default_value="", tag="tra_head")
                        dpg.add_text(default_value="", tag="tra_img")
                        dpg.add_text(default_value="", tag="tra_cls")     

            # Create tooltips
            with dpg.tooltip("tra_new", delay=0.5):
                dpg.add_text("Create a new training dataset from layer images.")
            with dpg.tooltip("tra_old", delay=0.5):
                dpg.add_text("Load latest dataset to manage and monitor training.")                                   
            with dpg.tooltip("tra_1", delay=0.5):
                dpg.add_text("Import segmentations from previous dataset or load layer images from manual segmentation.")
            with dpg.tooltip("tra_2", delay=0.5):
                dpg.add_text("Inspect the dataset statistics and select classes to be included in training.")
            with dpg.tooltip("tra_3", delay=0.5):
                dpg.add_text("Choose the pixel size at which you want to train the model.\nThis should not be smaller than the pixel size at which you collect MM maps.")
            with dpg.tooltip("tra_4", delay=0.5):
                dpg.add_text("Rescale and save the images and preprocess the training dataset.")
            with dpg.tooltip("tra_5", delay=0.5):
                dpg.add_text("Choose the device and fold to start a training job.\nnnU-Net trains 5 models (folds) with varying training/validation splits to optimize training data utilization.")                   

    dpg.add_image("logo", pos=(10, dpg.get_viewport_height() - 40 - logo_dims[0]), tag="logo_img")
    dpg.add_text(default_value="SPACEtomo TI", pos=(10 + logo_dims[1] / 2 - (40), dpg.get_viewport_height() - 40 - logo_dims[0] / 2), tag="logo_text")
    dpg.add_text(default_value="v" + versionSPACE, pos=(dpg.get_viewport_width() - 100, 10), tag="version_text")

#dpg.set_value("tabbar", "ins_tab")
dpg.set_primary_window("GUI", True)

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()