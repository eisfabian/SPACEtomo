#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_TI
# Purpose:      User interface for training SPACEtomo segmentation models using nnU-Netv2
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/03/20
# Revision:     v1.1
# Last Change:  2024/04/09: fixes after Rado Krios test, added default binning to speed up loading
# ===================================================================

##### SETTINGS #####

thumbnails = True               # takes time on startup to load all maps
default_binning = 1             # higher binning saves VRAM and loads faster at the cost of map detail

### END SETTINGS ###

import os
os.environ["__GLVND_DISALLOW_PATCHING"] = "1"           # helps to minimize Segmentation fault crashes on Linux when deleting textures
import sys
import copy
import glob
import json
try:
    import dearpygui.dearpygui as dpg
except:
    print("ERROR: DearPyGUI module not installed! If you cannot install it, please run the target selection GUI from an external machine.")
    sys.exit()
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import time
import argparse
from skimage import transform, draw

import SPACEtomo_functions_ext as space_ext

SPACE_DIR = os.path.dirname(__file__)

print("\n########################################\nRunning SPACEtomo Target Selection GUI [" + os.path.basename(__file__) + "]\n########################################\n")

# Process arguments
parser = argparse.ArgumentParser(description='Calls GUI to inspect lamella maps and select targets.')
parser.add_argument('map_dir', nargs="?", type=str, default="", help='Absolute path to folder containing lamella maps. This should be the same directory that was set in SPACEtomo on the SerialEM PC. (Default: Folder this script is run from)')
args = parser.parse_args()

if args.map_dir != "": 
    if os.path.exists(args.map_dir):
        CUR_DIR = args.map_dir
    else:
        print("ERROR: Folder does not exist!")
        sys.exit()
else:
    CUR_DIR = os.getcwd()
print("NOTE: Opening " + CUR_DIR)

dpg.create_context()

### FUNCTIONS ###

def mouseClick(sender, app_data):
    mouse_coords = np.array(dpg.get_plot_mouse_pos())
    mouse_coords_global = dpg.get_mouse_pos(local=False)    # need global coords, because plot coords give last value at edge of plot when clicking outside of plot

    # Left mouse button functions
    if dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and dpg.is_key_down(dpg.mvKey_Shift):
        tgt_mouseClick_left(mouse_coords, mouse_coords_global)

    # Right mouse button functions
    elif dpg.is_mouse_button_down(dpg.mvMouseButton_Right) or (dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and dpg.is_key_down(dpg.mvKey_D)):
        tgt_mouseClick_right(mouse_coords, mouse_coords_global)   

    # Middle mouse button functions
    elif dpg.is_mouse_button_down(dpg.mvMouseButton_Middle) or (dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and dpg.is_key_down(dpg.mvKey_G)):
        tgt_mouseClick_middle(mouse_coords, mouse_coords_global)

def window_size_change():
    # Update items anchored to side of window
    dpg.set_item_pos("logo_img", pos=(10, dpg.get_viewport_height() - 40 - logo_dims[0]))
    dpg.set_item_pos("logo_text", pos=(10 + logo_dims[1] / 2 - (30), dpg.get_viewport_height() - 40 - logo_dims[0] / 2))
    dpg.set_item_pos("version_text", pos=(10 + logo_dims[1] / 2 - (30), dpg.get_viewport_height() - 27 - logo_dims[0] / 2))

def cancel_callback():
    pass

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

def hideButtons():
    dpg.hide_item("tgt_butload")
    dpg.hide_item("tgt_butnext")    
    dpg.hide_item("tgt_butgrid")
    # Show spacer
    dpg.set_value("tgt_mapstatus", "")
    dpg.show_item("tgt_mapstatus")
    dpg.set_value("tgt_tgtstatus", "")
    dpg.show_item("tgt_tgtstatus")

    dpg.hide_item("tgt_butts")
    dpg.hide_item("tgt_buttsexp")
    dpg.hide_item("tgt_buttsins")    

def showButtons():
    dpg.show_item("tgt_butload")

    map_name = dpg.get_value("tgt_map")
    map_id = map_list.index(map_name)
    if map_id < len(map_list) - 1:
        dpg.show_item("tgt_butnext") 
    dpg.show_item("tgt_butgrid")

    dpg.hide_item("tgt_mapstatus")
    dpg.hide_item("tgt_tgtstatus")

    # find conditions
    if dpg.does_item_exist("tgt_butts"):
        dpg.show_item("tgt_butts")
    if dpg.does_item_exist("tgt_buttsins"):
        dpg.show_item("tgt_buttsins")

##### Inspection functions ######

def tgt_chooseMap(next_map=False):
    global map_name
    map_name = dpg.get_value("tgt_map")

    # Update map list on every button click
    tgt_updateMapList()

    if next_map:
        map_id = map_list.index(map_name) + 1
        if map_id >= len(map_list):
            map_id = 0
            print("WARNING: Reached end of list of maps. Loading first map!")
        map_name = map_list[map_id]

    dpg.set_value("tgt_map", map_name)

    tgt_loadMap()

def map_chooseMap(sender, app_data):
    global map_name
    map_name = dpg.get_item_label(sender)

    print("Loading " + map_name + "...")

    dpg.hide_item("map_window")

    # Update map list on every button click
    tgt_updateMapList()
    
    dpg.set_value("tgt_map", map_name)

    tgt_loadMap()


def tgt_loadMap():
    global map_name, dims, dims_microns, binning, inspected, target_areas
    file_path = os.path.join(CUR_DIR, map_name + ".png")

    # Hide buttons while loading
    hideButtons()

    # Delete previous target areas
    if "target_areas" in globals():
        del target_areas
        tgt_showTargets()

    # Show status
    dpg.set_value("tgt_mapstatus", "Loading map...")

    dpg.set_item_label("tgt_plot", os.path.basename(file_path) + " loading...")

    dpg.set_item_label("tgt_x_axis", "x [µm]")
    dpg.set_item_label("tgt_y_axis", "y [µm]")
 
    # Open map image
    map_img = Image.open(file_path)

    # Convert map image
    image = np.array(map_img).astype(float) / 255
    #image_orig = copy.deepcopy(image) # make copy for export
    dims = [image.shape[1], image.shape[0]]
    dims_microns = np.array(dims) * MM_model.pix_size / 1000
    binning = default_binning

    if np.max(dims) / binning > 16384:  # hard limit for texture sizes on apple GPU
        binning *= 2
        print("WARNING: Map is too large and will be binned by " + str(binning) + " (for display only)! Export will be unbinned.")

    if binning > 1:
        image = image[::binning, ::binning]
        dims = [image.shape[1], image.shape[0]]


    # Make texture
    image = np.ravel(np.dstack([image, image, image, np.ones(image.shape)]))

    if dpg.does_item_exist("tgt_img"): dpg.delete_item("tgt_img")
    time.sleep(0.1)             # helps to reduce Segmentation fault crashes
    if dpg.does_item_exist("tgt_imgplot"): dpg.delete_item("tgt_imgplot")
    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[0], height=dims[1], default_value=image, tag="tgt_img")
    dpg.add_image_series("tgt_img", bounds_min=(0, 0), bounds_max=np.array(dims) * MM_model.pix_size / 1000 * binning, parent="tgt_x_axis", tag="tgt_imgplot")
    dpg.fit_axis_data("tgt_x_axis")
    dpg.fit_axis_data("tgt_y_axis")

    # Make list of classes
    if dpg.does_item_exist("tgt_class"): dpg.delete_item("tgt_class")
    dpg.add_radio_button(["None"] + [key for key in MM_model.categories.keys()], horizontal=False, default_value="None", callback=tgt_loadMask, tag="tgt_class", parent="tgt_left", before="tgt_lfinal", show=False)

    # Load segmentation
    dpg.set_value("tgt_mapstatus", "Loading segmentation...")
    tgt_loadSeg(os.path.join(CUR_DIR, map_name + "_seg.png"))

    # Show tilt axis
    if mic_params is not None:
        theta = np.radians(mic_params.view_ta_rotation)
        rotM = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        axis_points = (np.array([rotM @ np.array([-dims[1] / 2, 0]), rotM @ np.array([dims[1] / 2, 0])]) + np.array([dims[1] / 2, dims[0] / 2])) * MM_model.pix_size / 1000 * binning
        if dpg.does_item_exist("tgt_taxis"): dpg.delete_item("tgt_taxis")
        dpg.add_line_series(list(axis_points[:, 1]), list(axis_points[:, 0]), label="Tilt axis", tag="tgt_taxis", parent="tgt_x_axis")
        dpg.bind_item_theme("tgt_taxis", "axis_theme")

    # Show list of classes
    dpg.show_item("tgt_cls")
    dpg.show_item("tgt_class")

    # Check for inspected file
    if os.path.exists(os.path.join(CUR_DIR, map_name + "_inspected.txt")):
        inspected = True
        dpg.show_item("tgt_ins")
    else:
        inspected = False
        dpg.hide_item("tgt_ins")
        tgt_showTgtMenu()

    # Show targets if point file exists
    tgt_showTargets(load_from_file=True)

    # Make thumbnail
    if not os.path.exists(os.path.join(CUR_DIR, "thumbnails", map_name + ".png")):
        tgt_makeThumbnail(map_name, map_img, thumbnail_size, override=True)

    dpg.set_item_label("tgt_plot", os.path.basename(file_path))

    # Show buttons again
    showButtons()

    # Reload map window
    tgt_makeMapGrid()


def tgt_loadSeg(file_path):
    global seg, seg_orig
    seg = np.array(Image.open(file_path))
    dims = seg.shape
    if binning > 1:  # hard limit for texture sizes on apple GPU
        seg_orig = copy.deepcopy(seg)
        dims = (dims[0] // binning, dims[1] // binning)
        seg = seg[::binning, ::binning]

    tgt_loadMask(class_list=["None"])


def tgt_loadMask(sender=None, class_list=[]):
    if not isinstance(class_list, list) or len(class_list) == 0:
        class_list = [dpg.get_value("tgt_class")]
    else:
        dpg.set_value("tgt_class", class_list[0])

    # Hide buttons while loading
    hideButtons()
    dpg.set_value("tgt_mapstatus", "Creating overlay...")

    if dpg.does_item_exist("tgt_seg"): dpg.delete_item("tgt_seg")
    if dpg.does_item_exist("tgt_segplot"): dpg.delete_item("tgt_segplot")

    mask = tgt_makeMask(seg, class_list)
    if np.max(mask) > 0:
        mask = np.ravel(np.dstack([mask, np.zeros(mask.shape), np.zeros(mask.shape), mask * np.full(mask.shape, 0.25)]))

        with dpg.texture_registry():
            dpg.add_static_texture(width=dims[0], height=dims[1], default_value=mask, tag="tgt_seg")
        dpg.add_image_series("tgt_seg", bounds_min=(0, 0), bounds_max=np.array(dims) * MM_model.pix_size / 1000 * binning, parent="tgt_x_axis", tag="tgt_segplot")

    showButtons()

# Create mask from segmentation and selected classes
def tgt_makeMask(seg, class_names):
    mask = np.zeros(seg.shape)
    if "None" not in class_names:
        for name in class_names:
            mask[seg == MM_model.categories[name]] = 1

    return mask

def tgt_showTgtMenu():
    if mic_params is not None and tgt_params is not None:
        dpg.show_item("tgt_tsmenu")
        dpg.hide_item("tgt_buttsexp")
    else:
        dpg.set_value("tgt_rfinal", "Target selection not possible.\n(Requires mic_params.json generated by SPACEtomo run!)")

def tgt_showTargets(load_from_file=False):
    global tgt_overlay_dims, geo_overlay_dims, target_areas

    if tgt_params is None:
        print("ERROR: No target parameters found!")
        return

    if load_from_file:
        # Load json data for all point files
        point_files = sorted(glob.glob(os.path.join(CUR_DIR, map_name + "_points*.json")))
        if len(point_files) > 0:
            target_areas = []
            for file in point_files:
                # Load json data
                with open(file, "r") as f:
                    target_areas.append(json.load(f, object_hook=space_ext.revertArray))
            print("NOTE: Loaded target coordinates from file.")
        else:
            print("NOTE: No target coordinates found.")
            if "target_areas" in globals():
                del target_areas
            return

    # Delete previous target plots
    stop_early = [False, False] # break when reaching the end of targets AND geo_points
    for i in range(1000):
        if dpg.does_item_exist("tgt_tgtplot" + str(i)): dpg.delete_item("tgt_tgtplot" + str(i))   
        if dpg.does_item_exist("tgt_tgtdrag" + str(i)): dpg.delete_item("tgt_tgtdrag" + str(i)) 
        if dpg.does_item_exist("tgt_tgtoverlayplot" + str(i)): dpg.delete_item("tgt_tgtoverlayplot" + str(i))
        else: stop_early[0] = True
        if dpg.does_item_exist("tgt_geoplot" + str(i)): dpg.delete_item("tgt_geoplot" + str(i))
        if dpg.does_item_exist("tgt_geodrag" + str(i)): dpg.delete_item("tgt_geodrag" + str(i)) 
        if dpg.does_item_exist("tgt_geooverlayplot" + str(i)): dpg.delete_item("tgt_geooverlayplot" + str(i))
        else: stop_early[1] = True
        if all(stop_early): break

    # Generate target overlay texture
    if not dpg.does_item_exist("tgt_tgtoverlay"):
        tgt_makeTargetOverlay()

    if "target_areas" not in globals():
        return
    
    tgt_counter = 0
    geo_counter = 0
    for t, target_area in enumerate(target_areas):
        if len(target_area["points"]) == 0: continue
        # Transform coords to plot
        x_vals = target_area["points"][:, 1] * MM_model.pix_size / 1000
        y_vals = dims_microns[1] - target_area["points"][:, 0] * MM_model.pix_size / 1000

        dpg.add_scatter_series(x_vals, y_vals, tag="tgt_tgtplot" + str(t), parent="tgt_x_axis")
        # Load color if not out of bounds of prepared themes
        if dpg.does_item_exist("scatter_theme" + str(t)):
            dpg.bind_item_theme("tgt_tgtplot" + str(t), "scatter_theme" + str(t))

        scaled_overlay_dims = tgt_overlay_dims * MM_model.pix_size / 1000
        for p in range(len(x_vals)):
            # add draggable point
            dpg.add_drag_point(label="tgt_" + str(p + 1).zfill(3), user_data="pt_" + str(t) + "_" + str(p), tag="tgt_tgtdrag" + str(tgt_counter), color=cluster_colors[t % len(cluster_colors)], default_value=(x_vals[p], y_vals[p]), callback=tgt_dragPointUpdate, parent="tgt_plot")

            bounds_min = (x_vals[p] - scaled_overlay_dims[1] / 2, y_vals[p] - scaled_overlay_dims[0] / 2)
            bounds_max = (x_vals[p] + scaled_overlay_dims[1] / 2, y_vals[p] + scaled_overlay_dims[0] / 2)
            if p == 0:
                dpg.add_image_series("tgt_trkoverlay", bounds_min=bounds_min, bounds_max=bounds_max, parent="tgt_x_axis", tag="tgt_tgtoverlayplot" + str(tgt_counter))
            else:
                dpg.add_image_series("tgt_tgtoverlay", bounds_min=bounds_min, bounds_max=bounds_max, parent="tgt_x_axis", tag="tgt_tgtoverlayplot" + str(tgt_counter))

            tgt_counter += 1

        if tgt_counter > 0 and len(target_areas[t]["geo_points"]) > 0:
            # Transform geo coords to plot
            x_vals = target_areas[t]["geo_points"][:, 1] * MM_model.pix_size / 1000
            y_vals = dims_microns[1] - target_areas[t]["geo_points"][:, 0] * MM_model.pix_size / 1000

            dpg.add_scatter_series(x_vals, y_vals, tag="tgt_geoplot" + str(t), parent="tgt_x_axis")
            # Load color if not out of bounds of prepared themes
            if dpg.does_item_exist("scatter_theme" + str(t)):
                dpg.bind_item_theme("tgt_geoplot" + str(t), "scatter_theme" + str(t))

            scaled_overlay_dims = geo_overlay_dims * MM_model.pix_size / 1000
            for p in range(len(x_vals)):
                # add draggable point
                dpg.add_drag_point(label="geo_" + str(p + 1).zfill(3), user_data="geo_" + str(t) + "_" + str(p), tag="tgt_geodrag" + str(geo_counter), color=cluster_colors[4], default_value=(x_vals[p], y_vals[p]), callback=tgt_dragPointUpdate, parent="tgt_plot")

                bounds_min = (x_vals[p] - scaled_overlay_dims[1] / 2, y_vals[p] - scaled_overlay_dims[0] / 2)
                bounds_max = (x_vals[p] + scaled_overlay_dims[1] / 2, y_vals[p] + scaled_overlay_dims[0] / 2)
                dpg.add_image_series("tgt_geooverlay", bounds_min=bounds_min, bounds_max=bounds_max, parent="tgt_x_axis", tag="tgt_geooverlayplot" + str(geo_counter))

                geo_counter += 1

# Generate target overlay texture
def tgt_makeTargetOverlay():
    global tgt_overlay_dims, geo_overlay_dims

    # Delete previous textures
    if dpg.does_item_exist("tgt_tgtoverlay"): dpg.delete("tgt_tgtoverlay")
    if dpg.does_item_exist("tgt_trkoverlay"): dpg.delete("tgt_trkoverlay")
    if dpg.does_item_exist("tgt_geooverlay"): dpg.delete("tgt_geooverlay")
    
    # Get camera dims
    rec_dims = np.array(tgt_params.weight.shape)

    # TGT
    # Create canvas with size of stretched beam diameter
    tgt_overlay = np.zeros([int(MM_model.beam_diameter), int(MM_model.beam_diameter / np.cos(np.radians(tgt_params.max_tilt)))])
    canvas = Image.fromarray(tgt_overlay).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    # Draw beam
    draw.ellipse((0, 0, tgt_overlay.shape[1] - 1, tgt_overlay.shape[0] - 1), outline="#ffd700", width=20)

    # Rotate tilt axis
    canvas = canvas.rotate(-mic_params.view_ta_rotation, expand=True)

    # Draw camera outline
    rect = ((canvas.width - rec_dims[1]) // 2, (canvas.height - rec_dims[0]) // 2, (canvas.width + rec_dims[1]) // 2, (canvas.height + rec_dims[0]) // 2)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(rect, outline="#578abf", width=20)

    # Convert to array
    tgt_overlay = np.array(canvas).astype(float) / 255

    # Draw camera outline for tracking target
    draw.rectangle(rect, outline="#c92b27", width=20)

    # Convert to array
    trk_overlay = np.array(canvas).astype(float) / 255

    # GEO
    # Create canvas for geo with unstretched beam diameter
    geo_overlay = np.zeros([int(MM_model.beam_diameter), int(MM_model.beam_diameter)])
    canvas = Image.fromarray(geo_overlay).convert('RGB')
    draw = ImageDraw.Draw(canvas)    

    # Draw beam and camera dims
    draw.ellipse((0, 0, geo_overlay.shape[1] - 1, geo_overlay.shape[0] - 1), outline="#ee8844", width=20)
    rect = ((canvas.width - rec_dims[1]) // 2, (canvas.height - rec_dims[0]) // 2, (canvas.width + rec_dims[1]) // 2, (canvas.height + rec_dims[0]) // 2)
    draw.rectangle(rect, outline="#ee8844", width=20)

    # Convert to array
    geo_overlay = np.array(canvas).astype(float) / 255

    # Make textures
    tgt_overlay_dims = np.array(tgt_overlay.shape)[:2]
    alpha = np.zeros(tgt_overlay.shape[:2])
    alpha[np.sum(tgt_overlay, axis=-1) > 0] = 1
    tgt_overlay_image = np.ravel(np.dstack([tgt_overlay, alpha]))
    trk_overlay_image = np.ravel(np.dstack([trk_overlay, alpha]))

    geo_overlay_dims = np.array(geo_overlay.shape)[:2]
    alpha = np.zeros(geo_overlay.shape[:2])
    alpha[np.sum(geo_overlay, axis=-1) > 0] = 1
    geo_overlay_image = np.ravel(np.dstack([geo_overlay, alpha]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=int(tgt_overlay_dims[1]), height=int(tgt_overlay_dims[0]), default_value=tgt_overlay_image, tag="tgt_tgtoverlay")
        dpg.add_static_texture(width=int(tgt_overlay_dims[1]), height=int(tgt_overlay_dims[0]), default_value=trk_overlay_image, tag="tgt_trkoverlay")
        dpg.add_static_texture(width=int(geo_overlay_dims[1]), height=int(geo_overlay_dims[0]), default_value=geo_overlay_image, tag="tgt_geooverlay")


def tgt_runTargetSelection():
    global tgt_params, mic_params, MM_model

    # Check for existing point files and delete them
    point_files = sorted(glob.glob(os.path.join(CUR_DIR, map_name + "_points*.json")))
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
    tgt_loadMask(None, tgt_params.target_list)
    hideButtons()       # buttons are shown after loadMask is finished
    dpg.set_value("tgt_tgtstatus", "Selecting targets...\n(see console for details)")

    # Run target selection
    space_ext.runTargetSelection(CUR_DIR, map_name, tgt_params, mic_params, MM_model, save_final_plot=False)

    tgt_showTargets(load_from_file=True)

    showButtons()

    # Hide save button (only activated when point was dragged)
    dpg.hide_item("tgt_buttsexp")


# Add points by shift + left clicking
def tgt_mouseClick_left(mouse_coords, mouse_coords_global):
    global target_areas
    
    # Check if mouse click was in plot range and if Shift is pressed (to not double signal when dragging)
    if "map_name" in globals() and np.all(mouse_coords > 0) and mouse_coords_global[0] > 200 and not inspected:
        # Transform mouse coords to px coords
        img_coords = np.array([(dims_microns[1] - mouse_coords[1]) / MM_model.pix_size * 1000, mouse_coords[0] / MM_model.pix_size * 1000])

        # Get camera dims
        rec_dims = np.array(tgt_params.weight.shape)

        # Check if coords are out of bounds
        if not rec_dims[0] <= img_coords[0] < dims[1] * binning - rec_dims[0] / 2 or not rec_dims[1] <= img_coords[1] < dims[0] * binning - rec_dims[1] / 2:
            return
        
        # Create target areas to allow manual picking without auto run
        if "target_areas" not in globals():
            target_areas = [{"points": [], "scores": [], "geo_points": []}]

        # Check if coords are too close to existing point
        closest_point_id, in_range = tgt_findClosestPoint(img_coords, "points", np.min(rec_dims))
        if in_range:
            print("WARNING: Target is too close to existing target! It will not be added.")
            return
        
        # Figure out which target area tracking targets is closest
        if len(target_areas) > 1:
            track_points = [target_area["points"][0] for target_area in target_areas]
            closest_area = np.argmin(np.linalg.norm(track_points - img_coords, axis=1))
        else:
            closest_area = 0

        # Add point        
        if len(target_areas[0]["points"]) > 0:
            target_areas[closest_area]["points"] = np.vstack([target_areas[closest_area]["points"], img_coords])
            target_areas[closest_area]["scores"] = np.append(target_areas[closest_area]["scores"], [100])
        else:
            target_areas[0]["points"] = img_coords[np.newaxis, :]
            target_areas[0]["scores"] = np.array([100])
        print("NOTE: Added new target!")

        tgt_showTargets()
        dpg.show_item("tgt_buttsexp")

# Delete points by right clicking
def tgt_mouseClick_right(mouse_coords, mouse_coords_global):
    global target_areas
    
    # Check if mouse click was in plot range
    if "target_areas" in globals() and np.all(mouse_coords > 0) and mouse_coords_global[0] > 200 and not inspected:
        # Transform mouse coords to px coords
        img_coords = np.array([(dims_microns[1] - mouse_coords[1]) / MM_model.pix_size * 1000, mouse_coords[0] / MM_model.pix_size * 1000])

        # Get camera dims
        rec_dims = np.array(tgt_params.weight.shape)

        # Check if coords are out of bounds
        if not 0 <= img_coords[0] < dims[1] * binning or not 0 <= img_coords[1] < dims[0] * binning:
            return

        update = False

        # Get closest point
        closest_point_id, in_range = tgt_findClosestPoint(img_coords, "points", np.min(rec_dims))

        # If closest point is in range
        if in_range:
            target_areas[closest_point_id[0]]["points"] = np.delete(target_areas[closest_point_id[0]]["points"], closest_point_id[1], axis=0)
            target_areas[closest_point_id[0]]["scores"] = np.delete(target_areas[closest_point_id[0]]["scores"], closest_point_id[1], axis=0)
            print("NOTE: Target was deleted!")
            # Remove target area if no points remain
            if len(target_areas[closest_point_id[0]]["points"]) == 0:
                target_areas.pop(closest_point_id[0])
            if len(target_areas) == 0:
                del target_areas
            update = True

        else:
            # Find closest geo_point
            closest_point_id, in_range = tgt_findClosestPoint(img_coords, "geo_points", np.min(rec_dims))
            if in_range:
                target_areas[closest_point_id[0]]["geo_points"] = np.delete(target_areas[closest_point_id[0]]["geo_points"], closest_point_id[1], axis=0)
                update = True

        if update:
            tgt_showTargets()
            dpg.show_item("tgt_buttsexp")

# Add geo point
def tgt_mouseClick_middle(mouse_coords, mouse_coords_global):
    global target_areas

    # Check if mouse click was in plot range
    if "target_areas" in globals() and np.all(mouse_coords > 0) and mouse_coords_global[0] > 200 and not inspected:
        # Transform mouse coords to px coords
        img_coords = np.array([(dims_microns[1] - mouse_coords[1]) / MM_model.pix_size * 1000, mouse_coords[0] / MM_model.pix_size * 1000])

        # Get camera dims
        rec_dims = np.array(tgt_params.weight.shape)

        # Check if coords are out of bounds
        if not rec_dims[0] <= img_coords[0] < dims[1] * binning - rec_dims[0] / 2 or not rec_dims[1] <= img_coords[1] < dims[0] * binning - rec_dims[1] / 2:
            return
        
        # Check if coords are too close to existing point
        closest_point_id, in_range = tgt_findClosestPoint(img_coords, "points", np.min(rec_dims))
        if in_range:
            print("WARNING: Target is too close to existing target! It will not be added.")
            return
                    
        # Add geo point
        for t in range(len(target_areas)):
            if len(target_areas[t]["geo_points"]) > 0:
                target_areas[t]["geo_points"] = np.vstack([target_areas[t]["geo_points"], img_coords])
            else:
                target_areas[t]["geo_points"] = img_coords[np.newaxis, :]
        if len(target_areas) > 0:
            print("NOTE: Added new geo point to " + str(len(target_areas)) + " target areas!")
        else:
            print("NOTE: Added new geo point!")

        tgt_showTargets()
        dpg.show_item("tgt_buttsexp")

# Only update scatter point when dragging (targets are updated upon release)
def tgt_dragPointUpdate(sender, app_data, user_data):
    if inspected: return

    coords = dpg.get_value(sender)[:2]
    if dpg.does_item_exist("tgt_tempplot"): dpg.delete_item("tgt_tempplot")
    dpg.add_scatter_series([coords[0]], [coords[1]], tag="tgt_tempplot", parent="tgt_x_axis")
    dpg.bind_item_theme("tgt_tempplot", "scatter_theme3")   # red theme

# Update target upon mouse release
def tgt_tgtUpdate():
    # Only execute when targets are loaded
    if "target_areas" not in globals() or inspected:
        return
    
    # Go through all points
    update = False
    for i in range(1000):
        stop_early = [False, False]
        # Check if point exists
        if dpg.does_item_exist("tgt_tgtdrag" + str(i)):
            update = update or tgt_checkDragPoint("tgt_tgtdrag" + str(i))
        else:
            stop_early[0] = True
        # Check if geo point exists
        if dpg.does_item_exist("tgt_geodrag" + str(i)):
            update = update or tgt_checkDragPoint("tgt_geodrag" + str(i))
        else:
            stop_early[1] = True
        if all(stop_early): break

    # Replot targets if any coords have changed
    if update:
        tgt_showTargets()
        dpg.show_item("tgt_buttsexp")

# Update target_areas if drag point has been moved
def tgt_checkDragPoint(drag_point_tag):
    # Get coords from drag point value
    coords = np.array(dpg.get_value(drag_point_tag)[:2])
    # Get area and point IDs from user data embedded in drag point
    user_data = dpg.get_item_user_data((drag_point_tag)).split("_")
    point_id = np.array(user_data[1:], dtype=int)
    # Check if drag point is geo point
    if user_data[0] == "geo":
        point_type = "geo_points"
    else:
        point_type = "points"
    # Transform points to plot points for comparison
    old_coords = np.array([target_areas[point_id[0]][point_type][point_id[1]][1] * MM_model.pix_size / 1000, dims_microns[1] - target_areas[point_id[0]][point_type][point_id[1]][0] * MM_model.pix_size / 1000])
    # Go to next points if coords have not changed
    if np.all(coords == old_coords):
        return False
    else:
        # Get camera dims in microns
        rec_dims = np.array(tgt_params.weight.shape) * MM_model.pix_size / 1000
        # Clip coords to map size
        coords[0] = np.clip(coords[0], rec_dims[1] / 2, dims_microns[0] - rec_dims[1] / 2)
        coords[1] = np.clip(coords[1], rec_dims[0] / 2, dims_microns[1] - rec_dims[0] / 2)
        # Update coords if they have changed
        target_areas[point_id[0]][point_type][point_id[1]][1] = coords[0] / MM_model.pix_size * 1000
        target_areas[point_id[0]][point_type][point_id[1]][0] = (dims_microns[1] - coords[1]) / MM_model.pix_size * 1000

        return True

# Find closest point within treshold
def tgt_findClosestPoint(coords, point_type, threshold):
    # Check for point within range
    if len(target_areas[0]["points"]) > 0:
        # Check if coords are too close to existing point (also allows for dragging to work without creating new point)
        closest_point_id = np.zeros(2)
        closest_point_dist = 1e9
        for t, target_area in enumerate(target_areas):
            for p, point in enumerate(target_area[point_type]):
                dist = np.linalg.norm(point - coords)
                if dist < closest_point_dist:
                    closest_point_dist = dist
                    closest_point_id = np.array([t, p])

        return closest_point_id, closest_point_dist < threshold
    else:
        return None, False

# Export points for SPACEtomo target file creation
def tgt_exportPoints():
    # Check for existing point files and delete them
    point_files = sorted(glob.glob(os.path.join(CUR_DIR, map_name + "_points*.json")))
    if len(point_files) > 0:
        print("NOTE: Previous targets were deleted.")
        for file in point_files:
            os.remove(file)

    if "target_areas" in globals() and len(target_areas) > 0:
        for t, target_area in enumerate(target_areas):
            with open(os.path.join(CUR_DIR, map_name + "_points" + str(t) + ".json"), "w+") as f:
                json.dump(target_area, f, indent=4, default=space_ext.convertArray)
    else:
        # Write empty points file to ensure empty targets file is written and map is considered processed
        with open(os.path.join(CUR_DIR, map_name + "_points.json"), "w+") as f:
            json.dump({"points": [], "scores": [], "geo_points": []}, f)    

    # Delete save button (only activated when point was dragged)
    dpg.hide_item("tgt_buttsexp")

# Create inspected file to give SPACEtomo the signal to continue
def tgt_inspect():
    global inspected
    tgt_exportPoints()
    # Create empty inspected file
    open(os.path.join(CUR_DIR, map_name + "_inspected.txt"), "w").close()

    inspected = True

    dpg.hide_item("tgt_tsmenu")
    dpg.show_item("tgt_ins")

    # Update map window to reflected inspected status
    tgt_updateMapList()
    tgt_makeMapGrid()

# Check for new maps in folder
def tgt_updateMapList():
    global map_list, map_list_tgt
    prev_len = len(map_list)
    # Find all maps with segmentation
    seg_list = sorted(glob.glob(os.path.join(CUR_DIR, "*_seg.png")))
    map_list = [os.path.basename(seg).split("_seg.png")[0] for seg in seg_list] 
    if dpg.does_item_exist("tgt_map"): 
        dpg.delete_item("tgt_map")
        dpg.add_combo(map_list, default_value=map_list[0], label="(" + str(len(map_list)) + " maps)", tag="tgt_map", parent="tgt_left", before="tgt_mapbutgrp")
        if "map_name" in globals():
            dpg.set_value("tgt_map", map_name)

    # Find selected targets per lamella
    map_list_tgt = []
    for m_name in map_list:
        point_files = sorted(glob.glob(os.path.join(CUR_DIR, m_name + "_points*.json")))
        point_num = 0
        if len(point_files) > 0:
            target_areas = []
            for file in point_files:
                # Load json data
                with open(file, "r") as f:
                    point_data = json.load(f, object_hook=space_ext.revertArray)
                    point_num += len(point_data["points"])
        map_list_tgt.append(point_num)


    if len(map_list) != prev_len:
        if prev_len > 0:
            tgt_loadThumbnails(thumbnail_size)
        else:
            tgt_loadThumbnails(thumbnail_size, first=True)

# Create thumbnails of lamella maps or load from file if they already exist
def tgt_loadThumbnails(size=(100, 100), first=False, override=False):
    global thumbnails

    if not dpg.does_item_exist("thumbnail_textures"):
        dpg.add_texture_registry(tag="thumbnail_textures")

    # Allow thumbnail setting to be changed by button
    if override:
        thumbnails = True
        if dpg.does_item_exist("map_butthumb"):
            dpg.configure_item("map_butthumb", label="Creating thumbnails...", enabled=False)
            #dpg.set_item_label("map_butthumb", "Creating thumbnails...")
            #dpg.delete_item("map_butthumb")

    if not thumbnails: 
        thumbnails = tgt_makeGenericThumbnail(size)


    prev_num = len(dpg.get_item_children("thumbnail_textures")[1])

    now = time.time()
    for m_name in map_list:
        tgt_makeThumbnail(m_name, size=size)
        
    if first:
        print("Thumbnails loaded in", round((time.time() - now) / len(map_list), 1), "s per map. (Set thumbnails=False in script settings to save time.)")

    # Create new window if thumbnails changed
    if len(dpg.get_item_children("thumbnail_textures")[1]) != prev_num:
        tgt_makeMapGrid()

# Make thumbnail for map
def tgt_makeThumbnail(m_name, map_img=None, size=(100, 100), override=False):
    # If texture already exists
    if dpg.does_item_exist("tgt_thumb_" + m_name): 
        return

    # Make folder 
    if not os.path.exists(os.path.join(CUR_DIR, "thumbnails")):
        os.makedirs(os.path.join(CUR_DIR, "thumbnails"))
    
    # If thumnail file already exists
    make_texture = True
    if os.path.exists(os.path.join(CUR_DIR, "thumbnails", m_name + ".png")):
        map_img = Image.open(os.path.join(CUR_DIR, "thumbnails", m_name + ".png"))
    elif thumbnails or override:
        # Open image if not opened yet
        if map_img is None:
            map_img = Image.open(os.path.join(CUR_DIR, m_name + ".png"))
        # Save thumbnail
        map_img.thumbnail(size)
        map_img.save(os.path.join(CUR_DIR, "thumbnails", m_name + ".png"))
    else:
        make_texture = False

    if make_texture:
        print("Creating thumbnail for " + m_name + "...")
        # Pad thumbnail to 100x100
        map_thumb = np.zeros(size, dtype=float)
        map_thumb[(size[1] - map_img.size[1]) // 2: (size[1] + map_img.size[1]) // 2, (size[0] - map_img.size[0]) // 2: (size[0] + map_img.size[0]) // 2] = np.array(map_img) / 255
        
        # Save as texture
        map_thumb = np.ravel(np.dstack([map_thumb, map_thumb, map_thumb, np.ones(map_thumb.shape)]))
        dpg.add_static_texture(width=size[0], height=size[1], default_value=map_thumb, tag="tgt_thumb_" + m_name, parent="thumbnail_textures")

# Make generic thumbnail
def tgt_makeGenericThumbnail(size=(100, 100)):
    thumbnail = np.zeros(size)
    thumbnail[size[0] // 3: size[0] // 3 * 2, 0: size[1] // 5] = 1
    thumbnail[size[0] // 3: size[0] // 3 * 2, size[1] // 5 * 4: size[1]] = 1
    thumbnail[size[0] // 3: size[0] // 3 * 2, size[1] // 5: size[1] // 5 * 4] = 0.6
    thumbnail[size[0] // 3: size[0] // 3 * 2, size[1] // 5: size[1] // 7 * 2] = 0.3

    # Expansion joints
    thumbnail[size[0] // 5: size[0] // 4, size[1] // 7: size[1] // 7 * 6] = 1
    thumbnail[size[0] // 4 * 3: size[0] // 5 * 4, size[1] // 7: size[1] // 7 * 6] = 1

    # Save as texture
    if not dpg.does_item_exist("thumbnail_textures"):
        dpg.add_texture_registry(tag="thumbnail_textures")
    map_thumb = np.ravel(np.dstack([thumbnail, thumbnail, thumbnail, np.ones(thumbnail.shape)]))
    dpg.add_static_texture(width=size[0], height=size[1], default_value=map_thumb, tag="tgt_thumb_gen", parent="thumbnail_textures")   

# Make window for map selection
def tgt_makeMapGrid(grid_cols=5):
    # Delete old window
    if dpg.does_item_exist("map_window"):
        dpg.delete_item("map_window")

    # Make new window
    with dpg.window(label="Lamella maps", tag="map_window", no_collapse=True, popup=True, show=False):
        with dpg.table(header_row=False):
            # Add table columns
            for j in range(grid_cols):
                dpg.add_table_column(init_width_or_weight=thumbnail_size[0] + 10, width_fixed=True)

            for i in range(int(np.ceil(len(map_list) / grid_cols))):
                with dpg.table_row():
                    for j in range(grid_cols):
                        m_id = i * grid_cols + j
                        if m_id >= len(map_list): break
                        m_name = map_list[m_id]
                        #dpg.add_image_button("tgt_thumb_" + m_name, label=m_name, callback=map_chooseMap, tag="map_thumb_" + m_name, pos=(20 + j * (thumbnail_size[0] + 20), 20 + i * (thumbnail_size[1] + 20)))  

                        if dpg.does_item_exist("tgt_thumb_" + m_name):
                            texture_name = "tgt_thumb_" + m_name 
                        else:
                            texture_name = "tgt_thumb_gen"

                        with dpg.table_cell():
                            if "map_name" in globals() and m_name == map_name:
                                dpg.add_image_button(texture_name, label=m_name, callback=map_chooseMap, tag="map_thumb_" + m_name, background_color=heading_color, enabled=False)
                                dpg.add_text(m_name, color=heading_color)
                            else:
                                dpg.add_image_button(texture_name, label=m_name, callback=map_chooseMap, tag="map_thumb_" + m_name)
                                dpg.add_text(m_name)
                            
                            if map_list_tgt[m_id] > 0:
                                dpg.add_text(str(map_list_tgt[m_id]) + " targets")
                            else:
                                dpg.add_text("No targets", color=error_color)
                            if os.path.exists(os.path.join(CUR_DIR, m_name + "_inspected.txt")):
                                dpg.add_text("Inspected", color=heading_color)
                            else:
                                dpg.add_text("Not inspected")
                                
        if not thumbnails:
            dpg.add_button(label="Generate thumbnails", tag="map_butthumb", callback=lambda: tgt_loadThumbnails(thumbnail_size, override=True))        

def tgt_showPointTooltip():
    if "target_areas" in globals():
        mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # Transform mouse coords to px coords
        img_coords = np.array([(dims_microns[1] - mouse_coords[1]) / MM_model.pix_size * 1000, mouse_coords[0] / MM_model.pix_size * 1000])

        # Get camera dims
        rec_dims = np.array(tgt_params.weight.shape)

        # Check if coords are too close to existing point
        closest_point_id, in_range = tgt_findClosestPoint(img_coords, "points", np.min(rec_dims))

        if in_range:
            dpg.set_value("tt_heading", "Target information:")
            info = ""
            if len(target_areas) > 1:
                info += "Area: " + str(closest_point_id[0] + 1) + "\n"
            info += "Target: " + str(closest_point_id[1] + 1) + "\n"
            if target_areas[closest_point_id[0]]["scores"][closest_point_id[1]] == 100:
                score = "manual"
            else:
                score = str(round(target_areas[closest_point_id[0]]["scores"][closest_point_id[1]], 2))
            info += "Score: " + score + "\n"
            dist = round(np.linalg.norm(target_areas[closest_point_id[0]]["points"][closest_point_id[1]] - target_areas[closest_point_id[0]]["points"][0]) * MM_model.pix_size / 1000, 2)
            info += "IS: " + str(dist) + " µm\n"

            dpg.set_value("tt_text", info)
            return

    dpg.set_value("tt_heading", "Target manipulation:")
    dpg.set_value("tt_text", "- Drag target to reposition\n- Shift + left click to add target\n- Right click to delete target\n- Middle click to add geo point")


# TODO add checkbox selection for target_list and avoid_list
def tgt_showListSelection():
    current_list = [cat.strip() for cat in dpg.get_value("target_list").split(",")]
    with dpg.window(label="Select from classes", tag="window_class_list", popup=True):
        for cat in MM_model.categories.keys():
            if cat in current_list:
                checked = True
            else:
                checked = False
            dpg.add_checkbox(label=cat, tag="sel_" + cat, default_value=checked)
            

def tgt_askForSave():
    if dpg.is_item_shown("tgt_buttsexp") and not dpg.does_item_exist("window_save"):
        with dpg.window(label="Save", tag="window_save", modal=True):
            dpg.add_text("There are unsaved changes to your targets.", color=heading_color)
            dpg.add_button(label="Save", callback=tgt_saveAndClose, pos=(10, 60))
            dpg.add_button(label="Discard", callback=dpg.stop_dearpygui, pos=(50, 60))
        dpg.set_item_pos("window_save", (dpg.get_item_width("GUI") // 3 , dpg.get_item_height("GUI") // 3))
    else:
        dpg.stop_dearpygui()

def tgt_saveAndClose():
    tgt_exportPoints()
    dpg.stop_dearpygui()


### END FUNCTIONS ###

heading_color = (255, 200, 0, 255)
error_color = (200, 0, 0, 255)
subtle_color = (255, 255, 255, 64)

# Make logo
logo, logo_dims = makeLogo()

# Find all maps with segmentation
thumbnail_size = (100, 100)
map_list = []
tgt_updateMapList()

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
        tgt_params = space_ext.TgtParams(["lamella"], ["black", "white", "crack", "ice"], MM_model, False, False, 0.3, 0.5, 60, mic_params, False, 10)

    
else:
    print("ERROR: No microscope parameters found. Make sure the mic_params.json file created by SPACEtomo is available!")
    sys.exit()

if len(map_list) == 0:
    print("ERROR: No segmentations found in " + CUR_DIR)
    sys.exit()


with dpg.texture_registry():
    dpg.add_static_texture(width=logo_dims[1], height=logo_dims[0], default_value=logo, tag="logo")

# Create themes for plots
cluster_colors = [(87, 138, 191, 255), (229, 242, 255, 255), (59, 92, 128, 255), (200, 0, 0, 255), (238, 136, 68, 255)]
for c, color in enumerate(cluster_colors):
    with dpg.theme(tag="scatter_theme" + str(c)):
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)

with dpg.theme(tag="axis_theme"):
    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 64), category=dpg.mvThemeCat_Plots)


# Create file dialogues

# Create event handlers

with dpg.handler_registry() as mouse_handler:
    m_click = dpg.add_mouse_click_handler(callback=mouseClick)
    m_release_left = dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=tgt_tgtUpdate)

with dpg.item_handler_registry(tag="point_tooltip_handler"):
    dpg.add_item_hover_handler(callback=tgt_showPointTooltip)


dpg.set_viewport_resize_callback(callback=window_size_change)

dpg.create_viewport(title="SPACEtomo Target Selection", disable_close=True)
dpg.setup_dearpygui()

with dpg.window(label="GUI", tag="GUI", no_scrollbar=True, no_scroll_with_mouse=True):

    with dpg.table(header_row=False):
        dpg.add_table_column(init_width_or_weight=200, width_fixed=True)
        dpg.add_table_column()
        dpg.add_table_column(init_width_or_weight=200, width_fixed=True)

        with dpg.table_row():
            with dpg.table_cell(tag="tgt_left"):
                dpg.add_text(default_value="Load your MM map", tag="tgt_1", color=heading_color)
                dpg.add_combo(map_list, default_value=map_list[0], label="(" + str(len(map_list)) + " maps)", tag="tgt_map")

                with dpg.group(horizontal=True, tag="tgt_mapbutgrp"):
                    dpg.add_button(label="Load map", tag="tgt_butload", callback=lambda: tgt_chooseMap())
                    dpg.add_button(label="Next", tag="tgt_butnext", callback=lambda: tgt_chooseMap(next_map=True))
                    if len(map_list) <= 1:
                        dpg.hide_item("tgt_butnext")
                    dpg.add_button(label="[]", tag="tgt_butgrid", callback=lambda: dpg.show_item("map_window"))
                    dpg.add_text(default_value="", tag="tgt_mapstatus", color=subtle_color, show=False)

                dpg.add_text(default_value="\nClasses:", tag="tgt_cls", show=False)

                dpg.add_text(default_value="", tag="tgt_lfinal", color=subtle_color)

            with dpg.table_cell(tag="tgt_tblplot"):
                with dpg.plot(label="Map", width=-1, height=-1, equal_aspects=True, no_menus=True, tag="tgt_plot"):
                    # REQUIRED: create x and y axes
                    dpg.add_plot_axis(dpg.mvXAxis, label="x [µm]", tag="tgt_x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label="y [µm]", tag="tgt_y_axis")

            with dpg.table_cell(tag="tgt_right"):
                dpg.add_text(default_value="Target selection", tag="tgt_r1", color=heading_color)
                with dpg.group(label="Target selection", tag="tgt_tsmenu", show=False, parent="tgt_right", before="tgt_rfinal"):
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

                    dpg.add_text("")
                    with dpg.group(horizontal=True, tag="tgt_butgrp"):
                        dpg.add_button(label="Select targets", callback=tgt_runTargetSelection, tag="tgt_butts")
                        dpg.add_button(label="Save", callback=tgt_exportPoints, tag="tgt_buttsexp", show=False)
                    dpg.add_button(label="Mark as inspected", callback=tgt_inspect, tag="tgt_buttsins", show=False)
                    dpg.add_text(default_value="", tag="tgt_tgtstatus", color=subtle_color, show=False)
                dpg.add_text("\nTargets for this map were\nalready inspected and\ncannot be changed anymore.", tag="tgt_ins", color=heading_color, show=False)
                dpg.add_text("", tag="tgt_rfinal")

    # Create tooltips
    with dpg.tooltip("tgt_1", delay=0.5):
        dpg.add_text("Select and load a map generated by SPACEtomo.")
    with dpg.tooltip("tgt_cls", delay=0.5):
        dpg.add_text("Choose a class to be displayed as overlay.")

    with dpg.tooltip("tgt_tblplot", delay=0.5, hide_on_activity=True):
        dpg.add_text(default_value="", color=heading_color, tag="tt_heading")
        dpg.add_text(default_value="", tag="tt_text")
    dpg.bind_item_handler_registry("tgt_plot", "point_tooltip_handler")

    with dpg.tooltip("tgt_r1", delay=0.5):
        dpg.add_text("Select targets based on segmentation.")
    with dpg.tooltip("target_list", delay=0.5):
        dpg.add_text("List of target classes (comma separated). For exhaustive acquisition use \"lamella\".")
    with dpg.tooltip("avoid_list", delay=0.5):
        dpg.add_text("List of classes to avoid (comma separated).")
    with dpg.tooltip("target_score_threshold", delay=0.5):
        dpg.add_text("Score threshold [0-1] below targets will be excluded.")
    with dpg.tooltip("penalty_weight", delay=0.5):
        dpg.add_text("Relative weight of avoided classes to target classes.")
    with dpg.tooltip("max_tilt", delay=0.5):
        dpg.add_text("Maximum tilt angle to consider electron beam exposure.")
    with dpg.tooltip("IS_limit", delay=0.5):
        dpg.add_text("Image shift limit for PACEtomo acquisition. If targets are further apart, target area will be split.")
    with dpg.tooltip("sparse_targets", delay=0.5):
        dpg.add_text("Target positions will be initialized only on target classes and refined independently (instead of grid based target target setup to minimize exposure overlap).")
    with dpg.tooltip("target_edge", delay=0.5):
        dpg.add_text("Targets will be centered on edge of segmented target instead of maximising coverage.")
    with dpg.tooltip("extra_tracking", delay=0.5):
        dpg.add_text("An extra target will be placed centrally for tracking.")

    with dpg.tooltip("tgt_buttsexp", delay=0.5):
        dpg.add_text("Save changes to targets. (SPACEtomo might already have read the targets if \"wait_for_inspection = False\".)")
    with dpg.tooltip("tgt_buttsins", delay=0.5):
        dpg.add_text("Mark targets as inspected. (No more changes can be made.)")

    dpg.add_image("logo", pos=(10, dpg.get_viewport_height() - 40 - logo_dims[0]), tag="logo_img")
    dpg.add_text(default_value="SPACEtomo", pos=(10 + logo_dims[1] / 2 - (30), dpg.get_viewport_height() - 40 - logo_dims[0] / 2), tag="logo_text")
    dpg.add_text(default_value="v" + space_ext.versionSPACE, pos=(10 + logo_dims[1] / 2 - (30), dpg.get_viewport_height() - 27 - logo_dims[0] / 2), tag="version_text")

# Make window for map thumbnails
tgt_makeMapGrid()

dpg.set_exit_callback(tgt_askForSave)

dpg.set_primary_window("GUI", True)
dpg.show_viewport()

# Render loop
next_update = time.time() + 60
while dpg.is_dearpygui_running():

    # Recheck folder for segmentation every minute
    now = time.time()
    if now > next_update:
        next_update = now + 60
        tgt_updateMapList()

    dpg.render_dearpygui_frame()

dpg.destroy_context()