#!/usr/bin/env pythonEisenstein
# ===================================================================
# ScriptName:   SPACEtomo_functions
# Purpose:      Functions necessary to run SPACEtomo.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/10/04
# Revision:     v1.0beta
# Last Change:  2023/12/12: fixed proximity removal, added version check
# ===================================================================

import serialem as sem
import os
import copy
import json
from datetime import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images
from scipy.ndimage import rotate
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
from skimage import transform, exposure
import mrcfile
from ultralytics import YOLO
import SPACEtomo_config as config

versionSPACE = "1.0beta"

sem.SuppressReports()
sem.SetNewFileType(0)		# set file type to mrc in case user changed default file type
CUR_DIR = sem.ReportDirectory()
SPACE_DIR = os.path.dirname(__file__)

##### WG functions #####

class WGModel:
    def __init__(self):
        self.model = YOLO(os.path.join(SPACE_DIR, config.WG_model_file))
        self.pix_size = config.WG_model_pix_size
        self.sidelen = config.WG_model_sidelen
        self.categories = config.WG_model_categories
        self.cat_colors = config.WG_model_colors
        self.cat_nav_colors = config.WG_model_nav_colors

# Merge overlapping bboxes
def mergeOverlappingBoxes(boxes):
    # Sort boxes by x values
    boxes = boxes[boxes[:, 0].argsort()]

    merged_boxes = []
    for box in boxes:
        if merged_boxes:
            # Get the previous merged box
            prev_box = merged_boxes[-1]

            size = [box[2] - box[0], box[3] - box[1]]

            # Check if the current box overlaps with the previous box within threshold
            if box[0] <= prev_box[2] + 0.1 * size[0] and box[1] <= prev_box[3] + 0.1 * size[1] and box[3] >= prev_box[1] - 0.1 * size[1]:
                # Update the previous box to encompass both boxes
                prev_box[2] = max(box[2], prev_box[2])
                prev_box[3] = max(box[3], prev_box[3])
                # Change class if proability is higher for new box OR if old box was class 0 ("broken")
                if prev_box[5] < box[5] or prev_box[4] == 0:
                    prev_box[4] = box[4]
                    prev_box[5] = box[5]
            else:
                merged_boxes.append(box)
        else:
            merged_boxes.append(box)
    final_boxes = np.array(merged_boxes)
    return final_boxes

# Collect montage on new grid and find lamellae
def findLamellae(grid_name, mic_params, montage_overlap, model, threshold=0.1, include_broken=False, plot=False):
    sem.Echo("Collecting whole grid map...")
    # Check if WG map already exists
    note_id = sem.NavIndexWithNote(grid_name + ".mrc")
    if note_id > 0:
        map_id = note_id
        sem.Echo("NOTE: Grid map already found in navigator. Skipping acquisiion.")
    else:
        sem.GoToImagingState(str(mic_params.WG_image_state))
        sem.SetupFullMontage(montage_overlap, grid_name + ".mrc")
        sem.M()
        map_id = sem.NewMap(0, grid_name + ".mrc")

    sem.ChangeItemLabel(int(map_id), grid_name[:6])
    sem.LoadOtherMap(int(map_id))
    sem.CloseFile()

    # Load buffer
    buffer_orig, *_ = sem.ReportCurrentBuffer()
    img_prop = sem.ImageProperties(buffer_orig)
    mic_params.WG_pix_size = float(img_prop[4])

    # Rescale to model pixel size
    rescale_factor = model.pix_size / mic_params.WG_pix_size
    sem.ReduceImage(buffer_orig, rescale_factor)                                # faster than tranform.rescale
    buffer = "A"
    montage = np.asarray(sem.bufferImage(buffer))

    # Split WG montage into windows for inference
    num_cols = montage.shape[0] // model.sidelen + 1
    num_rows = montage.shape[1] // model.sidelen + 1

    bboxes = []
    for c in range(num_cols):
        for r in range(num_rows):
            x = c * model.sidelen
            y = r * model.sidelen
            crop = montage[x: x + model.sidelen, y: y + model.sidelen]
            crop = np.dstack([crop, crop, crop])

            results = model.model(crop)                                         # YOLO inference
            
            if len(results[0].boxes) > 0:
                bbox = np.array(results[0].boxes.xyxy.to("cpu"))

                bbox[:, 0] += y
                bbox[:, 2] += y
                bbox[:, 1] += x
                bbox[:, 3] += x
                
                cat = np.reshape(results[0].boxes.cls.to("cpu"), (bbox.shape[0], 1))
                conf = np.reshape(results[0].boxes.conf.to("cpu"), (bbox.shape[0], 1))

                bbox = np.hstack([bbox, cat, conf])

                if len(bboxes) > 0:
                    bboxes = np.concatenate([bboxes, bbox])
                else:
                    bboxes = bbox
                
    bboxes = np.array(bboxes)
    sem.Echo("Lamellae found (initial): " + str(len(bboxes)))

    # Merge bboxes found on different crop windows
    bboxes = mergeOverlappingBoxes(bboxes)
    sem.Echo("Lamellae found (merged): " + str(len(bboxes)))

    # Clean lamellae based on user defined confidence threshold
    bboxes = np.array([bbox for bbox in bboxes if bbox[5] >= threshold])
    sem.Echo("Lamellae found (final): " + str(len(bboxes)))

    # Plot montage with bboxes
    #if plot:
    fig, ax = plt.subplots(figsize=(12, 16))
    plt.imshow(montage, cmap="gray")

    for bbox in bboxes:
        rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, color=model.cat_colors[int(bbox[4])])
        ax.add_artist(rect)
        t = plt.text(bbox[0], bbox[1] - 10, model.categories[int(bbox[4])] + " " + str(bbox[5]), color=model.cat_colors[int(bbox[4])])
        t.set_bbox(dict(facecolor='black', alpha=0.5, pad=0))

    fig.savefig(os.path.join(CUR_DIR, grid_name + "_lamella_detected.png"))
    #fig.savefig(grid_name + "_lamella_detected.svg", format="svg")
    plt.clf()

    # Generate nav item for each lamella
    nav_ids = []
    final_bboxes = []
    for b, bbox in enumerate(bboxes):
        if bbox[4] == model.categories.index("broken") and not include_broken:
            continue
        size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        center = [bbox[0] + size[0] / 2, bbox[1] + size[1] / 2]
        nav_ids.append(int(sem.AddImagePosAsNavPoint(buffer_orig, int(center[0] * rescale_factor), int(center[1] * rescale_factor), 0)))
        # Check if coords are within grid limits (default is +/- 990, but can be changed in SerialEM property "StageLimits", this property can not be read by script)
        _, x_coord, y_coord, *_ = sem.ReportOtherItem(nav_ids[-1])
        if abs(x_coord) > 990 or abs(y_coord) > 990:
            nav_ids.pop(-1)
            continue
        final_bboxes.append(bbox)
        sem.ChangeItemColor(nav_ids[-1], model.cat_nav_colors[int(bbox[4])])
        sem.ChangeItemNote(nav_ids[-1], model.categories[int(bbox[4])] + " (" + str(int(round(bbox[5] * 100, 0))) + "%)")
        sem.ChangeItemLabel(nav_ids[-1], "PL" + str(b + 1))

    return nav_ids, final_bboxes

def findOffset(lamella_ids, initial_bboxes, mic_params, model, keep_threshold=0.7):
    sem.GoToImagingState(str(mic_params.WG_image_state))
    if mic_params.IM_mag_index < 50:
        sem.SetMagIndex(mic_params.IM_mag_index)
    else:
        sem.SetMag(mic_params.IM_mag_index)
    new_lamella_ids = []
    new_bboxes = []
    shifted = False
    for l, lamella_id in enumerate(lamella_ids):
        sem.MoveToNavItem(lamella_id)
        sem.R()
        # Load buffer
        img_prop = sem.ImageProperties("A")
        mic_params.IM_pix_size = float(img_prop[4])

        # Rescale to model pixel size
        rescale_factor = model.pix_size / mic_params.IM_pix_size
        sem.ReduceImage("A", rescale_factor)                                # faster than tranform.rescale
        montage = exposure.rescale_intensity(np.asarray(sem.bufferImage("A")), out_range=(0, 255)).astype(np.uint8)
        padded_map = np.zeros((model.sidelen, model.sidelen))
        #padded_map[0: montage.shape[0], 0: montage.shape[1]] = montage 
        padded_map[(model.sidelen - montage.shape[0]) // 2: (model.sidelen + montage.shape[0]) // 2, (model.sidelen - montage.shape[1]) // 2: (model.sidelen + montage.shape[1]) // 2] = montage

        # Run YOLO model to detect bbox
        yolo_input = np.dstack([padded_map, padded_map, padded_map])
        results = model.model(yolo_input)   

        if len(results[0].boxes) > 0:
            bbox = np.array(results[0].boxes.xyxy.to("cpu"))

            bbox[:, 0] -= (model.sidelen - montage.shape[1]) // 2
            bbox[:, 2] -= (model.sidelen - montage.shape[1]) // 2
            bbox[:, 1] -= (model.sidelen - montage.shape[0]) // 2
            bbox[:, 3] -= (model.sidelen - montage.shape[0]) // 2
            
            cat = np.reshape(results[0].boxes.cls.to("cpu"), (bbox.shape[0], 1))
            conf = np.reshape(results[0].boxes.conf.to("cpu"), (bbox.shape[0], 1))

            bbox = np.hstack([bbox, cat, conf])
            if len(bbox) > 1:
                bbox = mergeOverlappingBoxes(bbox)
                if len(bbox) > 1:
                    bbox = sorted(bbox, key=lambda x: x[4], reverse=True)     # sort according to probability
                    sem.Echo("WARNING: Found " + str(len(bbox)) + " lamellae. Using lamella with highest probability.")
                else:
                    sem.Echo("NOTE: Merged found lamellae.")
            bbox = bbox[0]

            sem.Echo("Lamella bounding box: " +  str(bbox))
            sem.Echo("Lamella was categorized as: " + model.categories[int(bbox[4])] + " (" + str(round(bbox[5] * 100, 1)) + " %)")

            center = [(bbox[0] + bbox[2]) / 2,(bbox[1] + bbox[3]) / 2]

            new_lamella_ids.append(int(sem.AddImagePosAsNavPoint("B", int(center[0] * rescale_factor), int(center[1] * rescale_factor), 0)))
            
            new_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
            old_size = [initial_bboxes[l][2] - initial_bboxes[l][0], initial_bboxes[l][3] - initial_bboxes[l][1]]
            if new_size[0] * new_size[1] > old_size[0] * old_size[1]:
                new_bboxes.append(bbox)
            else:
                new_bboxes.append(initial_bboxes[l])
            
            if not shifted:
                _, x_old, y_old, *_ = sem.ReportOtherItem(lamella_id)
                _, x_new, y_new, *_ = sem.ReportOtherItem(new_lamella_ids[-1])
                shift = np.array([x_new - x_old, y_new - y_old])
            
                sem.Echo("Shifting items by " + str(shift))
                sem.ShiftItemsByMicrons(x_new - x_old, y_new - y_old)
                sem.DeleteNavigatorItem(new_lamella_ids[-1])
                new_lamella_ids[-1] = lamella_id

                shifted = True
            else:
                sem.ChangeItemColor(lamella_id, 5)
            sem.ChangeItemColor(new_lamella_ids[-1], model.cat_nav_colors[int(bbox[4])])
            sem.ChangeItemNote(new_lamella_ids[-1], model.categories[int(bbox[4])] + " (" + str(int(round(bbox[5] * 100, 0))) + "%)")
            sem.ChangeItemLabel(new_lamella_ids[-1], "FL" + str(l + 1))

        else:
            if initial_bboxes[l][5] > keep_threshold:
                new_lamella_ids.append(lamella_id)
                new_bboxes.append(initial_bboxes[l])
                sem.Echo("No lamella detected in intermediate mag. Keeping original lamella due to high confidence (" + str(initial_bboxes[l][5]) + ")...")
            else:
                sem.Echo("No lamella detected in intermediate mag. Removing lamella from list...")

    # Checking for lamellae too close to each other
    remove_ids = []
    lamella_coords = np.zeros((len(new_lamella_ids), 2))
    _, x, y, *_ = sem.ReportOtherItem(new_lamella_ids[0])
    lamella_coords[0, :] = x, y
    for lid1 in range(0, len(new_lamella_ids)):
        if lid1 in remove_ids: continue
        for lid2 in range(lid1 + 1, len(new_lamella_ids)):
            if lid2 in remove_ids: continue
            if np.array_equal(lamella_coords[lid2], np.zeros(2)):
                _, x, y, *_ = sem.ReportOtherItem(new_lamella_ids[lid2])
                lamella_coords[lid2, :] = x, y
            dist = np.linalg.norm(lamella_coords[lid2] - lamella_coords[lid1])
            if dist < 10:               # if distance is smaller than 10 microns, remove lamella with smaller probability
                if new_bboxes[lid1][5] > new_bboxes[lid2][5]:
                    remove_ids.append(lid2)
                else:
                    remove_ids.append(lid1)
    if len(remove_ids) > 0:
        sem.Echo("WARNING: Removed " + str(len(remove_ids)) + " lamellae due to close proximity.")
        for rid in remove_ids:
            new_lamella_ids.pop(rid)
            new_bboxes.pop(rid)

    sem.Echo(str(len(new_lamella_ids)) + " lamellae were confirmed in intermediate mag.")
    return new_lamella_ids, new_bboxes

##### MM functions #####

class MMModel:
    def __init__(self):
        self.script = os.path.join(SPACE_DIR, config.MM_model_script)
        self.dir = os.path.join(SPACE_DIR, config.MM_model_folder)
        self.pix_size = config.MM_model_pix_size
        self.max_runs = config.MM_model_max_runs

        # Get all classes from model folder
        with open(os.path.join(self.dir, "dataset.json"), "r") as f:
            dataset_json = json.load(f)
        self.categories = dataset_json["labels"]

    def setDimensions(self, mic_params):
        self.beam_diameter = mic_params.rec_beam_diameter * 1000 / self.pix_size
        self.cam_dims = (mic_params.cam_dims[[1, 0]] * mic_params.rec_pix_size / self.pix_size).astype(int)    # record camera dimensions in pixels on view montage

# Collect new montage
def collectMMMap(nav_id, map_name, mean_threshold, bbox, padding_factor, montage_overlap, mic_params, model):
    # Check if montage already exists
    note_id = int(sem.NavIndexWithNote(map_name + ".mrc"))
    if note_id > 0:
        map_id = note_id
        sem.Echo("MM montage already available. Skipping acquisition...")
    else:
        # Determine montage size
        size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]]) * padding_factor
        #mont_x = int(round(size[0] * model.pix_size / pix_size / (cam_dims[0] - (min(cam_dims) * montage_overlap)))) + 1
        #mont_y = int(round(size[1] * model.pix_size / pix_size / (cam_dims[1] - (min(cam_dims) * montage_overlap)))) + 1
        mont = np.round(size * model.pix_size / mic_params.view_pix_size / (mic_params.cam_dims - (min(mic_params.cam_dims) * montage_overlap))).astype(int) + np.ones(2, dtype=int)

        sem.MoveToNavItem(nav_id)

        # Check if targeting is totally off and allow for manual correction
        if mean_threshold > 0:
            sem.V()
            mean_counts = sem.ReportMeanCounts("A")
            if mean_counts < mean_threshold:
                user_input = sem.YesNoBox("\n".join(["DARK IMAGE", "", "The coordinates might be off target. Do you want to manually drag the image to a nearby lamella?"]))
                if user_input == 1:
                    sem.OKBox("\n".join(["Please center your target by dragging the image using the right mouse button! Press the <b> key when finished!", "Press <v> immediately following <b> if you want to take another View image."]))
                    sem.Echo("NOTE: Please center your target by dragging the image using the right mouse button! Press the <b> key when finished!")
                    user_confirm = False
                    while not user_confirm:
                        while not sem.KeyBreak():
                            sem.Delay(0.1, "s")
                        user_confirm = True
                        for i in range(10):
                            if sem.KeyBreak("v"):
                                user_confirm = False
                                sem.Echo("Take new view image!")
                                sem.V()
                                break
                            sem.Delay(0.1, "s")
                    sem.ResetImageShift()

        sem.Eucentricity(1)
        sem.UpdateItemZ(nav_id)
        
        sem.Echo("Collecting medium mag montage at [" + map_name.split("_")[-1] + "] (" + str(mont[0]) + "x" + str(mont[1]) + ")...")

        sem.OpenNewMontage(mont[0], mont[1], os.path.join(CUR_DIR, map_name + ".mrc"))
        sem.SetMontageParams(1, int(montage_overlap * min(mic_params.cam_dims)), int(montage_overlap * min(mic_params.cam_dims)), int(mic_params.cam_dims[0]), int(mic_params.cam_dims[1])) # stage move, overlap X, overlap Y, piece size X, piece size Y, skip correlation, binning

        sem.M()
        map_id = int(sem.NewMap(0, map_name + ".mrc"))
        sem.ChangeItemLabel(map_id, "L" + map_name.split("_L")[-1])  
        sem.CloseFile()
    return map_id

# Save montage as rescaled input image
def saveMMMap(map_id, map_dir, map_name, mic_params, model, overwrite=False):
    if os.path.exists(os.path.join(map_dir, map_name + ".png")) and not overwrite:
        sem.Echo("MM map already saved. Skipping saving...")
    else:
        sem.LoadOtherMap(map_id)
        buffer, *_ = sem.ReportCurrentBuffer()
        montage = np.asarray(sem.bufferImage(buffer))
        montage = transform.rescale(montage.astype(float) / 255, mic_params.view_pix_size / model.pix_size)
        montage = (montage * 255).astype(np.uint8)
        montage_img = Image.fromarray(montage)
        montage_img.save(os.path.join(map_dir, map_name + ".png"))

# Control queue for MM montage inference
def queueSpaceRun(model, new_map_name=None):
    # Check run file
    space_runs = []
    active_runs = []
    if os.path.exists(os.path.join(CUR_DIR, "SPACE_runs.txt")):
        with open(os.path.join(CUR_DIR, "SPACE_runs.txt"), "r") as f:
            space_lines = f.readlines()
        for line in space_lines:
            map_name = os.path.splitext(line)[0]
            space_runs.append(map_name)
            map_seg = os.path.join(CUR_DIR, map_name + "_seg.png")
            if not os.path.exists(map_seg):
                sem.Echo("SPACEtomo inference of " + line + " is still running...")
                active_runs.append(map_name)
            else:
                sem.Echo("SPACEtomo inference of " + line + " was completed.")
    # Check queue file
    space_queue = []
    if os.path.exists(os.path.join(CUR_DIR, "SPACE_queue.txt")):
        with open(os.path.join(CUR_DIR, "SPACE_queue.txt"), "r") as f:
            space_lines = f.readlines()
        for line in space_lines:
            sem.Echo("SPACEtomo inference of " + line + " is queued.")
            map_name = os.path.splitext(line)[0]
            space_queue.append(map_name)
    # Add new map to queue
    if new_map_name is not None:
        map_seg = os.path.join(CUR_DIR, new_map_name + "_seg.png")
        if os.path.exists(map_seg):
            sem.Echo("SPACEtomo was already run on this lamella. Skipping " + new_map_name + "...")
        else:
            space_queue.append(new_map_name)
    # Submit new run
    while len(active_runs) < model.max_runs and len(space_queue) > 0:
        map_name = space_queue.pop(0)
        out_file = open(os.path.join(CUR_DIR, map_name + "_SPACE.log"), "w")
        subprocess.Popen(["python", model.script, os.path.join(CUR_DIR, map_name + ".png")], stdout=out_file, stderr=subprocess.STDOUT, text=True)
        space_runs.append(map_name)
        active_runs.append(map_name)
    # Write runs file
    if len(space_runs) > 0:     
        space_output = ""
        for map_name in space_runs:
            space_output += map_name + ".png" + "\n"
        with open(os.path.join(CUR_DIR, "SPACE_runs.txt"), "w") as f:
            f.write(space_output)
    # Write queue file
    space_output = ""
    for map_name in space_queue:
        space_output += map_name + ".png" + "\n"
    with open(os.path.join(CUR_DIR, "SPACE_queue.txt"), "w") as f:
        f.write(space_output)

class Lamella:
    def __init__(self, map_name, map_dir, target_list, penalty_list, model, score_weights, score_weights_edge, grid_vecs, mic_params, max_tilt, plot=False):
        self.map_name = map_name
        self.model = model
        self.penalty_list = penalty_list
        if target_list[0] == "lamella":
            self.target_list = [name for name in self.model.categories if name not in self.penalty_list and name not in ["black", "white", "crack", "dynabeads"]]
        else:
            self.target_list = target_list
        self.score_weights = score_weights
        self.score_weights_edge = score_weights_edge
        self.vecs = grid_vecs
        self.mic_params = mic_params
        self.max_tilt = max_tilt
        self.plot = plot

        self.map = np.array(Image.open(os.path.join(map_dir, map_name + ".png")))
        self.segmentation = np.array(Image.open(os.path.join(map_dir, map_name + "_seg.png")))
        sem.Echo("Generating target mask...")
        self.target_mask = self.makeMask(self.target_list)
        sem.Echo("Generating background mask...")
        self.geo_mask = self.makeMask([name for name in self.model.categories if name not in self.target_list and name not in self.penalty_list and name not in ["black", "white"]])
        sem.Echo("Generating penalty mask...")
        self.penalty_mask = self.makeMask(self.penalty_list)

        self.points = []
        self.point_scores = []
        self.clusters = []
        self.geo_points = []

    # Create mask from segmentation and selected classes
    def makeMask(self, class_names):
        mask = np.zeros(self.segmentation.shape)
        for name in class_names:
            if name not in self.model.categories:
                sem.Echo("Error: Unknown target class: " + name)
                sem.Echo("Possible classes are: ")
                sem.Echo(str(self.model.categories.keys()))
                break
            mask[self.segmentation == self.model.categories[name]] = 1
        if self.plot:
            sem.Echo("Classes: " +  str(class_names))
            fig, ax = plt.subplots()
            plt.imshow(mask)
            fig.savefig(os.path.join(CUR_DIR, self.map_name + "_mask_" + class_names[0] + ".png"))
            plt.clf()
        return mask

    # Define hexagonal grid of points depending on beam diamter
    def definePoints_grid(self, penalty_weight, threshold=0.0, alternative_mask=None):
        x1, x2 = self.vecs
        rec_dims = np.array(self.score_weights.shape)
        if alternative_mask is None:
            mask = self.target_mask
        else:
            mask = alternative_mask
        points = []
        point_scores = []
        max_rows = int(mask.shape[0] / abs(x1[0]) * 1.5)
        max_cols = int(mask.shape[1] / abs(x2[1]) * 1.5)
        for i in range(max_rows):
            for j in range(1, max_cols):
                point = i * x1 + j * x2 - max_rows // 2 * x1 - max_cols // 2 * x2 + np.array(mask.shape) // 2
            
                score = calcScore_point(point, mask, self.penalty_mask, penalty_weight, self.score_weights)
                if score > threshold:
                    points.append(point)
                    point_scores.append(score)
        return np.array(points), point_scores

    # Define sparse points depending on segmentation
    def definePoints_sparse(self, penalty_weight, threshold=0.0, alternative_mask=None):
        rec_dims = np.array(self.score_weights.shape)
        if alternative_mask is None:
            mask = self.target_mask
        else:
            mask = alternative_mask
        points = []
        point_scores = []
        i = rec_dims[0] / 2
        while i < mask.shape[0]:
            j = rec_dims[1] / 2
            while j < mask.shape[1]:
                point = np.array([i, j])
                score = calcScore_point(point, mask, self.penalty_mask, penalty_weight, self.score_weights)
                if score > threshold:
                    overlap = False
                    for pt in points:
                        dist = np.linalg.norm(pt - point)
                        if dist < self.model.beam_diameter:
                            overlap = True
                            break
                    if not overlap:
                        points.append(point)
                        point_scores.append(score)
                j += rec_dims[1]
            i += rec_dims[0]
        return np.array(points), point_scores
    
    # Add additional points surrounding existing points
    def addAdjacentPoints(self, penalty_weight, threshold=0.0):
        x1, x2 = self.vecs
        new_points = []
        new_point_scores = []
        for p, point in enumerate(self.points):
            new_points.append(point)
            new_point_scores.append(self.point_scores[p])
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == j == 0:
                        continue
                    new_point = point + i * x1 + j * x2
                    for other_point in new_points:
                        dif_real = other_point - new_point
                        dif_squeezed = (self.mic_params.view_rotM.T @ dif_real) * np.array([1, np.cos(np.radians(self.max_tilt))])   # scale distance according to maximum tilt angle
                        if np.linalg.norm(dif_squeezed) < self.model.beam_diameter:
                            break  
                    score = calcScore_point(new_point, self.target_mask, self.penalty_mask, penalty_weight, self.score_weights)
                    if score > threshold:
                        new_points.append(new_point)
                        new_point_scores.append(score)
        sem.Echo("Added " + str(len(new_points) - len(self.points)) + " adjacent points!")
        sem.Echo(str(len(new_points)) + " points remaining.")
        self.points = np.array(new_points)
        self.point_scores = new_point_scores

    # Find clusters of points based on beam diameter
    def findClusters(self, points):
        if len(points) < 2:
            return [[0]]
        squeezed_points = np.array([(self.mic_params.view_rotM.T @ point) * np.array([1, np.cos(np.radians(self.max_tilt))]) for point in points])
        distM = distance_matrix(squeezed_points, squeezed_points)
        clusters = []
        for p1 in range(len(points)):
            clusters.append([p[0] for p in np.argwhere(distM[p1] <= 1.01 * self.model.beam_diameter)])
        
        clusters = sorted(clusters)
        
        for i in range(3):
            if len(clusters) > 1:
                final_clusters = [clusters[0]]
                for cluster in clusters[1:]:
                    cluster_set = set(cluster)
                    overlap = False
                    for fc in range(len(final_clusters)):
                        fcluster_set = set(final_clusters[fc])
                        if len(cluster_set & fcluster_set) != 0:
                            final_clusters[fc] = list(cluster_set | fcluster_set)
                            overlap = True
                            break
                    if not overlap:
                        final_clusters.append(cluster)  
                clusters = final_clusters
            
        sem.Echo("Clusters found: " + str(len(clusters)))
        return clusters    

    # Clean points based on score threshold
    def cleanPoints_score(self, penalty_weight, threshold=0.0, clusters=None):
        cleaned_points = []
        cleaned_point_scores = []
        removed_point_ids = []
        for p, point in enumerate(self.points):
            score = calcScore_point(point, self.target_mask, self.penalty_mask, penalty_weight, self.score_weights)
            if score > threshold:
                cleaned_points.append(point)
                cleaned_point_scores.append(score)
            else:
                removed_point_ids.append(p)
        if clusters is not None:
            removed_point_ids = sorted(removed_point_ids, reverse=True)
            for p in removed_point_ids:
                cleaned_clusters = []
                for cluster in clusters:
                    cleaned_clusters.append([])
                    for i in cluster:
                        if i == p: continue
                        if i > p: cleaned_clusters[-1].append(i - 1)
                        else: cleaned_clusters[-1].append(i)
                clusters = [cluster for cluster in cleaned_clusters if cluster != []]
                
        sem.Echo("Cleaned " + str(len(removed_point_ids)) + " points by score threshold!")
        sem.Echo(str(len(cleaned_points)) + " points remaining.")
        self.points = np.array(cleaned_points)
        self.point_scores = cleaned_point_scores
        if clusters is not None:
            self.clusters = clusters

    # Clean points based on beam overlap (needs more flexible input and output to clean against other groups of points, e.g. geo points)
    def cleanPoints_dist(self, points, max_tilt=0, tolerance=0.1, point_scores=None, clusters=None):
        if point_scores is not None:    # sort points accoring to score to remove low score points first
            point_zip = sorted(zip(list(range(len(points))), points, point_scores), key=lambda x: x[2])
            points = [p for _, p, _ in point_zip]
            original_point_ids = [pid for pid, _, _ in point_zip]
        else:
            original_point_ids = list(range(len(points)))
        removed_point_ids = []
        moved_point_ids = []
        for p1 in range(len(points)):
            too_close = False
            for p2 in range(len(points)):
                if p1 == p2 or original_point_ids[p2] in removed_point_ids: continue
                dif_real = points[p2] - points[p1]
                dif_squeezed = (self.mic_params.view_rotM.T @ dif_real) * np.array([1, np.cos(np.radians(max_tilt))])   # scale distance according to maximum tilt angle
                if np.linalg.norm(dif_squeezed) < self.model.beam_diameter * (1 - 2 * tolerance * np.linalg.norm(dif_squeezed) / np.linalg.norm(dif_real)) or (np.linalg.norm(dif_squeezed) < 0.99 * self.model.beam_diameter and (p1 in moved_point_ids or p2 in moved_point_ids)):
                    too_close = True
                    break
                elif np.linalg.norm(dif_squeezed) < self.model.beam_diameter:
                    overlap = (self.model.beam_diameter - np.linalg.norm(dif_squeezed))
                    move = self.mic_params.view_rotM @ (overlap * dif_squeezed / np.linalg.norm(dif_squeezed) / np.array([1, np.cos(np.radians(max_tilt))]) / 2)
                    points[p1] = points[p1] - move
                    points[p2] = points[p2] + move
                    moved_point_ids.extend([p1, p2])
            if too_close:
                removed_point_ids.append(original_point_ids[p1])
        if point_scores is not None:
            point_zip = sorted(zip(original_point_ids, points), key=lambda x: x[0])
            points = [p for _, p in point_zip]
        cleaned_points = [point for p, point in enumerate(points) if p not in removed_point_ids]
        
        if clusters is not None:
            removed_point_ids = sorted(removed_point_ids, reverse=True)
            for p in removed_point_ids:
                cleaned_clusters = []
                for cluster in clusters:
                    cleaned_clusters.append([])
                    for i in cluster:
                        if i == p: continue
                        if i > p: cleaned_clusters[-1].append(i - 1)
                        else: cleaned_clusters[-1].append(i)
                clusters = [cluster for cluster in cleaned_clusters if cluster != []]
                
        sem.Echo("Cleaned " + str(len(removed_point_ids)) + " points by distance threshold!")
        sem.Echo(str(len(cleaned_points)) + " points remaining.")
        return np.array(cleaned_points), clusters
    
    # Remove single point from from beam overlap
    def distancePoint(self, point):
        moved = False
        for p in range(len(self.points)):
            dif_real = self.points[p] - point
            dif_squeezed = (self.mic_params.view_rotM.T @ dif_real) * np.array([1, np.cos(np.radians(self.max_tilt))])   # scale distance according to maximum tilt angle
            if np.linalg.norm(dif_squeezed) < self.model.beam_diameter:
                overlap = (self.model.beam_diameter - np.linalg.norm(dif_squeezed))
                move = self.mic_params.view_rotM @ (overlap * dif_squeezed / np.linalg.norm(dif_squeezed) / np.array([1, np.cos(np.radians(self.max_tilt))]))
                point = point - move         
                moved = True
        return point, moved
    
    # Create plot of targets on montage
    def plotTargets(self, offsets=[], clusters=[], tracking_id=None, overlay=None, save="temp.png"):
        rec_dims = np.array(self.score_weights.shape)
        if len(self.points) > 0:
            if tracking_id is None:
                middle_point = [np.max(self.points[:, 0]) / 2 + np.min(self.points[:, 0]) / 2, np.max(self.points[:, 1]) / 2 + np.min(self.points[:, 1]) / 2]
                closest_middle = self.points[np.argmin(np.linalg.norm(self.points - middle_point, axis=1))]
            else:
                closest_middle = self.points[tracking_id]
        else:
            closest_middle = np.array(self.map.shape) // 2
        fig, ax = plt.subplots(figsize=(16, 12))
        
        if overlay is None:
            plt.imshow(self.map, cmap="gray")
        else:
            alpha = 1 / 4
            overlay_map = np.clip(np.dstack([self.map / 255 * (1 - alpha) + overlay * alpha, self.map / 255 * (1 - alpha), self.map / 255 * (1 - alpha)]), 0, 1)
            plt.imshow(overlay_map)
        plt.axline(closest_middle[[1, 0]], slope=-1 / np.tan(np.radians(self.mic_params.view_ta_rotation)), linestyle="--", color="#ffffff")
        
        if len(offsets) > 0 and len(clusters) > 0:
            for c, cluster in enumerate(clusters):
                data = np.array(self.points[cluster, :])
                plt.scatter(data[:, 1], data[:, 0], s=100)
                offset_cluster_pixel = offsets[c] * max(rec_dims) * 10
                plt.arrow(self.points[cluster, :][0][1] - offset_cluster_pixel[1], self.points[cluster, :][0][0] - offset_cluster_pixel[0], offset_cluster_pixel[1], offset_cluster_pixel[0], length_includes_head=True, width=10, color="#ff7d00")
        
        for p, point in enumerate(self.points):
            ellipse = Ellipse((point[1], point[0]), self.model.beam_diameter / np.cos(np.radians(self.max_tilt)), self.model.beam_diameter, angle=self.mic_params.view_ta_rotation, fill=False, linewidth=1, color="#ffd700")
            ax.add_artist(ellipse)
            if np.array_equal(point, closest_middle):
                color = "#c92b27"
            else:
                color = "#578abf"
            rect = Rectangle([point[1] - rec_dims[1] / 2, point[0] - rec_dims[0] / 2], rec_dims[1], rec_dims[0], fill=False, linewidth=2, color=color)
            ax.add_artist(rect)
            plt.text(point[1] - 10, point[0] - 10, str(p + 1), color=color)

        if len(offsets) == 1:
            offset_pixel = offsets[0] * max(rec_dims) * 10
            plt.arrow(closest_middle[1] - offset_pixel[1], closest_middle[0] - offset_pixel[0], offset_pixel[1], offset_pixel[0], length_includes_head=True, width=10, color="#ff7d00")
        plt.axis("equal")
        plt.savefig(save, bbox_inches="tight", dpi=300)
        plt.clf()

    # Find targets based on mask
    def findPoints(self, sparse=False, penalty_weight=0.5, threshold=0.0, iterations=10, extra_tracking=False):
        rec_dims = np.array(self.score_weights.shape)
        # Setup initial points
        if sparse:
            self.points, self.point_scores = self.definePoints_sparse(penalty_weight=0)
        else:
            self.points, self.point_scores = self.definePoints_grid(penalty_weight=0)

        sem.Echo("Initial points found: " + str(len(self.points)))

        if self.plot: self.plotTargets(save=self.map_name + "_points_initial.png")

        if len(self.points) == 0:
            return False

        # Global translation optimization
        if not sparse:
            start_offset = np.zeros(2)
            offset = minimize(calcScore_cluster, start_offset, args=(self.points, self.target_mask, self.penalty_mask, 0.0, self.score_weights, self.score_weights_edge), method="nelder-mead", bounds=((-0.1, 0.1), (-0.1, 0.1)))
            offset_pixel = offset.x * max(rec_dims) * 10
            self.points += offset_pixel
            
            if self.plot: self.plotTargets(offsets=[offset.x], save=self.map_name + "_points_globalSGD.png")
            self.cleanPoints_score(penalty_weight=0, threshold=threshold)

        # Find clusters
        self.clusters = self.findClusters(self.points)

        # Iterative optimization
        if sparse: #len(self.score_weights_edge) > 0:
            adjacent_points = False
        else:
            adjacent_points = True
            
        for it in range(iterations):
            sem.Echo("____________________")
            sem.Echo("Iteration: " + str(it + 1))
            
            # Translation optimization for each cluster
            cluster_offsets = []
            total_offset = 0
            for cluster in self.clusters:
                start_offset = np.zeros(2)
                offset_cluster = minimize(calcScore_cluster, start_offset, args=(self.points[cluster, :], self.target_mask, self.penalty_mask, penalty_weight, self.score_weights, self.score_weights_edge), method="nelder-mead", bounds=((-0.1, 0.1), (-0.1, 0.1)))
                offset_cluster_pixel = offset_cluster.x * max(rec_dims) * 10
                self.points[cluster, :] += offset_cluster_pixel
                cluster_offsets.append(offset_cluster.x)
                total_offset += np.linalg.norm(offset_cluster_pixel)

            if self.plot: self.plotTargets(offsets=cluster_offsets, clusters=self.clusters, save=self.map_name + "_points_it" + str(it + 1).zfill(2) + ".png")
            
            # Clean points
            prev_number = len(self.points)
            self.cleanPoints_score(penalty_weight, threshold, self.clusters)
            self.points, self.clusters = self.cleanPoints_dist(self.points, self.max_tilt, tolerance=0.1, point_scores=self.point_scores, clusters=self.clusters)
            new_number = len(self.points)
            
            # Find new clusters
            if new_number != prev_number:
                if new_number == 0:
                    break
                self.clusters = self.findClusters(self.points)
            
            # Check for convergence
            sem.Echo("Total offset: " + str(total_offset))
            if total_offset < 1 and new_number == prev_number:
                if not adjacent_points and it < iterations - 1:
                    sem.Echo("Adding additional adjacent points...")
                    self.addAdjacentPoints(penalty_weight)
                    self.clusters = self.findClusters(self.points)
                    adjacent_points = True
                if len(self.points) == prev_number:
                    sem.Echo("##### Converged! #####")
                    sem.Echo("")
                    break
        
        if len(self.points) > 0:
            sem.Echo("Choosing tracking target...")
            # Find middle point, choose tracking target closest to it and move it to start of list of points
            middle_point = [np.max(self.points[:, 0]) / 2 + np.min(self.points[:, 0]) / 2, np.max(self.points[:, 1]) / 2 + np.min(self.points[:, 1]) / 2]

            # Consider tilted FOV to avoid ice coming in and compromising tracking
            expanded_dims = (self.mic_params.view_rotM @ ((self.mic_params.view_rotM.T @ np.array(self.score_weights.shape)) / np.array([1, np.cos(np.radians(self.max_tilt))]))).astype(int)

            if extra_tracking: # or len(self.points) == 1:
                for it in range(3):
                    # Use middle point as initial tracking point and refine based on expanded dims and penalty mask only 
                    start_offset = np.random.random(2)   # add a random offset in case middle point coincides with only point
                    offset = minimize(calcScore_cluster, start_offset, args=([middle_point], np.zeros(self.target_mask.shape), self.penalty_mask, 1, np.ones(expanded_dims)), method="nelder-mead", bounds=((-0.1, 0.1), (-0.1, 0.1)))
                    offset_pixel = offset.x * max(rec_dims) * 10
                    middle_point += offset_pixel
                    # In case of overlap remove tracking point from other point
                    middle_point, moved = self.distancePoint(middle_point)
                    # No need for additional iterations if point was not moved after minimize
                    if not moved:
                        break
                self.points = np.vstack([middle_point, self.points])
                sem.Echo("Successfully added additional tracking target.")
            else:                
                closest_middle_ids = np.argsort(np.linalg.norm(self.points - middle_point, axis=1))
                # Check 3 closest points to middle for ice
                ice_scores = []
                for min_id in range(0, min(3, len(closest_middle_ids))):
                    # Calculate score using expanded dims and only penalty mask
                    ice_scores.append(calcScore_point(self.points[closest_middle_ids[min_id]], np.zeros(self.target_mask.shape), self.penalty_mask, 1, np.ones(expanded_dims)))
                sem.Echo("Candidate ice scores: " + str(ice_scores))
                # Use point with least ice for tracking
                min_ice = np.argmax(ice_scores)

                closest_middle = self.points[closest_middle_ids[min_ice]]
                self.points = np.vstack([closest_middle, np.delete(self.points, closest_middle_ids[min_ice], 0)])
                sem.Echo("Successfully chose tracking target.")

    # Find points to measure geometry
    def findGeoPoints(self):
        sem.Echo("Finding geo points...")
        # Start from grid of points with high threshold
        geo_points, _ = self.definePoints_grid(penalty_weight=0.5, threshold=0.8, alternative_mask=self.geo_mask)

        # Remove points overlapping with target points at zero tilt
        all_points, _ = self.cleanPoints_dist(np.concatenate([geo_points, self.points]), max_tilt=0, tolerance=0.0)
        geo_points = all_points[:-len(self.points), :]

        # Remove points beyond IS limits from tracking point (assuming default of 15 microns)
        prev_number = len(geo_points)
        geo_points = np.array([point for point in geo_points if np.linalg.norm(point - self.points[0]) * self.model.pix_size < 15000])
        if len(geo_points) < prev_number:
            sem.Echo("NOTE: Removed " + str(prev_number - len(geo_points)) + " geo points due to image shift limit.")

        # Only use 1 point per cluster if >=5 clusters
        if len(geo_points) > 5:
            geo_clusters = self.findClusters(geo_points)
            if len(geo_clusters) >= 5:
                geo_points = geo_points[[cluster[0] for cluster in geo_clusters]]

        if len(geo_points) > 10:
            # Choose 10 random geo points (TODO: update to use extreme points)
            geo_points = np.random.default_rng().choice(geo_points, 10, replace=False)

        sem.Echo(str(len(geo_points)) + " geo points were selected.")
        if len(geo_points) < 3:
            sem.Echo("WARNING: Not enough geo points found to measure the sample geometry. The user defined pretilt and rotation values will be used.")
        self.geo_points = geo_points

    # Save targets for PACEtomo   
    def saveAsTargets(self, buffer, penalty_weight):
        # Load map from buffer
        map_image = np.asarray(sem.bufferImage(buffer), dtype=float)

        targets = []
        nav_ids = []
        nav_virt_maps = []
        nav_virt_maps_stats = []
        geo_SS = []
        
        if len(self.points) > 0:
            for i, point in enumerate(self.points):
                # Make virtual map
                virt_map_name = self.map_name + "_tgt_" + str(i + 1).zfill(3) + "_view.mrc"
                image_crop = np.flip(cropImage(map_image, (int(point[0] * self.model.pix_size / self.mic_params.view_pix_size), int(point[1] * self.model.pix_size / self.mic_params.view_pix_size)), self.mic_params.cam_dims[[1, 0]]), axis=0).astype(np.float32)
                nav_virt_maps_stats.append([np.min(image_crop), np.max(image_crop)])
                writeMrc(virt_map_name, image_crop, self.mic_params.view_pix_size)
                nav_virt_maps.append(virt_map_name)

                # Add nav item
                nav_ids.append(int(sem.AddImagePosAsNavPoint(buffer, int(point[1] * self.model.pix_size / self.mic_params.view_pix_size), int(point[0] * self.model.pix_size / self.mic_params.view_pix_size))))
                sem.ChangeItemNote(nav_ids[-1], virt_map_name)
                nav_info = sem.ReportOtherItem(nav_ids[-1])
                if i == 0:
                    stage0 = np.array([nav_info[1], nav_info[2]])
                    stage = stage0
                    SS = [0, 0]
                else:
                    stage = np.array([nav_info[1], nav_info[2]])
                    SS = self.mic_params.rec_s2ssMatrix @ (stage - stage0)

                # Calculate final SPACE score
                score = calcScore_point(point, self.target_mask, self.penalty_mask, penalty_weight, self.score_weights, self.score_weights_edge)

                # Check if target is within IS limits
                if SS[0] > 15 or SS[1] > 15:
                    sem.Echo("WARNING: Point " + str(i + 1) + " requires image shifts (" + str(round(SS[0], 1)) + "|" + str(round(SS[1], 1)) + ") beyond the default image shift limit (15)!")

                # Add target
                targets.append({"tsfile": self.map_name + "_ts_" + str(i + 1).zfill(3) + ".mrc", "viewfile": self.map_name + "_tgt_" + str(i + 1).zfill(3) + "_view.mrc", "SSX": SS[0], "SSY": SS[1], "stageX": stage[0], "stageY": stage[1], "SPACEscore": score, "skip": "False"})

            # Get specimen coords of geo points
            for point in self.geo_points:
                # Add nav item
                geo_id = int(sem.AddImagePosAsNavPoint(buffer, int(point[1] * self.model.pix_size / self.mic_params.view_pix_size), int(point[0] * self.model.pix_size / self.mic_params.view_pix_size)))
                sem.ChangeItemColor(geo_id, 5)
                geo_item = sem.ReportOtherItem(geo_id)
                geo_coords = self.mic_params.rec_s2ssMatrix @ (np.array([geo_item[1], geo_item[2]]) - stage0)
                geo_SS.append({"SSX": geo_coords[0], "SSY": geo_coords[1]})

            # Set up nav for acquisition
            sem.ChangeItemNote(int(nav_ids[0]), self.map_name + "_tgts.txt")
            sem.SetItemAcquire(int(nav_ids[0]))

            # Change tgt points to virtual view maps
            changeNavPtsToMaps(nav_ids, nav_virt_maps, nav_virt_maps_stats)

        # Write PACE target file
        tgtsFilePath = os.path.join(CUR_DIR, self.map_name + "_tgts.txt")
        writeTargets(tgtsFilePath, targets, geo_SS)

        sem.Echo("Target selection completed! " + str(len(targets)) + " targets were selected.")



##### UTILITY FUNCTIONS #####

class MicParams:
    def __init__(self, WG_image_state, IM_mag_index, MM_image_state):
        self.WG_image_state = WG_image_state
        self.IM_mag_index = IM_mag_index
        self.MM_image_state = MM_image_state

        self.WG_pix_size = None
        self.IM_pix_size = None

    def getViewParams(self):
        cam_props = sem.CameraProperties()
        self.cam_dims = np.array([cam_props[0], cam_props[1]])
        self.view_pix_size = cam_props[4]   # [nm]

        self.view_c2ssMatrix = np.array(sem.CameraToSpecimenMatrix(0)).reshape((2, 2))
        self.view_ta_rotation = 90 - np.degrees(np.arctan(self.view_c2ssMatrix[0, 1] / self.view_c2ssMatrix[0, 0]))
        sem.Echo("Tilt axis rotation: " + str(self.view_ta_rotation))
        self.view_rotM = np.array([[np.cos(np.radians(self.view_ta_rotation)), np.sin(np.radians(self.view_ta_rotation))], [-np.sin(np.radians(self.view_ta_rotation)), np.cos(np.radians(self.view_ta_rotation))]])

    def getRecParams(self):
        self.rec_s2ssMatrix = np.array(sem.StageToSpecimenMatrix(0)).reshape((2, 2))
        self.rec_beam_diameter = sem.ReportIlluminatedArea() * 100
        cam_props = sem.CameraProperties()
        self.rec_pix_size = cam_props[4]   # [nm]       

# Save settings for post action script
def saveSettings(filename, vars, start="SEMflush", end="sem"):
    output = "# SPACEtomo settings from " + datetime.now().strftime("%d.%m.%Y %H:%M:%S") + "\n"
    save = False
    for var in vars:                                    # globals() is ordered by creation, start and end points might have to be adjusted if script changes
        if var == end:                                  # first var after settings vars
            break
        if save:
            if isinstance(vars[var], str):
                output += var + " = r'" + vars[var] + "'" + "\n"
            else:
                output += var + " = " + str(vars[var]) + "\n"
        if var == start:                                   # last var before settings vars
            save = True
    with open(filename + "_settings.txt", "w") as f:
        f.write(output)

# Load grid and report name
def loadGrid(grid_slot, grid_default_name=""):
    # Check grid slot
    load_grid = False
    slot_status = sem.ReportSlotStatus(grid_slot)
    if grid_slot > 0 and slot_status[0] > 0:
        load_grid = True
        if len(slot_status) > 1 and isinstance(slot_status[-1], str):
            grid_name = slot_status[-1]
        else:
            grid_name = "G" + str(grid_slot).zfill(2)
    elif grid_slot == 0:
        if grid_default_name != "":
            grid_name = grid_default_name
        else:
            grid_name = "GX"
    else:
        sem.OKBox("Grid not found in slot. Please set grid_slot = 0 if it is already loaded!")
        sem.Exit()

    # Open new navigator and log
    openNav(grid_name)
    sem.SaveLogOpenNew(grid_name)

    sem.Echo("SPACEtomo Version " + versionSPACE)
    sem.ProgramTimeStamps()

    # Initiate loading
    if load_grid:
        sem.Echo("Loading grid [" + grid_name + "] from slot " + str(grid_slot) + "...")
        sem.LoadCartridge(grid_slot)

    return grid_name

# Open new nav file if necessary
def openNav(grid_name):
    # Check if nav file is open
    nav_status = sem.ReportIfNavOpen()
    if nav_status < 2:
        if nav_status == 1:
            sem.SaveNavigator("temp.nav")
            sem.CloseNavigator()
            sem.Echo("WARNING: Open navigator was saved as temp.nav and closed!")
        nav_file = os.path.join(CUR_DIR, grid_name + ".nav")
        sem.OpenNavigator(nav_file)
    else:
        # Check if current nav file is from loaded grid
        nav_file = sem.ReportNavFile()
        if os.path.splitext(os.path.basename(nav_file))[0] != grid_name:
            sem.SaveNavigator()
            nav_file = os.path.join(CUR_DIR, grid_name + ".nav")
            if os.path.exists(nav_file):
                sem.ReadNavFile(nav_file)
            else:
                sem.OpenNavigator(nav_file)

# Find vectors for hexagonal grid
def findGridVecs(model, max_tilt, mic_params):
    rotM_pattern = np.array([[np.cos(np.radians(60)), -np.sin(np.radians(60))], [np.sin(np.radians(60)), np.cos(np.radians(60))]])
    x1 = np.array([model.beam_diameter, 0])
    x2 = rotM_pattern @ x1
    x2[1] /= np.cos(np.radians(max_tilt))

    #rotM = np.array([[np.cos(np.radians(mic_params.view_ta_rotation)), np.sin(np.radians(ta_rotation))], [-np.sin(np.radians(ta_rotation)), np.cos(np.radians(ta_rotation))]])
    x1 = mic_params.view_rotM @ x1
    x2 = mic_params.view_rotM @ x2
    if abs(mic_params.view_ta_rotation) > 45:
        xt = x1[:]
        x1 = x2[:]
        x2 = xt

    return np.vstack([x1, x2])

# Create score weights
def makeScoreWeights(model, edge=False, edge_angle_sampling=45):
    # Setup weight mask
    weight_mask = np.zeros(model.cam_dims)
    x, y = np.meshgrid(np.linspace(-model.cam_dims[1] // 2, model.cam_dims[1] // 2, model.cam_dims[1]), np.linspace(-model.cam_dims[0] // 2, model.cam_dims[0] // 2, model.cam_dims[0]))
    d = np.sqrt(x*x+y*y)
    sigma, mu = max(model.cam_dims) // 2, 0
    weight_mask = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    weight_mask /= np.sum(weight_mask) / np.prod(model.cam_dims)

    # Setup edge masks
    edge_weight_masks = []
    if edge:
        x, y = np.meshgrid(np.linspace(-max(model.cam_dims), max(model.cam_dims), 2 * max(model.cam_dims)), np.linspace(-max(model.cam_dims), max(model.cam_dims), 2 * max(model.cam_dims)))
        d = np.sqrt(x*x+y*y)
        sigma, mu = max(model.cam_dims) // 2, 0
        square_weight = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        square_weight /= np.sum(weight_mask) / np.prod(model.cam_dims) / 2
        
        edge_weight = np.array(square_weight)
        edge_weight[0:square_weight.shape[0] // 2, :] =  -square_weight[0:square_weight.shape[0] // 2, :]
        
        dif_shape = np.array(square_weight.shape) - np.array(weight_mask.shape)
        
        for i in range(0, 360, edge_angle_sampling):
            edge_weight_masks.append(rotate(edge_weight, angle=i, reshape=False)[dif_shape[0] // 2: dif_shape[0] // 2 + weight_mask.shape[0], dif_shape[1] // 2: dif_shape[1] // 2 + weight_mask.shape[1]])

    return weight_mask, edge_weight_masks

# Calculate score of a point depening on camera dimensions
def calcScore_point(point, mask, penalty_mask, penalty_weight, weight_mask, edge_weight_masks=[]):
    rec_dims = np.array(weight_mask.shape)
    if not rec_dims[0] / 2 <= point[0] < mask.shape[0] - rec_dims[0] / 2 or not rec_dims[1] / 2 <= point[1] < mask.shape[1] - rec_dims[1] / 2:
        return 0
    if len(edge_weight_masks) > 0:
        edge_scores = []
        for edge_mask in edge_weight_masks:
            score = np.sum(mask[int(point[0] - rec_dims[0] / 2): int(point[0] + rec_dims[0] / 2), int(point[1] - rec_dims[1] / 2): int(point[1] + rec_dims[1] / 2)] * edge_mask) / np.prod(rec_dims)
            score -= np.sum(penalty_mask[int(point[0] - rec_dims[0] / 2): int(point[0] + rec_dims[0] / 2), int(point[1] - rec_dims[1] / 2): int(point[1] + rec_dims[1] / 2)] * weight_mask) / np.prod(rec_dims) * penalty_weight
            edge_scores.append(score)
        score = np.max(edge_scores) 
    else:
        score = np.sum(mask[int(point[0] - rec_dims[0] / 2): int(point[0] + rec_dims[0] / 2), int(point[1] - rec_dims[1] / 2): int(point[1] + rec_dims[1] / 2)] * weight_mask) / np.prod(rec_dims)
        score -= np.sum(penalty_mask[int(point[0] - rec_dims[0] / 2): int(point[0] + rec_dims[0] / 2), int(point[1] - rec_dims[1] / 2): int(point[1] + rec_dims[1] / 2)] * weight_mask) / np.prod(rec_dims) * penalty_weight
    return score

# Calculate the total score of a cluster of points
def calcScore_cluster(offset, points, mask, penalty_mask, penalty_weight, weight_mask, edge_weight_masks=[]):
    rec_dims = np.array(weight_mask.shape)
    pixel_offset = offset * max(rec_dims) * 10
    total_score = 0
    for pt in points:
        point = pt + pixel_offset
        score = calcScore_point(point, mask, penalty_mask, penalty_weight, weight_mask, edge_weight_masks)
        total_score -= score
    return total_score   

# Write PACEtomo target file
def writeTargets(targetFile, targets, geoPoints=[], savedRun=False, resume={"sec": 0, "pos": 0}, settings={}):
    output = ""
    if settings != {}:
        for key, val in settings.items():
            if val != "":
                output += "_set " + key + " = " + str(val) + "\n"
        output += "\n"
    if resume["sec"] > 0 or resume["pos"] > 0:
        output += "_spos = " + str(resume["sec"]) + "," + str(resume["pos"]) + "\n" * 2
    for pos in range(len(targets)):
        output += "_tgt = " + str(pos + 1).zfill(3) + "\n"
        for key in targets[pos].keys():
            output += key + " = " + str(targets[pos][key]) + "\n"
        if savedRun:
            output += "_pbr" + "\n"
            for key in savedRun[pos][0].keys():
                output += key + " = " + str(savedRun[pos][0][key]) + "\n"
            output += "_nbr" + "\n"
            for key in savedRun[pos][1].keys():
                output += key + " = " + str(savedRun[pos][1][key]) + "\n"       
        output += "\n"
    for pos in range(len(geoPoints)):
        output += "_geo = " + str(pos + 1) + "\n"
        for key in geoPoints[pos].keys():
            output += key + " = " + str(geoPoints[pos][key]) + "\n"
        output += "\n"
    with open(targetFile, "w") as f:
        f.write(output)

# Helper functions for virtual maps:

# Make sure nav contains template images
def getTemplateID():
    view_id = int(sem.NavIndexWithNote("Template View"))
    if view_id == 0:
        sem.SetColumnOrGunValve(0)
        sem.AllowFileOverwrite(1)
        sem.OpenNewFile("template_view.mrc")
        sem.V()
        sem.S()
        view_id = int(sem.NewMap(0, "Template View"))
        sem.ChangeItemLabel(view_id, "TV")
        sem.CloseFile()
        sem.SetColumnOrGunValve(1)
        sem.AllowFileOverwrite(0)
    return view_id

# Crop and pad the virtual map
def cropImage(image, coords, fov):
	imageCrop = image[max(0, int(coords[0] - fov[0] / 2)):min(image.shape[0], int(coords[0] + fov[0] / 2)), max(0, int(coords[1] - fov[1] / 2)):min(image.shape[1], int(coords[1] + fov[1] / 2))]
	if not np.array_equal(imageCrop.shape, fov):
		mean = np.mean(imageCrop)
		if imageCrop.shape[0] < fov[0]:
			padding = np.full((int(fov[0] - imageCrop.shape[0]), imageCrop.shape[1]), mean)
			if int(coords[0] - fov[0] / 2) < 0:
				imageCrop = np.concatenate((padding, imageCrop), axis=0)
			if int(coords[0] + fov[0] / 2) > image.shape[0]:
				imageCrop = np.concatenate((imageCrop, padding), axis=0)
		if imageCrop.shape[1] < fov[1]:
			padding = np.full((imageCrop.shape[0], int(fov[1] - imageCrop.shape[1])), mean)
			if int(coords[1] - fov[1] / 2) < 0:
				imageCrop = np.concatenate((padding, imageCrop), axis=1)
			if int(coords[1] + fov[1] / 2) > image.shape[1]:
				imageCrop = np.concatenate((imageCrop, padding), axis=1)		
		sem.Echo("WARNING: Target position is close to the edge of the map and was padded.")
	return imageCrop

def writeMrc(outfilename, image, pix_size):
	with mrcfile.new(os.path.join(CUR_DIR, outfilename), overwrite=True) as mrc:
		mrc.set_data(image)
		mrc.voxel_size = (pix_size * 10, pix_size * 10, pix_size * 10)
		mrc.update_header_from_data()
		mrc.update_header_stats()

def parseNav(navFile):
	with open(navFile) as f:
		navContent = f.readlines()
	header = []
	items = []
	newItem = {}
	index = 0

	for line in navContent:
		col = line.rstrip().split(" ")
		if col[0] == "": 
			if "Item" in newItem.keys():
				items.append(newItem)
				newItem = {}
				continue
			else:
				continue
		if line.startswith("[Item"):
			index += 1
			newItem = {"index": index, "Item": col[2].strip("]")}
		elif "Item" in newItem.keys():
			newItem[col[0]] = [val for val in col[2:]]
		else:
			header.append(line)
	if "Item" in newItem.keys():	#append last target
		items.append(newItem)
	return header, items

def writeNav(header, items, filename):
	text = ""
	for line in header:
		text += line
	text += os.linesep
	for item in items:
		text += "[Item = " + item["Item"] + "]" + os.linesep
		item.pop("Item")
		item.pop("index")
		for key, attr in item.items():
			text += key + " = "
			for val in attr:
				text += str(val) + " "
			text += os.linesep
		text += os.linesep
	with open(filename, "w", newline="") as f:
		f.write(text)
	return

# Read nav file and change entries to map entries using the virtual maps
def changeNavPtsToMaps(nav_ids, nav_maps, nav_virt_maps_stats):
    template_id = getTemplateID()   # make sure template exists before saving nav

    # Read nav file
    sem.SaveNavigator()
    nav_file = sem.ReportNavFile()
    nav_header, nav_items = parseNav(nav_file)

    # Make copy of view template entry
    template = copy.deepcopy(nav_items[template_id - 1])
    template.pop("RawStageXY")
    template.pop("SamePosId")

    # Determine image dimenstions for polygons
    ptsDX = np.array(np.array(template["PtsX"], dtype=float) - float(template["StageXYZ"][0]))
    ptsDY = np.array(np.array(template["PtsY"], dtype=float) - float(template["StageXYZ"][1]))    

    new_nav_ids = []
    for n, nav_id in enumerate(nav_ids):
        nav_file_id = nav_id - 1

        # Create new item with data from template and from nav point
        new_item = copy.deepcopy(template)
        new_item["Item"] = str(n + 1).zfill(3)
        new_item["StageXYZ"] = nav_items[nav_file_id]["StageXYZ"]
        new_item["Note"] = nav_items[nav_file_id]["Note"]
        if "Acquire" in nav_items[nav_file_id].keys(): 
            new_item["Acquire"] = nav_items[nav_file_id]["Acquire"]
        new_item["MapFile"] = [nav_maps[n]]
        new_item["MapMinMaxScale"] = nav_virt_maps_stats[n]    
        new_item["PtsX"] = ptsDX + float(nav_items[nav_file_id]["StageXYZ"][0])
        new_item["PtsY"] = ptsDY + float(nav_items[nav_file_id]["StageXYZ"][1])
    
        # Create new map ID
        unique_id = int(sem.GetUniqueNavID())
        while unique_id in new_nav_ids:
            unique_id = int(sem.GetUniqueNavID())
        new_nav_ids.append(unique_id)
        new_item["MapID"] = [unique_id]
        new_item["SamePosId"] = [unique_id]
    
        # Overwrite nav point with new nav item
        nav_items[nav_file_id] = copy.deepcopy(new_item)

    # Write nav file
    sem.SaveNavigator(nav_file)# + "~")						# backup unaltered nav file
    sem.CloseNavigator()
    writeNav(nav_header, nav_items, nav_file)
    sem.ReadNavFile(nav_file)
    
    sem.Echo("Updated navigator file with target maps.")