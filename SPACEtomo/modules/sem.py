#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_functions
# Purpose:      Functions necessary to run SPACEtomo.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/10/04
# Revision:     v1.1
# Last Change:  2024/04/10: check if tgts file already exists, added additional montage settings
#               2024/04/09: saved absolute path for virt maps in nav
#               2024/04/02: make inspected file when saving targets
#               2024/03/26: added check for GPU using torch
#               2024/03/25: fixes after Krios 3 test, added check for IM step already done (using WG_distance_threshold), fixed log management
#               2024/03/19: fixes after Krios 3 test
#               2024/03/08: outsourced target selection to updateQueue (also replaces queueSpaceRun)
#               2024/03/07: added import of more than one target area per lamella
#               2024/03/06: added check for stage is busy before WG montage, added check for no geo points before cleaning them up
#               2024/02/14: split IM routine to run lamella detection externally
#               2024/02/13: split collectWGMap from lamella detection
#               2023/12/12: fixed proximity removal, added version check
#               2023/12/11: added removal of lamellae due to close proximity
#               2023/11/28: fixes after Krios test
#               2023/11/15: added center padding for IM lamella detection and consider multiple hits, removed lamellae close to grid edge, removed geo points beyond IS limit, fixes and improvements to findOffset
#               2023/11/06: used IM lamella detection as confirmation
#               2023/11/04: fixes after Krios test
#               2023/10/31: fixes after Krios test
#               2023/10/27: added lamella option to select all classes, added ice check for tracking point, adjusted plotting option
#               2023/10/25: added nav file management
#               2023/10/23: added extra tracking
#               2023/10/18: added virtual map for targeting
#               2023/10/16: fixes after Krios test, removed polygon (listToSEMarray not found in external script)
#               2023/10/12: fixes after Krios test, added findOffset for more lamellae if lamella not found
#               2023/10/04: outsourcing of functions from main SPACEtomo script
# ===================================================================

import serialem as sem
import os
import glob
import copy
import json
from datetime import datetime
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None                                   # removes limit for large images
from skimage import transform, exposure
import mrcfile
import SPACEtomo.modules.ext as space_ext
import SPACEtomo.modules.utils as utils
from SPACEtomo.modules.utils import log

versionSPACE = "1.2dev"

sem.SuppressReports()
sem.SetNewFileType(0)		                                    # set file type to mrc in case user changed default file type
CUR_DIR = sem.ReportDirectory()
SPACE_DIR = os.path.dirname(__file__)

# Check if torch and GPU are available
try:
    from torch.cuda import is_available
    if is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except:
    DEVICE = "cpu"

##### WG functions #####

# Collect montage on new grid
""" MOVED TO modules.scope
def collectWGMap(grid_name, mic_params, montage_overlap):
    log("Collecting whole grid map...")

    # Check if WG map already exists
    note_id = sem.NavIndexWithNote(grid_name + ".mrc")
    if note_id > 0:
        map_id = note_id
        log("NOTE: Grid map already found in navigator. Skipping acquisiion.")
    else:
        sem.GoToImagingState(str(mic_params.WG_image_state))
        if sem.ReportLowDose()[0] > 0:
             log("WARNING: Whole grid montage image state is in Low Dose! This might result in unreasonably large grid maps.")
        sem.SetupFullMontage(montage_overlap, grid_name + ".mrc")

        # Make sure stage is ready (sometimes stage is busy error popped up when starting montage)
        while sem.ReportStageBusy():
            sem.Delay(1, "s")

        sem.M()
        map_id = sem.NewMap(0, grid_name + ".mrc")
        sem.CloseFile()

    sem.ChangeItemLabel(int(map_id), grid_name[:6])
    if len(grid_name) > 6:
         log("WARNING: Grid name is longer than 6 characters and will be truncated for labels in the Navigator.")
"""

# Save WG montage as rescaled image
""" MOVED TO modules.buf
def saveWGMap(grid_name, map_dir, mic_params, model):
    log("Loading whole grid map...")

    map_id = int(sem.NavIndexWithNote(grid_name + ".mrc"))
    sem.LoadOtherMap(map_id)

    # Load buffer
    log("Processing whole grid map...")
    buffer_orig, *_ = sem.ReportCurrentBuffer()
    img_prop = sem.ImageProperties(buffer_orig)
    mic_params.WG_pix_size = float(img_prop[4])

    # Rescale to model pixel size
    rescale_factor = model.pix_size / mic_params.WG_pix_size
    sem.ReduceImage(buffer_orig, rescale_factor)                # faster than tranform.rescale
    buffer = "A"
    montage = np.asarray(sem.bufferImage(buffer))

    # Save WG montage to file
    log("Saving whole grid map...")
    montage_img = Image.fromarray(montage.astype(np.uint8))
    montage_img.save(os.path.join(map_dir, grid_name + '_wg.png'))
    log(f"Saved at: {os.path.join(map_dir, grid_name + '_wg.png')}")
"""

# Load lamella detection results and draw nav points
""" SPLIT AND MOVED TO main script, buf.py, nav.py, mod_wg.py
def drawNavPoints(grid_name, map_dir, mic_params, model, exclude_classes=[], distance_threshold=5):
    # Read bboxes
    with open(os.path.join(map_dir, grid_name + "_boxes.json"), "r") as f:
        bboxes = json.load(f, object_hook=utils.revertArray)

    # Load map to draw points
    map_id = int(sem.NavIndexWithNote(grid_name + ".mrc"))
    sem.LoadOtherMap(map_id)
    buffer, *_ = sem.ReportCurrentBuffer()
    rescale_factor = model.pix_size / mic_params.WG_pix_size

    # Load all PL and FL nav points
    PL_nav_items = []
    FL_nav_items = []
    for b, bbox in enumerate(bboxes):
        # Check nav for preliminary lamella items
        nav_id = int(sem.NavIndexWithLabel("PL" + str(b + 1)))
        if nav_id > 0:
             item = sem.ReportOtherItem(nav_id)
             PL_nav_items.append({"id": nav_id, "coords": np.array([item[1], item[2]]), "bbox": bbox})

        # Check nav for final lamella items
        nav_id = int(sem.NavIndexWithLabel("FL" + str(b + 1)))
        if nav_id > 0:
             item = sem.ReportOtherItem(nav_id)
             FL_nav_items.append({"id": nav_id, "coords": np.array([item[1], item[2]]), "bbox": bbox})

    # If lamellae already detected return
    if len(PL_nav_items) > 0 or len(FL_nav_items) > 0:
        log("WARNING: Already found navigator items for detected lamellae. Skipping intermediate mag images.")

    # Generate nav item for each lamella
    nav_ids = []
    final_bboxes = []
    for b, bbox in enumerate(bboxes):
        #if bbox[4] == model.categories.index("broken") and not include_broken:
        if model.categories[bbox[4]] in exclude_classes:
            continue
        size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        center = [bbox[0] + size[0] / 2, bbox[1] + size[1] / 2]

        nav_ids.append(int(sem.AddImagePosAsNavPoint(buffer, int(center[0] * rescale_factor), int(center[1] * rescale_factor), 0)))
        _, x_coord, y_coord, *_ = sem.ReportOtherItem(nav_ids[-1])

        # Check if coords are within grid limits (default is +/- 990, but can be changed in SerialEM property "StageLimits", this property can not be read by script)
        if abs(x_coord) > 990 or abs(y_coord) > 990:
            nav_ids.pop(-1)
            continue

        # Check if point already existed in final lamella items or preliminary lamella items and keep bbox (should not differ too much)
        if len(FL_nav_items) > 0:
             check_nav_items = FL_nav_items
        elif len(PL_nav_items) > 0:
             check_nav_items = PL_nav_items
        else:
             check_nav_items = []

        for item in check_nav_items:
            if np.linalg.norm(np.array([x_coord, y_coord]) - item["coords"]) < distance_threshold:
                sem.DeleteNavigatorItem(nav_ids[-1])
                nav_ids.pop(-1)
                nav_ids.append(item["id"])
                break          

        final_bboxes.append(bbox)
        sem.ChangeItemColor(nav_ids[-1], model.cat_nav_colors[int(bbox[4])])
        sem.ChangeItemNote(nav_ids[-1], model.categories[int(bbox[4])] + " (" + str(int(round(bbox[5] * 100, 0))) + "%)")
        sem.ChangeItemLabel(nav_ids[-1], "PL" + str(b + 1))

    return nav_ids, final_bboxes
"""

"""MOVED TO main script
def findOffset(grid_name, map_dir, lamella_ids, initial_bboxes, mic_params, model, keep_threshold=0.7, distance_threshold=5, external=False):
    # Check if findOffset was already run and skip
    IM_map_files = glob.glob(os.path.join(map_dir, grid_name + "_IM*_wg.png"))
    if len(IM_map_files) > 0:
         log("WARNING: Intermediate mag images were already recorded. Skipping...")
         return lamella_ids, initial_bboxes
    
    # Go to WG image state to get beam settings
    sem.GoToImagingState(str(mic_params.WG_image_state))
    sem.Delay(1, "s")       # Add delay for DM/GIF to catch up (DM crashes ocasionally on Krios 3)

    # Switch to IM mag depending on if mag index or mag value was given
    if mic_params.IM_mag_index < 50:
        sem.SetMagIndex(mic_params.IM_mag_index)
    else:
        sem.SetMag(mic_params.IM_mag_index)

    new_lamella_ids = []
    new_bboxes = []
    shifted = False
    for l, lamella_id in enumerate(lamella_ids):
        log(f"Moving to lamella {l + 1}...")
        sem.MoveToNavItem(lamella_id)
        sem.R()
        # Load buffer
        img_prop = sem.ImageProperties("A")
        mic_params.IM_pix_size = float(img_prop[4])

        # Rescale to model pixel size
        log("Rescaling map for lamella detection...")
        rescale_factor = model.pix_size / mic_params.IM_pix_size
        sem.ReduceImage("A", rescale_factor)                    # faster than tranform.rescale
        log("Rescaling intensity...")
        montage = exposure.rescale_intensity(np.asarray(sem.bufferImage("A")), out_range=(0, 255)).astype(np.uint8)

        # Save WG montage to file
        log("Saving rescaled map...")
        montage_img = Image.fromarray(montage)
        montage_img.save(os.path.join(map_dir, grid_name + "_IM" + str(l + 1) + "_wg.png"))   

        # Run findLamellae
        log("Finding lamella...")
        if not external:
            bboxes = model.findLamellae(map_dir, grid_name + "_IM" + str(l + 1), device=DEVICE)
        else:
            # Wait for boxes file to be written
            while not os.path.exists(os.path.join(map_dir, grid_name + "_IM" + str(l + 1) + "_boxes.json")):
                log("Waiting for external lamella detection...")
                sem.Delay(1, "s")
            # Read bboxes
            with open(os.path.join(map_dir, grid_name + "_IM" + str(l + 1) + "_boxes.json"), "r") as f:
                bboxes = json.load(f, object_hook=utils.revertArray)

        # Deal with more than one lamella per image
        bboxes = sorted(bboxes, key=lambda x: np.linalg.norm(np.array([(x[0] + x[2]) / 2,(x[1] + x[3]) / 2]) - np.array([model.sidelen // 2, model.sidelen // 2])))     # sort according to distance from center

        for bbox in bboxes:
            log(f"Lamella bounding box: {bbox}")
            log(f"Lamella was categorized as: {model.categories[int(bbox[4])]} ({round(bbox[5] * 100, 1)} %)")

            center = [(bbox[0] + bbox[2]) / 2,(bbox[1] + bbox[3]) / 2]

            # Add nav point
            new_lamella_ids.append(int(sem.AddImagePosAsNavPoint("B", int(center[0] * rescale_factor), int(center[1] * rescale_factor), 0)))

            new_bboxes.append(bbox)
                
            # Shift by marker if not done yet
            if not shifted:
                _, x_old, y_old, *_ = sem.ReportOtherItem(lamella_id)
                _, x_new, y_new, *_ = sem.ReportOtherItem(new_lamella_ids[-1])
                shift = np.array([x_new - x_old, y_new - y_old])
            
                log(f"Shifting items by {shift}")
                sem.ShiftItemsByMicrons(x_new - x_old, y_new - y_old)
                sem.DeleteNavigatorItem(new_lamella_ids[-1])
                new_lamella_ids[-1] = lamella_id

                shifted = True
            else:
                sem.ChangeItemColor(lamella_id, 5)
            sem.ChangeItemColor(new_lamella_ids[-1], model.cat_nav_colors[int(bbox[4])])
            sem.ChangeItemNote(new_lamella_ids[-1], model.categories[int(bbox[4])] + " (" + str(int(round(bbox[5] * 100, 0))) + "%)")
            sem.ChangeItemLabel(new_lamella_ids[-1], "FL" + str(l + 1))
                
        # Deal with situation where high confidence lamella was found in WG map but not in IM
        if len(bboxes) == 0:
            if initial_bboxes[l][5] > keep_threshold:
                new_lamella_ids.append(lamella_id)
                new_bboxes.append(initial_bboxes[l])
                log(f"No lamella detected in intermediate mag. Keeping original lamella due to high confidence ({initial_bboxes[l][5]})...")
            else:
                log("No lamella detected in intermediate mag. Removing lamella from list...")

    # Remove duplicates
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
            if dist < distance_threshold:                       # if distance is smaller than 5 microns, remove lamella with smaller probability
                if new_bboxes[lid1][5] > new_bboxes[lid2][5]:
                    remove_ids.append(lid2)
                else:
                    remove_ids.append(lid1)
    if len(remove_ids) > 0:
        log(f"WARNING: Removed {len(remove_ids)} lamellae due to close proximity.")
        for rid in remove_ids:
            new_lamella_ids.pop(rid)
            new_bboxes.pop(rid)

    log(f"NOTE: {len(new_lamella_ids)} lamellae were confirmed in intermediate mag.")
    return new_lamella_ids, new_bboxes
"""

##### MM functions #####

# Collect new montage
""" MOVED TO scope.py
def collectMMMap(nav_id, map_name, mean_threshold, bbox, padding_factor, montage_overlap, mic_params, model):
    # Check if montage already exists
    note_id = int(sem.NavIndexWithNote(map_name + ".mrc"))
    if note_id > 0:
        map_id = note_id
        log("MM montage already available. Skipping acquisition...")
    else:
        # Determine montage size
        size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]]) * padding_factor
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
                    log("NOTE: Please center your target by dragging the image using the right mouse button! Press the <b> key when finished!")
                    user_confirm = False
                    while not user_confirm:
                        while not sem.KeyBreak():
                            sem.Delay(0.1, "s")
                        user_confirm = True
                        for i in range(10):
                            if sem.KeyBreak("v"):
                                user_confirm = False
                                log("Take new view image!")
                                sem.V()
                                break
                            sem.Delay(0.1, "s")
                    sem.ResetImageShift()

        sem.Eucentricity(1)
        sem.UpdateItemZ(nav_id)
        
        log(f"Collecting medium mag montage at [{map_name.split("_")[-1]}] ({mont[0]}x{mont[1]})...")

        # Set montage parameters
        sem.ParamSetToUseForMontage(2)
        sem.SetMontPanelParams(1, 1, 1, 1)      # check all Montage control panel options
        sem.OpenNewMontage(mont[0], mont[1], os.path.join(CUR_DIR, map_name + ".mrc"))
        sem.SetMontageParams(1, int(montage_overlap * min(mic_params.cam_dims)), int(montage_overlap * min(mic_params.cam_dims)), int(mic_params.cam_dims[0]), int(mic_params.cam_dims[1]), 0, 1) # stage move, overlap X, overlap Y, piece size X, piece size Y, skip correlation, binning

        sem.M()
        map_id = int(sem.NewMap(0, map_name + ".mrc"))
        sem.ChangeItemLabel(map_id, "L" + map_name.split("_L")[-1])  
        sem.CloseFile()
    return map_id
"""

# Save montage as rescaled input image
""" MOVED TO buf.py
def saveMMMap(map_id, map_dir, map_name, mic_params, model, overwrite=False):
    if os.path.exists(os.path.join(map_dir, map_name + ".png")) and not overwrite:
        log("MM map already saved. Skipping saving!")
    else:
        sem.LoadOtherMap(map_id)
        buffer, *_ = sem.ReportCurrentBuffer()
        montage = np.asarray(sem.bufferImage(buffer))
        montage = transform.rescale(montage.astype(float) / 255, mic_params.view_pix_size / model.pix_size)
        montage = (montage * 255).astype(np.uint8)
        montage_img = Image.fromarray(montage)
        montage_img.save(os.path.join(map_dir, map_name + ".png"))
"""

# Save targets for PACEtomo   
""" MOVED TO main, buf.py, tgt.pt, nav.py
def saveAsTargets(buffer, map_dir, map_name, model, mic_params):
    # Check if tgts file for lamella already exists
    tgts_files = sorted(glob.glob(os.path.join(CUR_DIR, map_name + "*_tgts.txt")))
    if len(tgts_files) > 0:
         log(f"WARNING: Targets file for {map_name} already exists. Skipping target setup!")
         return

    # Load json data for all point files
    point_files = sorted(glob.glob(os.path.join(map_dir, map_name + "_points*.json")))
    target_areas = []
    for file in point_files:
        # Load json data
        with open(file, "r") as f:
            target_areas.append(json.load(f, object_hook=utils.revertArray))

    # Load map from buffer
    map_image = np.asarray(sem.bufferImage(buffer), dtype=float)

    # Loop over all found areas
    map_name_ori = map_name
    for ta, target_area in enumerate(target_areas):
        # Add area number to map_name if necessary
        if len(target_areas) > 1:
             map_name = map_name_ori + "_" + str(ta + 1)

        # Initiate lists
        targets = []
        nav_ids = []
        nav_virt_maps = []
        nav_virt_maps_stats = []
        geo_SS = []
        settings = {}
        if len(target_area["points"]) > 0:
            for i, point in enumerate(target_area["points"]):
                # Make virtual map
                virt_map_name = map_name + "_tgt_" + str(i + 1).zfill(3) + "_view.mrc"
                image_crop = np.flip(cropImage(map_image, (int(point[0] * model.pix_size / mic_params.view_pix_size), int(point[1] * model.pix_size / mic_params.view_pix_size)), mic_params.cam_dims[[1, 0]]), axis=0).astype(np.float32)
                nav_virt_maps_stats.append([np.min(image_crop), np.max(image_crop)])
                writeMrc(virt_map_name, image_crop, mic_params.view_pix_size)
                nav_virt_maps.append(virt_map_name)

                # Add nav item
                nav_ids.append(int(sem.AddImagePosAsNavPoint(buffer, int(point[1] * model.pix_size / mic_params.view_pix_size), int(point[0] * model.pix_size / mic_params.view_pix_size))))
                sem.ChangeItemNote(nav_ids[-1], virt_map_name)
                nav_info = sem.ReportOtherItem(nav_ids[-1])
                if i == 0:
                    stage0 = np.array([nav_info[1], nav_info[2]])
                    stage = stage0
                    SS = [0, 0]
                else:
                    stage = np.array([nav_info[1], nav_info[2]])
                    SS = mic_params.rec_s2ssMatrix @ (stage - stage0)

                # Check if target is within IS limits
                if SS[0] > mic_params.IS_limit or SS[1] > mic_params.IS_limit:
                    log(f"WARNING: Point {i + 1} requires image shifts ({round(SS[0], 1)}|{round(SS[1], 1)}) beyond the default image shift limit (15)!")

                # Add target
                targets.append({"tsfile": map_name + "_ts_" + str(i + 1).zfill(3) + ".mrc", "viewfile": map_name + "_tgt_" + str(i + 1).zfill(3) + "_view.mrc", "SSX": SS[0], "SSY": SS[1], "stageX": stage[0], "stageY": stage[1], "SPACEscore": target_area["scores"][i], "skip": "False"})

            # Get specimen coords of geo points
            for point in target_area["geo_points"]:
                # Add nav item
                geo_id = int(sem.AddImagePosAsNavPoint(buffer, int(point[1] * model.pix_size / mic_params.view_pix_size), int(point[0] * model.pix_size / mic_params.view_pix_size)))
                sem.ChangeItemColor(geo_id, 5)
                geo_item = sem.ReportOtherItem(geo_id)
                geo_coords = mic_params.rec_s2ssMatrix @ (np.array([geo_item[1], geo_item[2]]) - stage0)
                geo_SS.append({"SSX": geo_coords[0], "SSY": geo_coords[1]})

            # Set up nav for acquisition
            sem.ChangeItemNote(int(nav_ids[0]), map_name + "_tgts.txt")
            sem.SetItemAcquire(int(nav_ids[0]))

            # Change tgt points to virtual view maps
            changeNavPtsToMaps(nav_ids, nav_virt_maps, nav_virt_maps_stats)

            # Check if points file contains settings (use first occurence of settings, since they should be the same)
            if not settings and "settings" in target_area.keys():
                 settings = target_area["settings"]

        # Write PACE target file
        tgtsFilePath = os.path.join(CUR_DIR, map_name + "_tgts.txt")
        writeTargets(tgtsFilePath, targets, geo_SS, settings=settings)

        log(f"Target selection completed! {len(targets)} targets were selected.")

    # Make inspected file to prohibit user changing targets from GUI
    open(os.path.join(map_dir, map_name + "_inspected.txt"), "w").close()
"""

##### UTILITY FUNCTIONS #####

"""MOVED TO scope.py
class MicParams:
    def __init__(self, WG_image_state, IM_mag_index, MM_image_state, IS_limit):
        self.WG_image_state = WG_image_state
        self.IM_mag_index = IM_mag_index
        self.MM_image_state = MM_image_state

        self.IS_limit = IS_limit

        self.WG_pix_size = None
        self.IM_pix_size = None

    def getViewParams(self):
        cam_props = sem.CameraProperties()
        self.cam_dims = np.array([cam_props[0], cam_props[1]])
        self.view_pix_size = cam_props[4]   # [nm]

        self.view_c2ssMatrix = np.array(sem.CameraToSpecimenMatrix(0)).reshape((2, 2))
        self.view_ta_rotation = 90 - np.degrees(np.arctan(self.view_c2ssMatrix[0, 1] / self.view_c2ssMatrix[0, 0]))
        log(f"Tilt axis rotation: {self.view_ta_rotation}")
        self.view_rotM = np.array([[np.cos(np.radians(self.view_ta_rotation)), np.sin(np.radians(self.view_ta_rotation))], [-np.sin(np.radians(self.view_ta_rotation)), np.cos(np.radians(self.view_ta_rotation))]])

    def getRecParams(self):
        self.rec_s2ssMatrix = np.array(sem.StageToSpecimenMatrix(0)).reshape((2, 2))
        self.rec_beam_diameter = sem.ReportIlluminatedArea() * 100
        cam_props = sem.CameraProperties()
        self.rec_pix_size = cam_props[4]   # [nm]   

    def export(self, map_dir):
        with open(os.path.join(map_dir, "mic_params.json"), "w+") as f:
             json.dump(vars(self), f, indent=4, default=utils.convertArray)
        log(f"NOTE: Saved microscope parameters at {os.path.join(map_dir, 'mic_params.json')}")
"""

# Save settings for post action script
"""MOVED TO utils.py
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
"""
# Get grid name
""" MOVED TO scope.py
def getGridName(grid_slot, grid_default_name=""):
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

    # Save old log before folder is changed (open empty log that will be discarded)
    sem.SaveLogOpenNew()

    return grid_name, load_grid
"""
# Load grid
""" MOVED TO scope.py
def loadGrid(grid_slot, grid_name, load_grid):
    # Open new navigator and log
    openNav(grid_name)
    sem.CloseLogOpenNew(1)      # Log was already saved in getGridName so force close here
    sem.SaveLog(0, grid_name)   # Set file name for new log

    log(f"SPACEtomo Version {versionSPACE}")
    sem.ProgramTimeStamps()

    if DEVICE == "cuda":
         log("NOTE: GPU will be used.")
    else:
         log("NOTE: No GPU found. CPU will be used.")

    # Initiate loading
    if load_grid:
        log(f"Loading grid [{grid_name}] from slot {grid_slot}...")
        sem.LoadCartridge(grid_slot)
"""
# Open new nav file if necessary
""" MOVED to nav.py
def openNav(grid_name):
    # Check if nav file is open
    nav_status = sem.ReportIfNavOpen()
    if nav_status < 2:
        if nav_status == 1:
            sem.SaveNavigator("temp.nav")
            sem.CloseNavigator()
            log("WARNING: Open navigator was saved as temp.nav and closed!")
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
                sem.CloseNavigator()
                sem.OpenNavigator(nav_file)
"""
# Write PACEtomo target file
""" MOVED TO utils.py
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
"""

# Helper functions for virtual maps:

# Make sure nav contains template images
""" NOT NEEDED ANYMORE
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
"""
# Crop and pad the virtual map
""" MOVED TO buf.py
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
		log("WARNING: Target position is close to the edge of the map and was padded.")
	return imageCrop
"""
""" MOVED TO utils.py
def writeMrc(outfilename, image, pix_size):
	with mrcfile.new(os.path.join(CUR_DIR, outfilename), overwrite=True) as mrc:
		mrc.set_data(image)
		mrc.voxel_size = (pix_size * 10, pix_size * 10, pix_size * 10)
		mrc.update_header_from_data()
		mrc.update_header_stats()
"""
""" MOVED TO nav.py
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
"""
# Read nav file and change entries to map entries using the virtual maps
""" MOVED TO nav.py
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
        new_item["MapFile"] = [os.path.join(CUR_DIR, nav_maps[n])]
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
    
    log("Updated navigator file with target maps.")
"""