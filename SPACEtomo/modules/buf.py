#!/usr/bin/env python
# ===================================================================
# ScriptName:   buf
# Purpose:      Interface for SerialEM buffer.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/14
# Revision:     v1.2
# Last Change:  2024/09/04: added new polygon via nav to avoid listToSEMarray, fixed rollBuffers
#               2024/08/23: added BufferDummy
#               2024/08/19: added getCropImage from sem.py and addNavPoint
#               2024/08/16: changed nav_id convention for consistency, added check before file saving
#               2024/08/15: added buffer rolling, added scaling for addNavBoxes 
# ===================================================================

# Check SerialEM availability
try:
    import serialem as sem
    # Check if PySEMSocket is connected
    try:
        sem.Delay(0, "s")
        SERIALEM = True
    except sem.SEMmoduleError:
        SERIALEM = False
except ModuleNotFoundError:
    SERIALEM = False

import sys
import time
import json
import weakref
import numpy as np
import concurrent.futures
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images
from skimage import transform, exposure

from SPACEtomo import config
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log

class Buffer:

    # Keep track of all instances to remove buffer letters when letters are overwritten
    instances = []

    # Get rolling buffers
    if SERIALEM:
        rolling_buffers = [chr(i + 65) for i in range(int(sem.ReportUserSetting("BuffersToRollOnAcquire")) + 1)]
    else:
        rolling_buffers = []

    # Reference to active nav
    nav = None

    def __init__(self, buffer="A", nav_id=None, img=None) -> None:
        """Buffer can be initialized in three ways:
            1. Using a navigator ID of a map.
            2. Using a buffer containing an image.
            3. Using a buffer and an image to place into it."""

        # Initialize attributes
        self.img = None             # Image as numpy array
        self.img_mod = None         # Modified image as numpy array

        # Load nav map
        if nav_id is not None:
            self.nav_id = nav_id
            sem.LoadOtherMap(self.nav_id + 1, buffer)
            self.buf, *_ = sem.ReportCurrentBuffer()
            self.buf_mod = self.buf

        # Use loaded buffer
        elif buffer and not img:
            self.nav_id = None
            self.buf = buffer
            self.buf_mod = self.buf
            sem.Show(self.buf)
        
        # Use provided image
        elif buffer and img:
            self.nav_id = None
            self.buf = buffer
            self.buf_mod = self.buf
            self.img = img
            self.imgToBuffer()
            # possibly sem.Show(self.buf) might be necessary to change to filled buffer

        else:
            raise ValueError("Need buffer letter (and optional image) or navigator item ID to initialize!")
        
        # Get image info
        img_prop = sem.ImageProperties(self.buf)
        self.shape = np.array([img_prop[0], img_prop[1]])
        self.binning = int(img_prop[2])
        self.pix_size = float(img_prop[4])

        # Get timestamp of image to make sure no other image was loaded in buffer when returning
        self.timestamp = sem.GetVariable("imageTickTime")

        # Add weak reference to instance list
        self.__class__.instances.append(weakref.ref(self))

    def show(self, reload=False):
        """Show buffer and assert that buffer is unchanged."""

        sem.Show(self.buf)

        # Get image info
        img_prop = sem.ImageProperties(self.buf)

        # Get timestamp
        timestamp = sem.GetVariable("imageTickTime")

        if timestamp != self.timestamp or reload:
            # Reload from map if buffer was overwritten
            if self.nav_id:
                sem.LoadOtherMap(self.nav_id + 1, self.buf)
            # Or place image in buffer
            elif self.img is not None:
                self.imgToBuffer()
            else:
                log(f"ERROR: Buffer is no longer available!")

    def getMetadata(self, key):
        """Gets specific mdoc entry."""

        return sem.ReportMetadataValues(self.buf, key)

    def imgFromBuffer(self, mod=False, reload=False):
        """Converts buffer to image as numpy array."""

        # Wrap in loop because it sometimes fails and add timeout call in case it freezes without error
        for i in range(10):
            try:
                if not mod:
                    if self.img is None or reload:
                        log(f"DEBUG: Getting image from buffer {self.buf}...")
                        self.img = utils.timeoutCall(lambda: np.asarray(sem.bufferImage(self.buf)))
                else:
                    if self.img_mod is None or reload:
                        log(f"DEBUG: Getting modified image from buffer {self.buf_mod}...")
                        self.img_mod = utils.timeoutCall(lambda: np.asarray(sem.bufferImage(self.buf_mod)))
                break
            except (sem.SEMmoduleError, SystemError):
                if i < 9:
                    log(f"WARNING: Could not obtain image from buffer. Trying again... [{i + 1}]")
                    time.sleep(1)
                else:
                    log(f"ERROR: Could not obtain image from buffer! There might be a problem with SerialEM.")
                    sys.exit()

    def imgToBuffer(self, mod=False, buffer=None):
        """Shows img in buffer."""

        if not buffer:
            buffer = self.buf

        if mod:
            sem.PutImageInBuffer(self.img_mod, buffer, *self.img_mod.shape, self.buf)
        else:
            sem.PutImageInBuffer(self.img, buffer, *self.img.shape, self.buf)

        # Roll buffers only if put in A (see script commands page for PutImageInBuffer)
        if buffer == "A":
            self.rollBuffers(skip=["A"])
        
    def saveImg_old(self, file_path, target_pix_size=None, save_meta_data=True, overwrite=False):
        """Saves buffer image as png image (optionally rescales it)."""

        if file_path.exists():
            if not overwrite:
                log(f"WARNING: {file_path.name} already exists! Skipping saving.")
                return
            else:
                log(f"WARNING: {file_path.name} already exists! It will be overwritten.")

        if target_pix_size:
            # Rescale image
            rescale_factor = target_pix_size / self.pix_size
            log(f"DEBUG: Rescaling image by factor: {rescale_factor}")
            if rescale_factor > 1:
                # Use reduce in case of downscaling (faster)
                self.reduceBuffer(rescale_factor)
                self.imgFromBuffer(mod=True, reload=True)
            else:
                # Use rescale in case of upscaling
                self.imgFromBuffer()
                self.rescaleImage(rescale_factor)
            # Use modified image
            img = self.img_mod
        else:
            # Use unmodified image
            self.imgFromBuffer()
            img = self.img

        # Check intensity range and rescale if not between 0 and 255
        log(f"DEBUG: Image min/max: {np.min(img)}/{np.max(img)}")
        if np.min(img) < 0 or not 200 < np.max(img) <= 255:
            log(f"DEBUG: Rescaling image intensity to out_range=(0, 255)") 
            img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

        log(f"DEBUG: Saving image to file...")
        Image.fromarray(img).save(file_path)

        if save_meta_data:
            log(f"DEBUG: Saving meta data to {file_path.parent / (file_path.stem + '.json')}")
            self.saveMetaData(file_path=file_path.parent / (file_path.stem + ".json"), target_pix_size=target_pix_size)

    def saveImg(self, file_path, target_pix_size=None, save_meta_data=True, overwrite=False):
        """Saves buffer image as png image (optionally rescales it)."""

        if file_path.exists():
            if not overwrite:
                log(f"WARNING: {file_path.name} already exists! Skipping saving.")
                return
            else:
                log(f"WARNING: {file_path.name} already exists! It will be overwritten.")

        if target_pix_size:
            # Rescale image
            rescale_factor = target_pix_size / self.pix_size
            log(f"DEBUG: Rescaling image by factor: {rescale_factor}")
            if rescale_factor > 1:
                # Use reduce in case of downscaling (faster)
                self.reduceBuffer(rescale_factor)
                self.imgFromBuffer(mod=True, reload=True)
            else:
                # Use rescale in case of upscaling
                self.imgFromBuffer()
                self.rescaleImage(rescale_factor)
            # Use modified image
            img = self.img_mod
        else:
            # Use unmodified image
            self.imgFromBuffer()
            img = self.img

        # Internal function that performs the actual save operation
        def save_operation(img):
            # Check intensity range and rescale if not between 0 and 255
            log(f"DEBUG: Image min/max: {np.min(img)}/{np.max(img)}")
            if np.min(img) < 0 or not 200 < np.max(img) <= 255:
                log(f"DEBUG: Rescaling image intensity to out_range=(0, 255)") 
                img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

            log(f"DEBUG: Saving image to file...")
            Image.fromarray(img).save(file_path)
            log(f"DEBUG: Finished saving {file_path}")

        # Start the background thread to run save_operation
        executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(save_operation, img)

        if save_meta_data:
            log(f"DEBUG: Saving meta data to {file_path.parent / (file_path.stem + '.json')}")
            self.saveMetaData(file_path=file_path.parent / (file_path.stem + ".json"), target_pix_size=target_pix_size)

        return future

    def saveMetaData(self, file_path, target_pix_size=None):
        """Saves meta data for buffer image as json file."""

        meta_data = {"datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "img_size": self.shape}

        if target_pix_size:
            meta_data["pix_size"] = target_pix_size
            meta_data["img_size"] = np.array(self.img_mod.shape)
            scaling_factor = target_pix_size / self.pix_size
        else:
            meta_data["pix_size"] = self.pix_size
            scaling_factor = 1

        # Get additional data from nav
        if self.nav and self.nav_id:
            meta_data["stage_coords"] = np.array(self.nav.items[self.nav_id].entries["StageXYZ"])
            if "MapScaleMat" in self.nav.items[self.nav_id].entries.keys():
                matrix = np.array([float(val) for val in self.nav.items[self.nav_id].entries["MapScaleMat"]]).reshape((2, 2))
                meta_data["s2img_matrix"] = matrix / scaling_factor
            if "MapFile" in self.nav.items[self.nav_id].entries.keys():
                meta_data["original_map"] = Path(" ".join(self.nav.items[self.nav_id].entries["MapFile"])).as_posix()
            if "MapTiltAngle" in self.nav.items[self.nav_id].entries.keys():
                meta_data["tilt_angle"] = float(self.nav.items[self.nav_id].entries["MapTiltAngle"][0])
            # Can be extended if necessary

        with open(file_path, "w") as f:
            json.dump(meta_data, f, indent=4, default=utils.convertArray)

    def rescaleImage(self, factor):
        """Rescales image using skimage (slower, but can also upscale)."""

        if self.img is not None:
            self.img_mod = (transform.rescale(self.img, 1 / factor) * 255).astype(np.uint8)
        else:
            raise ValueError("Image has not been converted from buffer yet!")

    def reduceBuffer(self, factor):
        """Downscales image using SerialEM's implementation of antialiased reduction with a Lanczos 2 filter."""

        sem.ReduceImage(self.buf, factor)
        self.rollBuffers()      # Roll before adding buf_mod to list
        self.buf_mod = "A"

    def addNavPoint(self, point):
        """Adds buffer coords as nav point."""

        # Show buffer
        self.show()

        # Add point
        nav_id = int(sem.AddImagePosAsNavPoint(self.buf, point[1], point[0]))

        return nav_id

    def addNavBoxes(self, boxes, labels=[], padding_factor=1):
        """Adds polygons from boxes in image coords."""

        # Show buffer
        self.show()

        # Find scale factor
        if boxes.pix_size:
            factor = boxes.pix_size / self.pix_size
        else:
            factor = 1

        # Add boxes as polygon
        for b, box in enumerate(boxes.boxes):
            # Scale box coords
            box = box * factor

            # Check for stage limits (currently hardcoded as +/-990, TODO, once SerialEM property becomes readable by script command)
            stage_coords = self.px2stage(box.center) 
            if not (-990 < stage_coords[0] < 990 and -990 < stage_coords[1] < 990):
                continue

            # Pad box
            box.scale(padding_factor)

            # Choose label
            label = ""
            if labels and b < len(labels):
                label = labels[b]
            elif box.label:
                label = box.label

            # Add polygon
            if self.nav is not None:
                # Get z coordinate
                if self.nav_id:
                    stage_z = self.nav.items[self.nav_id].stage[2]
                else:
                    stage_z = 0.0
                # Get xy coordinates
                box_min = self.px2stage(box.xyxycc[:2])
                box_max = self.px2stage(box.xyxycc[2:4])
                pts_x = [box_min[0], box_min[0], box_max[0], box_max[0], box_min[0]]
                pts_y = [box_min[1], box_max[1], box_max[1], box_min[1], box_min[1]]
                # Add nav item
                nav_id = self.nav.newPolygon(pts_x, pts_y, stage_z, label=label, note=f"{label}: {config.WG_model_categories[box.cat]} ({round(box.prob * 100)} %)", color=config.WG_model_nav_colors[box.cat], update=False)
                # Add confidence as UserValue1 for nav item
                self.nav.items[nav_id].addEntry("UserValue1", box.prob)
            else:
                # Add point via SerialEM
                try:
                    nav_id = int(sem.AddImagePosAsNavPoint(self.buf, box.center[0], box.center[1])) - 1
                    # Adjust nav item
                    sem.ChangeItemColor(nav_id + 1, config.WG_model_nav_colors[box.cat])
                    if label:
                        sem.ChangeItemNote(nav_id + 1, f"{label}: {config.WG_model_categories[box.cat]} ({round(box.prob * 100)} %)")
                        sem.ChangeItemLabel(nav_id + 1, label)
                    else:
                        sem.ChangeItemNote(nav_id + 1, f"{config.WG_model_categories[box.cat]} ({round(box.prob * 100)} %)")
                except:
                    log(f"ERROR: Could not create navigator point at {stage_coords}.")

        # Deferred update after adding all boxes
        if self.nav:
            self.nav.push()


    def px2stage(self, point):
        """Converts image coords to stage coords using buffer specific image and montage shifts."""

        return sem.BufImagePosToStagePos(self.buf, 1, point[0], point[1])[:2]
       
    def stage2px(self, point):
        """Converts stage coords to image coords using buffer specific image and montage shifts."""

        return sem.StagePosToBufImagePos(self.buf, 1, point[0], point[1])
    
    def rollBuffers(self, skip=[]):
        """Rolls buffers of all instances of Buffer."""

        # Check all instances
        for buf_ref in self.instances:

            # Get object from weakref object
            buf = buf_ref()

            # Ignore dead refs
            if not buf: continue

            # Move buffer by 1 while in rolling buffers and remove if it moves beyond
            if buf.buf in self.rolling_buffers and buf.buf not in skip:
                buf.buf = chr(ord(buf.buf) + 1)
                if buf.buf not in self.rolling_buffers:
                    buf.buf = None

            # Do the same for the modified buffers
            if buf.buf_mod in self.rolling_buffers and buf.buf_mod not in skip:
                buf.buf_mod = chr(ord(buf.buf_mod) + 1)
                if buf.buf_mod not in self.rolling_buffers:
                    buf.buf_mod = None

    def getCropImage(self, coords, fov):
        """Crops out virtual map at given coordinates and field of view."""

        if self.img is None:
            self.imgFromBuffer()

        log(f"DEBUG:\nMap dimensions: {self.img.shape}\nCrop coords:\nx: {int(coords[0] - fov[0] / 2)}, {int(coords[0] + fov[0] / 2)}\ny: {int(coords[1] - fov[1] / 2)}, {int(coords[1] + fov[1] / 2)}")

        imageCrop = self.img[max(0, int(coords[0] - fov[0] / 2)):min(self.img.shape[0], int(coords[0] + fov[0] / 2)), max(0, int(coords[1] - fov[1] / 2)):min(self.img.shape[1], int(coords[1] + fov[1] / 2))]
        if not np.array_equal(imageCrop.shape, fov):
            mean = np.mean(imageCrop)
            if imageCrop.shape[0] < fov[0]:
                padding = np.full((int(fov[0] - imageCrop.shape[0]), imageCrop.shape[1]), mean)
                if int(coords[0] - fov[0] / 2) < 0:
                    imageCrop = np.concatenate((padding, imageCrop), axis=0)
                if int(coords[0] + fov[0] / 2) > self.img.shape[0]:
                    imageCrop = np.concatenate((imageCrop, padding), axis=0)
            if imageCrop.shape[1] < fov[1]:
                padding = np.full((imageCrop.shape[0], int(fov[1] - imageCrop.shape[1])), mean)
                if int(coords[1] - fov[1] / 2) < 0:
                    imageCrop = np.concatenate((padding, imageCrop), axis=1)
                if int(coords[1] + fov[1] / 2) > self.img.shape[1]:
                    imageCrop = np.concatenate((imageCrop, padding), axis=1)		
            log("WARNING: Target position is close to the edge of the map and was padded.")
        return imageCrop
    
    def __del__(self):
        """Cleans list of instances (but cannot delete current instance)."""

        remaining_refs = []
        for ref in self.__class__.instances:
            if ref():
                remaining_refs.append(ref)
        self.__class__.instances = remaining_refs
    