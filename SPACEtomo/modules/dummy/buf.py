#!/usr/bin/env python
# ===================================================================
# ScriptName:   buf
# Purpose:      Dummy interface for SerialEM buffer.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/23
# Revision:     v1.2
# Last Change:  2024/08/26: split from modules/buf.py
#               2024/08/23: added BufferDummy
# ===================================================================
 
import time
import json
import weakref
import numpy as np
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images
from skimage import transform, exposure
import mrcfile

from SPACEtomo import config
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log

class BufferDummy:

    # Keep track of all instances to remove buffer letters when letters are overwritten
    instances = []

    # Get rolling buffers
    rolling_buffers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

    # Dummy info refs
    nav = None
    imaging_params = None

    # Keep file name of last image created by MicroscopeDummy
    last_dummy_image = None

    def __init__(self, buffer="A", nav_id=None, img=None) -> None:
        """Buffer can be initialized in three ways:
            1. Using a navigator ID of a map.
            2. Using a buffer containing an image.
            3. Using a buffer and an image to place into it."""

        # Initialize attributes
        self.img = None             # Image as numpy array
        self.img_mod = None         # Modified image as numpy array

        self.pix_size = None

        # Load nav map
        if nav_id is not None and self.nav:
            self.nav_id = nav_id
            self.buf, *_ = "A", None
            self.buf_mod = self.buf
            self.imgFromBuffer()

        # Use loaded buffer
        elif buffer and not img:
            self.nav_id = None
            self.buf = buffer
            self.buf_mod = self.buf

            if self.buf == "A" and self.last_dummy_image is not None:
                with mrcfile.open(self.last_dummy_image) as mrc:
                    self.img = np.flip(mrc.data, axis=0)
                    self.pix_size = mrc.voxel_size.x / 10

                log(f"#DUMMY: Loaded image from {self.last_dummy_image}")

            log(f"#DUMMY: Showing buffer {self.buf}!")
        
        # Use provided image
        elif buffer and img:
            self.nav_id = None
            self.buf = buffer
            self.buf_mod = self.buf
            self.img = img
            self.imgToBuffer()
            log(f"#DUMMY: Showing image in buffer {self.buf}!")

        else:
            raise ValueError("Need buffer letter (and optional image) or navigator item ID to initialize!")
        
        # Get image info
        if img:
            self.shape = np.array(img.shape)
        else:
            self.shape = self.imaging_params.cam_dims
        self.binning = 1
        if not self.pix_size:
            self.pix_size = 1

        # Get timestamp of image to make sure no other image was loaded in buffer when returning
        self.timestamp = time.time()

        # Add weak reference to instance list
        self.__class__.instances.append(weakref.ref(self))

    def show(self, reload=False):
        """Show buffer and assert that buffer is unchanged."""

        log(f"#DUMMY: Showing buffer {self.buf}!")

    def getMetadata(self, key):
        """Gets specific mdoc entry."""

        log(f"#DUMMY: Getting metadata entry from buffer for {key}. Returning 1.")

        return 1

    def imgFromBuffer(self, mod=False, reload=False):
        """Converts buffer to image as numpy array."""

        if not mod:
            if self.img is None or reload:
                if self.nav_id is not None:
                    map_file = self.nav.items[self.nav_id].entries["MapFile"][0]
                    with mrcfile.open(map_file) as mrc:
                        self.img = np.flip(mrc.data, axis=0)
                        self.pix_size = mrc.voxel_size.x / 10
                    log(f"#DUMMY: Loaded buffer image from {map_file}!")

                    log(f"DEBUG: Min val: {np.min(self.img)}, Max val: {np.max(self.img)}")
                else:
                    self.img = np.random.rand(*self.imaging_params.cam_dims)
                    log(f"#DUMMY: Generated random buffer image!")
        else:
            if self.img_mod is None or reload:
                self.img_mod = np.random.rand(*self.imaging_params.cam_dims)
                log(f"#DUMMY: Generated random modified buffer image!")

    def imgToBuffer(self, mod=False, buffer=None):
        """Shows img in buffer."""

        if not buffer:
            buffer = self.buf

        log(f"#DUMMY: Loaded image into buffer {buffer}!")

        # Roll buffers only if put in A (see script commands page for PutImageInBuffer)
        if buffer == "A":
            self.rollBuffers(skip=["A"])
            log(f"#DUMMY: Rolling buffers!")
        
    def saveImg(self, file_path, target_pix_size=None, save_meta_data=True, overwrite=False):
        """Saves buffer image as png image (optionally rescales it)."""

        # Check if file exists and handle overwriting
        if file_path.exists():
            if not overwrite:
                log(f"WARNING: {file_path.name} already exists! Skipping saving.")
                return
            else:
                log(f"WARNING: {file_path.name} already exists! It will be overwritten.")

        # Process image
        self.imgFromBuffer()

        if target_pix_size:
            # Rescale image
            rescale_factor = target_pix_size / self.pix_size
            self.rescaleImage(rescale_factor)
            img = self.img_mod  # Use modified image
        else:
            img = self.img  # Use unmodified image

        import concurrent.futures
        # Internal function that performs the actual save operation
        def save_operation(img):

            # Check intensity range and rescale if needed
            log(f"DEBUG: Image min/max: {np.min(img)}/{np.max(img)}")
            if np.min(img) < 0 or not 200 < np.max(img) < 255:
                log(f"DEBUG: Rescaling image intensity to out_range=(0, 255)")
                img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

            Image.fromarray(img).save(file_path)

            if save_meta_data:
                log(f"DEBUG: Saving meta data to {file_path.parent / (file_path.stem + '.json')}")
                self.saveMetaData(file_path=file_path.parent / (file_path.stem + ".json"), target_pix_size=target_pix_size)

        # Start the background thread to run `save_operation`
        executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(save_operation, img)


        """
        if file_path.exists():
            if not overwrite:
                log(f"WARNING: {file_path.name} already exists! Skipping saving.")
                return
            else:
                log(f"WARNING: {file_path.name} already exists! It will be overwritten.")

        self.imgFromBuffer()

        if target_pix_size:
            # Rescale image
            rescale_factor = target_pix_size / self.pix_size
            # Use rescale in case of upscaling
            self.rescaleImage(rescale_factor)
            # Use modified image
            img = self.img_mod
        else:
            # Use unmodified image
            img = self.img

        # Check intensity range and rescale if not between 0 and 255
        log(f"DEBUG: Image min/max: {np.min(img)}/{np.max(img)}")
        if np.min(img) < 0 or not 200 < np.max(img) < 255:
            log(f"DEBUG: Rescaling image intensity to out_range=(0, 255)") 
            img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

        Image.fromarray(img).save(file_path)

        """

        if save_meta_data:
            log(f"DEBUG: Saving meta data to {file_path.parent / (file_path.stem + '.json')}")
            self.saveMetaData(file_path=file_path.parent / (file_path.stem + ".json"), target_pix_size=target_pix_size)

        return future

    def saveMetaData(self, file_path, target_pix_size=None):
        """Saves meta data for buffer image as json file."""

        meta_data = {"datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "img_size": self.shape}

        if target_pix_size and target_pix_size != self.pix_size:
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

        self.rescaleImage(factor)
        self.buf_mod = "A"

        self.rollBuffers(skip=[self.buf_mod])

    def addNavPoint(self, point):
        """Adds buffer coords as nav point."""

        # Show buffer
        self.show()

        # Add point
        nav_id = self.nav.newPoint(str(len(self.nav)), np.random.rand(2) * 1800 - 900, "Dummy point")

        return nav_id

    def addNavBoxes(self, boxes, labels=[], padding_factor=1):
        """Adds polygons from boxes in image coords."""

        # Show buffer
        self.show()

        # Find scale factor
        if boxes.pix_size:
            factor = self.pix_size / boxes.pix_size
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

            # Create polygon points
            pts_x = [box.x_min, box.x_min, box.x_max, box.x_max, box.x_min]
            pts_y = [box.y_min, box.y_max, box.y_max, box.y_min, box.x_min]


            # Add polygon
            nav_id = self.nav.newPoint([*stage_coords, 0], box.label, "Dummy polygon")
            
            # Try to get stage rough stage coords
            if boxes.pix_size:
                factor = boxes.pix_size / 1000
            else:
                factor = 1

            self.nav.items[nav_id].entries["Type"] = ["1"]
            self.nav.items[nav_id].entries["PtsX"] = [str(x * factor) for x in pts_x]
            self.nav.items[nav_id].entries["PtsY"] = [str(y * factor) for y in pts_y]
            self.nav.items[nav_id].entries["NumPts"] = [str(len(pts_x))]

            # Adjust nav item
            self.nav.items[nav_id].entries["Color"] = [str(config.WG_model_nav_colors[box.cat])]

            # Choose label
            label = ""
            if labels and b < len(labels):
                label = labels[b]
            elif box.label:
                label = box.label
            
            if label:
                self.nav.items[nav_id].label = label
                self.nav.items[nav_id].note = f"{label}: {config.WG_model_categories[box.cat]} ({round(box.prob * 100)} %)"
            else:
                self.nav.items[nav_id].note = f"{config.WG_model_categories[box.cat]} ({round(box.prob * 100)} %)"

        self.nav.push()

    def px2stage(self, point):
        """Converts image coords to stage coords using buffer specific image and montage shifts."""

        if self.nav_id:
            stage = self.nav.items[self.nav_id].stage[:2]
            stage += point * self.pix_size / 1000
        else:
            stage = np.random.rand(2) * 1800 - 900

        return stage
       
    def stage2px(self, point):
        """Converts stage coords to image coords using buffer specific image and montage shifts."""

        return np.random.rand(2) * self.img.shape
    
    def rollBuffers(self, skip=[]):
        """Rolls buffers of all instances of Buffer."""

        # Check all instances
        for buf in self.instances:
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