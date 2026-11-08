#!/usr/bin/env python
# ===================================================================
# ScriptName:   buf
# Purpose:      Dummy interface for SerialEM buffer.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/23
# Revision:     v1.4
# Last Change:  2026/11/07: stage geometry for px2stage/stage2px/addNavBoxes, cross-correlation alignTo, fixed file loading
#               2026/11/07: fixed shape/magnification/polygon/nav_id bugs, added reset for tests
#               2025/03/04: updated to make dummy run work again
#               2024/12/23: added alignTo function
#               2024/08/26: split from modules/buf.py
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
from skimage.feature import match_template
import mrcfile

from SPACEtomo import config
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.dummy.specimen import stageToPixelMatrix

class BufferDummy:

    # Keep track of all instances to remove buffer letters when letters are overwritten
    instances = []

    # Get rolling buffers
    rolling_buffers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

    # Dummy info refs
    nav = None
    microscope = None
    imaging_params = None

    # Keep file name (or array) of last image created by MicroscopeDummy and where it was taken
    last_dummy_image = None
    last_dummy_info = None

    @classmethod
    def reset(cls):
        """Clears class-level state (references, instances, last dummy image), e.g. between tests."""
        cls.instances = []
        cls.nav = None
        cls.microscope = None
        cls.imaging_params = None
        cls.last_dummy_image = None
        cls.last_dummy_info = None

    def __init__(self, buffer="A", nav_id=None, img=None, file_path=None, section=None) -> None:
        """Buffer can be initialized in three ways:
            1. Using a navigator ID of a map.
            2. Using a buffer containing an image.
            3. Using a buffer and an image to place into it."""

        # Initialize attributes
        self.img = None             # Image as numpy array
        self.img_mod = None         # Modified image as numpy array
        self.section = section      # Section of mrc file if any
        self.timestamp = None       # Timestamp of image
        self.grid_vectors = None    # Grid vectors [optional]

        self.pix_size = None
        self.mdoc = {}

        # Geometry: nominal stage position of the image center and stage (µm) to pixel matrix (None if unknown)
        self.stage_center = None
        self.s2px = None

        # Load nav map
        if nav_id is not None and self.nav:
            self.nav_id = nav_id
            self.buf = "A"
            self.buf_mod = self.buf
            self.imgFromBuffer()

            # Geometry from nav item (maps created without template have an identity MapScaleMat)
            item = self.nav.items[self.nav_id]
            self.stage_center = np.array(item.stage[:2], dtype=float)
            self.s2px = np.array(item.entries.get("MapScaleMat", ["1", "0", "0", "1"]), dtype=float).reshape((2, 2))
            if np.allclose(self.s2px, np.identity(2)):
                self.s2px = stageToPixelMatrix(self.pix_size)

        # Load from file
        elif buffer and file_path:
            self.nav_id = None
            self.buf = buffer
            self.buf_mod = self.buf
            self.loadFile(file_path, section)

        # Use loaded buffer
        elif buffer and img is None:
            self.nav_id = None
            self.buf = buffer
            self.buf_mod = self.buf

            if self.buf == "A" and self.last_dummy_image is not None:
                if isinstance(self.last_dummy_image, Path) and self.last_dummy_image.exists():
                    with mrcfile.open(self.last_dummy_image) as mrc:
                        self.img = np.flip(mrc.data, axis=0)
                        self.pix_size = float(mrc.voxel_size.x) / 10 # nm/px

                    log(f"#DUMMY: Loaded image from {self.last_dummy_image}")
                elif isinstance(self.last_dummy_image, np.ndarray):
                    self.img = self.last_dummy_image
                    self.pix_size = self.last_dummy_info["pix_size"] if self.last_dummy_info else self.microscope.camera_pix_size
                    log(f"#DUMMY: Loaded image from last dummy image.")
                self._geometryFromLastImage()

            log(f"#DUMMY: Showing buffer {self.buf}!")
        
        # Use provided image
        elif buffer and img is not None:
            self.nav_id = None
            self.buf = buffer
            self.buf_mod = self.buf
            self.img = img
            if self.last_dummy_info:
                self.pix_size = self.last_dummy_info["pix_size"]
                self._geometryFromLastImage()
            self.imgToBuffer()
            log(f"#DUMMY: Showing image in buffer {self.buf}!")

        else:
            raise ValueError("Need buffer letter (and optional image) or navigator item ID to initialize!")
        
        # Get image info (shape as [x, y] like ImageProperties)
        if self.img is not None:
            self.shape = np.flip(np.array(self.img.shape[:2]))
        else:
            self.shape = np.array(self.imaging_params.cam_dims)
        self.binning = 1
        if not self.pix_size:
            self.pix_size = 1
        self.dose = 1
        self.magnification = int(12323 / self.pix_size)
        self.mean_counts = np.random.randint(50, 200)

        # Get timestamp of image to make sure no other image was loaded in buffer when returning
        self.timestamp = time.time()

        # Add weak reference to instance list
        self.__class__.instances.append(weakref.ref(self))

    def _geometryFromLastImage(self):
        """Uses where MicroscopeDummy took the last image as geometry of this buffer."""
        if self.last_dummy_info:
            self.stage_center = np.array(self.last_dummy_info["stage"], dtype=float)
            self.s2px = stageToPixelMatrix(self.pix_size)

    def show(self, reload=False):
        """Show buffer and assert that buffer is unchanged."""

        log(f"#DUMMY: Showing buffer {self.buf}!")

    def getMetadata(self, key):
        """Gets specific mdoc entry."""

        # List for speicific keys
        if key in ["FilterSlitAndLoss"]:
            result = [0, 0, 0]
        else:
            result = 1

        log(f"#DUMMY: Getting metadata entry from buffer for {key}. Returning {result}.")

        return result

    def imgFromBuffer(self, mod=False, reload=False):
        """Converts buffer to image as numpy array."""

        if not mod:
            if self.img is None or reload:
                if self.nav_id is not None and "MapFile" in self.nav.items[self.nav_id].entries.keys():
                    map_file = self.nav.items[self.nav_id].entries["MapFile"][0]
                    with mrcfile.open(map_file) as mrc:
                        self.img = np.flip(mrc.data, axis=0)
                        self.pix_size = float(mrc.voxel_size.x) / 10 # nm/px
                    log(f"#DUMMY: Loaded buffer image from {map_file}!")

                    log(f"DEBUG: Min val: {np.min(self.img)}, Max val: {np.max(self.img)}")
                else:
                    # Figure out what kind of dummy image to generate
                    self.img = None
                    if self.microscope.dummy_state["imaging_state"] == self.imaging_params.WG_image_state:
                        # Generate LM image
                        log(f"#DUMMY: Generating WG image...")
                        self.img = self.microscope.dummyImageWG()
                        self.pix_size = self.imaging_params.WG_pix_size
                    elif self.microscope.dummy_state['imaging_state'] == self.imaging_params.IM_image_state:
                        # Generate IM image
                        log(f"#DUMMY: Generating IM image...")
                        self.img = self.microscope.dummyImageIM()
                        self.pix_size = self.imaging_params.IM_pix_size
                    elif self.microscope.dummy_state["imaging_state"] == self.imaging_params.MM_image_state or self.microscope.dummy_state["imaging_state"] in self.imaging_params.MM_image_state:
                        if self.microscope.low_dose_area == "V":
                            # Generate View image
                            log(f"#DUMMY: Generating View image...")
                            self.pix_size = self.imaging_params.view_pix_size
                        elif self.microscope.low_dose_area == "R":
                            # Generate Record image
                            log(f"#DUMMY: Generating Record image...")
                            self.pix_size = self.imaging_params.rec_pix_size
                        else:
                            self.pix_size = 0.1
                    else:
                        self.pix_size = 0.1

                    if self.img is None:
                        # Generate random noise image
                        self.img = np.random.rand(*np.flip(self.imaging_params.cam_dims)).astype(np.float32)
                        log(f"#DUMMY: Generated random buffer image!")
        else:
            if self.img_mod is None or reload:
                self.img_mod = np.random.rand(*np.flip(self.imaging_params.cam_dims))
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

    def loadFile(self, file_path, section):
        """Opens file and reads section."""

        log(f"DEBUG: Reading {file_path}[{section}] into buffer {self.buf}...")

        file_path = Path(file_path)
        if file_path.exists():
            with mrcfile.open(file_path) as mrc:
                self.pix_size = float(mrc.voxel_size.x) / 10  # nm/px
                self.shape = np.array([mrc.header.nx, mrc.header.ny], dtype=int)
                self.binning = 1
                self.dose = 0
                self.magnification = None
                if mrc.data.ndim == 3 and section < mrc.data.shape[0]:
                    self.img = np.flip(mrc.data[section], axis=0)
                elif mrc.data.ndim == 2:
                    self.img = np.flip(mrc.data, axis=0)

            # Get mdoc metadata if available
            from SPACEtomo.modules import utils
            mdoc_path = file_path.with_suffix(".mrc.mdoc")
            if mdoc_path.exists():
                self.mdoc = utils.parseMdoc(mdoc_path)[1][section]
                self.binning = self.mdoc.get("Binning", 1)
                self.dose = self.mdoc.get("ExposureDose", 0)
                self.magnification = self.mdoc.get("Magnification", None)
                self.timestamp = self.mdoc.get("TimeStamp", None)

        log(f"#DUMMY: Loaded image into buffer {self.buf}!")
        
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
            if not self.rescaleImage(rescale_factor):
                target_pix_size = self.pix_size
            img = self.img_mod  # Use modified image
        else:
            img = self.img  # Use unmodified image

        import concurrent.futures
        # Internal function that performs the actual save operation
        def save_operation(img):

            # Check intensity range and rescale if needed
            log(f"DEBUG: Image min/max: {np.min(img)}/{np.max(img)}")
            if np.min(img) < 0 or not 200 < np.max(img) <= 255:
                log(f"DEBUG: Rescaling image intensity to out_range=(0, 255)")
                img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

            Image.fromarray(img).save(file_path)

        # Start the background thread to run `save_operation`
        executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(save_operation, img)



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
        if self.nav and self.nav_id is not None:
            meta_data["stage_coords"] = np.array(self.nav.items[self.nav_id].entries["StageXYZ"])
            if "MapScaleMat" in self.nav.items[self.nav_id].entries.keys():
                matrix = np.array([float(val) for val in self.nav.items[self.nav_id].entries["MapScaleMat"]]).reshape((2, 2))
                meta_data["s2img_matrix"] = matrix / scaling_factor * float(self.nav.items[self.nav_id].entries["MontBinning"][0]) / float(self.nav.items[self.nav_id].entries["MapBinning"][0]) # TODO: check if additional "MontBinning"/"MapBinning" factor is needed
            if "MapFile" in self.nav.items[self.nav_id].entries.keys():
                meta_data["original_map"] = Path(" ".join(self.nav.items[self.nav_id].entries["MapFile"])).as_posix()
            if "MapTiltAngle" in self.nav.items[self.nav_id].entries.keys():
                meta_data["tilt_angle"] = float(self.nav.items[self.nav_id].entries["MapTiltAngle"][0])
            # Can be extended if necessary

        with open(file_path, "w") as f:
            json.dump(meta_data, f, indent=4, default=utils.convertToTaggedString)

    def rescaleImage(self, factor):
        """Rescales image using skimage (slower, but can also upscale)."""

        if factor > 100 or factor < 0.01:
            log(f"ERROR: Buffer image cannot be resized. Rescale factor {factor} is out of range. Please a pixel size closer to the target pixel size!")
            self.img_mod = self.img
            return False
        
        if self.img is not None:
            self.img_mod = (transform.rescale(self.img, 1 / factor) * 255).astype(np.uint8)
        else:
            raise ValueError("Image has not been converted from buffer yet!")
        return True
    
    def reduceBuffer(self, factor):
        """Downscales image using SerialEM's implementation of antialiased reduction with a Lanczos 2 filter."""

        self.rescaleImage(factor)
        self.buf_mod = "A"

        self.rollBuffers(skip=[self.buf_mod])

    def addNavPoint(self, point):
        """Adds buffer coords as nav point."""

        # Show buffer
        self.show()

        # Add point (point given as (row, col) like the real Buffer passes to AddImagePosAsNavPoint)
        nav_id = self.nav.newPoint([*self.px2stage(np.flip(point)), 0], str(len(self.nav)), "Dummy point")

        return nav_id

    def addNavBoxes(self, boxes, labels=[], padding_factor=1, label_prefix=None):
        """Adds polygons from boxes in image coords."""

        # Show buffer
        self.show()

        # Find scale factor
        if boxes.pix_size:
            factor = boxes.pix_size / self.pix_size
        else:
            factor = 1

        # Get z coordinate
        if self.nav_id is not None:
            stage_z = self.nav.items[self.nav_id].stage[2]
        else:
            stage_z = 0.0

        # Add boxes as polygon
        for b, box in enumerate(boxes.boxes):
            # Scale box coords
            box = box * factor

            # Choose nav label (custom label is kept as display label)
            if labels and b < len(labels):
                label = labels[b]
            elif label_prefix is not None:
                label = label_prefix + str(b + 1).zfill(2)
            else:
                label = box.label
            display_label = box.label if box.label else label

            # Pad box and convert corners (x, y) to stage coords
            box.scale(padding_factor)
            corners = [(box.x_min, box.y_min), (box.x_min, box.y_max), (box.x_max, box.y_max), (box.x_max, box.y_min), (box.x_min, box.y_min)]
            pts = np.array([self.px2stage(np.array(corner)) for corner in corners])

            note = f"{display_label}: {config.WG_model_categories[box.cat]} ({round(box.prob * 100)} %)" if display_label else f"{config.WG_model_categories[box.cat]} ({round(box.prob * 100)} %)"
            nav_id = self.nav.newPolygon(pts[:, 0], pts[:, 1], stage_z, label=label, note=note, color=config.WG_model_nav_colors[box.cat], update=False)

            # Add confidence as UserValue1 and custom label as UserValue2 for nav item
            self.nav.items[nav_id].addEntry("UserValue1", box.prob)
            self.nav.items[nav_id].addEntry("UserValue2", box.label)

            log(f"DEBUG: Added new polygon [{label}]")

        self.nav.push()

    def px2stage(self, point):
        """Converts image coords (x, y) to nominal stage coords."""

        if self.stage_center is None:
            log(f"ERROR: Buffer {self.buf} has no stage geometry. Returning current stage position.")
            return np.array(self.microscope.stage[:2], dtype=float)

        return self.stage_center + np.linalg.solve(self.s2px, np.array(point, dtype=float) - self.shape / 2)
       
    def stage2px(self, point):
        """Converts nominal stage coords to image coords (x, y)."""

        if self.stage_center is None:
            log(f"ERROR: Buffer {self.buf} has no stage geometry. Returning image center.")
            return self.shape / 2

        return self.s2px @ (np.array(point[:2], dtype=float) - self.stage_center) + self.shape / 2
    
    def rollBuffers(self, skip=[]):
        """Rolls buffers of all instances of Buffer."""

        # Check all instances
        for buf_ref in self.instances:

            # Get object from weakref object
            buf = buf_ref()

            if buf is None:
                continue

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

    getCropImage = Buffer.getCropImage     # same cropping as the real Buffer
    
    def alignTo(self, buffer, avoid_image_shift=False, max_shift_um=30):
        """Aligns current buffer to given buffer by cross-correlation at the coarser pixel size (like AlignTo/AlignBetweenMags).

        Returns the shift in pixels of this buffer and the specimen shift (µm) of this image's content relative to the reference,
        i.e. the stage has to be moved by minus this shift to center the reference content.
        """

        if isinstance(buffer, BufferDummy):
            ref_img, ref_pix_size = buffer.img, buffer.pix_size
        else:
            log(f"WARNING: Can only align dummy buffers. Returning no shift.")
            return np.zeros(2), np.zeros(2)
        if self.img is None or ref_img is None:
            log(f"ERROR: Missing image for alignment. Returning no shift.")
            return np.zeros(2), np.zeros(2)

        # Bring both images to the coarser pixel size
        pix_size = max(self.pix_size, ref_pix_size)
        img = transform.rescale(self.img.astype(np.float32), self.pix_size / pix_size, anti_aliasing=True) if self.pix_size < pix_size else self.img.astype(np.float32)
        ref = transform.rescale(ref_img.astype(np.float32), ref_pix_size / pix_size, anti_aliasing=True) if ref_pix_size < pix_size else ref_img.astype(np.float32)

        # Search smaller image in the central region of the larger one
        sign = 1
        if img.shape[0] > ref.shape[0] or img.shape[1] > ref.shape[1]:
            img, ref, sign = ref, img, -1
        search = int(max_shift_um * 1000 / pix_size)
        r0 = max(0, (ref.shape[0] - img.shape[0]) // 2 - search)
        c0 = max(0, (ref.shape[1] - img.shape[1]) // 2 - search)
        region = ref[r0: min(ref.shape[0], r0 + img.shape[0] + 2 * search), c0: min(ref.shape[1], c0 + img.shape[1] + 2 * search)]

        cc = match_template(region, img)
        peak_row, peak_col = np.unravel_index(np.argmax(cc), cc.shape)
        d_row = r0 + peak_row + img.shape[0] / 2 - ref.shape[0] / 2
        d_col = c0 + peak_col + img.shape[1] / 2 - ref.shape[1] / 2

        # Image rows run along -y
        ss_shift = sign * np.array([d_col, -d_row]) * pix_size / 1000
        pixel_shift = ss_shift * 1000 / self.pix_size
        log(f"#DUMMY: Aligned image (CC peak {cc.max():.2f}): shift {ss_shift} µm")

        return pixel_shift, ss_shift
    
    def clearAlignment(self):
        """Clears alignment of current buffer. (Only works if buffer is A.)"""

        if self.buf == "A":
            log("#DUMMY: Cleared alignment.")
        else:
            log("DEBUG: Cannot clear alignment of non-A buffer.")

    def fitCTF(self, search_range, pretilt=0):
        """Fits CTF to current buffer."""

        cplot = [np.random.uniform(*search_range)]
        log(f"#DUMMY: Fitted CTF: {cplot[0]} (Range: {search_range}).")

        return cplot[0]
    
    def findGrid(self, spacing_nm=3000):
        """Finds periodic grid in current buffer image and returns grid vectors [nm]. (WARNING: Always returns vectors, even if no grid is present.)"""

        self.grid_vectors = np.array([[spacing_nm, 0], [0, spacing_nm]])
        log(f"DEBUG: Found grid vectors: {self.grid_vectors} [nm]")
    
    def __del__(self):
        """Cleans list of instances (but cannot delete current instance)."""

        remaining_refs = []
        for ref in self.__class__.instances:
            if ref():
                remaining_refs.append(ref)
        self.__class__.instances = remaining_refs