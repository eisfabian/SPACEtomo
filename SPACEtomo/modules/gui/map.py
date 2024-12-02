#!/usr/bin/env python
# ===================================================================
# ScriptName:   gui_map
# Purpose:      User interface utility to handle maps and segmentations
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/09
# Revision:     v1.2
# Last Change:  2024/11/19: added loadTif
#               2024/08/09: separated from gui.py 
# ===================================================================

import os
import time
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import mrcfile
import tifffile
import subprocess
from skimage import exposure, transform
import dearpygui.dearpygui as dpg

from SPACEtomo.modules.gui.gui import showInfoBox
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log

# Check if IMOD is available (needed for mrc file import)
if "IMOD" in os.environ["PATH"] or shutil.which("newstack") is not None:
    IMOD = True
else:
    IMOD = False

class MMap:
    def __init__(self, file_path, pix_size=1, stitched=True, tile_size=(1024, 1024), quantile=0.01, status=None) -> None:
        self.file = Path(file_path)
        self.pix_size = pix_size
        self.stitched = stitched
        self.tile_size = np.array(tile_size)
        self.status = status # for GUI status line update
        self.img = np.zeros((1, 1))
        self.img_bin = self.img
        self.img_mod = None # Option to hold modified version of image
        self.binning = 1

        if self.file.suffix.lower() in [".mrc", ".map"]:
            if self.loadMrc(quantile=quantile):
                return

        if self.file.suffix.lower() in [".png", ".tif"]:
            start = time.time()
            if self.file.suffix.lower() == ".tif":
                if not self.loadTif():
                    return
            else:
                # Catch truncated files
                try:
                    #self.img = np.array(Image.open(self.file)).astype(float) / 255
                    self.img = utils.toNumpy(Image.open(self.file))
                except OSError:
                    log(f"ERROR: Image file [{self.file}] did not finish saving. Please try again later or make sure the file was not corrupted!")
                    return
            log(f"DEBUG: Loaded map in {time.time() - start} s. [{self.img.dtype}]")

        # Fill also when loadMrc fails
        self.pix_size = pix_size        # Angstrom / px
        self.tile_num = (self.img.shape[:2] / self.tile_size).astype(int)
        self.calculateTileBounds()

        # Dims in microns
        self.dims_microns = np.flip(self.img.shape[:2]) * self.pix_size / 10000

    def loadMrc(self, quantile):
        """Loads mrc file and assembles montage. (TODO: outsource to Mrc class)"""

        arg_pix_size = self.pix_size
        arg_tile_size = self.tile_size

        # If IMOD is available: extract tile coordinates from header
        if IMOD:
            pieces_file = self.file.parent / (self.file.stem + "_pieces.txt")
            subprocess.run(["extractpieces", "-input", self.file, "-output", pieces_file], stdout=subprocess.DEVNULL)
            piece_data = np.array(utils.loadColFile(pieces_file, int))
            if len(piece_data) > 0:
                # Delete file
                pieces_file.unlink()

                # Check if file contains more than 1 montage
                if piece_data.shape[1] > 2:
                    unique = np.unique(piece_data[:, 2])
                    if len(unique) > 1:
                        log(f"WARNING: File contains {len(unique)} montage maps. Use IMOD's edmont command to split the file!")
                        showInfoBox("WARNING", f"File contains {len(unique)} montage maps.\nDo you want to run IMOD's edmont to split the file by montage?", splitMontageFile, ["Yes", "No"], [self.file, ""])
                        self.img = np.zeros((1, 1))
                        return
        else:
            piece_data = []

        # Load mrc file
        with mrcfile.open(self.file, permissive=True) as mrc:
            # Get pixel data
            data = mrc.data
            self.pix_size = float(mrc.voxel_size.x) #[A]
        
            # Add third dimension in case of single image
            if data.ndim < 3:
                data = np.expand_dims(data, 0)

            # Get size of single tile
            self.tile_size = np.array(data.shape[1:3], dtype=int)

            # Pre-bin mrc data x2 in case pixel size is <4x model size and large montages do speed up scaling
            # Thresholds of 1/4 model pixel size and 10 montage pieces were chosen arbitrarily
            prebinned = False
            if self.pix_size * 3 < arg_pix_size and data.shape[0] > 10:
                data = data[:, ::2, ::2]
                self.pix_size = self.pix_size * 2
                self.tile_size = self.tile_size // 2
                prebinned = True
                log("WARNING: Map was prebinned by 2 to speed up processing.")

            # Figure out scaling based on quantiles
            vals = np.round(np.array((np.min(data), np.max(data), np.mean(data)), dtype=np.float32), 2)
            if self.status is not None:
                self.status.update("Scaling mrc file...", box=True)
            log("Loading mrc file...")
            log("# Map statistics:")
            log(f"# Min, max, mean: {vals}")
            log(f"# Quantile: {round(quantile, 5)}")
            log(f"# Cutoffs: {np.quantile(data, quantile)}, {np.round(np.quantile(data, 1 - quantile), 2)}")
            data = exposure.rescale_intensity(data, in_range=(np.quantile(data, quantile), np.quantile(data, 1 - quantile)), out_range=(0, 1))

            if self.status is not None:
                self.status.update("Assembling tiles...", box=True)
            log("Assembling tiles...")
            montage_nums = []

            # If IMOD is available: use tile coordinates
            if len(piece_data) > 0:
                log("Using extracted piece coordinates...")
                tile_coords = []
                self.tile_bounds = [[], []]
                for piece in piece_data:
                    tile_coords.append([piece[1], piece[0]])
                    # Add coords also to tile bounds
                    #if piece[0] not in self.tile_bounds[0]:
                    #    self.tile_bounds[0].append(piece[0])
                    #if piece[1] not in self.tile_bounds[1]:
                    #    self.tile_bounds[1].append(piece[1])    

                tile_coords = np.array(tile_coords)

                if prebinned:
                    tile_coords = tile_coords // 2

                # Figure out montage dimensions from coordinates
                unique_y, counts = np.unique(tile_coords[:, 0], return_counts=True)
                self.tile_num = np.array([counts[-1], data.shape[0] / counts[-1]], dtype=int)

                # Get tile_bounds by extracting unique coords
                unique_x = np.unique(tile_coords[:, 1])
                self.tile_bounds = [unique_x, unique_y]

                # Assemble image
                self.img = np.zeros((np.max(tile_coords[:, 0]) + self.tile_size[0], np.max(tile_coords[:, 1]) + self.tile_size[1])) #self.tile_num * self.tile_size)
                for t, tile in enumerate(data):
                    # Stop when end of tile_coords is reached (in case there are more montages in file)
                    if t >= len(tile_coords): 
                        break
                    self.img[tile_coords[t][0]: tile_coords[t][0] + self.tile_size[0], tile_coords[t][1]: tile_coords[t][1] + self.tile_size[1]] = tile

                #else:
                #    print("WARNING: Loaded file is no montage!")
                #    self.img = data[0]
                #    self.tile_num = np.array([1, 1], dtype=int)
                #    self.calculateTileBounds()

                self.stitched = True
                #self.tile_size = arg_tile_size
                #self.calculateTileBounds()

            # If IMOD is not available: guess montage dimensions
            else:
                self.tile_num = utils.guessMontageDims(data.shape[0])
                if self.tile_size[0] < self.tile_size[1]:
                    self.tile_num = np.flip(self.tile_num)

                if not IMOD:
                    if self.status is not None:
                        self.status.update()
                    log(f"WARNING: No IMOD installation found! Cannot access montage tile coordinates. Best guess for montage dimensions is: {self.tile_num[0]}x{self.tile_num[1]}")
                    showInfoBox("WARNING", f"No IMOD installation found! Cannot access montage tile coordinates.\nBest guess for montage dimensions is: {self.tile_num[0]}x{self.tile_num[1]}\nPlease make sure IMOD is on the PATH to get better results!")
                else:
                    log(f"WARNING: No piece coordinates found. Best guess for montage dimensions is: {self.tile_num[0]}x{self.tile_num[1]}")

                # Assemble image
                self.img = np.zeros(self.tile_num * self.tile_size)
                for t, tile in enumerate(data):
                    x = t % self.tile_num[0] * self.tile_size[0]
                    y = t // self.tile_num[0] * self.tile_size[1]

                    # Account for serpentine acquisition (at least for WG maps)
                    if (t // self.tile_num[0]) % 2 == 1:
                        x = self.img.shape[0] - x - self.tile_size[0]

                    self.img[x: x + self.tile_size[0], y: y + self.tile_size[1]] = tile

                if self.stitched:
                    log("WARNING: Map was not stitched. Using tile images as tiles!")
                self.stitched = False

                self.calculateTileBounds()
     
            # Flip y
            self.img = np.flip(self.img, axis=0)

            #self.pix_size = float(mrc.voxel_size.x) #[A]

            return True

    def loadTif(self):
        """Loads tif file. TODO: outsource to allow for multiple MMaps from single file"""

        img = tifffile.imread(self.file)

        # Check channels
        if img.ndim > 3:                # ZCXY format
            log(f"ERROR: Loading of multi-channel stacks is currently not supported. Please provide a single image per channel!")
            return False
        elif img.ndim > 2:              # CXY or XYC format
            if img.shape[0] <= 10:      # CXY
                log(f"DEBUG: TIF: CXY with {img.shape[0]} channels")
                log(f"WARNING: TIF file contains multiple channels. Channels have been split into separate files and only first channel was loaded. Please load the other channels manually!")
                for c, channel in enumerate(img):
                    channel_rescaled = exposure.rescale_intensity(channel, out_range=(0, 255))
                    if c == 0:
                        self.img = channel_rescaled
                    Image.fromarray(channel_rescaled.astype(np.uint8)).save(self.file.parent / (self.file.stem + f"_ch{c}.tif"))

            elif img.shape[2] <= 10:    # XYC
                log(f"DEBUG: TIF: XYC with {img.shape[2]} channels")
                self.img = img
        else:                           # XY format
            log(f"DEBUG: TIF: Single channel tif")
            self.img = exposure.rescale_intensity(img, out_range=(0, 255))
        return True

    def checkBinning(self, default_binning=1):
        """Checks if map has to be binned."""

        start = time.time()
        self.binning = default_binning
        dims = np.flip(np.array(self.img.shape[:2])) // self.binning
        while np.max(dims) > 16384:  # hard limit for texture sizes on apple GPU
            self.binning += 1
            dims = np.flip(np.array(self.img.shape[:2])) // self.binning

        if self.binning > 1:
            log("WARNING: Map is too large and will be binned by " + str(self.binning) + " (for display only)! Export will be unbinned.")
            self.img_bin = self.img[::self.binning, ::self.binning]
            #dims = np.flip(np.array(self.img_bin.shape))
        else:
            self.img_bin = self.img
        log(f"Map binning took {time.time() - start} s.")

    def padDims(self, dims, transparency=True):
        """Adds channel dimensions to img."""

        if self.img_bin.ndim < 3:
            self.img_bin = np.dstack([self.img_bin] * dims)
        elif self.img_bin.shape[2] < 3:
            self.img_bin = np.dstack([self.img_bin[:, :, 0]] * dims)
        elif self.img_bin.shape[2] < 4 or transparency:
            self.img_bin = np.dstack([self.img_bin[:, :, 0], self.img_bin[:, :, 1], self.img_bin[:, :, 2], np.average(self.img_bin[:, :, :3], axis=2)])

        if not transparency:
            self.img_bin[:, :, 3] = np.full(self.img_bin.shape[:2], 255 if np.max(self.img_bin) > 1 else 1)

    def getTexture(self, mod=False):
        """Creates texture of map."""

        if not mod:
            img = self.img_bin
        elif self.img_mod is not None:
            img = self.img_mod
        else:
            log(f"ERROR: Modified image not available!")
            return np.zeros(1)

        start = time.time()
        # Convert to float
        if img.dtype == np.uint8 or np.max(img) > 1:
            img = img.astype(np.float32) / 255
        
        # Create RGBA img and flatten
        if np.ndim(img) == 3 and img.shape[2] > 3:
            # Image with transparency: just flatten
            texture_img = np.ravel(img)
        elif np.ndim(img) == 3 and img.shape[2] == 3:
            # Image with RGB: keep channels and add ones for alpha
            texture_img = np.ravel(np.dstack([img[:, :, :3], np.ones(img.shape[:2])]))
        elif np.ndim(img) == 3 and img.shape[2] < 3:
            # Image with less than 3 channels but 3 dims
            texture_img = np.ravel(np.dstack([img[:, :, :1]] * 3 + [np.ones(img.shape[:2])]))
        else:
            # Image as 2D array without 3rd dim
            texture_img = np.ravel(np.dstack([img] * 3 + [np.ones(img.shape[:2])]))
        log(f"Map texture was created in {time.time() - start} s.")

        # Get dims for texture
        dims = np.flip(np.array(img.shape[:2])).tolist()

        start = time.time()
        # Load texture
        with dpg.texture_registry():
            texture = dpg.add_static_texture(width=dims[0], height=dims[1], default_value=texture_img)
        log(f"DEBUG: Map texture was loaded in {time.time() - start} s.")

        return texture
    
    def getTileTextures(self, split=(4, 4)):
        """Creates tiled textures of map instead of single texture. (Speeds up loading, but slows unloading in case of too many tiles, could also be used for plotting unbinned map.)"""

        start = time.time()
        # Convert to float
        if self.img_bin.dtype == np.uint8:
            self.img_bin = self.img_bin.astype(np.float32) / 255
        
        # Get tile size and crop img to be divisible by tile size
        tile_size = np.array(self.img_bin.shape) // np.array(split)
        self.img_bin = self.img_bin[0: tile_size[0] * split[0], 0: tile_size[1] * split[1]]

        # Get dims for texture
        dims = np.flip(np.array(tile_size)).tolist()

        # Loop over all tiles
        textures = []
        bounds = []
        for x, v in enumerate(np.vsplit(self.img_bin, split[0])):
            for y, h in enumerate(np.hsplit(v, split[1])):
                # Create RGBA image and flatten
                texture_img = np.ravel(np.dstack([h] * 3 + [np.ones(h.shape)]))

                # Get tile bounds
                bound_min = np.array([self.img_bin.shape[0] - (x + 1) * tile_size[0], y * tile_size[1]])
                bound_max = bound_min + tile_size
                bounds.append([[bound_min[1], bound_max[1]], [bound_min[0], bound_max[0]]])

                # Load texture
                with dpg.texture_registry():
                    textures.append(dpg.add_static_texture(width=dims[0], height=dims[1], default_value=texture_img))

        log(f"Map texture was created in {time.time() - start} s.")
        
        bounds = np.array(bounds) * self.pix_size / 10000 * self.binning
        return textures, bounds

    def rescale(self, rescale_pix_size, padding=(0, 0)):
        """Rescales map to specific pixel size."""

        rescale_factor = self.pix_size / rescale_pix_size
        # Don't rescale if factor is less than 1%
        if abs(rescale_factor - 1) <= 0.01 and self.img.shape[0] > padding[0] and self.img.shape[1] > padding[1]:
            log("ERROR: New pixel size is unchanged. No rescaling was performed.")
            showInfoBox("ERROR", "New pixel size is unchanged. No rescaling was performed.")
            return
        
        if self.status is not None:
            self.status.update("Rescaling map...", box=True)
        
        self.img = transform.rescale(self.img, rescale_factor)

        self.pix_size = rescale_pix_size

        # Padding (optional)
        if self.img.shape[0] < padding[0] or self.img.shape[1] < padding[1]:
            img_pad = np.zeros([max(self.img.shape[0], padding[0]), max(self.img.shape[1], padding[1])])
            img_pad[(img_pad.shape[0] - self.img.shape[0]) // 2: (img_pad.shape[0] + self.img.shape[0]) // 2, (img_pad.shape[1] - self.img.shape[1]) // 2: (img_pad.shape[1] + self.img.shape[1]) // 2] = self.img
            self.img = img_pad
            self.tile_size = padding
            self.tile_num = np.array([1, 1], dtype=int)
        else:
            if not self.stitched:
                # Crop image for even tiles
                self.img = self.img[0:self.img.shape[0] - self.img.shape[0] % self.tile_num[0], 0:self.img.shape[1] - self.img.shape[1] % self.tile_num[1]]
                self.tile_size = (np.array(self.img.shape) / self.tile_num).astype(int)
            else:
                self.tile_num = (np.array(self.img.shape) / self.tile_size).astype(int)

        self.calculateTileBounds()

    def stitch(self):
        if self.status is not None:
            self.status.update("Stitching map...", box=True)
        # Stitch map based on cross correlation of overlaps
        # TODO
        pass

    def calculateTileBounds(self):
        self.tile_bounds = [[], []]
        # x
        for i in range(self.tile_num[1] + 1):
            self.tile_bounds[0].append(i * self.tile_size[1])
        # y
        for i in range(self.tile_num[0] + 1):
            self.tile_bounds[1].append(i * self.tile_size[0])

    def changeTiling(self, tile_size):
        self.tile_size = tile_size
        self.tile_num = (self.img.shape / self.tile_size).astype(int)
        self.calculateTileBounds()

    def returnTile(self, i, j, padding=True, padding_threshold=0.8):
        bounds = [[i * self.tile_size[0], (i + 1) * self.tile_size[0]], [j * self.tile_size[1], (j + 1) * self.tile_size[1]]]

        #print(i, j, ":", (bounds[0][1] - self.img.shape[0]) / self.tile_size[0], (bounds[1][1] - self.img.shape[1]) / self.tile_size[1])
        # If lower bounds are out of image: return False
        if bounds[0][0] < 0 or bounds[0][0] > self.img.shape[0] or bounds[1][0] < 0 or bounds[1][0] > self.img.shape[1]:
            return False, np.zeros([2, 2])
        # If larger bounds are out of image: return padded tile (if tile itself is larger than padding_threshold of normal tile)
        elif (bounds[0][1] > self.img.shape[0] or bounds[1][1] > self.img.shape[1]) and padding:
            if padding_threshold > 0 and (bounds[0][1] - self.img.shape[0] > padding_threshold * self.tile_size[0] or bounds[1][1] - self.img.shape[1] > padding_threshold * self.tile_size[1]):
                return False, np.zeros([2, 2])
            tile = np.zeros(self.tile_size)
            tile[0: min([self.tile_size[0], self.tile_size[0] - bounds[0][1] + self.img.shape[0]]), 0: min([self.tile_size[1], self.tile_size[1] - bounds[1][1] + self.img.shape[1]])] = self.img[bounds[0][0]: bounds[0][1], bounds[1][0]: bounds[1][1]]
            return tile, bounds
        # If all bounds within image: return tile
        else:
            return self.img[bounds[0][0]: bounds[0][1], bounds[1][0]: bounds[1][1]], bounds
        
    def px2microns(self, coords_px):
        """Converts img coordinates to plot coordinates."""

        return np.array([coords_px[1] * self.pix_size / 10000, self.dims_microns[1] - coords_px[0] * self.pix_size / 10000])

    def microns2px(self, coords_micron):
        """Converts plot coordinates to image coordinates."""

        return np.array([self.img.shape[0] - coords_micron[1] / self.pix_size * 10000, coords_micron[0] / self.pix_size * 10000])
    

class Segmentation:
    def __init__(self, file_path, model, pix_size=1) -> None:
        self.file = Path(file_path)
        # Check if file is .png file
        if file_path.suffix.lower() in [".png"]:
            start = time.time()
            self.img = utils.toNumpy(Image.open(self.file))
            log(f"DEBUG: Segmentation was loaded in {time.time() - start} s. [{self.img.dtype}]")
            self.valid = True
        else:
            showInfoBox("ERROR", "Can only read segmentations from .png files!")
            self.valid = False
            return
        # Check if segmentation contains data (manual_selection creates empty segmentation)
        if not np.any(self.img):
            self.valid = False
            return
        # Check if image has to be binned
        self.pix_size = pix_size
        self.checkBinning()
        # Get model for categories
        self.model = model

    # Bin map if hard threshold is surpassed
    def checkBinning(self, default_binning=1):
        self.binning = default_binning
        dims = np.flip(np.array(self.img.shape)) // self.binning
        while np.max(dims) > 16384:  # hard limit for texture sizes on apple GPU
            self.binning += 1
            dims = np.flip(np.array(self.img.shape)) // self.binning

        if self.binning > 1:
            self.img_bin = self.img[::self.binning, ::self.binning]
        else:
            self.img_bin = self.img

    # Create binary mask from selected classes
    def getMask(self, cats, unbinned=False):
        if unbinned:
            img = self.img
        else:
            img = self.img_bin

        mask = np.zeros(img.shape)
        for name in cats:
            mask[img == self.model.categories[name]] = 1

        return mask
    
    def getMaskTexture(self, cats, color=(255, 0, 0, 64)):

        # Get binary mask from classes
        start = time.time()
        mask = self.getMask(cats)
        log(f"Created mask in {time.time() - start} s.")

        # Get color image for texture
        start = time.time()
        texture_img = np.ravel(np.dstack([mask] * 3 + [np.ones(mask.shape)]) * (np.array(color) / 255))
        log(f"Created texture in {time.time() - start} s.")

        # Get dims
        dims = np.flip(np.array(mask.shape)).tolist()   # texture requires list, not numpy

        # Make texture
        start = time.time()
        with dpg.texture_registry():
            texture = dpg.add_static_texture(width=dims[0], height=dims[1], default_value=texture_img)
        log(f"DEBUG: Loaded texture in {time.time() - start} s.")
        return texture

    def getMaskTileTextures(self, cats, color=(255, 0, 0, 64), split=(4, 4)):
        """Creates tiled textures of map instead of single texture. (Speeds up loading, but slows unloading in case of too many tiles, could also be used for plotting unbinned map.)"""

        # Get binary mask from classes
        start = time.time()
        mask = self.getMask(cats)
        log(f"Created mask in {time.time() - start} s.")

        start = time.time()
        # Get tile size and crop img to be divisible by tile size
        tile_size = np.array(mask.shape) // np.array(split)
        mask = mask[0: tile_size[0] * split[0], 0: tile_size[1] * split[1]]

        # Get dims and color for texture
        dims = np.flip(np.array(tile_size)).tolist()
        color = np.array(color) / 255

        # Loop over all tiles
        textures = []
        bounds = []
        for x, v in enumerate(np.vsplit(mask, split[0])):
            for y, h in enumerate(np.hsplit(v, split[1])):
                # Create RGBA image and flatten
                texture_img = np.ravel(np.dstack([h] * 3 + [np.ones(h.shape)]) * color)

                # Get tile bounds
                bound_min = np.array([mask.shape[0] - (x + 1) * tile_size[0], y * tile_size[1]])
                bound_max = bound_min + tile_size
                bounds.append([[bound_min[1], bound_max[1]], [bound_min[0], bound_max[0]]])

                # Load texture
                with dpg.texture_registry():
                    textures.append(dpg.add_static_texture(width=dims[0], height=dims[1], default_value=texture_img))

        log(f"DEBUG: Loaded texture in {time.time() - start} s.")
        
        bounds = np.array(bounds) * self.pix_size / 10000 * self.binning
        return textures, bounds



# TODO: make file for mrc processing functions?
def splitMontageFile(sender, app_data, user_data):
    if user_data is not None and dpg.does_item_exist(user_data[0]):
        dpg.delete_item(user_data[0])
        dpg.split_frame()

    file = Path(user_data[1])    
    if len(user_data) > 1 and file.exists() and IMOD:
        processing_box = showInfoBox("PROCESSING", "Splitting mrc file into montages...", options=None)

        # Extract pieces
        pieces_file = file.parent / (file.stem + "_pieces.txt")
        subprocess.run(["extractpieces", "-input", file, "-output", pieces_file], stdout=subprocess.DEVNULL)
        if pieces_file.exists():
            with open(pieces_file, "r") as f:
                pieces = f.readlines()
            tile_nums = []
            tile_coords = []
            for piece in pieces:
                col = piece.strip().split()
                # If end of first montage is reached, reset tile_coords
                if len(col) > 2:
                    if int(col[2]) > len(tile_nums):
                        tile_nums.append(len(tile_coords))
                        tile_coords = []
                tile_coords.append([int(col[1]), int(col[0])])
            # Add final montage
            tile_nums.append(len(tile_coords))

            # If more than 1 montage, run edmont command
            if len(tile_nums) > 1:
                cmd = ["edmont", "-imin", file]
                for i in range(len(tile_nums)):
                    cmd = cmd + ["-imout", file.parent / (file.stem + "_" + str(i + 1) + file.suffix)]
                subprocess.run(cmd, stdout=subprocess.DEVNULL)

                # Clean up
                #if not os.path.exists(os.path.join(os.path.dirname(file), "multi_montage_files")): 
                if not (file.parent / "multi_montage_files").exists():
                    (file.parent / "multi_montage_files").mkdir()
                    #os.makedirs(os.path.join(os.path.dirname(file), "multi_montage_files"))
                #shutil.move(file, os.path.join(os.path.dirname(file), "multi_montage_files", os.path.basename(file)))
                shutil.move(file, file.parent / "multi_montage_files" / file.name)

        dpg.delete_item(processing_box)
        dpg.split_frame()

        log(f"{file.name} was split into {len(tile_nums)} files!")
        showInfoBox("INFO", f"{file.name} was split into {len(tile_nums)} montage files.\nPlease try loading one of them!")