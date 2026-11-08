#!/usr/bin/env python
# ===================================================================
# ScriptName:   sim
# Purpose:      Simulated microscope behind the mock serialem module: optics, camera buffers, image files,
#               montages and navigator, all imaging the synthetic grid (specimen.py).
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein / Claude
# Created:      2026/11/08
# Revision:     v1.4
# Last Change:  2026/11/08: created
# ===================================================================
#
# Geometry (see specimen.py): stage and specimen axes coincide; camera columns run along +x and rows along -y,
# so the camera-to-specimen matrix is diag(p, -p) (p = pixel size in µm) and the measured tilt axis rotation is 90°.
# Each buffer knows the nominal stage position of its image center and its stage-to-pixel matrix; an image taken at
# stage s shows the world at s + offset of the current mag (simulated image shift offsets between mags).

import json
import time
import numpy as np
from pathlib import Path

import mrcfile
from skimage import transform
from skimage.feature import match_template

from SPACEtomo.modules.dummy.specimen import Specimen, stageToPixelMatrix, toCounts

CAMERA_DIMS = (5760, 4092)              # K3 (x, y)
DETECTOR_PIX_SIZE = 5.0                 # µm

# Mag index -> pixel size (nm)
MAGS = {5: 400.0, 15: 10.0, 17: 12.0, 20: 3.905, 31: 0.25, 33: 0.20}
OFFSET_KEY = {5: "WG", 15: "IM"}        # all other mags use the "MM" offset of the specimen

# Low Dose areas: mag index, illuminated area (µm)
LD_AREAS = {"V": (20, 38.2), "F": (33, 1.5), "T": (33, 1.5), "R": (31, 1.8), "S": (17, 90.0)}
NON_LD_BEAM = {5: 3000.0, 15: 120.0}

# Imaging states: name -> (number, type (-1 non-LD, 0-5 LD area V F T R S R-mont), camera, mag index)
IMAGING_STATES = {"WG": (1, -1, 1, 5), "IM": (2, -1, 1, 15), "MM1": (3, 0, 1, 20)}

ROLLING_BUFFERS = [chr(c) for c in range(ord("A"), ord("K") + 1)]


def _resample(img, factor):
    """Downsamples by factor (block averaging like binning, then fine rescaling)."""
    if factor <= 1:
        return img
    block = int(factor)
    if block > 1:
        img = transform.downscale_local_mean(img, (block, block))
    return transform.rescale(img, block / factor, anti_aliasing=False) if factor / block > 1.01 else img


def magnification(mag_index):
    return round(DETECTOR_PIX_SIZE * 1000 / MAGS[mag_index])


class ImageBuffer:
    """Image in a buffer (display orientation: row 0 at the top) with its geometry."""

    def __init__(self, img, pix_size, stage=None, s2px=None, mag_index=None, binning=1):
        self.img = np.asarray(img)          # keeps acquired type (int16 counts) like SerialEM buffers
        self.pix_size = float(pix_size)
        self.stage = None if stage is None else np.array(stage, dtype=float)       # nominal stage (x, y, z) of image center
        self.s2px = stageToPixelMatrix(pix_size) if s2px is None else np.array(s2px, dtype=float)
        self.mag_index = mag_index
        self.binning = binning
        self.tick = time.time()

    @property
    def shape_xy(self):
        return np.array([self.img.shape[1], self.img.shape[0]], dtype=float)

    def px2stage(self, x, y):
        offset = np.linalg.solve(self.s2px, np.array([x, y], dtype=float) - self.shape_xy / 2)
        return self.stage[:2] + offset

    def stage2px(self, sx, sy):
        return self.s2px @ (np.array([sx, sy], dtype=float) - self.stage[:2]) + self.shape_xy / 2

    def copy(self):
        buf = ImageBuffer(self.img.copy(), self.pix_size, self.stage, self.s2px, self.mag_index, self.binning)
        buf.tick = self.tick
        return buf


class Sim:
    """State of the simulated microscope."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mag_index = 5
        self.low_dose = False
        self.ld_area = None
        self.imaging_state = None
        self.stage = np.zeros(3)
        self.specimen = None
        self.loaded_slot = None
        self.maps = []                      # acquired maps for the ground truth file

        self.buffers = {}                   # letter -> ImageBuffer
        self.align_shift = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self.directory = Path.cwd()
        self.files = []                     # open image files (Path)
        self.current_file = None
        self.montage = None                 # pending montage setup

        self.nav_file = None
        self.nav_open = False
        self.nav_items = []                 # SPACEtomo NavItem objects (mock owns the navigator like SerialEM)
        self.nav_header = "AdocVersion = 2.00\n"
        self.current_nav_index = 0
        self.map_id_counter = 1000

    # --- Optics ---

    @property
    def pix_size(self):
        return MAGS[self.mag_index]

    def areaMagIndex(self, area):
        return LD_AREAS[area][0] if self.low_dose and area in LD_AREAS else self.mag_index

    def illuminatedArea(self):
        if self.low_dose and self.ld_area in LD_AREAS:
            return LD_AREAS[self.ld_area][1]
        return NON_LD_BEAM.get(self.mag_index, 10.0)

    def goToImagingState(self, name):
        name = str(name)
        match = IMAGING_STATES.get(name) or next((props for props in IMAGING_STATES.values() if str(props[0]) == name), None)
        if match is None:
            return 1
        number, state_type, camera, mag_index = match
        self.imaging_state = number
        if state_type >= 0:
            self.low_dose = True
            self.goToArea("VFTRSR"[state_type])
        else:
            self.low_dose = False
            self.ld_area = None
            self.mag_index = mag_index
        return 0

    def goToArea(self, area):
        self.ld_area = area
        self.mag_index = LD_AREAS[area][0]

    # --- Specimen and ground truth ---

    def getSpecimen(self):
        if self.specimen is None:
            self.specimen = Specimen(seed=np.random.randint(2**31))
        return self.specimen

    def worldCenter(self, stage_xy):
        specimen = self.getSpecimen()
        return np.array(stage_xy, dtype=float) + specimen.offsets[OFFSET_KEY.get(self.mag_index, "MM")]

    def saveGroundTruth(self, directory):
        self.getSpecimen().save(Path(directory) / "dummy_ground_truth.json", extra={"maps": self.maps})

    # --- Buffers ---

    def putInBuffer(self, letter, image_buffer):
        """Places image in buffer; a new image in A rolls the rolling buffers first (like Buffer.rollBuffers)."""
        if letter == "A":
            for index in range(len(ROLLING_BUFFERS) - 1, 0, -1):
                src = ROLLING_BUFFERS[index - 1]
                if src in self.buffers:
                    self.buffers[ROLLING_BUFFERS[index]] = self.buffers.pop(src)
        self.buffers[letter] = image_buffer

    def getBuffer(self, letter):
        if letter not in self.buffers:
            raise KeyError(letter)
        return self.buffers[letter]

    def acquire(self, area=None):
        """Camera image at the current stage position into buffer A."""
        mag_index = self.areaMagIndex(area) if area else self.mag_index
        previous = self.mag_index
        self.mag_index = mag_index
        pix_size = self.pix_size
        world = self.worldCenter(self.stage[:2])
        self.mag_index = previous
        img = self.getSpecimen().render(world, (CAMERA_DIMS[1], CAMERA_DIMS[0]), pix_size, detail=pix_size < 50)
        self.putInBuffer("A", ImageBuffer(toCounts(img), pix_size, self.stage.copy(), mag_index=mag_index))
        return self.buffers["A"]

    # --- Image files ---

    def path(self, file):
        file = Path(file)
        return file if file.is_absolute() else self.directory / file

    def saveToFile(self, buffer, file=None):
        """Appends buffer image as new section (files store rows bottom-up)."""
        file = self.path(file) if file else self.current_file
        data = np.flip(self.buffers[buffer].img, axis=0)
        if file.exists() and file.stat().st_size > 0:
            with mrcfile.open(file, mode="r+") as mrc:
                stack = mrc.data if mrc.data.ndim == 3 else mrc.data[np.newaxis]
                if stack.shape[1:] == data.shape:
                    mrc.set_data(np.concatenate((stack, data[np.newaxis]), axis=0))
                    return
        with mrcfile.new(file, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = self.buffers[buffer].pix_size * 10

    def readFromFile(self, file, section=0):
        with mrcfile.open(self.path(file), permissive=True) as mrc:
            data = mrc.data[section] if mrc.data.ndim == 3 else mrc.data
            return np.flip(np.array(data), axis=0), float(mrc.voxel_size.x) / 10

    # --- Montages ---

    def collectMontage(self):
        """Renders the montage that was set up and writes it as single image to the montage file."""

        setup = self.montage
        specimen = self.getSpecimen()
        if setup["kind"] == "full":
            pix_size = self.pix_size
            center = specimen.center
            extent = np.array([specimen.region, specimen.region], dtype=float)
            detail = False
        else:
            item = self.nav_items[setup["poly_index"] - 1]
            pts = np.array([[float(x) for x in item.entries["PtsX"]], [float(y) for y in item.entries["PtsY"]]])
            center = (pts.min(axis=1) + pts.max(axis=1)) / 2
            extent = pts.max(axis=1) - pts.min(axis=1)
            mag_index = LD_AREAS["V"][0] if setup.get("view", True) else self.mag_index
            pix_size = MAGS[mag_index]
            detail = True

        # Maps are loaded at full resolution like SerialEM montage maps (MapBinning 1); only huge montages are binned
        binning = max(1, int(np.ceil(np.sqrt(np.prod(extent * 1000 / pix_size) / 100e6))))
        pix_size *= binning
        shape = np.round(np.flip(extent) * 1000 / pix_size).astype(int)

        if setup["kind"] == "full":
            world = center + specimen.offsets["WG"]
        else:
            world = center + specimen.offsets["MM"]
        img = toCounts(specimen.render(world, shape, pix_size, detail=detail))

        stage = np.array([center[0], center[1], self.stage[2]])
        self.putInBuffer("A", ImageBuffer(img, pix_size, stage))
        with mrcfile.new(setup["file"], overwrite=True) as mrc:
            mrc.set_data(np.flip(img, axis=0).astype(np.int16))
            mrc.voxel_size = pix_size * 10
        if setup["kind"] == "polygon":
            self.maps.append({"name": setup["file"].stem, "nominal_center": center.tolist(), "world_center": world.tolist(),
                              "extent": extent.tolist(), "pix_size": pix_size})
        setup["acquired"] = True

    # --- Navigator ---

    def readNav(self, file):
        from SPACEtomo.modules.nav import NavItem
        self.nav_file = Path(file)
        self.nav_open = True
        self.nav_items, header = [], ""
        if self.nav_file.exists():
            with open(self.nav_file) as f:
                for line in f:
                    col = line.strip().split(" ")
                    if col[0] == "":
                        continue
                    if col[0] == "[Item":
                        item = NavItem(index=len(self.nav_items) + 1, label=col[2].strip("]"))
                        item.readBlock(f)
                        item.convertEntries()
                        self.nav_items.append(item)
                    else:
                        header += line
        self.nav_header = header or "AdocVersion = 2.00\n"
        self.current_nav_index = 1 if self.nav_items else 0
        ids = [int(item.entries["MapID"][0]) for item in self.nav_items if "MapID" in item.entries]
        self.map_id_counter = max(ids + [self.map_id_counter]) + 1

    def writeNav(self, file=None):
        file = Path(file) if file else self.nav_file
        file.parent.mkdir(parents=True, exist_ok=True)
        text = self.nav_header + "\n" + "".join(item.getBlock() for item in self.nav_items)
        file.write_text(text)

    def newMapId(self):
        self.map_id_counter += 1
        return self.map_id_counter

    def addItem(self, item):
        self.nav_items.append(item)
        item.nav_index = len(self.nav_items)
        self.current_nav_index = len(self.nav_items)
        return len(self.nav_items)

    def newMap(self, note):
        """Map item of the image in A (montage or single image) saved in the current file."""
        from SPACEtomo.modules.nav import NavItem
        buffer = self.buffers["A"]
        file = self.montage["file"] if self.montage and self.montage.get("acquired") else self.current_file
        overrides = {"MapScaleMat": [str(val) for val in np.ravel(buffer.s2px)],
                     "MapMagInd": [str(buffer.mag_index if buffer.mag_index is not None else self.mag_index)],
                     "MapLDConSet": [str("VFTRS".index(self.ld_area)) if self.low_dose and self.ld_area in "VFTRS" else "0"],
                     "MapMontage": ["1" if self.montage and self.montage.get("acquired") else "0"]}
        item = NavItem(len(self.nav_items) + 1, label=str(len(self.nav_items) + 1))
        item.createMap(None, buffer.stage, file, self.newMapId(), note or file.name, entry_overrides=overrides)
        index = self.addItem(item)
        if self.montage and self.montage.get("acquired"):
            self.montage = None
        self.saveGroundTruth(file.parent)
        return index

    def loadMap(self, index, letter):
        item = self.nav_items[index - 1]
        img, pix_size = self.readFromFile(item.map_file, int(item.entries.get("MapSection", ["0"])[0]))
        s2px = np.array(item.entries["MapScaleMat"], dtype=float).reshape((2, 2)) * float(item.entries.get("MapBinning", ["1"])[0]) / float(item.entries.get("MontBinning", ["1"])[0])
        buffer = ImageBuffer(img, pix_size, item.stage, s2px, int(item.entries.get("MapMagInd", ["0"])[0]))
        self.buffers[letter] = buffer       # loading a map does not roll buffers (as assumed by Buffer)

    # --- Alignment ---

    def align(self, ref_letter, max_shift_um=30):
        """Cross-correlates A with the reference at the coarser pixel size; stores the shift like ReportAlignShift."""

        cur, ref = self.buffers["A"], self.buffers[ref_letter]
        pix_size = max(cur.pix_size, ref.pix_size)
        img = _resample(cur.img.astype(np.float32), pix_size / cur.pix_size)
        ref_img = _resample(ref.img.astype(np.float32), pix_size / ref.pix_size)

        sign = 1
        if img.shape[0] > ref_img.shape[0] or img.shape[1] > ref_img.shape[1]:
            img, ref_img, sign = ref_img, img, -1
        search = int(max_shift_um * 1000 / pix_size)
        r0 = max(0, (ref_img.shape[0] - img.shape[0]) // 2 - search)
        c0 = max(0, (ref_img.shape[1] - img.shape[1]) // 2 - search)
        region = ref_img[r0: r0 + img.shape[0] + 2 * search, c0: c0 + img.shape[1] + 2 * search]
        cc = match_template(region, img)
        peak_row, peak_col = np.unravel_index(np.argmax(cc), cc.shape)
        d_row = r0 + peak_row + img.shape[0] / 2 - ref_img.shape[0] / 2
        d_col = c0 + peak_col + img.shape[1] / 2 - ref_img.shape[1] / 2

        ss_shift = sign * np.array([d_col, -d_row]) * pix_size / 1000         # µm, content of A relative to reference
        pixel_shift = ss_shift * 1000 / cur.pix_size * np.array([1, -1])
        self.align_shift = (pixel_shift[0], pixel_shift[1], 0.0, 0.0, ss_shift[0] * 1000, ss_shift[1] * 1000)
        return cc.max()


sim = Sim()
