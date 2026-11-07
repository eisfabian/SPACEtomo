#!/usr/bin/env python
# ===================================================================
# ScriptName:   scope
# Purpose:      Dummy interface for microscope functions.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/09/02
# Revision:     v1.3
# Last Change:  2026/11/07: fixed imaging state/beam/tilt bugs, int16 images, shared lamella generation
#               2026/07/17: added mag_index property to mirror real Microscope
#               2025/03/04: updated to make dummy run work again
#               2024/09/02: separated from modules/scope.py
# ===================================================================

import mrcfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import SPACEtomo.config as config
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.dummy.buf import BufferDummy
from SPACEtomo.modules.utils import log, writeMrc

### DEV SETTINGS ###
# 
# Dummy image generation
cats = {
    "background": 0,
    "white": 1,
    "black": 2,
    "crack": 3,
    "coating": 4,
    "cell": 5,
    "cellwall": 6,
    "nucleus": 7,
    "vacuole": 8,
    "mitos": 9,
    "lipiddroplets": 10,
    "vesicles": 11,
    "multivesicles": 12,
    "membranes": 13,
    "dynabeads": 14,
    "ice": 15,
    "cryst": 16
}
cats_intensity = {
    "background": 0.5,
    "white": 0.9,
    "black": 0.1,
    "crack": 0.9,
    "coating": 0.25,
    "cell": 0.4,
    "cellwall": 0.35,
    "nucleus": 0.37,
    "vacuole": 0.32,
    "mitos": 0.3,
    "lipiddroplets": 0.25,
    "vesicles": 0.33,
    "multivesicles": 0.33,
    "membranes": 0.3,
    "dynabeads": 0.2,
    "ice": 0.11,
    "cryst": 0.2
}
#
### END SETTINGS ###


def _toCounts(img, mean_counts=2000):
    """Converts a relative intensity image (0-1) into int16 camera counts, as stored in real maps."""
    return np.clip(img * 2 * mean_counts, 0, np.iinfo(np.int16).max).astype(np.int16)

def _addNoise(img, fraction=0.25):
    """Multiplies image with uniform noise (fraction of intensity)."""
    return img * (1 - fraction + fraction * np.random.rand(*img.shape).astype(np.float32))

def _lamellaStamp(lamella_dims_px, transpose=False):
    """Lamella as seen at low mag: ice slab (0.5) between two milled trenches (1.0) in a 2x larger patch (0.25)."""
    width, height = int(lamella_dims_px[0]), int(lamella_dims_px[1])
    stamp = np.full((height * 2, width * 2), 0.25, dtype=np.float32)
    rows = slice((stamp.shape[0] - height) // 2, (stamp.shape[0] + height) // 2)
    cols = slice((stamp.shape[1] - width) // 2, (stamp.shape[1] + width) // 2)
    stamp[rows, cols] = 0.5
    stamp[:rows.start, cols] = 1
    stamp[rows.stop:, cols] = 1
    return stamp.T if transpose else stamp

def _placeStamp(img, stamp, top_left):
    """Pastes stamp into img at top_left (row, col), clipped to the image bounds."""
    r0, c0 = max(0, int(top_left[0])), max(0, int(top_left[1]))
    r1, c1 = min(img.shape[0], int(top_left[0]) + stamp.shape[0]), min(img.shape[1], int(top_left[1]) + stamp.shape[1])
    if r1 > r0 and c1 > c0:
        img[r0:r1, c0:c1] = stamp[r0 - int(top_left[0]):r1 - int(top_left[0]), c0 - int(top_left[1]):c1 - int(top_left[1])]

def _gridImage(shape, pix_size, lamella_dims_um, transpose, grid_pitch_um=100, n_lamellae=None, margin_squares=1):
    """Grid squares (0.25) separated by bars (0) with lamellae centred in random grid squares (never on bars)."""
    img = np.full(shape, 0.25, dtype=np.float32)
    pitch = grid_pitch_um / pix_size * 1000
    bar_width = pitch / 3
    for i in range(int(max(shape) / pitch) + 1):
        img[:, int(i * pitch): int(i * pitch + bar_width)] = 0
        img[int(i * pitch): int(i * pitch + bar_width), :] = 0

    stamp = _lamellaStamp(np.array(lamella_dims_um) / pix_size * 1000, transpose)
    squares = [(r, c) for r in range(margin_squares, int(shape[0] / pitch) - margin_squares) for c in range(margin_squares, int(shape[1] / pitch) - margin_squares)]
    if n_lamellae is None:
        n_lamellae = np.random.randint(1, 5)
    for idx in np.random.permutation(len(squares))[:n_lamellae]:
        r, c = squares[idx]
        center = np.array([(r + 0.5) * pitch + bar_width / 2, (c + 0.5) * pitch + bar_width / 2])
        _placeStamp(img, stamp, center - np.array(stamp.shape) / 2)
    return img


class MicroscopeDummy:

    def __init__(self, cur_dir=None, map_dir=None, nav=None, imaging_params=None) -> None:

        self.low_dose = False
        self.low_dose_area = None

        # Limits
        self.tilt_limit = 60.0
        self.is_limit = 15
        self.stage_limit = 990

        # Properties
        self.ta_offset = 0.0
        self.use_coma_vs_is = False

        # Check autoloader
        self.loaded_grid = None
        self.autoloader = {}
        #self.checkAutoloader()

        # Dummy state
        if not cur_dir:
            self.cur_dir = Path.cwd()
        else:
            self.cur_dir = Path(cur_dir)
        if not map_dir:
            self.map_dir = Path.cwd() / "SPACE_maps"
        else:
            self.map_dir = Path(map_dir)
        if not nav:
            self.nav = Navigator(Path("temp.nav"))
        else:
            self.nav = nav
        if not imaging_params:
            self.imaging_params = None

        self.dummy_state = {}
        self.dummy_state["stage"] = np.zeros(3)
        self.dummy_state["tilt_angle"] = 0.0
        self.dummy_state["image_shift"] = np.zeros(2)
        self.dummy_state["specimen_shift"] = np.zeros(2)
        self.dummy_state["defocus"] = 0.0
        self.dummy_state["spot_size"] = 6
        self.dummy_state["condenser_lens"] = 2.0
        self.dummy_state["condenser_aperture"] = 50
        self.dummy_state["objective_aperture"] = 100
        self.dummy_state["beam_size"] = 2.0
        self.dummy_state["beam_shift"] = np.zeros(2)
        self.dummy_state["beam_tilt"] = np.zeros(2)
        self.dummy_state["micro_probe"] = 1
        self.dummy_state["magnification"] = 33000
        self.dummy_state["mag_index"] = 25
        self.dummy_state["camera"] = [0, None] # camera_id and read_mode
        self.dummy_state["energy_filter"] = [20, 0, 1] # slit_width, energy_loss, slit_in
        self.dummy_state["exposure_time"] = 1.0

        self.dummy_state["imaging_state"] = 1

    ### Properties

    @property
    def stage(self):
        return self.dummy_state["stage"]
    
    @property
    def tilt_angle(self):
        return self.dummy_state["tilt_angle"]
    
    @property
    def image_shift(self):
        return self.dummy_state["image_shift"]
    
    @property
    def specimen_shift(self):
        return self.dummy_state["specimen_shift"]

    @property
    def defocus(self):
        return self.dummy_state["defocus"]
    
    @property
    def spot_size(self):
        return self.dummy_state["spot_size"]
    
    @property
    def condenser_lens(self):
        return self.dummy_state["condenser_lens"]
    
    @property
    def condenser_aperture(self):
        return self.dummy_state["condenser_aperture"]

    @property
    def objective_aperture(self):
        return self.dummy_state["objective_aperture"]

    @property
    def beam_size(self):
        return self.dummy_state["beam_size"]

    @property
    def beam_shift(self):
        return self.dummy_state["beam_shift"]
    
    @property
    def beam_tilt(self):
        return self.dummy_state["beam_tilt"]
    
    @property
    def micro_probe(self):
        return self.dummy_state["micro_probe"]
    
    @property
    def magnification(self):
        return self.dummy_state["magnification"]

    @property
    def mag_index(self):
        return self.dummy_state["mag_index"]

    @property
    def camera(self):
        camera_id, read_mode = self.dummy_state["camera"]
        return camera_id, read_mode

    @property
    def camera_dims(self):
        if self.imaging_params is not None and getattr(self.imaging_params, "cam_dims", None) is not None:
            return np.array(self.imaging_params.cam_dims, dtype=int)
        return np.array([5760, 4092], dtype=int)
    
    @property
    def camera_pix_size(self):
        return 12323 / self.magnification
    
    @property
    def exposure_time(self):
        return self.dummy_state["exposure_time"]
    
    @property
    def energy_filter(self):
        slit_width, energy_loss, slit_in = self.dummy_state["energy_filter"]
        return slit_in, slit_width, energy_loss

    @property
    def has_energy_filter(self):
        return True
    
    def getMatrices(self, mag_index=0, camera=None):
        matrices = {}
        matrices["s2ss"] = np.array(((-1, 0), (0, -1)))
        matrices["ss2s"] = np.array(((-1, 0), (0, -1)))
        matrices["is2ss"] = np.array(((1, 0), (0, 1)))
        matrices["ss2is"] = np.array(((1, 0), (0, 1)))
        matrices["c2ss"] = np.array(((1, 0), (0, 1)))
        matrices["ss2c"] = np.array(((1, 0), (0, 1)))
        return matrices
    
    ### Setters

    def setImageShift(self, image_shift, relative=False):
        """Applies image shift."""

        if relative:
            self.dummy_state["image_shift"] += image_shift
            log(f"#DUMMY: Image shift was changed by {image_shift} units to {self.dummy_state['image_shift']}!")
        else:
            self.dummy_state["image_shift"] = image_shift
            log(f"#DUMMY: Image shift was changed to {self.dummy_state['image_shift']} units!")

    def setSpecimenShift(self, specimen_shift, relative=False):
        """Applies image shift in specimen coordinates."""
        
        if not relative:
            if self.low_dose_area == "R":
                try:
                    image_shift = self.imaging_params.rec_ss2is_matrix @ specimen_shift
                    self.dummy_state["image_shift"] = image_shift
                except (AttributeError, ValueError):
                    log(f"ERROR: Could not convert specimen shift to image shift.")
                    return
            elif self.low_dose_area == "V":
                try:
                    image_shift = self.imaging_params.view_ss2is_matrix @ specimen_shift
                    self.dummy_state["image_shift"] = image_shift
                except (AttributeError, ValueError):
                    log(f"ERROR: Could not convert specimen shift to image shift.")
                    return
            self.dummy_state["specimen_shift"] = specimen_shift
            log(f"#DUMMY: Image shift was changed to {self.dummy_state['image_shift']} units!")
        else:
            self.dummy_state["specimen_shift"] += specimen_shift
            log(f"#DUMMY: Image shift was changed by {specimen_shift} microns!")

    def setDefocus(self, defocus, relative=False):
        """Applies defocus."""

        if not relative:
            self.dummy_state["defocus"] = defocus
            log(f"#DUMMY: Defocus was set to {defocus} microns!")
        else:
            self.dummy_state["defocus"] += defocus
            log(f"#DUMMY: Defocus was changed by {defocus} microns!")

    def setSpotSize(self, spot_size: int):
        """Sets spot size."""

        self.dummy_state["spot_size"] = spot_size
        log(f"#DUMMY: Spot size was changed to {self.dummy_state['spot_size']}!")

    def setCondenserLens(self, value):
        """Sets condenser lens/illuminated area."""

        self.dummy_state["condenser_lens"] = value
        log(f"#DUMMY: C2/3 was changed to {self.dummy_state['condenser_lens']}!")

    def setBeamShift(self, beam_shift, relative=False):
        """Applies beam shift."""

        if relative:
            self.dummy_state["beam_shift"] = self.dummy_state["beam_shift"] + np.array(beam_shift)
        else:
            self.dummy_state["beam_shift"] = np.array(beam_shift)

        log(f"#DUMMY: Beam shift was changed to {self.dummy_state['beam_shift']}!")

    def setBeamTilt(self, tilt_xy=None, auto=False):
        """Sets beam tilt."""

        if tilt_xy is not None and not auto:
            self.dummy_state["beam_tilt"] = np.array(tilt_xy)
        elif auto:
            log(f"#DUMMY: Beam tilt was adjusted automatically.")
        else:
            log(f"ERROR: No beam tilt value given!")

        log(f"#DUMMY: Beam tilt was changed to {self.dummy_state['beam_tilt']}!")

    def setMicroProbe(self, probe_mode: int):
        """Turns micro-probe on (1) or off (0)."""

        self.dummy_state["micro_probe"] = probe_mode
        log(f"#DUMMY: Micro-probe was changed to {self.dummy_state['micro_probe']}!")

    def setMag(self, mag: int):
        """Switches mag by number or index."""

        self.dummy_state["magnification"] = mag

        log(f"#DUMMY: Mag was changed to {self.dummy_state['magnification']}!")

    def setCamera(self, camera_id: int, read_mode: int= None):
        """Selects camera."""

        self.dummy_state["camera"] = [camera_id, read_mode]
        log(f"#DUMMY: Camera was changed to {self.dummy_state['camera'][0]} (Read mode: {self.dummy_state['camera'][1]}!")

    def setEnergyFilter(self, slit_in, slit_width, energy_loss=0):
        """Sets energy filter slit settings."""

        self.dummy_state["energy_filter"] = [slit_width, energy_loss, slit_in]
        log(f"#DUMMY: Energy filter slit was changed to {slit_in} (width: {slit_width}, loss: {energy_loss})!")

    def setExposureTime(self, time):
        """Sets exposure time."""

        self.dummy_state["exposure_time"] = time
        log(f"#DUMMY: Exposure time was changed to {self.dummy_state['exposure_time']}!")

    def moveStage(self, xyz, relative=False):
        """Moves stage to x, y, z."""

        xyz = np.array(xyz, dtype=float)
        if len(xyz) == 2:
            xyz = np.append(xyz, 0.0 if relative else self.dummy_state["stage"][2])

        if relative:
            self.dummy_state["stage"] += xyz
        else:
            self.dummy_state["stage"] = xyz
        log(f"#DUMMY: Stage was moved to {self.stage}!")

    def tiltStage(self, tilt, backlash=True, threshold=0.1, force=False):
        """Tilts stage to target angle."""

        # Check if tilt changed
        if abs(tilt - self.tilt_angle) <= threshold and not force:
            log(f"DEBUG: Tilt angle change is <={threshold} degrees and tilt action will be skipped.")
            return

        # Check tilt limit
        if abs(tilt) > self.tilt_limit:
            log(f"ERROR: {tilt} goes beyond maximum stage tilt [{self.tilt_limit}]!")
            return False

        # Tilt to angle
        self.dummy_state["tilt_angle"] = tilt
        log(f"#DUMMY: Stage was tilted to {self.tilt_angle}!")

    ### Simple actions

    def record(self, save="", view=False):
        """Records an image and optionally saves it to file."""

        # Figure out what kind of dummy image to generate
        img_array = None
        if self.dummy_state["imaging_state"] == self.imaging_params.WG_image_state:
            # Generate LM image
            img_array = self.dummyImageWG()
            pix_size = self.imaging_params.WG_pix_size or 400.0
        elif self.dummy_state['imaging_state'] == self.imaging_params.IM_image_state:
            # Generate IM image
            img_array = self.dummyImageIM()
            pix_size = self.imaging_params.IM_pix_size or 10.0
        elif self.dummy_state["imaging_state"] == self.imaging_params.MM_image_state or self.dummy_state["imaging_state"] in self.imaging_params.MM_image_state:
            if self.low_dose_area == "V" or view:
                # Generate View image
                pix_size = self.imaging_params.view_pix_size
            elif self.low_dose_area == "R":
                # Generate Record image
                pix_size = self.imaging_params.rec_pix_size
            else:
                pix_size = 0.1
        else:
            pix_size = 0.1

        if img_array is None:
            # Generate random noise image
            img_array = _toCounts(np.random.rand(*np.flip(self.camera_dims)).astype(np.float32))

        # Find file to save to
        if save:
            save = Path(save)
            if save.exists():
                with mrcfile.open(save, mode="r+") as mrc:
                    if mrc.data.ndim == 2:
                        # If single 2D image, convert it to a stack
                        mrc.set_data(np.stack([mrc.data, img_array], axis=0))
                    else:
                        # If already a stack of images, concatenate new image
                        mrc.set_data(np.concatenate((mrc.data, img_array[np.newaxis, ...]), axis=0))
                    mrc.update_header_from_data()
                    mrc.update_header_stats()
                log(f"#DUMMY: Image was appended to {save}!")
            else:
                writeMrc(save, img_array, pix_size)
                log(f"#DUMMY: Image was saved to {save}!")

            BufferDummy.last_dummy_image = save
        else:
            BufferDummy.last_dummy_image = img_array

        log(f"#DUMMY: Image {save} was recorded!")

    def autofocus(self, target=0.0, measure=False):
        """Uses beam tilt routine to determine defocus and optionally adjusts it to target defocus."""
        
        drift = np.random.uniform(0.1, 1)
        if not measure:
            self.dummy_state["defocus"] = target
            measured = self.dummy_state["defocus"]
            log(f"#DUMMY: Focus was adjusted to target: {self.dummy_state['defocus']}")
        else:
            measured = self.dummy_state["defocus"] + np.random.uniform(-1, 1)
            log(f"#DUMMY: Focus was measured: {measured} (drift: {drift}!")

        return measured, drift

    def eucentricity(self, rough=True):
        """Call eucentricity routine."""

        log(f"#DUMMY: Stage was moved to eucentric height!")

    def findZLP(self, interval_min=0):
        """Runs zero loss peak routine."""

        log(f"#DUMMY: ZLP was refined!")

    def changeImagingState(self, target_state, low_dose_expected=None):
        """Changes SerialEM imaging state."""

        # Deal with list of imaging states for different low dose areas
        if not isinstance(target_state, list):
            target_state = [target_state]

        mm_states = self.imaging_params.MM_image_state if isinstance(self.imaging_params.MM_image_state, list) else [self.imaging_params.MM_image_state]
        for state in target_state:
            self.dummy_state["imaging_state"] = state
            # WG and IM states are not Low Dose, MM states are Low Dose with View as initial area
            if state in mm_states:
                self.low_dose = True
                self.low_dose_area = "V"
                self.dummy_state["magnification"] = 12323 / self.imaging_params.view_pix_size
            else:
                self.low_dose = False
                self.low_dose_area = None
                if state == self.imaging_params.WG_image_state:
                    self.dummy_state["magnification"] = 12323 / (self.imaging_params.WG_pix_size or 400.0)
                elif state == self.imaging_params.IM_image_state:
                    self.dummy_state["magnification"] = 12323 / (getattr(self.imaging_params, "IM_pix_size", None) or 10.0)
            log(f"#DUMMY: Changed image state to {state}!")

        # Check if imaging state fits expectation
        if low_dose_expected is not None:
            if self.low_dose != low_dose_expected:
                log(f"WARNING: Imaging state does not fit expected low dose state!")
                return False
        return True

    def changeLowDoseArea(self, area):
        """Changes Low Dose area."""

        if self.low_dose:
            self.low_dose_area = area
            log(f"#DUMMY: Switched Low Dose Area to {self.low_dose_area}!")
        else:
            log("ERROR: Imaging state is not in Low Dose!")

    def exitLowDose(self, delay=1):
        """Exits Low Dose mode."""

        self.low_dose = False
        self.low_dose_area = None

    def changeC2Aperture(self, size=0):
        """Inserts C2 aperture of given size or retracts aperture if size is 0."""

        # Cannot take out C2 aperture
        if not size:
            return
        
        self.dummy_state["condenser_aperture"] = size
        log(f"#DUMMY: Inserted C2 aperture with size {size}!")

    def changeObjAperture(self, size=0):
        """Inserts objective aperture of given size or retracts aperture if size is 0."""

        self.dummy_state["objective_aperture"] = size
        log(f"#DUMMY: Inserted objective aperture with size {size}!")

    def openValves(self, open=True):
        if open:
            # Open column valves
            log(f"#DUMMY: Opened column valves!")
        else:
            # Close column valves
            log(f"#DUMMY: Closed column valves!")

    def checkAutoloader(self):
        """Checks autoloader slots."""

        for slot in range(1, 13):
            self.autoloader[slot] = "G" + str(slot).zfill(2)

        log(f"#DUMMY: Autoloader was checked!")


    def unloadGrid(self):
        """Unloads the current grid from the stage."""
        if self.loaded_grid:
            log(f"#DUMMY: Unloaded grid {self.autoloader[self.loaded_grid]}!")
            self.loaded_grid = None
        else:
            log(f"WARNING: No grid currently loaded.")

    def loadGrid(self, grid_slot):

        if not len(self.autoloader):
            self.checkAutoloader()

        # Initiate loading
        if grid_slot != self.loaded_grid:
            self.loaded_grid = grid_slot
            log(f"#DUMMY: Loaded grid {self.autoloader[self.loaded_grid]}!")
        else:
            log(f"Grid [{self.autoloader[grid_slot]}] is already loaded.")

    def realignGrid(self, nav: Navigator, map_id: int):
        """Realigns grid to previously taken whole grid map and transforms nav items accordingly."""

        log(f"WARNING: Grid was realigned to previously collected whole grid map. Targets selected previously might still be subject to residual shifts. Please check the realign to item procedure before starting data collection.")

    ### Complex actions

    def collectViewBeamTiltPair(self, beam_tilt_mrad, tilt_x: int=1, tilt_y: int=0, file_path=None):
        """Collects pair of View images with +/- beam tilt in mrad."""

        if isinstance(beam_tilt_mrad, (float, int)):
            applied_beam_tilt = beam_tilt_mrad * np.array([tilt_x, tilt_y]) / np.linalg.norm(np.array([tilt_x, tilt_y]))
        elif isinstance(beam_tilt_mrad, (list, tuple, np.ndarray)):
            applied_beam_tilt = np.array(beam_tilt_mrad)
        else:
            log(f"ERROR: Beam tilt value not recognized!")
            return None, None

        # Generate two dummy images
        img1 = _toCounts(np.random.rand(*np.flip(self.camera_dims)).astype(np.float32))
        img2 = _toCounts(np.random.rand(*np.flip(self.camera_dims)).astype(np.float32))

        if file_path:
            from SPACEtomo.modules.utils import writeMrc
            writeMrc(file_path, np.stack([img1, img2], axis=0), self.imaging_params.view_pix_size or 1.0)

        BufferDummy.last_dummy_image = img1
        buf1 = BufferDummy("A", img=img1)
        BufferDummy.last_dummy_image = img2
        buf2 = BufferDummy("A", img=img2)

        log(f"#DUMMY: Collected beam tilt pair with {beam_tilt_mrad} mrad!")
        return buf1, buf2

    def collectFullMontage(self, imaging_params, model, overlap):
        """Collects full grid montage."""

        if not len(self.autoloader):
            self.checkAutoloader()
            
        grid_name = self.autoloader[self.loaded_grid]

        self.changeImagingState(imaging_params.WG_image_state)
        if self.low_dose:
            log("WARNING: Whole grid montage image state is in Low Dose! This might result in unreasonably large grid maps.")

        # Figure out appropriate binning to get close to model pix size
        pix_size = model.pix_size
        binning = 1

        # Save pix size
        imaging_params.WG_pix_size = pix_size * binning

        # Generate grid map (1 mm region, lamellae in grid squares)
        map_size = 1000 # microns
        grid_map = _gridImage((int(map_size / pix_size * 1000), int(map_size / pix_size * 1000)), pix_size, [15, 25], transpose=abs(imaging_params.view_ta_rotation) < 45)
        grid_map = _toCounts(_addNoise(grid_map))

        # Save map
        grid_map_file = self.cur_dir / (grid_name + ".mrc")
        with mrcfile.new(grid_map_file, overwrite=True) as mrc:
            mrc.set_data(grid_map)
            mrc.voxel_size = pix_size * 10 # Angstrom

        map_id = self.nav.newMapFromImg(label=grid_name[:6], template_id=None, coords=self.stage[:2], img_file=grid_map_file, note=grid_map_file.name)

        BufferDummy.last_dummy_image = grid_map_file

        log(f"#DUMMY: Collected whole grid montage at [{grid_map_file.stem}]!")

        return map_id

    def collectPolygonMontage(self, poly_id, file, overlap):
        """Collects montage at polygon using View mode."""

        self.changeLowDoseArea("V")
        # Setup polygon montage
        pix_size = self.imaging_params.view_pix_size

        map_size = np.array([35, 40]) / pix_size * 1000       # width, height in pixels
        lamella_dims = np.array([15, 25]) / pix_size * 1000   # width, height in pixels

        # Generate map with black background
        lamella_map = np.full((int(map_size[1]), int(map_size[0])), cats_intensity["black"], dtype=np.float32)
        seg_map = np.full((int(map_size[1]), int(map_size[0])), cats["black"], dtype=np.uint8)

        # Generate lamella
        lamella_width = int(lamella_dims[0])
        lamella_height = int(lamella_dims[1])

        # Lamella background
        lamella_map[int(lamella_map.shape[0] - lamella_height) // 2: int(lamella_map.shape[0] + lamella_height) // 2, int(lamella_map.shape[1] - lamella_width) // 2: int(lamella_map.shape[1] + lamella_width) // 2] = cats_intensity["background"]
        seg_map[int(seg_map.shape[0] - lamella_height) // 2: int(seg_map.shape[0] + lamella_height) // 2, int(seg_map.shape[1] - lamella_width) // 2: int(seg_map.shape[1] + lamella_width) // 2] = cats["background"]

        # Lamella trenches
        lamella_map[:int(lamella_map.shape[0] - lamella_height) // 2, int(lamella_map.shape[1] - lamella_width) // 2: int(lamella_map.shape[1] + lamella_width) // 2] = cats_intensity["white"]
        lamella_map[int(lamella_map.shape[0] + lamella_height) // 2:, int(lamella_map.shape[1] - lamella_width) // 2: int(lamella_map.shape[1] + lamella_width) // 2] = cats_intensity["white"]

        seg_map[:int(seg_map.shape[0] - lamella_height) // 2, int(seg_map.shape[1] - lamella_width) // 2: int(seg_map.shape[1] + lamella_width) // 2] = cats["white"]
        seg_map[int(seg_map.shape[0] + lamella_height) // 2:, int(seg_map.shape[1] - lamella_width) // 2: int(seg_map.shape[1] + lamella_width) // 2] = cats["white"]

        # Make drawing canvas
        lamella_canvas = Image.fromarray(lamella_map)
        lamella_draw = ImageDraw.Draw(lamella_canvas)
        seg_canvas = Image.fromarray(seg_map)
        seg_draw = ImageDraw.Draw(seg_canvas)

        # Draw cell
        cell_size = np.array([5, 7]) / pix_size * 1000
        random_coords = np.random.rand(30, 2) * (lamella_dims - cell_size) + (np.flip(lamella_map.shape) - lamella_dims) // 2
        cell_coords = []
        for c, coords in enumerate(random_coords):
            if min([np.linalg.norm(coords - random) for random in random_coords[c + 1:]] + [1e9]) < min(cell_size): 
                continue
            lamella_draw.ellipse((*coords, *(coords + cell_size)), fill=cats_intensity["cell"], outline=cats_intensity["cellwall"], width=int(0.5 / pix_size * 1000))
            seg_draw.ellipse((*coords, *(coords + cell_size)), fill=cats["cell"], outline=cats["cellwall"], width=int(0.5 / pix_size * 1000))
            cell_coords.append(coords)

        # Draw organelles
        mito_size = np.array([1, 2]) / pix_size * 1000
        nuc_size = np.array([3, 3]) / pix_size * 1000
        vac_size = np.array([3, 3]) / pix_size * 1000

        for cell in cell_coords:
            # Mitos
            random_coords = np.random.rand(3, 2) * (0.8 * cell_size - mito_size) + cell + cell_size * 0.1
            for coords in random_coords:
                lamella_draw.ellipse((*coords, *(coords + mito_size)), fill=cats_intensity["mitos"])
                seg_draw.ellipse((*coords, *(coords + mito_size)), fill=cats["mitos"])

            # Nucleus
            random_coords = np.random.rand(1, 2) * (0.8 * cell_size - nuc_size) + cell + cell_size * 0.1
            for coords in random_coords:
                size = nuc_size * np.random.uniform(0.3, 1)
                lamella_draw.ellipse((*coords, *(coords + size)), fill=cats_intensity["nucleus"])
                seg_draw.ellipse((*coords, *(coords + size)), fill=cats["nucleus"])

            # Vacuole
            random_coords = np.random.rand(1, 2) * (0.8 * cell_size - vac_size) + cell + cell_size * 0.1
            for coords in random_coords:
                size = vac_size * np.random.uniform(0.5, 1)
                lamella_draw.ellipse((*coords, *(coords + size)), fill=cats_intensity["vacuole"])
                seg_draw.ellipse((*coords, *(coords + size)), fill=cats["vacuole"])

        # Draw ice blobs
        max_size = np.array([3, 3]) / pix_size * 1000
        random_coords = np.random.rand(10, 2) * (lamella_dims - max_size) + (np.flip(lamella_map.shape) - lamella_dims) // 2
        for c, coords in enumerate(random_coords):
            size = np.random.uniform() * max_size
            lamella_draw.ellipse((*coords, *(coords + size)), fill=cats_intensity["ice"])
            seg_draw.ellipse((*coords, *(coords + size)), fill=cats["ice"])

        # Revert back to array
        lamella_map = np.array(lamella_canvas)
        seg_map = np.array(seg_canvas)

        # Lamella coating
        seg_map[int((seg_map.shape[0] + lamella_height) // 2 - seg_map.shape[0] / 10): int(seg_map.shape[0] + lamella_height) // 2, int(seg_map.shape[1] - lamella_width) // 2: int(seg_map.shape[1] + lamella_width) // 2] = cats["coating"]
        lamella_map[int((lamella_map.shape[0] + lamella_height) // 2 - lamella_map.shape[0] / 10): int(lamella_map.shape[0] + lamella_height) // 2, int(lamella_map.shape[1] - lamella_width) // 2: int(lamella_map.shape[1] + lamella_width) // 2] = cats_intensity["coating"]

        # Add 50% noise
        lamella_map = _toCounts(_addNoise(lamella_map, fraction=0.5))

        # Align lamella with tilt axis
        if abs(self.imaging_params.view_ta_rotation) < 45:
            lamella_map = lamella_map.T
            seg_map = seg_map.T

        # Save map
        with mrcfile.new(file, overwrite=True) as mrc:
            mrc.set_data(lamella_map)
            mrc.voxel_size = pix_size * 10 # Angstrom

        # Save seg in model pix size
        model_size = np.round(np.flip(seg_map.shape) * pix_size / config.MM_model_pix_size).astype(int)
        Image.fromarray(np.flip(seg_map, axis=0)).resize(tuple(model_size), resample=Image.Resampling.NEAREST).save(self.map_dir / (file.stem + "_seg.png"))

        lamella_name = file.stem.split("_")[-1]
        map_id = self.nav.newMapFromImg(label=lamella_name, template_id=None, coords=self.stage[:2], img_file=file, note=file.name)

        BufferDummy.last_dummy_image = file

        log(f"#DUMMY: Collected medium mag montage at [{file.stem}]!")

        return map_id
    


    # Dummy image generation
    def dummyImageWG(self):
        """Camera image at whole grid mag (grid bars and 1-4 lamellae in grid squares)."""

        pix_size = self.imaging_params.WG_pix_size or 400.0
        map_img = _gridImage(np.flip(self.camera_dims), pix_size, [15, 25], transpose=abs(self.imaging_params.view_ta_rotation) < 45, margin_squares=0)

        log(f"#DUMMY: Generated LM mag image!")
        return _toCounts(_addNoise(map_img))

    def dummyImageIM(self):
        """Camera image at intermediate mag with one lamella at a random position."""

        pix_size = self.imaging_params.IM_pix_size or 10.0
        map_img = np.full(np.flip(self.camera_dims), 0.25, dtype=np.float32)

        stamp = _lamellaStamp(np.array([15, 25]) / pix_size * 1000, transpose=abs(self.imaging_params.view_ta_rotation) < 45)
        stamp = stamp[:map_img.shape[0], :map_img.shape[1]]
        top_left = np.random.rand(2) * np.maximum(0, np.array(map_img.shape) - np.array(stamp.shape))
        _placeStamp(map_img, stamp, top_left)

        log(f"#DUMMY: Generated IM mag image!")
        return _toCounts(_addNoise(map_img))
