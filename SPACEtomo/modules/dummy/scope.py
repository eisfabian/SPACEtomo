#!/usr/bin/env python
# ===================================================================
# ScriptName:   scope
# Purpose:      Dummy interface for microscope functions.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/09/02
# Revision:     v1.2
# Last Change:  2024/09/02: separated from modules/scope.py
# ===================================================================

import mrcfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import SPACEtomo.config as config
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.dummy.buf import BufferDummy
from SPACEtomo.modules.utils import log

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

class MicroscopeDummy:

    def __init__(self, imaging_params, cur_dir=None, map_dir=None, nav=None) -> None:

        self.imaging_params = imaging_params
        self.low_dose = False
        self.low_dose_area = None

        # Limits
        self.tilt_limit = 60.0
        self.is_limit = 15
        self.stage_limit = 990

        self.imaging_params.IS_limit = self.is_limit

        # Properties
        self.ta_offset = 0.0

        # Check autoloader
        self.loaded_grid = None
        self.autoloader = {}
        self.checkAutoloader()

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

        self.dummy_state = {}
        self.dummy_state["stage"] = np.zeros(3)
        self.dummy_state["tilt_angle"] = 0.0
        self.dummy_state["image_shift"] = np.zeros(2)
        self.dummy_state["specimen_shift"] = np.zeros(2)
        self.dummy_state["defocus"] = 0.0
        self.dummy_state["spot_size"] = 6
        self.dummy_state["condenser_lens"] = 2.0
        self.dummy_state["beam_tilt"] = np.zeros(2)
        self.dummy_state["micro_probe"] = 1
        self.dummy_state["magnification"] = 33000
        self.dummy_state["camera"] = [0, None] # camera_id and read_mode
        self.dummy_state["energy_filter"] = [20, 0, 1] # slit_width, energy_loss, slit_in

        if isinstance(self.imaging_params.MM_image_state, list):
            self.dummy_state["imaging_state"] = self.imaging_params.MM_image_state[0]
        else:
            self.dummy_state["imaging_state"] = self.imaging_params.MM_image_state

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
    def speciment_shift(self):
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
    def beam_tilt(self):
        return self.dummy_state["beam_tilt"]
    
    @property
    def micro_probe(self):
        return self.dummy_state["micro_probe"]
    
    @property
    def magnification(self):
        return self.dummy_state["magnification"]
    
    @property
    def camera(self):
        camera_id, read_mode = self.dummy_state["camera"]
        return camera_id, read_mode
    
    @property
    def energy_filter(self):
        slit_width, energy_loss, slit_in = self.dummy_state["energy_filter"]
        return slit_in, slit_width, energy_loss
    
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

    def setBeamTilt(self, tilt_xy):
        """Sets beam tilt."""

        self.dummy_state["beam_tilt"] = np.array(tilt_xy)
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

    def moveStage(self, xyz):
        """Moves stage to x, y, z."""
        
        self.dummy_state["stage"] = np.array(xyz)
        log(f"#DUMMY: Stage was moved to {self.stage}!")

    def tiltStage(self, tilt, backlash=True):
        """Tilts stage to target angle."""

        # Check tilt limit
        if abs(tilt) > self.tilt_limit:
            log(f"ERROR: {tilt} goes beyond maximum stage tilt [{self.tilt_limit}]!")
            return

        # Tilt to angle
        self.dummy_state["tilt_angle"] = tilt
        log(f"#DUMMY: Stage was tilted to {self.tilt_angle}!")

    ### Simple actions

    def record(self, save=""):
        """Records an image and optionally saves it to file."""

        # Figure out what kind of dummy image to generate
        if self.dummy_state["imaging_state"] == self.imaging_params.WG_image_state:
            # Generate LM image
            img_array = self.dummyImageWG()
            pix_size = self.imaging_params.WG_pix_size
        elif self.dummy_state['magnification'] == self.imaging_params.IM_mag_index:
            # Generate IM image
            img_array = self.dummyImageIM()
            pix_size = self.imaging_params.IM_pix_size
        elif self.dummy_state["imaging_state"] == self.imaging_params.MM_image_state or self.dummy_state["imaging_state"] in self.imaging_params.MM_image_state:
            if self.low_dose_area == "V":
                # Generate View image
                pix_size = self.imaging_params.view_pix_size
            elif self.low_dose_area == "R":
                # Generate Record image
                pix_size = self.imaging_params.rec_pix_size
        else:
            # Generate random noise image
            img_array = np.random.rand(*self.imaging_params.cam_dims).astype(np.float32) 
            pix_size = 0.1


        # Find file to save to
        if save:
            save = Path(save)        
            with mrcfile.new(save, overwrite=True) as mrc:
                mrc.set_data(img_array)
                mrc.voxel_size = pix_size * 10 # Angstrom

            BufferDummy.last_dummy_image = save

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

    def changeImagingState(self, target_state, low_dose_expected=None):
        """Changes SerialEM imaging state."""

        # Consider state 1 LM state and all others Low Dose states
        if target_state > 1:
            self.low_dose = True
            self.low_dose_area = "R"

        # Deal with list of imaging states for different low dose areas
        if not isinstance(target_state, list):
            target_state = [target_state]

        for state in target_state:
            self.dummy_state["imaging_state"] = state
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

    def changeObjAperture(self, size=0):
        """Inserts objective aperture of given size or retracts aperture if size is 0."""

        log(f"#DUMMY: Inserted objective aperture with size {size}!")

    def openValves(open=True):
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


    def loadGrid(self, grid_slot):
        # Initiate loading
        if grid_slot != self.loaded_grid:
            self.loaded_grid = grid_slot
            log(f"#DUMMY: Loaded grid {self.autoloader[self.loaded_grid]}!")
        else:
            log(f"Grid [{self.autoloader[grid_slot]}] is already loaded.")

    ### Complex actions

    def collectFullMontage(self, model, overlap):
        """Collects full grid montage."""
        
        grid_name = self.autoloader[self.loaded_grid]

        self.changeImagingState(self.imaging_params.WG_image_state)
        if self.low_dose:
            log("WARNING: Whole grid montage image state is in Low Dose! This might result in unreasonably large grid maps.")

        # Figure out appropriate binning to get close to model pix size
        pix_size = model.pix_size
        binning = 1

        # Save pix size
        self.imaging_params.WG_pix_size = pix_size * binning

        # Setup montage
        grid_bar_num = 20
        map_size = 2000 # microns
        lamella_dims = np.array([15, 25]) # width, height in microns

        # Generate grid map
        grid_map = np.full((int(map_size / pix_size * 1000), int(map_size / pix_size * 1000)), 0.25, dtype=np.float32)
        grid_bar_dist = max(grid_map.shape) / grid_bar_num
        grid_bar_width = max(grid_map.shape) / grid_bar_num / 3
        for i in range(grid_bar_num):
            grid_map[:, int(i * grid_bar_dist): int(i * grid_bar_dist + grid_bar_width)] = 0
            grid_map[int(i * grid_bar_dist): int(i * grid_bar_dist + grid_bar_width), :] = 0

        # Generate lamella img
        lamella_width = int(lamella_dims[0] / map_size * max(grid_map.shape))
        lamella_height = int(lamella_dims[1] / map_size * max(grid_map.shape))
        lamella_img = np.full((lamella_height * 2, lamella_width * 2), 0.25, dtype=np.float32)
        # Lamella ice
        lamella_img[int(lamella_img.shape[0] - lamella_height) // 2: int(lamella_img.shape[0] + lamella_height) // 2, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 0.5
        # Lamella trenches
        lamella_img[:int(lamella_img.shape[0] - lamella_height) // 2, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 1
        lamella_img[int(lamella_img.shape[0] + lamella_height) // 2:, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 1

        # Align lamella with tilt axis
        if abs(self.imaging_params.view_ta_rotation) < 45:
            lamella_img = lamella_img.T

        # Add lamellae at random coords
        random_coords = np.random.rand(np.random.randint(1, 5), 2)
        for coords in random_coords:
            x = int(coords[0] * grid_map.shape[0] * 0.8 + 0.1 * grid_map.shape[0])
            y = int(coords[1] * grid_map.shape[1] * 0.8 + 0.1 * grid_map.shape[1])
            grid_map[x: x + lamella_img.shape[0], y: y + lamella_img.shape[1]] = lamella_img

        # Add 25% noise
        noise = np.random.rand(*grid_map.shape)
        grid_map = grid_map * (0.75 + 0.25 * noise.astype(np.float32))

        # Save map
        grid_map_file = self.cur_dir / (grid_name + ".mrc")
        with mrcfile.new(grid_map_file, overwrite=True) as mrc:
            mrc.set_data(grid_map)
            mrc.voxel_size = pix_size * 10

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
        noise = np.random.rand(*lamella_map.shape)
        lamella_map = lamella_map * (0.5 + 0.5 * noise.astype(np.float32))

        # Align lamella with tilt axis
        if abs(self.imaging_params.view_ta_rotation) < 45:
            lamella_map = lamella_map.T
            seg_map = seg_map.T

        # Save map
        with mrcfile.new(file, overwrite=True) as mrc:
            mrc.set_data(lamella_map)
            mrc.voxel_size = pix_size * 10

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
        # Get proper pix size
        pix_size = self.imaging_params.WG_pix_size

        # Define lamella dims
        lamella_dims = np.array([15, 25]) / pix_size * 1000 # width, height in microns

        # Define grid bar distance and width
        grid_bar_dist = 100 / pix_size * 1000 # microns
        grid_bar_width = grid_bar_dist / 3
        grid_bar_num = int(max(self.imaging_params.cam_dims) / grid_bar_dist / pix_size * 1000)

        # Generate map image
        map_img = np.full(self.imaging_params.cam_dims, 0.25, dtype=np.float32)

        # Add grid bars
        for i in range(grid_bar_num):
            map_img[:, int(i * grid_bar_dist): int(i * grid_bar_dist + grid_bar_width)] = 0
            map_img[int(i * grid_bar_dist): int(i * grid_bar_dist + grid_bar_width), :] = 0

        # Generate lamella image
        lamella_width = int(lamella_dims[0])
        lamella_height = int(lamella_dims[1])
        lamella_img = np.full((lamella_height * 2, lamella_width * 2), 0.25, dtype=np.float32)
        # Lamella ice
        lamella_img[int(lamella_img.shape[0] - lamella_height) // 2: int(lamella_img.shape[0] + lamella_height) // 2, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 0.5
        # Lamella trenches
        lamella_img[:int(lamella_img.shape[0] - lamella_height) // 2, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 1
        lamella_img[int(lamella_img.shape[0] + lamella_height) // 2:, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 1

        # Align lamella with tilt axis
        if abs(self.imaging_params.view_ta_rotation) < 45:
            lamella_img = lamella_img.T

        # Add lamellae at random coords
        random_coords = np.random.rand(np.random.randint(1, 5), 2)
        for coords in random_coords:
            x = int(coords[0] * map_img.shape[0] * 0.8 + 0.1 * map_img.shape[0])
            y = int(coords[1] * map_img.shape[1] * 0.8 + 0.1 * map_img.shape[1])
            map_img[x: x + lamella_img.shape[0], y: y + lamella_img.shape[1]] = lamella_img

        # Add 25% noise
        noise = np.random.rand(*map_img.shape)
        map_img = map_img * (0.75 + 0.25 * noise.astype(np.float32))

        log(f"#DUMMY: Generated LM mag image!")
        return map_img

    def dummyImageIM(self):
        # Get proper pix size
        pix_size = self.imaging_params.IM_pix_size

        # Define lamella dims
        lamella_dims = np.array([15, 25]) / pix_size * 1000 # width, height in pixels

        # Generate map image
        map_img = np.full(np.flip(self.imaging_params.cam_dims).astype(int), 0.25, dtype=np.float32)

        # Generate lamella image
        lamella_width = int(lamella_dims[0])
        lamella_height = int(lamella_dims[1])
        lamella_img = np.full((lamella_height * 2, lamella_width * 2), 0.25, dtype=np.float32)
        # Lamella ice
        lamella_img[int(lamella_img.shape[0] - lamella_height) // 2: int(lamella_img.shape[0] + lamella_height) // 2, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 0.5
        # Lamella trenches
        lamella_img[:int(lamella_img.shape[0] - lamella_height) // 2, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 1
        lamella_img[int(lamella_img.shape[0] + lamella_height) // 2:, int(lamella_img.shape[1] - lamella_width) // 2: int(lamella_img.shape[1] + lamella_width) // 2] = 1

        # Align lamella with tilt axis
        if abs(self.imaging_params.view_ta_rotation) < 45:
            lamella_img = lamella_img.T

        # Add lamellae at random coords
        random_coords = np.random.rand(1, 2)
        for coords in random_coords:
            x = int(coords[0] * (map_img.shape[0] - lamella_img.shape[0]))
            y = int(coords[1] * (map_img.shape[1] - lamella_img.shape[1]))
            map_img[x: x + lamella_img.shape[0], y: y + lamella_img.shape[1]] = lamella_img

        # Add 25% noise
        noise = np.random.rand(*map_img.shape)
        map_img = map_img * (0.75 + 0.25 * noise.astype(np.float32))

        log(f"#DUMMY: Generated IM mag image!")
        return map_img

    def dummyImageMM(self):
        pass

    def dummyImageView(self):
        pass

    def dummyImageRec(self):
        pass