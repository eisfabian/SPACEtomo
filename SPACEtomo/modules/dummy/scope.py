#!/usr/bin/env python
# ===================================================================
# ScriptName:   scope
# Purpose:      Dummy interface for microscope functions.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/09/02
# Revision:     v1.3
# Last Change:  2026/11/07: images rendered from synthetic grid (specimen.py) with per-mag offsets and ground truth file
#               2026/11/07: fixed imaging state/beam/tilt bugs, int16 images, shared lamella generation
#               2026/07/17: added mag_index property to mirror real Microscope
#               2025/03/04: updated to make dummy run work again
#               2024/09/02: separated from modules/scope.py
# ===================================================================

import mrcfile
import numpy as np
from pathlib import Path
from PIL import Image

import SPACEtomo.config as config
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.dummy.buf import BufferDummy
from SPACEtomo.modules.utils import log, writeMrc

from SPACEtomo.modules.dummy.specimen import Specimen, toCounts, stageToPixelMatrix

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

        # Synthetic grid (created at first image of a grid) and record of acquired maps for the ground truth file
        self.specimen = None
        self.acquired_maps = []

    ### Properties

    @property
    def stage(self):
        return self.dummy_state["stage"].copy()     # copy like a fresh SerialEM report (callers keep it as initial position)
    
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
        # Specimen and stage axes coincide in the synthetic world (see dummy/specimen.py)
        identity = np.identity(2)
        return {key: identity.copy() for key in ("s2ss", "ss2s", "is2ss", "ss2is", "c2ss", "ss2c")}
    
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
        """Records an image of the synthetic grid and optionally saves it to file."""

        img_array, pix_size = self._renderCamera(view=view)

        # Find file to save to
        if save:
            save = Path(save)
            if save.exists():
                with mrcfile.open(save, mode="r+") as mrc:
                    if mrc.data.ndim == 2:
                        # If single 2D image, convert it to a stack
                        mrc.set_data(np.stack([mrc.data, np.flip(img_array, axis=0)], axis=0))
                    else:
                        # If already a stack of images, concatenate new image
                        mrc.set_data(np.concatenate((mrc.data, np.flip(img_array, axis=0)[np.newaxis, ...]), axis=0))
                    mrc.update_header_from_data()
                    mrc.update_header_stats()
                log(f"#DUMMY: Image was appended to {save}!")
            else:
                writeMrc(save, np.flip(img_array, axis=0), pix_size)
                log(f"#DUMMY: Image was saved to {save}!")

            BufferDummy.last_dummy_image = save
        else:
            BufferDummy.last_dummy_image = img_array
        BufferDummy.last_dummy_info = {"stage": self.stage[:2].copy(), "pix_size": pix_size}

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
            self.specimen = None        # New grid
            self.acquired_maps = []
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

        # Two View images of the same area (no beam tilt induced shift is simulated)
        img1, pix_size = self._renderCamera(view=True)
        img2, _ = self._renderCamera(view=True)

        if file_path:
            writeMrc(file_path, np.stack([np.flip(img1, axis=0), np.flip(img2, axis=0)], axis=0), pix_size)

        BufferDummy.last_dummy_info = {"stage": self.stage[:2].copy(), "pix_size": pix_size}
        BufferDummy.last_dummy_image = img1
        buf1 = BufferDummy("A", img=img1)
        BufferDummy.last_dummy_image = img2
        buf2 = BufferDummy("A", img=img2)

        log(f"#DUMMY: Collected beam tilt pair with {beam_tilt_mrad} mrad!")
        return buf1, buf2

    def collectFullMontage(self, imaging_params, model, overlap):
        """Collects whole grid montage of the synthetic grid at the model pixel size."""

        if not len(self.autoloader):
            self.checkAutoloader()

        grid_name = self.autoloader[self.loaded_grid]

        self.changeImagingState(imaging_params.WG_image_state)
        if self.low_dose:
            log("WARNING: Whole grid montage image state is in Low Dose! This might result in unreasonably large grid maps.")

        # Use model pix size and save it
        pix_size = model.pix_size
        imaging_params.WG_pix_size = pix_size

        specimen = self._getSpecimen()
        center = self.stage[:2].copy()
        size_px = int(specimen.region / pix_size * 1000)
        grid_map = toCounts(specimen.render(center + specimen.offsets["WG"], (size_px, size_px), pix_size))

        # Save map (files store rows bottom-up)
        grid_map_file = self.cur_dir / (grid_name + ".mrc")
        writeMrc(grid_map_file, np.flip(grid_map, axis=0), pix_size)

        map_id = self.nav.newMapFromImg(label=grid_name[:6], template_id=None, coords=center, img_file=grid_map_file, note=grid_map_file.name,
                                        entry_overrides={"MapScaleMat": [str(val) for val in np.ravel(stageToPixelMatrix(pix_size))]})

        BufferDummy.last_dummy_image = grid_map_file
        BufferDummy.last_dummy_info = {"stage": center, "pix_size": pix_size}
        self._saveGroundTruth()

        log(f"#DUMMY: Collected whole grid montage at [{grid_map_file.stem}]!")

        return map_id

    def collectPolygonMontage(self, poly_id, file, overlap, max_pixels=16e6):
        """Collects View montage of the polygon area of the synthetic grid and writes its ground truth segmentation."""

        self.changeLowDoseArea("V")

        # Polygon extent in stage coords
        polygon = self.nav.items[poly_id]
        pts = np.array([[float(x) for x in polygon.entries["PtsX"]], [float(y) for y in polygon.entries["PtsY"]]])
        center = (pts.min(axis=1) + pts.max(axis=1)) / 2
        extent = pts.max(axis=1) - pts.min(axis=1)

        # Bin montage if it would be very large
        pix_size = self.imaging_params.view_pix_size
        binning = max(1, int(np.ceil(np.sqrt(np.prod(extent * 1000 / pix_size) / max_pixels))))
        pix_size *= binning
        shape = np.round(np.flip(extent) * 1000 / pix_size).astype(int)

        specimen = self._getSpecimen()
        world_center = center + specimen.offsets["MM"]
        lamella_map, seg_map = specimen.render(world_center, shape, pix_size, detail=True, seg=True)
        lamella_map = toCounts(lamella_map)

        # Save map (files store rows bottom-up)
        writeMrc(file, np.flip(lamella_map, axis=0), pix_size)

        # Save ground truth segmentation in model pix size
        model_size = np.round(np.flip(seg_map.shape) * pix_size / config.MM_model_pix_size).astype(int)
        Image.fromarray(seg_map).resize(tuple(model_size), resample=Image.Resampling.NEAREST).save(self.map_dir / (file.stem + "_seg.png"))

        lamella_name = file.stem.split("_")[-1]
        map_id = self.nav.newMapFromImg(label=lamella_name, template_id=None, coords=center, img_file=file, note=file.name,
                                        entry_overrides={"MapScaleMat": [str(val) for val in np.ravel(stageToPixelMatrix(pix_size))]})

        self.acquired_maps.append({"name": file.stem, "nominal_center": center.tolist(), "world_center": world_center.tolist(), "pix_size": pix_size})
        self._saveGroundTruth()

        BufferDummy.last_dummy_image = file
        BufferDummy.last_dummy_info = {"stage": center, "pix_size": pix_size}

        log(f"#DUMMY: Collected medium mag montage at [{file.stem}]!")

        return map_id
    


    # Dummy image generation
    def dummyImageWG(self):
        """Camera image of the synthetic grid at whole grid mag at the current stage position."""
        return self._renderCamera(state_key="WG")[0]

    def dummyImageIM(self):
        """Camera image of the synthetic grid at intermediate mag at the current stage position."""
        return self._renderCamera(state_key="IM")[0]

    ### Synthetic grid

    def _stateKey(self):
        """Offset key of the current imaging state."""
        if self.dummy_state["imaging_state"] == self.imaging_params.WG_image_state:
            return "WG"
        if self.dummy_state["imaging_state"] == self.imaging_params.IM_image_state:
            return "IM"
        return "MM"

    def _getSpecimen(self):
        """Synthetic grid of the loaded grid (created around the current stage position on first use)."""
        if self.specimen is None:
            self.specimen = Specimen(seed=np.random.randint(2**31), center=self.stage[:2],
                                     vertical=abs(self.imaging_params.view_ta_rotation) >= 45)
            log(f"#DUMMY: Created synthetic grid with {len(self.specimen.lamellae)} lamellae (seed {self.specimen.seed}).")
        return self.specimen

    def _saveGroundTruth(self):
        """Writes lamellae, image offsets and acquired maps for test assertions."""
        self._getSpecimen().save(self.cur_dir / "dummy_ground_truth.json", extra={"maps": self.acquired_maps})

    def _renderCamera(self, state_key=None, view=False):
        """Renders a camera image (int16 counts) at the current stage position and returns it with its pixel size."""

        state_key = state_key or self._stateKey()
        if state_key == "WG":
            pix_size = self.imaging_params.WG_pix_size or 400.0
        elif state_key == "IM":
            pix_size = self.imaging_params.IM_pix_size or 10.0
        elif self.low_dose_area == "R" and not view:
            pix_size = self.imaging_params.rec_pix_size
        else:
            pix_size = self.imaging_params.view_pix_size

        specimen = self._getSpecimen()
        img = specimen.render(self.stage[:2] + specimen.offsets[state_key], np.flip(self.camera_dims), pix_size, detail=state_key == "MM")
        log(f"#DUMMY: Generated {state_key} image!")
        return toCounts(img), pix_size
