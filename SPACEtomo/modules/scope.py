#!/usr/bin/env python
# ===================================================================
# ScriptName:   scope
# Purpose:      Interface for microscope functions (inspired by Jonathan Bouvette).
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/13
# Revision:     v1.2
# Last Change:  2024/09/04: fixed autoloader check, fixed delay during imaging state change
#               2024/09/02: moved MicroscopeDummy to dummy modules
#               2024/08/27: added generation of detailed MM montage, made MicroscopeDummy work
#               2024/08/23: added MicroscopeDummy
#               2024/08/16: added save option to record, added polygon montage
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
from pathlib import Path
import numpy as np

from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log, serialem_check, dummy_skip

class Microscope:

    def __init__(self, imaging_params) -> None:

        self.imaging_params = imaging_params

        self.low_dose = bool(sem.ReportLowDose()[0])
        self.low_dose_area = None

        # Limits
        self.tilt_limit = sem.ReportProperty("MaximumTiltAngle")
        self.is_limit = sem.ReportProperty("ImageShiftLimit")
        #self.stage_limit = sem.ReportProperty("StageLimits")       # not accessible by script command

        self.imaging_params.IS_limit = self.is_limit

        # Properties
        self.ta_offset = sem.ReportProperty("TiltAxisOffset")
        self.use_coma_vs_is = False

        # Set user settings
        sem.SetUserSetting("DriftProtection", 1)
        sem.SetUserSetting("ShiftToTiltAxis", 1)

        # Check autoloader
        self.loaded_grid = None
        self.autoloader = {}
        self.checkAutoloader()

    ### Properties

    @property
    def stage(self):
        return np.array(sem.ReportStageXYZ())
    
    @property
    def tilt_angle(self):
        return sem.ReportTiltAngle()
    
    @property
    def image_shift(self):
        isx, isy, *_ = sem.ReportImageShift()
        return np.array((isx, isy))
    
    @property
    def speciment_shift(self):
        return np.array(sem.ReportSpecimenShift())
    
    @property
    def defocus(self):
        return float(sem.ReportDefocus())
    
    @property
    def spot_size(self):
        return int(sem.ReportSpotSize())
    
    @property
    def condenser_lens(self):
        return float(sem.ReportPercentC2()[0]) # Returns fractional lens strength as second value
    
    @property
    def beam_tilt(self):
        return np.array(sem.ReportBeamTilt())
    
    @property
    def micro_probe(self):
        return int(sem.ReportProbeMode())
    
    @property
    def magnification(self):
        return int(sem.ReportMag()[0]) # Returns low mag mode as second value, and TEM, EFTEM, STEM as third value
    
    @property
    def camera(self):
        try:
            area = self.low_dose_area if self.low_dose_area else "R"
            read_mode = int(sem.ReportReadMode(area))
        except sem.SEMmoduleError:
            read_mode = None

        return int(sem.CameraProperties()[5]), read_mode
    
    @property
    def energy_filter(self):
        slit_width, energy_loss, slit_in = sem.ReportEnergyFilter()
        return int(slit_in), slit_width, energy_loss
    
    def getMatrices(self, mag_index=0, camera=None):
        matrices = {}
        matrices["s2ss"] = np.array(sem.StageToSpecimenMatrix(mag_index)).reshape((2, 2))
        matrices["ss2s"] = np.array(sem.SpecimenToStageMatrix(mag_index)).reshape((2, 2))
        matrices["is2ss"] = np.array(sem.ISToSpecimenMatrix(mag_index)).reshape((2, 2))
        matrices["ss2is"] = np.array(sem.SpecimenToISMatrix(mag_index)).reshape((2, 2))
        matrices["c2ss"] = np.array(sem.CameraToSpecimenMatrix(mag_index)).reshape((2, 2))
        matrices["ss2c"] = np.array(sem.SpecimenToCameraMatrix(mag_index)).reshape((2, 2))
        return matrices
        
    ### Setters

    def setImageShift(self, image_shift, relative=False):
        """Applies image shift."""

        if not relative:
            sem.SetImageShift(image_shift[0], image_shift[1], 0, int(self.use_coma_vs_is))
        else:
            sem.ImageShiftByUnits(image_shift[0], image_shift[1], 0, int(self.use_coma_vs_is))

    def setSpecimenShift(self, specimen_shift, relative=False):
        """Applies image shift in specimen coordinates."""

        if not relative:
            if self.low_dose_area == "R":
                try:
                    image_shift = self.imaging_params.rec_ss2is_matrix @ specimen_shift
                except (AttributeError, ValueError):
                    log(f"ERROR: Could not convert specimen shift to image shift.")
                    return
            elif self.low_dose_area == "V":
                try:
                    image_shift = self.imaging_params.view_ss2is_matrix @ specimen_shift
                except (AttributeError, ValueError):
                    log(f"ERROR: Could not convert specimen shift to image shift.")
                    return
            sem.SetImageShift(image_shift[0], image_shift[1], 0, int(self.use_coma_vs_is))
        else:
            sem.ImageShiftByMicrons(specimen_shift[0], specimen_shift[1], 0, int(self.use_coma_vs_is))

    def setDefocus(self, defocus, relative=False):
        """Applies defocus."""

        if not relative:
            sem.SetDefocus(defocus)
        else:
            sem.ChangeFocus(defocus)

    def setSpotSize(self, spot_size: int):
        """Sets spot size."""

        sem.SetSpotSize(spot_size)

    def setCondenserLens(self, value):
        """Sets condenser lens/illuminated area."""

        sem.SetPercentC2(value)

    def setBeamTilt(self, tilt_xy):
        """Sets beam tilt."""

        sem.SetBeamTilt(*tilt_xy)

    def setMicroProbe(self, probe_mode: int):
        """Turns micro-probe on (1) or off (0)."""

        if probe_mode != self.micro_probe:
            sem.SetProbeMode(str(probe_mode))

    def setMag(self, mag: int):
        """Switches mag by number or index."""

        # Switch to IM mag depending on if mag index or mag value was given
        if mag < 50:
            sem.SetMagIndex(mag)
        else:
            sem.SetMag(mag)

    def setCamera(self, camera_id: int, read_mode: int= None):
        """Selects camera."""

        sem.SelectCamera(camera_id)
        if read_mode is not None:
            area = self.low_dose_area if self.low_dose_area else "R"
            sem.SetK2ReadMode(area, read_mode)

    def setEnergyFilter(self, slit_in, slit_width, energy_loss=0):
        """Sets energy filter slit settings."""

        sem.SetSlitIn(slit_in)
        sem.SetSlitWidth(slit_width)
        sem.SetEnergyLoss(energy_loss)

    def moveStage(self, xyz):
        """Moves stage to x, y, z."""
        
        sem.MoveStageTo(*xyz)

    def tiltStage(self, tilt, backlash=True):
        """Tilts stage to target angle."""

        # Check tilt limit
        if abs(tilt) > self.tilt_limit:
            log(f"ERROR: {tilt} goes beyond maximum stage tilt [{self.tilt_limit}]!")
            return False

        # Do backlash correction
        if backlash and tilt < self.tilt_angle:
            if abs(tilt - 2) > self.tilt_limit:
                log(f"WARNING: Backlash step was skipped to remain within tilt limits.")
            else:
                sem.TiltTo(tilt - 2)
        
        # Tilt to angle
        sem.TiltTo(tilt)

    ### Simple actions

    def record(self, save=""):
        """Records an image and optionally saves it to file."""

        # Find file to save to
        if save:
            save = Path(save)
            # Check if file is open
            if sem.IsImageFileOpen(str(save)):
                # Search for file number and switch to matching file
                for f in range(1, sem.ReportNumOpenFiles() + 1):
                    if sem.ReportOpenImageFile(f) == save:
                        sem.SwitchToFile(f)
                        break
            else:
                # Open file or create new file
                if save.exists():
                    sem.OpenOldFile(str(save))
                else:
                    sem.OpenNewFile(str(save))

        sem.R()
        if save:
            sem.S()

    def autofocus(self, target=0.0, measure=False):
        """Uses beam tilt routine to determine defocus and optionally adjusts it to target defocus."""

        if not measure:
            sem.SetTargetDefocus(target)
            sem.G(1, -1)
        else:
            sem.G(-2, -1)
        defocus, *_ = sem.ReportAutoFocus()
        drift = np.linalg.norm(sem.ReportFocusDrift())
        return defocus, drift

    def eucentricity(self, rough=True):
        """Call eucentricity routine."""

        if rough:
            sem.Eucentricity(1)         # Rough
        else:
            sem.Eucentricity(6)         # Refine and realign

    def changeImagingState(self, target_state, low_dose_expected=None):
        """Changes SerialEM imaging state."""

        if self.low_dose or self.low_dose_area is None:               # Exit Low Dose and wait to avoid occasional DM crash when switching from LD to LM
            sem.SetLowDoseMode(0)
            sem.SetSlitIn(0)
            time.sleep(1)

        # Deal with list of imaging states for different low dose areas
        if not isinstance(target_state, list):
            target_state = [target_state]

        for state in target_state:
            err = sem.GoToImagingState(str(state))
            if err > 0:
                if err == 1 or err == 3:
                    log(f"ERROR: Imaging state [{state}] not found!")
                elif err == 2:
                    log(f"ERROR: Imaging state [{state}] ambigous!")
                elif err >= 4:
                    log(f"ERROR: Imaging state [{state}] could not be set!")

        # Check if imaging state is a low dose state.
        low_dose, area = sem.ReportLowDose()
        self.low_dose = bool(low_dose)
        self.low_dose_area = area

        # Check if imaging state fits expectation
        if low_dose_expected is not None:
            if self.low_dose != low_dose_expected:
                log(f"WARNING: Imaging state does not fit expected low dose state!")
                return False
        return True

    def changeLowDoseArea(self, area):
        """Changes Low Dose area."""

        if self.low_dose:
            sem.GoToLowDoseArea(area)
        else:
            log("ERROR: Imaging state is not in Low Dose!")

    def changeObjAperture(self, size=0):
        """Inserts objective aperture of given size or retracts aperture if size is 0."""

        try:
            sem.SetApertureSize(2, size)
        except:
            log(f"WARNING: Could not insert objective aperture. Please check if SerialEM has aperture control!")

    def openValves(open=True):
        if open:
            # Open column valves
            sem.SetColumnOrGunValve(1)
        else:
            # Close column valves
            sem.SetColumnOrGunValve(0)

    def checkAutoloader(self):
        """Checks autoloader slots."""

        for slot in range(1, 13):
            slot_status = sem.ReportSlotStatus(slot)
            log(f"DEBUG: Slot status: {slot_status}")
            if slot_status[0] > 0:
                if len(slot_status) > 1 and isinstance(slot_status[-1], str) and slot_status[-1] != "!NONAME!":
                    self.autoloader[slot] = slot_status[-1]
                else:
                    self.autoloader[slot] = "G" + str(slot).zfill(2)
            else:
                if len(slot_status) > 1 and isinstance(slot_status[-1], str) and slot_status[-1] != "!NONAME!":
                    log(f"DEBUG: Slot label: {slot_status[-1]}, Type: {type(slot_status[-1])}, Len: {len(slot_status[-1])}")
                    self.autoloader[slot] = slot_status[-1]
                    if not self.loaded_grid:
                        on_stage = sem.YesNoBox("\n".join(["GRID LOADED?", "", f"Is the grid from slot {slot} [{self.autoloader[slot]}] currently on the stage?"]))
                        if on_stage:
                            self.loaded_grid = slot
                        else:
                            log(f"WARNING: Please only enter names for occupied slots in the Autoloader panel!")

    def loadGrid(self, grid_slot):
        # Initiate loading
        if grid_slot != self.loaded_grid:
            log(f"Loading grid [{self.autoloader[grid_slot]}] from slot {grid_slot}...")
            sem.LoadCartridge(grid_slot)
            self.loaded_grid = grid_slot
        else:
            log(f"Grid [{self.autoloader[grid_slot]}] is already loaded.")

    ### Complex actions

    def collectFullMontage(self, model, overlap):
        """Collects full grid montage."""
        
        grid_name = self.autoloader[self.loaded_grid]

        self.changeImagingState(self.imaging_params.WG_image_state, low_dose_expected=False)
        if self.low_dose:
            log("WARNING: Whole grid montage image state is in Low Dose! This might result in unreasonably large grid maps.")

        # Figure out appropriate binning to get close to model pix size
        pix_size = sem.CameraProperties()[4]
        binning = 1
        while model.pix_size / (pix_size * binning) > 2:
            binning *= 2
        if binning > 2:
            log(f"WARNING: Whole grid montage could be binned by {binning}. Consider using a smaller mag to safe time in the future!")
        #sem.NoMessageBoxOnError(1)
        #sem.SetBinning("R", binning)
        binning = sem.ReportBinning("R") # Temporarily disable setting of binning due to error message in Log

        # Save pix size
        self.imaging_params.WG_pix_size = pix_size * binning

        # Setup montage
        sem.SetupFullMontage(overlap, grid_name + ".mrc")
        #sem.NoMessageBoxOnError(0) # might have to be moved lower down

        # Make sure stage is ready (sometimes stage is busy error popped up when starting montage)
        while sem.ReportStageBusy():
            sem.Delay(1, "s")

        # Collect montage
        log(f"Collecting low mag whole grid montage...")
        sem.M()
        log(f"Making new map from montage...")
        map_id = int(sem.NewMap(0, grid_name + ".mrc")) - 1
        sem.CloseFile()

        sem.ChangeItemLabel(map_id + 1, grid_name[:6])
        if len(grid_name) > 6:
            log(f"WARNING: Grid name [{grid_name}] will be truncated [{grid_name[:6]}] for labels in the Navigator.")

        return map_id

    def collectPolygonMontage(self, poly_id, file, overlap):
        """Collects montage at polygon using View mode."""

        self.changeLowDoseArea("V")
        # Setup polygon montage
        sem.ParamSetToUseForMontage(2)
        sem.SetMontPanelParams(1, 1, 1, 1)      # check all Montage control panel options
        sem.SetupPolygonMontage(poly_id + 1, 0, str(file))
        sem.SetMontageParams(1, int(overlap * min(self.imaging_params.cam_dims)), int(overlap * min(self.imaging_params.cam_dims)), *self.imaging_params.cam_dims, 0, 1) # stage move, overlap X, overlap Y, piece size X, piece size Y, skip correlation, binning

        pieces = int(sem.ReportNumMontagePieces())
        log(f"Collecting medium mag montage at [{file.stem}] ({pieces} pieces)...")

        sem.M()
        map_id = int(sem.NewMap(0, file.name)) - 1
        sem.CloseFile()

        return map_id


class ImagingParams:
    """Previously MicParams."""
    
    file_name = "mic_params.json"

    def __init__(self, WG_image_state, IM_mag_index, MM_image_state, file_dir=None):

        # User settings
        self.WG_image_state = WG_image_state
        self.IM_mag_index = IM_mag_index
        self.MM_image_state = MM_image_state

        # Initialize all attributes
        self.IS_limit = None
        self.cam_dims = np.zeros(2, dtype=int)

        # LM properties
        self.WG_pix_size = None
        self.IM_pix_size = None

        # Mag independent properties
        self.s2ss_matrix = None

        # View properties
        self.view_pix_size = None
        self.view_c2ss_matrix = None
        self.view_ss2is_matrix = None           # Should be independent between mags, but different for LM, so safe both, view and rec, to be safe
        self.view_ta_rotation = None
        self.view_rotM = None
        self.view_beam_diameter = None

        # Rec properties
        self.rec_pix_size = None
        self.rec_c2ss_matrix = None
        self.rec_ss2is_matrix = None
        self.rec_ta_rotation = None
        self.rec_rotM = None
        self.rec_beam_diameter = None
        
        # Read and overwrite params with values from file
        if file_dir and (file_dir / self.file_name).exists():
            self.readFromFile(Path(file_dir))
            log(f"NOTE: Instantiated imaging parameters from {self.file_name} in {file_dir}!")
        else:
            log(f"NOTE: Instantiated blank imaging paramters because no {self.file_name} was found.")

    @dummy_skip
    @serialem_check
    def getViewParams(self, final_attempt=False):
        low_dose, area = sem.ReportLowDose()
        if low_dose and area == 0:
            cam_props = sem.CameraProperties()
            self.cam_dims = np.array([cam_props[0], cam_props[1]], dtype=int)

            self.view_pix_size = cam_props[4]   # [nm]
            self.s2ss_matrix = np.array(sem.StageToSpecimenMatrix(0)).reshape((2, 2))
            self.view_c2ss_matrix = np.array(sem.CameraToSpecimenMatrix(0)).reshape((2, 2))
            self.view_ss2is_matrix = np.array(sem.SpecimenToISMatrix(0)).reshape((2, 2))
            self.view_ta_rotation = 90 - np.degrees(np.arctan(self.view_c2ss_matrix[0, 1] / self.view_c2ss_matrix[0, 0]))
            self.view_rotM = np.array([[np.cos(np.radians(self.view_ta_rotation)), np.sin(np.radians(self.view_ta_rotation))], [-np.sin(np.radians(self.view_ta_rotation)), np.cos(np.radians(self.view_ta_rotation))]])
            self.view_beam_diameter = sem.ReportIlluminatedArea() * 100
        else:
            if not final_attempt:
                log(f"WARNING: Cannot get View parameters in current Low Dose state {low_dose}, {area}. Trying to switch to medium mag imaging state...")
                if isinstance(self.MM_image_state, list):
                    for state in self.MM_image_state:
                        sem.GoToImagingState(str(state))
                else:
                    sem.GoToImagingState(str(self.MM_image_state))
                sem.GoToLowDoseArea(0)
                self.getViewParams(final_attempt=True)
            else:
                log(f"ERROR: Cannot get View parameters in current Low Dose state {low_dose}, {area}. Check your imaging state settings!")
                sys.exit()


    @dummy_skip
    @serialem_check
    def getRecParams(self, final_attempt=False):
        low_dose, area = sem.ReportLowDose()
        if low_dose and area == 3:
            cam_props = sem.CameraProperties()
            self.cam_dims = np.array([cam_props[0], cam_props[1]], dtype=int)

            self.rec_pix_size = cam_props[4]    # [nm]
            self.s2ss_matrix = np.array(sem.StageToSpecimenMatrix(0)).reshape((2, 2))
            self.rec_c2ss_matrix = np.array(sem.CameraToSpecimenMatrix(0)).reshape((2, 2))
            self.rec_ss2is_matrix = np.array(sem.SpecimenToISMatrix(0)).reshape((2, 2))
            self.rec_ta_rotation = 90 - np.degrees(np.arctan(self.rec_c2ss_matrix[0, 1] / self.rec_c2ss_matrix[0, 0]))
            self.rec_rotM = np.array([[np.cos(np.radians(self.rec_ta_rotation)), np.sin(np.radians(self.rec_ta_rotation))], [-np.sin(np.radians(self.rec_ta_rotation)), np.cos(np.radians(self.rec_ta_rotation))]])
            self.rec_beam_diameter = sem.ReportIlluminatedArea() * 100

            if self.rec_beam_diameter < 0:
                log(f"ERROR: Illuminated area/beam diameter is negative! Please check your imaging state setup!")
                sys.exit()
        else:
            if not final_attempt:
                log(f"WARNING: Cannot get Record parameters in current Low Dose state {low_dose}, {area}. Trying to switch to medium mag imaging state...")
                if isinstance(self.MM_image_state, list):
                    for state in self.MM_image_state:
                        sem.GoToImagingState(str(state))
                else:
                    sem.GoToImagingState(str(self.MM_image_state))
                sem.GoToLowDoseArea(3)
                self.getRecParams(final_attempt=True)
            else:
                log(f"ERROR: Cannot get Record parameters in current Low Dose state {low_dose}, {area}. Check your imaging state settings!")
                sys.exit()

    def export(self, dir):
        params_file = Path(dir) / self.file_name
        with open(params_file, "w+") as f:
             json.dump(vars(self), f, indent=4, default=utils.convertArray)
        log(f"NOTE: Saved imaging parameters at {params_file}")
        
    def readFromFile(self, dir):
        params_file = Path(dir) / self.file_name
        with open(params_file, "r") as f:
            params = json.load(f, object_hook=utils.revertArray)
        for key, value in params.items():
            vars(self)[key] = value
        log(f"NOTE: Read imaging parameters from {params_file}.")
