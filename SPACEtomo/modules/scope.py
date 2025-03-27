#!/usr/bin/env python
# ===================================================================
# ScriptName:   scope
# Purpose:      Interface for microscope functions (inspired by Jonathan Bouvette).
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/13
# Revision:     v1.3
# Last Change:  2025/02/19: added beam blanking
#               2024/09/04: fixed autoloader check, fixed delay during imaging state change
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

import SPACEtomo.modules.utils_sem as usem
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log, serialem_check, dummy_skip

from SPACEtomo import config

class Microscope:

    def __init__(self) -> None:

        #self.imaging_params = imaging_params

        self.low_dose = bool(sem.ReportLowDose()[0])
        self.low_dose_area = None

        # Limits
        self.tilt_limit = sem.ReportProperty("MaximumTiltAngle")
        self.is_limit = sem.ReportProperty("ImageShiftLimit")
        #self.stage_limit = sem.ReportProperty("StageLimits")       # not accessible by script command
        self.stage_limit = 990

        #self.imaging_params.IS_limit = self.is_limit

        # Properties
        self.ta_offset = sem.ReportProperty("TiltAxisOffset")
        self.use_coma_vs_is = False

        # Set user settings
        sem.SetUserSetting("DriftProtection", 1)
        sem.SetUserSetting("ShiftToTiltAxis", 1)

        # Check autoloader
        self.loaded_grid = None
        self.autoloader = {}
        #self.checkAutoloader()

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
    def specimen_shift(self):
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
    def condenser_aperture(self):
        try:
            sem.StartTry(1)
            return int(sem.ReportApertureSize(1))
        except sem.SEMerror:
            log(f"WARNING: Could not access C2 aperture. Please check if SerialEM has aperture control!")
            return 0
        finally:
            sem.EndTry()

    @property
    def objective_aperture(self):
        try:
            sem.StartTry(1)
            return int(sem.ReportApertureSize(2))
        except sem.SEMerror:
            log(f"WARNING: Could not access objective aperture. Please check if SerialEM has aperture control!")
            return 0
        finally:
            sem.EndTry()

    @property
    def beam_size(self):

        # Check if illuminated area is accessible
        if sem.ReportProperty("UseIlluminatedAreaForC2"):
            return sem.ReportIlluminatedArea() * 100
        else:
            # Calculate beam size from C2 aperture and config
            log(f"DEBUG: Could not access beam size. Calculating from config and C2 aperture...")
            beam_size = config.beam_sizes[self.micro_probe] * self.condenser_aperture / config.smallest_c2_aperture
            log(f"DEBUG: Calculated beam size: {beam_size}")
            return beam_size

    @property
    def beam_shift(self):
        return np.array(sem.ReportBeamShift())

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
            sem.StartTry(1)
            area = self.low_dose_area if self.low_dose_area else "R"
            read_mode = int(sem.ReportReadMode(area))
        except sem.SEMmoduleError:
            read_mode = None
        finally:
            sem.EndTry()

        return int(sem.CameraProperties()[5]), read_mode
    
    @property
    def camera_dims(self):
        camera_properties = sem.CameraProperties()
        return np.array([camera_properties[0], camera_properties[1]], dtype=int)
    
    @property
    def camera_pix_size(self):
        return sem.CameraProperties()[4]
    
    @property
    def exposure_time(self):
        area = self.low_dose_area if self.low_dose_area else "R"
        exposure_time, drift_settling = sem.ReportExposure(area)
        #binning = sem.ReportBinning(area)
        return exposure_time

    @property
    def energy_filter(self):
        if not self.has_energy_filter:
            return 0, 0, 0
        
        slit_width, energy_loss, slit_in = sem.ReportEnergyFilter()
        return int(slit_in), slit_width, energy_loss

    @property
    def has_energy_filter(self):
        try:
            sem.StartTry(1)
            sem.ReportEnergyFilter()
        except sem.SEMerror:
            return False
        finally:
            sem.EndTry()
        return True

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
            try:
                image_shift = self.getMatrices()["ss2is"] @ specimen_shift
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
            if defocus != 0:
                sem.ChangeFocus(defocus)

    def setSpotSize(self, spot_size: int):
        """Sets spot size."""

        sem.SetSpotSize(spot_size)

    def setCondenserLens(self, value):
        """Sets condenser lens/illuminated area."""

        sem.SetPercentC2(value)

    def setBeamShift(self, beam_shift, relative=False):
        """Applies beam shift."""

        if not relative:
            sem.SetBeamShift(*beam_shift)
        else:
            sem.MoveBeamByMicrons(*beam_shift)

    def setBeamTilt(self, tilt_xy=None, auto=False):
        """Sets beam tilt."""

        if tilt_xy and not auto:
            sem.SetBeamTilt(*tilt_xy)
        elif auto:
            try:
                sem.StartTry(1)
                sem.RestoreBeamTilt()
            except sem.SEMerror:
                log(f"DEBUG: No need to restore beam tilt.")
            finally:
                sem.EndTry()

            sem.AdjustBeamTiltforIS()
        else:
            log(f"ERROR: No beam tilt value given!")

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

        if not self.has_energy_filter:
            return

        sem.SetSlitIn(slit_in)
        sem.SetSlitWidth(slit_width)
        sem.SetEnergyLoss(energy_loss)

    def setExposureTime(self, time):
        """Sets exposure time."""

        area = self.low_dose_area if self.low_dose_area else "R"
        sem.SetExposure(area, time)

    def moveStage(self, xyz, relative=False):
        """Moves stage to x, y, z."""
        
        # Add z coord if missing
        if len(xyz) == 2:
            xyz = np.append(xyz, 0 if relative else self.stage[2])

        if relative:
            sem.MoveStage(*xyz)
        else:
            sem.MoveStageTo(*xyz)

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

        # Do backlash correction
        if backlash and tilt < self.tilt_angle:
            if abs(tilt - 2) > self.tilt_limit:
                log(f"WARNING: Backlash step was skipped to remain within tilt limits.")
            else:
                sem.TiltTo(tilt - 2)
        
        # Tilt to angle
        sem.TiltTo(tilt)

    ### Simple actions

    def record(self, save="", view=False):
        """Records an image and optionally saves it to file."""

        # Check if dewars are refilling
        if self.magnification > 10000 and not view:
            log(f"DEBUG: Checking dewars and pumps...")
            sem.ManageDewarsAndPumps(1)

        # Find file to save to
        if save:
            save = Path(save)
            # Check if file is open
            if sem.IsImageFileOpen(str(save)):
                log(f"DEBUG: File already open.")
                usem.switchToFile(save)
            else:
                # Open file or create new file
                if save.exists():
                    log(f"DEBUG: Reopening file.")
                    usem.openOldFile(str(save))
                    #sem.OpenOldFile(str(save))
                else:
                    log(f"DEBUG: Creating new file.")
                    sem.OpenNewFile(str(save))

        sem.SetBeamBlank(0)
        if not view:
            sem.R()
        else:
            sem.V()
        sem.SetBeamBlank(1)
        if save:
            sem.S()

        # Roll buffers
        Buffer.rollBuffers()#skip=["A"])

    def autofocus(self, target=0.0, measure=False):
        """Uses beam tilt routine to determine defocus and optionally adjusts it to target defocus."""

        sem.SetBeamBlank(0)
        if not measure:
            sem.SetTargetDefocus(target)
            sem.G(1, -1)
        else:
            sem.G(-1, -1)
        sem.SetBeamBlank(1)
        defocus, *_ = sem.ReportAutoFocus()
        drift = np.linalg.norm(sem.ReportFocusDrift())
        return defocus, drift

    def eucentricity(self, rough=True):
        """Call eucentricity routine."""

        sem.SetBeamBlank(0)
        if rough:
            sem.Eucentricity(1)         # Rough
        else:
            sem.Eucentricity(6)         # Refine and realign
        sem.SetBeamBlank(1)

    def findZLP(self, interval_min=0):
        """Runs zero loss peak routine."""
        sem.SetBeamBlank(0)
        sem.RefineZLP(interval_min, -1) # -1 for Preview instead of Trial
        sem.SetBeamBlank(1)

    def changeImagingState(self, target_state, low_dose_expected=None):
        """Changes SerialEM imaging state."""

        if self.low_dose or self.low_dose_area is not None:               # Exit Low Dose and wait to avoid occasional DM crash when switching from LD to LM
            self.exitLowDose()

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
            self.low_dose_area = area
            return sem.ReportLDDefocusOffset("V")
        else:
            log("ERROR: Imaging state is not in Low Dose!")

    def exitLowDose(self, delay=1):
        """Exits Low Dose mode."""

        sem.SetLowDoseMode(0)
        if self.has_energy_filter:
            sem.SetSlitIn(0)

        self.low_dose = False
        self.low_dose_area = None

        time.sleep(delay)

    def changeC2Aperture(self, size=0):
        """Inserts C2 aperture of given size or retracts aperture if size is 0."""

        # Cannot take out C2 aperture
        if not size:
            return
        
        try:
            sem.StartTry(1)
            sem.SetApertureSize(1, size)
        except sem.SEMerror:
            log(f"WARNING: Could not insert C2 aperture. Please check if SerialEM has aperture control!")
        finally:
            sem.EndTry()

    def changeObjAperture(self, size=0):
        """Inserts objective aperture of given size or retracts aperture if size is 0."""

        try:
            sem.StartTry(1)
            sem.SetApertureSize(2, size)
        except sem.SEMerror:
            log(f"WARNING: Could not insert objective aperture. Please check if SerialEM has aperture control!")
        finally:
            sem.EndTry()

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

        # Deal with missing cassette and grid on stage
        if not len(self.autoloader):
            log(f"WARNING: Autoloader cassete is not present or empty!")
            if not self.loaded_grid:
                on_stage = sem.YesNoBox("\n".join(["GRID LOADED?", "", f"Is the desired grid currently on the stage?"]))
                if on_stage:
                    self.loaded_grid = "stage"
                    sem.EnterString("grid_name", "Please provide a grid name for the currently loaded grid!")
                    grid_name = sem.GetVariable("grid_name").strip()
                    self.autoloader[self.loaded_grid] = grid_name
                else:
                    log(f"ERROR: No grid in the microscope!")
                    sys.exit()

    def loadGrid(self, grid_slot):
        """Loads grid from autoloader slot."""

        if not len(self.autoloader):
            self.checkAutoloader()

        # Initiate loading
        if grid_slot != self.loaded_grid:
            log(f"Loading grid [{self.autoloader[grid_slot]}] from slot {grid_slot}...")
            sem.LoadCartridge(grid_slot)
            self.loaded_grid = grid_slot
        else:
            log(f"Grid [{self.autoloader[grid_slot]}] is already loaded.")

    def realignGrid(self, nav: Navigator, map_id: int):
        """Realigns grid to previously taken whole grid map and transforms nav items accordingly."""

        # Get coords of grid map
        initial_coords = nav.items[map_id].stage

        # Run realignment routine
        start = time.time()
        sem.RealignReloadedGrid(map_id + 1, 0, 0, 0, 1)
        log(f"DEBUG: Realignment took {time.time() - start} seconds.")

        nav.pull()
        new_coords = nav.items[map_id].stage
        if not np.array_equal(initial_coords, new_coords):
            log(f"WARNING: Grid was realigned to previously collected whole grid map. Targets selected previously might still be subject to residual shifts. Please check the realign to item procedure before starting data collection.")
            return
        log(f"ERROR: Grid realignment took too long. Please check the realignment manually before starting data collection.")

    ### Complex actions

    def collectViewBeamTiltPair(self, beam_tilt_mrad, tilt_x: int=1, tilt_y: int=0, file_path=None):
        """Collects pair of View images with +/- beam tilt in mrad."""

        current_beam_tilt = np.array(sem.ReportBeamTilt())

        if isinstance(beam_tilt_mrad, (float, int)):
            applied_beam_tilt = beam_tilt_mrad * np.array([tilt_x, tilt_y]) / np.linalg.norm(np.array([tilt_x, tilt_y]))
        elif isinstance(beam_tilt_mrad, (list, tuple, np.ndarray)):
            applied_beam_tilt = beam_tilt_mrad
        else:
            log(f"ERROR: Beam tilt value not recognized!")
            return None, None

        if file_path:
            if not file_path.exists():
                sem.OpenNewFile(str(file_path))
            else:
                usem.openOldFile(file_path)
        
        energy_filter = self.energy_filter
        if energy_filter[0]:
            self.setEnergyFilter(0, *energy_filter[1:])
        sem.SetBeamTilt(*(current_beam_tilt - applied_beam_tilt))
        sem.SetBeamBlank(0)
        sem.V()
        sem.SetBeamBlank(1)

        # Roll buffers
        Buffer.rollBuffers()

        if file_path:
            sem.S()
        
        sem.SetBeamTilt(*(current_beam_tilt + applied_beam_tilt))
        sem.SetBeamBlank(0)
        sem.V()
        sem.SetBeamBlank(1)
        if energy_filter[0]:
            self.setEnergyFilter(*energy_filter)

        # Roll buffers
        Buffer.rollBuffers()

        if file_path:
            sem.S()
            sem.CloseFile()
        
        sem.SetBeamTilt(*current_beam_tilt)

        buf2, buf1 = Buffer("A"), Buffer("B")
        buf1.imgFromBuffer()
        buf2.imgFromBuffer()

        return buf1, buf2

    def collectFullMontage(self, imaging_params, model, overlap):
        """Collects full grid montage."""

        if not len(self.autoloader):
            self.checkAutoloader()

        grid_name = self.autoloader[self.loaded_grid]

        self.changeImagingState(imaging_params.WG_image_state, low_dose_expected=False)
        self.changeC2Aperture(config.c2_apertures[0])
        if self.low_dose:
            log("WARNING: Whole grid montage image state is in Low Dose! This might result in unreasonably large grid maps.")

        # Figure out appropriate binning to get close to model pix size
        pix_size = sem.CameraProperties()[4] # nm/px
        binning = 1
        while model.pix_size / (pix_size * binning) > 2:
            binning *= 2
        if binning > 2:
            log(f"WARNING: Whole grid montage could be binned by {binning}. Consider using a smaller mag to safe time in the future!")
        #sem.NoMessageBoxOnError(1)
        #sem.SetBinning("R", binning)
        binning = sem.ReportBinning("R") # Temporarily disable setting of binning due to error message in Log

        # Save pix size
        imaging_params.WG_pix_size = pix_size * binning

        # Setup montage
        sem.SetupFullMontage(overlap, grid_name + ".mrc")
        #sem.NoMessageBoxOnError(0) # might have to be moved lower down

        # Make sure stage is ready (sometimes stage is busy error popped up when starting montage)
        while sem.ReportStageBusy():
            sem.Delay(1, "s")

        # Collect montage
        log(f"Collecting low mag whole grid montage...")
        sem.SetBeamBlank(0)
        sem.M()
        sem.SetBeamBlank(1)
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

        # Setup frame name for View frames (Cannot activate frame saving from script at the moment.)
        sem.SetFrameBaseName(0, 1, 0, "View_" + file.name) # change frame name

        # Setup polygon montage
        sem.ParamSetToUseForMontage(2)
        sem.SetMontPanelParams(1, 1, 1, 1)      # check all Montage control panel options
        sem.SetupPolygonMontage(poly_id + 1, 0, str(file))
        sem.SetMontageParams(-1, int(overlap * min(self.camera_dims)), int(overlap * min(self.camera_dims)), *self.camera_dims, 0, 1) # stage move, overlap X, overlap Y, piece size X, piece size Y, skip correlation, binning

        pieces = int(sem.ReportNumMontagePieces())
        log(f"Collecting medium mag montage at [{file.stem}] ({pieces} pieces)...")
        sem.SetBeamBlank(0)
        sem.M()
        sem.SetBeamBlank(1)
        log(f"Making new map from montage...")
        map_id = int(sem.NewMap(0, file.name)) - 1
        sem.CloseFile()

        return map_id


class ImagingParams:
    """Previously MicParams."""
    
    file_name = "mic_params.json"

    def __init__(self, WG_image_state, IM_image_state, MM_image_state, file_dir=None):

        # User settings
        self.WG_image_state = WG_image_state
        self.IM_image_state = IM_image_state
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

        # Focus properties
        self.focus_pix_size = None
        self.focus_c2ss_matrix = None
        self.focus_ss2is_matrix = None
        self.focus_ta_rotation = None
        self.focus_rotM = None
        self.focus_beam_diameter = None
        
        # Read and overwrite params with values from file
        if file_dir and (file_dir / self.file_name).exists():
            self.readFromFile(Path(file_dir))
            log(f"NOTE: Instantiated imaging parameters from {self.file_name} in {file_dir}!")
        else:
            log(f"NOTE: Instantiated blank imaging paramters because no {self.file_name} was found.")

    @dummy_skip
    @serialem_check
    def getViewParams(self, microscope: Microscope, final_attempt=False):

        if microscope.low_dose and microscope.low_dose_area == "V":
            self.cam_dims = microscope.camera_dims

            self.view_pix_size = microscope.camera_pix_size
            matrices = microscope.getMatrices()
            self.s2ss_matrix = matrices["s2ss"]
            self.view_c2ss_matrix = matrices["c2ss"]
            self.view_ss2is_matrix = matrices["ss2is"]
            self.view_ta_rotation = 90 - np.degrees(np.arctan(self.view_c2ss_matrix[0, 1] / self.view_c2ss_matrix[0, 0]))
            self.view_rotM = np.array([[np.cos(np.radians(self.view_ta_rotation)), np.sin(np.radians(self.view_ta_rotation))], [-np.sin(np.radians(self.view_ta_rotation)), np.cos(np.radians(self.view_ta_rotation))]])
            self.view_beam_diameter = microscope.beam_size
        else:
            if not final_attempt:
                log(f"WARNING: Cannot get View parameters in current Low Dose state {microscope.low_dose}, {microscope.low_dose_area}. Trying to switch to medium mag imaging state...")
                microscope.changeImagingState(self.MM_image_state, low_dose_expected=True)
                microscope.changeC2Aperture(config.c2_apertures[2])
                microscope.changeLowDoseArea("V")
                self.getViewParams(microscope, final_attempt=True)
            else:
                log(f"ERROR: Cannot get View parameters in current Low Dose state {microscope.low_dose}, {microscope.low_dose_area}. Check your imaging state settings!")
                sys.exit()


    @dummy_skip
    @serialem_check
    def getRecParams(self, microscope: Microscope, final_attempt=False):

        if microscope.low_dose and microscope.low_dose_area == "R":
            self.cam_dims = microscope.camera_dims

            self.rec_pix_size = microscope.camera_pix_size
            matrices = microscope.getMatrices()
            self.s2ss_matrix = matrices["s2ss"]
            self.rec_c2ss_matrix = matrices["c2ss"]
            self.rec_ss2is_matrix = matrices["ss2is"]
            self.rec_ta_rotation = 90 - np.degrees(np.arctan(self.rec_c2ss_matrix[0, 1] / self.rec_c2ss_matrix[0, 0]))
            self.rec_rotM = np.array([[np.cos(np.radians(self.rec_ta_rotation)), np.sin(np.radians(self.rec_ta_rotation))], [-np.sin(np.radians(self.rec_ta_rotation)), np.cos(np.radians(self.rec_ta_rotation))]])
            self.rec_beam_diameter = microscope.beam_size

            if self.rec_beam_diameter < 0:
                log(f"ERROR: Illuminated area/beam diameter is negative! Please check your imaging state setup!")
                sys.exit()
        else:
            if not final_attempt:
                log(f"WARNING: Cannot get Record parameters in current Low Dose state {microscope.low_dose}, {microscope.low_dose_area}. Trying to switch to medium mag imaging state...")
                microscope.changeImagingState(self.MM_image_state, low_dose_expected=True)
                microscope.changeC2Aperture(config.c2_apertures[2])
                microscope.changeLowDoseArea("R")
                self.getRecParams(microscope, final_attempt=True)
            else:
                log(f"ERROR: Cannot get Record parameters in current Low Dose state {microscope.low_dose}, {microscope.low_dose_area}. Check your imaging state settings!")
                sys.exit()

    @dummy_skip
    @serialem_check
    def getFocusParams(self, microscope: Microscope, final_attempt=False):

        # Check if Focus mag differs from Rec
        rec_pix_size = sem.ReportCurrentPixelSize("R") / sem.ReportBinning("R")
        focus_pix_size = sem.ReportCurrentPixelSize("F") / sem.ReportBinning("F")

        if rec_pix_size != focus_pix_size:
            microscope.changeLowDoseArea("F")

            if microscope.low_dose and microscope.low_dose_area == "F":
                self.cam_dims = microscope.camera_dims

                self.focus_pix_size = microscope.camera_pix_size
                matrices = microscope.getMatrices()
                self.s2ss_matrix = matrices["s2ss"]
                self.focus_c2ss_matrix = matrices["c2ss"]
                self.focus_ss2is_matrix = matrices["ss2is"]
                self.focus_ta_rotation = 90 - np.degrees(np.arctan(self.rec_c2ss_matrix[0, 1] / self.rec_c2ss_matrix[0, 0]))
                self.focus_rotM = np.array([[np.cos(np.radians(self.rec_ta_rotation)), np.sin(np.radians(self.rec_ta_rotation))], [-np.sin(np.radians(self.rec_ta_rotation)), np.cos(np.radians(self.rec_ta_rotation))]])
                self.focus_beam_diameter = microscope.beam_size

                if self.rec_beam_diameter < 0:
                    log(f"ERROR: Illuminated area/beam diameter is negative! Please check your imaging state setup!")
                    sys.exit()
                return
            else:
                if not final_attempt:
                    log(f"WARNING: Cannot get Focus parameters in current Low Dose state {microscope.low_dose}, {microscope.low_dose_area}. Trying to switch to medium mag imaging state...")
                    microscope.changeImagingState(self.MM_image_state, low_dose_expected=True)
                    microscope.changeC2Aperture(config.c2_apertures[2])
                    microscope.changeLowDoseArea("F")
                    self.getFocusParams(microscope, final_attempt=True)
                    return

        # Use Rec params as backup
        self.focus_pix_size = self.rec_pix_size
        self.focus_c2ss_matrix = self.rec_c2ss_matrix
        self.focus_ss2is_matrix = self.rec_ss2is_matrix
        self.focus_ta_rotation = self.rec_ta_rotation
        self.focus_rotM = self.rec_rotM
        self.focus_beam_diameter = self.rec_beam_diameter

    def export(self, dir):
        params_file = Path(dir) / self.file_name
        with open(params_file, "w+") as f:
             json.dump(vars(self), f, indent=4, default=utils.convertToTaggedString)
        log(f"NOTE: Saved imaging parameters at {params_file}")
        
    def readFromFile(self, dir):
        params_file = Path(dir) / self.file_name
        with open(params_file, "r") as f:
            params = json.load(f, object_hook=utils.revertTaggedString)

        # Check if imaging states are the same, discard loaded params if not
        if ((self.WG_image_state and self.WG_image_state != params["WG_image_state"]) 
            or (self.IM_image_state and self.IM_image_state != params["IM_image_state"]) 
            or (self.MM_image_state and self.MM_image_state != params["MM_image_state"])):
            log(f"WARNING: Given imaging states do not match saved parameters file [{params_file}]! Skipping reading parameters from file...")
            return

        for key, value in params.items():
            vars(self)[key] = value
        log(f"NOTE: Read imaging parameters from {params_file}.")
