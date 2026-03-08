#!/usr/bin/env python
# ===================================================================
# ScriptName:   serialem (mock)
# Purpose:      Mock implementation of the SerialEM Python module for
#               testing without a running SerialEM instance.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein / Claude
# Created:      2026/03/06
# Revision:     v1.4
# ===================================================================
#
# This module emulates the `serialem` Python module (PySEMSocket).
# It maintains internal state for microscope parameters and provides
# return values consistent with what SerialEM would return, allowing
# the SPACEtomo codebase to run end-to-end without a real microscope.
#
# Usage:
#   Before any real `import serialem` would occur, ensure this module
#   is importable as `serialem`, e.g.:
#       import sys
#       from SPACEtomo.modules.dummy import serialem as _sem_mock
#       sys.modules["serialem"] = _sem_mock
#
# ===================================================================

import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Custom exceptions matching real serialem module
# ---------------------------------------------------------------------------

class SEMmoduleError(Exception):
    """Raised for connection / module-level errors."""
    pass

class SEMerror(Exception):
    """Raised when a SerialEM script command fails (used with StartTry/EndTry)."""
    pass


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

class _State:
    """Mutable singleton holding all mock microscope state."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Stage
        self.stage_xyz = [0.0, 0.0, 0.0]
        self.tilt_angle = 0.0
        self.stage_busy = False

        # Image shift (IS units and specimen shift)
        self.image_shift = [0.0, 0.0]
        self.specimen_shift = [0.0, 0.0]

        # Defocus
        self.defocus = 0.0
        self.target_defocus = -3.0

        # Beam
        self.beam_shift = [0.0, 0.0]
        self.beam_tilt = [0.0, 0.0]
        self.beam_blank = 1
        self.spot_size = 6
        self.percent_c2 = 50.0
        self.fractional_c2 = 0.12
        self.illuminated_area = 2.0
        self.probe_mode = 1  # 1 = nanoprobe, 0 = microprobe

        # Magnification
        self.mag = 33000
        self.mag_index = 25
        self.low_mag_mode = 0  # 0 = regular mag

        # Camera
        self.camera_id = 0
        self.camera_dims = [5760, 4092]
        self.camera_pix_size = 0.3735  # nm/px at default mag
        self.camera_rotation = 0.0
        self.detector_pix_size = 5.0  # physical detector pixel in nm
        self.read_modes = {"V": 1, "F": 1, "T": 1, "R": 1, "P": 0, "S": 1, "M": 0}
        self.binnings = {"V": 1, "F": 1, "T": 1, "R": 1, "P": 1, "S": 1, "M": 1}
        self.exposures = {"V": 1.0, "F": 0.5, "T": 0.5, "R": 1.0, "P": 0.1, "S": 0.5, "M": 1.0}
        self.drift_settling = {"V": 0.0, "F": 0.0, "T": 0.0, "R": 0.0, "P": 0.0, "S": 0.0, "M": 0.0}

        # Energy filter
        self.slit_in = 1
        self.slit_width = 20.0
        self.energy_loss = 0.0
        self.has_energy_filter = True

        # Apertures
        self.c_aperture = 50
        self.o_aperture = 100

        # Low Dose
        self.low_dose_mode = 0
        self.low_dose_area = "R"
        self.ld_defocus_offsets = {"V": 0.0, "F": 0.0, "T": 0.0, "R": 0.0, "S": 0.0}

        # Imaging states — registry of named states
        # Each entry: {name: (index, low_dose, camera, mag_index)}
        # low_dose: -1 = not low dose, 0+ = low dose area index
        self.imaging_states = {
            "WG": (1, -1, 0, 10),    # Whole grid: not low dose
            "IM": (2, -1, 0, 20),    # Intermediate mag: not low dose
            "MM1": (3, 0, 0, 25),    # Medium mag: low dose (Record area)
        }
        self.current_imaging_state = 1

        # Navigator
        self.nav_open = True
        self.nav_file = "temp.nav"
        self.nav_items = []  # list of dicts
        self.nav_acquire_count = 0

        # Files
        self.open_files = []  # list of file path strings
        self.current_file_index = 0
        self.file_overwrite_allowed = False

        # Persistent variables
        self.persistent_vars = {}
        self.script_vars = {"imageTickTime": str(time.time())}

        # User settings
        self.user_settings = {
            "BuffersToRollOnAcquire": 10,
            "DriftProtection": 0,
            "ShiftToTiltAxis": 0,
            "ShowStateNumbers": 0,
        }

        # Properties
        self.properties = {
            "MaximumTiltAngle": 60.0,
            "ImageShiftLimit": 15.0,
            "TiltAxisOffset": 0.0,
            "UseIlluminatedAreaForC2": 1,
        }

        # Try/Catch stack
        self._try_depth = 0

        # Navigator save callback
        self._save_nav_callback = None

        # Buffer images (letter -> numpy array)
        self.buffers = {}

        # Last alignment shift
        self.align_shift = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Autofocus results
        self.last_autofocus = -3.0
        self.last_focus_drift = [0.0, 0.0]

        # Column valve state
        self.column_valve_open = True


_state = _State()


def _get_state():
    return _state


# ---------------------------------------------------------------------------
# Connection & general
# ---------------------------------------------------------------------------

def ConnectToSEM(port=48888, ip="127.0.0.1"):
    pass

def Delay(duration, unit="s"):
    """Mock delay - does nothing for speed, but accepts the call."""
    pass

def SuppressReports():
    pass

def ErrorsToLog():
    pass

def SetNewFileType(file_type):
    pass

def ProgramTimeStamps():
    pass

def SaveSettings():
    pass

def Exit():
    pass

def KeyBreak():
    return 0


# ---------------------------------------------------------------------------
# Try/Catch
# ---------------------------------------------------------------------------

def StartTry(level=1):
    _state._try_depth += 1

def EndTry():
    if _state._try_depth > 0:
        _state._try_depth -= 1


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

def IsVariableDefined(name):
    if name in _state.persistent_vars or name in _state.script_vars:
        return 1
    return 0

def GetVariable(name):
    if name in _state.script_vars:
        return _state.script_vars[name]
    if name in _state.persistent_vars:
        return _state.persistent_vars[name]
    raise SEMerror(f"Variable '{name}' is not defined")

def SetPersistentVar(name, value):
    _state.persistent_vars[name] = str(value)

def ClearPersistentVars():
    _state.persistent_vars.clear()

def CLearPersistentVars():
    """Alias for the typo variant used in some scripts."""
    ClearPersistentVars()


# ---------------------------------------------------------------------------
# User settings & properties
# ---------------------------------------------------------------------------

def ReportUserSetting(name):
    return _state.user_settings.get(name, 0)

def SetUserSetting(name, value, *args):
    _state.user_settings[name] = value

def ReportProperty(name):
    return _state.properties.get(name, 0)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def ReportStageXYZ():
    return tuple(_state.stage_xyz)

def MoveStageTo(x, y, z=None):
    _state.stage_xyz[0] = float(x)
    _state.stage_xyz[1] = float(y)
    if z is not None:
        _state.stage_xyz[2] = float(z)

def MoveStage(dx, dy, dz=0):
    _state.stage_xyz[0] += float(dx)
    _state.stage_xyz[1] += float(dy)
    _state.stage_xyz[2] += float(dz)

def ReportTiltAngle():
    return _state.tilt_angle

def TiltTo(angle):
    _state.tilt_angle = float(angle)

def ReportStageBusy():
    return 0

def ReportAxisPosition(area="F"):
    return (0.0,)


# ---------------------------------------------------------------------------
# Image shift
# ---------------------------------------------------------------------------

def ReportImageShift():
    return (*_state.image_shift, 0.0, 0.0)

def SetImageShift(x, y, *args):
    _state.image_shift = [float(x), float(y)]

def ImageShiftByUnits(dx, dy, *args):
    _state.image_shift[0] += float(dx)
    _state.image_shift[1] += float(dy)

def ImageShiftByMicrons(dx, dy, *args):
    _state.image_shift[0] += float(dx)
    _state.image_shift[1] += float(dy)

def ResetImageShift():
    _state.image_shift = [0.0, 0.0]

def ReportSpecimenShift():
    return tuple(_state.specimen_shift)


# ---------------------------------------------------------------------------
# Defocus
# ---------------------------------------------------------------------------

def ReportDefocus():
    return _state.defocus

def SetDefocus(value):
    _state.defocus = float(value)

def ChangeFocus(delta):
    _state.defocus += float(delta)

def SetTargetDefocus(value):
    _state.target_defocus = float(value)

def ReportAutoFocus():
    _state.last_autofocus = _state.target_defocus + np.random.uniform(-0.1, 0.1)
    return (_state.last_autofocus,)

def ReportFocusDrift():
    return (np.random.uniform(0.01, 0.5), np.random.uniform(0.01, 0.5))

def ReportLDDefocusOffset(area):
    return _state.ld_defocus_offsets.get(area, 0.0)


# ---------------------------------------------------------------------------
# Beam
# ---------------------------------------------------------------------------

def ReportBeamShift():
    return tuple(_state.beam_shift)

def SetBeamShift(x, y):
    _state.beam_shift = [float(x), float(y)]

def MoveBeamByMicrons(dx, dy):
    _state.beam_shift[0] += float(dx)
    _state.beam_shift[1] += float(dy)

def ReportBeamTilt():
    return tuple(_state.beam_tilt)

def SetBeamTilt(x, y):
    _state.beam_tilt = [float(x), float(y)]

def AdjustBeamTiltforIS():
    pass

def RestoreBeamTilt():
    _state.beam_tilt = [0.0, 0.0]

def SetBeamBlank(on):
    _state.beam_blank = int(on)

def ReportSpotSize():
    return _state.spot_size

def SetSpotSize(size):
    _state.spot_size = int(size)

def ReportPercentC2():
    return (_state.percent_c2, _state.fractional_c2)

def SetPercentC2(value):
    _state.percent_c2 = float(value)

def ReportIlluminatedArea():
    return _state.illuminated_area

def ReportProbeMode():
    return _state.probe_mode

def SetProbeMode(mode):
    _state.probe_mode = int(mode)

def SetColumnOrGunValve(state):
    _state.column_valve_open = bool(state)


# ---------------------------------------------------------------------------
# Magnification
# ---------------------------------------------------------------------------

def ReportMag():
    return (_state.mag, _state.low_mag_mode, 0)

def SetMag(mag):
    _state.mag = int(mag)
    _state.camera_pix_size = 12323.0 / _state.mag

def SetMagIndex(index):
    _state.mag_index = int(index)
    # Approximate mag from index
    _state.mag = int(100 * (1.2 ** index))
    _state.camera_pix_size = 12323.0 / _state.mag


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def SelectCamera(camera_id):
    _state.camera_id = int(camera_id)

def CameraProperties(camera=None, mag_index=None):
    """Returns: sizeX, sizeY, rotation, numCameras, pixelSize, cameraNumber."""
    return (_state.camera_dims[0], _state.camera_dims[1],
            _state.camera_rotation, 1, _state.camera_pix_size, _state.camera_id)

def ReportReadMode(area="R"):
    return _state.read_modes.get(area, 0)

def SetK2ReadMode(area, mode):
    _state.read_modes[area] = int(mode)

def ReportBinning(area="R"):
    return _state.binnings.get(area, 1)

def SetBinning(area, binning):
    _state.binnings[area] = int(binning)

def ReportExposure(area="R"):
    return (_state.exposures.get(area, 1.0), _state.drift_settling.get(area, 0.0))

def SetExposure(area, exp_time, drift_settling=None):
    _state.exposures[area] = float(exp_time)
    if drift_settling is not None:
        _state.drift_settling[area] = float(drift_settling)

def ReportCurrentPixelSize(area="R"):
    """Returns pixel size in nm for given area (before binning)."""
    return _state.camera_pix_size

def SetFrameBaseName(index, suffix_flag, digits, name):
    pass

def SetDoseFracParams(area, *args):
    pass


# ---------------------------------------------------------------------------
# Energy filter
# ---------------------------------------------------------------------------

def ReportEnergyFilter():
    if not _state.has_energy_filter:
        raise SEMerror("No energy filter available")
    return (_state.slit_width, _state.energy_loss, _state.slit_in)

def SetSlitIn(state):
    _state.slit_in = int(state)

def SetSlitWidth(width):
    _state.slit_width = float(width)

def SetEnergyLoss(loss):
    _state.energy_loss = float(loss)

def RefineZLP(interval_min=0, trial_or_preview=-1):
    pass


# ---------------------------------------------------------------------------
# Apertures
# ---------------------------------------------------------------------------

def ReportApertureSize(aperture_type):
    if aperture_type == 'C':
        return _state.c_aperture
    elif aperture_type == 'O':
        return _state.o_aperture
    return 0

def SetApertureSize(aperture_type, size):
    if aperture_type == 'C':
        _state.c_aperture = int(size)
    elif aperture_type == 'O':
        _state.o_aperture = int(size)


# ---------------------------------------------------------------------------
# Low Dose
# ---------------------------------------------------------------------------

def ReportLowDose():
    return (_state.low_dose_mode, _state.low_dose_area)

def SetLowDoseMode(mode):
    _state.low_dose_mode = int(mode)
    if not mode:
        _state.low_dose_area = None

def GoToLowDoseArea(area):
    _state.low_dose_area = str(area)

def GetLowDoseAreaParams(area, var_name):
    """Stores low dose params into a script variable."""
    ld_areas = {"V": 0, "F": 1, "T": 2, "R": 3, "S": 4}
    params = [
        ld_areas.get(area, 0),  # area number
        _state.mag_index,       # mag index
        _state.spot_size,       # spot size
        _state.percent_c2 / 100.0,  # intensity
        0.0,                    # axis offset
        0,                      # mode (0=TEM)
        _state.slit_in,         # filter slit in
        _state.slit_width,      # slit width
        _state.energy_loss,     # energy loss
        0,                      # zero loss flag
        0.0, 0.0,              # beam X/Y offset
        -999.0,                # alpha (JEOL)
        -999.0,                # diffraction focus
        _state.beam_tilt[0], _state.beam_tilt[1],  # beam tilt
        _state.probe_mode,     # probe mode
        0,                     # dark field mode
        0.0, 0.0,             # dark field tilt
        100.0,                 # dose modulation
    ]
    _state.script_vars[var_name] = params


# ---------------------------------------------------------------------------
# Imaging states
# ---------------------------------------------------------------------------

def GoToImagingState(state_name_or_number):
    name = str(state_name_or_number)
    if name in _state.imaging_states:
        index, low_dose, camera, mag_index = _state.imaging_states[name]
    elif name.isdigit():
        index = int(name)
        # Find by index
        for sname, props in _state.imaging_states.items():
            if props[0] == index:
                _, low_dose, camera, mag_index = props
                break
        else:
            low_dose, camera, mag_index = -1, 0, 10
    else:
        return 1  # error: not found

    _state.current_imaging_state = index
    if low_dose >= 0:
        _state.low_dose_mode = 1
        _state.low_dose_area = "R"
    else:
        _state.low_dose_mode = 0
        _state.low_dose_area = None
    return 0

def ImagingStateProperties(state_name_or_number):
    """Returns: error, index, lowDose, camera, magIndex, name."""
    name = str(state_name_or_number)
    # Look up by name
    if name in _state.imaging_states:
        index, low_dose, camera, mag_index = _state.imaging_states[name]
        return (0, index, low_dose, camera, mag_index, name)
    # Look up by index
    if name.isdigit():
        idx = int(name)
        for sname, props in _state.imaging_states.items():
            if props[0] == idx:
                return (0, idx, props[1], props[2], props[3], sname)
        # Also support numeric states beyond the registry (for getImagingStates enumeration)
        if idx > len(_state.imaging_states):
            return 1  # not found
        return (0, idx, -1, 0, 10, f"State{idx}")
    return 1  # not found


# ---------------------------------------------------------------------------
# Image acquisition
# ---------------------------------------------------------------------------

def _generate_dummy_image():
    """Generate a random noise image matching camera dims."""
    img = np.random.rand(_state.camera_dims[1], _state.camera_dims[0]).astype(np.float32) * 200 + 28
    _state.buffers["A"] = img
    _state.script_vars["imageTickTime"] = str(time.time())
    return img

def R():
    """Record."""
    return _generate_dummy_image()

def V():
    """View."""
    return _generate_dummy_image()

def F():
    """Focus."""
    return _generate_dummy_image()

def T():
    """Trial."""
    return _generate_dummy_image()

def S(section=None):
    """Save current buffer to file."""
    pass

def M(prescan=None):
    """Montage."""
    return _generate_dummy_image()

def Search():
    """Search."""
    return _generate_dummy_image()

def G(action=1, trial=-1):
    """Autofocus."""
    _state.defocus = _state.target_defocus + np.random.uniform(-0.1, 0.1)
    return _state.defocus


# ---------------------------------------------------------------------------
# Eucentricity & alignment
# ---------------------------------------------------------------------------

def Eucentricity(mode=1):
    pass

def AlignTo(buffer, avoid_is=0, *args):
    _state.align_shift = [np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0, 0,
                          np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]

def AlignBetweenMags(buffer, *args):
    _state.align_shift = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0, 0,
                          np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)]

def ReportAlignShift():
    return tuple(_state.align_shift)

def ClearAlignment(flag=0):
    _state.align_shift = [0, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Buffer operations
# ---------------------------------------------------------------------------

def Show(buffer_letter):
    pass

def Copy(src, dst):
    if src in _state.buffers:
        _state.buffers[dst] = _state.buffers[src].copy()

def LoadOtherMap(nav_id, buffer="A"):
    """Load map into buffer. In mock, generates noise."""
    _generate_dummy_image()

def PutImageInBuffer(img_array, buffer, sizeX, sizeY, source_buf="A"):
    _state.buffers[buffer] = np.array(img_array)
    _state.script_vars["imageTickTime"] = str(time.time())

def bufferImage(buffer_letter):
    """Returns image from buffer as numpy array."""
    if buffer_letter in _state.buffers:
        return _state.buffers[buffer_letter]
    return _generate_dummy_image()

def ImageProperties(buffer="A"):
    """Returns: sizeX, sizeY, binning, exposure, pixelSize."""
    if buffer in _state.buffers:
        img = _state.buffers[buffer]
        return (img.shape[1], img.shape[0], 1, 1.0, _state.camera_pix_size)
    return (_state.camera_dims[0], _state.camera_dims[1], 1, 1.0, _state.camera_pix_size)

def ImageConditions(buffer="A"):
    """Returns: dose, magnification, ..."""
    return (1.0, _state.mag)

def ReportMeanCounts(buffer="A"):
    if buffer in _state.buffers:
        return float(np.mean(_state.buffers[buffer]))
    return np.random.uniform(50, 200)

def ReduceImage(buffer, factor):
    """Reduce image by factor."""
    if buffer in _state.buffers:
        from skimage import transform as skt
        img = _state.buffers[buffer]
        reduced = skt.rescale(img, 1.0 / factor, anti_aliasing=True)
        _state.buffers["A"] = reduced.astype(np.float32)

def AddBufToStackWindow(buffer, *args):
    pass


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def BufImagePosToStagePos(buffer, flag, x, y):
    """Convert buffer image position to stage coordinates. Returns stage X, Y, Z."""
    # Simple linear approximation
    cx = _state.camera_dims[0] / 2
    cy = _state.camera_dims[1] / 2
    pix = _state.camera_pix_size / 1000.0  # nm/px -> µm/px
    sx = _state.stage_xyz[0] + (x - cx) * pix
    sy = _state.stage_xyz[1] + (y - cy) * pix
    return (sx, sy, _state.stage_xyz[2])

def StagePosToBufImagePos(buffer, flag, sx, sy):
    """Convert stage coordinates to buffer image position."""
    cx = _state.camera_dims[0] / 2
    cy = _state.camera_dims[1] / 2
    pix = _state.camera_pix_size / 1000.0
    x = cx + (sx - _state.stage_xyz[0]) / pix
    y = cy + (sy - _state.stage_xyz[1]) / pix
    return (x, y)


# ---------------------------------------------------------------------------
# Matrix transforms
# ---------------------------------------------------------------------------

def _identity_flat():
    return (1.0, 0.0, 0.0, 1.0)

def StageToSpecimenMatrix(mag_index=0):
    return (-1.0, 0.0, 0.0, -1.0)

def SpecimenToStageMatrix(mag_index=0):
    return (-1.0, 0.0, 0.0, -1.0)

def ISToSpecimenMatrix(mag_index=0):
    return _identity_flat()

def SpecimenToISMatrix(mag_index=0):
    return _identity_flat()

def CameraToSpecimenMatrix(mag_index=0):
    # Return a realistic rotation matrix based on camera rotation
    angle = np.radians(_state.camera_rotation)
    scale = _state.camera_pix_size / 1000.0  # nm -> µm
    return (scale * np.cos(angle), scale * np.sin(angle),
            -scale * np.sin(angle), scale * np.cos(angle))

def SpecimenToCameraMatrix(mag_index=0):
    angle = np.radians(-_state.camera_rotation)
    scale = 1000.0 / _state.camera_pix_size  # µm -> nm -> px
    return (scale * np.cos(angle), scale * np.sin(angle),
            -scale * np.sin(angle), scale * np.cos(angle))

def ReportComaVsISmatrix():
    return _identity_flat()

def ReportTiltAxisOffset():
    return (_state.properties.get("TiltAxisOffset", 0.0),)


# ---------------------------------------------------------------------------
# Navigator
# ---------------------------------------------------------------------------

def OpenNavigator(path=None):
    _state.nav_open = True
    if path:
        _state.nav_file = str(path)
        # Create the file if it doesn't exist
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("AdocVersion = 2.00\nLastSavedAs = " + str(p) + "\n\n")

def CloseNavigator():
    _state.nav_open = False

def ReadNavFile(path):
    _state.nav_file = str(path)
    _state.nav_open = True

def SaveNavigator(path=None):
    """Save navigator file. In mock mode, calls registered callback to write nav state to disk."""
    if path:
        _state.nav_file = str(path)
    if _state._save_nav_callback is not None:
        _state._save_nav_callback(_state.nav_file)
    else:
        # No callback registered — ensure file exists so readFromFile won't fail
        p = Path(_state.nav_file) if _state.nav_file else None
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text("AdocVersion = 2.00\nLastSavedAs = " + str(p) + "\n\n")

def RegisterNavSaveCallback(callback):
    """Register a callback(path) that writes the navigator state to disk.
    Called by SaveNavigator in mock mode. The Navigator class should register
    this to ensure its in-memory state is persisted before re-reading."""
    _state._save_nav_callback = callback

def ReportNavFile():
    return _state.nav_file

def ReportIfNavOpen():
    return 2 if _state.nav_open else 0  # 2 = open and saved, 1 = open but not saved, 0 = not open

def ReportNumNavAcquire():
    return (_state.nav_acquire_count, 0)

def NavIndexWithLabel(label):
    """Returns 1-based index of first nav item with matching label, or 0."""
    for i, item in enumerate(_state.nav_items):
        if item.get("label") == label:
            return i + 1
    return 0

def NavIndexWithNote(note_substr):
    for i, item in enumerate(_state.nav_items):
        if note_substr in item.get("note", ""):
            return i + 1
    return 0

def ReportNavItem(index=None):
    """Returns x, y, z, type, ..."""
    if index and 0 < index <= len(_state.nav_items):
        item = _state.nav_items[index - 1]
        return (item.get("x", 0), item.get("y", 0), item.get("z", 0), item.get("type", 0))
    return (0.0, 0.0, 0.0, 0)

def ReportOtherItem(index, key=None):
    if index and 0 < index <= len(_state.nav_items):
        item = _state.nav_items[index - 1]
        return item.get(key, "")
    return ""

def AddImagePosAsNavPoint(buffer, x, y):
    """Adds a navigator point from buffer image position. Returns 1-based index."""
    _state.nav_items.append({"label": str(len(_state.nav_items) + 1),
                             "x": float(x), "y": float(y), "z": 0.0,
                             "type": 0, "note": ""})
    return len(_state.nav_items)

def DeleteNavigatorItem(index):
    if 0 < index <= len(_state.nav_items):
        _state.nav_items.pop(index - 1)

def ChangeItemLabel(index, label):
    if 0 < index <= len(_state.nav_items):
        _state.nav_items[index - 1]["label"] = str(label)

def ChangeItemNote(index, note):
    if 0 < index <= len(_state.nav_items):
        _state.nav_items[index - 1]["note"] = str(note)

def ChangeItemColor(index, color):
    if 0 < index <= len(_state.nav_items):
        _state.nav_items[index - 1]["color"] = int(color)

def ChangeItemDraw(index, draw):
    if 0 < index <= len(_state.nav_items):
        _state.nav_items[index - 1]["draw"] = int(draw)

def SetItemAcquire(index, state=1):
    if 0 < index <= len(_state.nav_items):
        _state.nav_items[index - 1]["acquire"] = int(state)

def SetSelectedNavItem(index):
    pass

def ShiftItemsByMicrons(start, end, dx, dy):
    pass

def NewMap(flag=0, filename=""):
    """Returns 1-based nav index of new map."""
    _state.nav_items.append({"label": "Map", "x": _state.stage_xyz[0],
                             "y": _state.stage_xyz[1], "z": _state.stage_xyz[2],
                             "type": 2, "note": filename, "map_file": filename})
    return len(_state.nav_items)

def GetUniqueNavID():
    return len(_state.nav_items) + 1000

def RealignReloadedGrid(map_id, *args):
    pass


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def OpenNewFile(path):
    _state.open_files.append(str(path))
    _state.current_file_index = len(_state.open_files)

def OpenOldFile(path):
    if str(path) not in _state.open_files:
        _state.open_files.append(str(path))
    _state.current_file_index = _state.open_files.index(str(path)) + 1

def CloseFile():
    if _state.open_files and _state.current_file_index > 0:
        idx = _state.current_file_index - 1
        if idx < len(_state.open_files):
            _state.open_files.pop(idx)
        _state.current_file_index = max(0, len(_state.open_files))

def SwitchToFile(file_number):
    _state.current_file_index = int(file_number)

def ReadFile(section, buffer="A"):
    _generate_dummy_image()

def IsImageFileOpen(path):
    return 1 if str(path) in _state.open_files else 0

def ReportFileNumber():
    return len(_state.open_files)

def ReportFileZsize():
    return 1

def ReportNumOpenFiles():
    return len(_state.open_files)

def ReportOpenImageFile(file_number):
    idx = int(file_number) - 1
    if 0 <= idx < len(_state.open_files):
        return _state.open_files[idx]
    return ""

def AllowFileOverwrite(flag=1):
    _state.file_overwrite_allowed = bool(flag)

def SetFrameBaseName(*args):
    pass


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------

def SetupFullMontage(overlap, filename):
    pass

def SetupPolygonMontage(nav_id, flag, filename):
    pass

def OpenNewMontage(sizeX, sizeY, filename):
    OpenNewFile(filename)

def SetMontageParams(stage_move, overlapX, overlapY, sizeX, sizeY, skip_corr, binning):
    pass

def SetMontPanelParams(*args):
    pass

def ParamSetToUseForMontage(param_set):
    pass

def ReportNumMontagePieces():
    return 4  # dummy: always 4 pieces


# ---------------------------------------------------------------------------
# Autoloader
# ---------------------------------------------------------------------------

def ReportSlotStatus(slot):
    """Returns slot status. Simulates Thermo/FEI style: 1=occupied, 0=empty."""
    if 1 <= slot <= 12:
        return (1, f"G{slot:02d}")
    return (-1,)

def LoadCartridge(slot):
    pass

def UnloadCartridge():
    pass

def LongOperation(operation_type):
    pass


# ---------------------------------------------------------------------------
# Metadata & autodoc
# ---------------------------------------------------------------------------

def ReportMetadataValues(buffer, key):
    return ""

def AddToAutodoc(key, value):
    pass

def WriteAutodoc():
    pass


# ---------------------------------------------------------------------------
# CTF & autocorrelation
# ---------------------------------------------------------------------------

def Ctfplotter(buffer, low, high, *args):
    """Returns (defocus, fit_to, ...) - 6 values."""
    defocus = np.random.uniform(low, high)
    return (defocus, 0, 0, 0, 0, 0)

def AutoCorrPeakVectors(buffer, spacing_px, flag1, flag2):
    """Returns (peak, v1x, v1y, v2x, v2y, ...)."""
    spacing = spacing_px
    return (1.0, spacing, 0.0, 0.0, spacing, 0.0)


# ---------------------------------------------------------------------------
# Logging / UI
# ---------------------------------------------------------------------------

def SetStatusLine(line, text):
    pass

def ClearStatusLine(line=0):
    pass

def SetNextLogOutputStyle(style, color=0):
    pass

def EchoBreakLines(text_or_flag):
    """In mock mode, print to stdout so log messages are visible."""
    if isinstance(text_or_flag, str):
        print(text_or_flag)

def NoMessageBoxOnError(flag):
    pass

def OKBox(text):
    pass

def YesNoBox(text):
    """Always returns 1 (Yes) in mock mode."""
    return 1

def Pause(text):
    pass

def EnterString(var_name, prompt):
    _state.script_vars[var_name] = "MockInput"

def EnterOneNumber(prompt, default=0):
    return default


# ---------------------------------------------------------------------------
# Log operations
# ---------------------------------------------------------------------------

def SaveLogOpenNew(name=None):
    pass

def CloseLogOpenNew(flag=0):
    pass

def SaveLog(flag=0, name=None):
    pass


# ---------------------------------------------------------------------------
# Acquisition control
# ---------------------------------------------------------------------------

def RunScriptAfterNavAcquire(script_id):
    pass

def StartNavAcquireAtEnd():
    pass

def NavAcqAtEndUseParams(area):
    pass

def SetNavAcqAtEndParams(key, value):
    pass

def ReportIfNavAcquiring():
    return 0

def ReportNumNavAcquire():
    return (_state.nav_acquire_count, 0)

def ManageDewarsAndPumps(flag):
    pass


# ---------------------------------------------------------------------------
# Directory
# ---------------------------------------------------------------------------

def ReportDirectory():
    import os
    return os.getcwd()

def SetDirectory(path):
    pass

def UserSetDirectory(prompt=""):
    return str(Path.cwd())


# ---------------------------------------------------------------------------
# Version checking
# ---------------------------------------------------------------------------

def IsVersionAtLeast(version, date_string):
    return 1


# ---------------------------------------------------------------------------
# Utility: reset state for testing
# ---------------------------------------------------------------------------

def reset_mock_state():
    """Reset all mock state. Useful between test runs."""
    _state.reset()
