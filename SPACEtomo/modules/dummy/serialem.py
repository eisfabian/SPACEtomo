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

from SPACEtomo.modules.dummy import sim as _sim_module
from SPACEtomo.modules.dummy.sim import sim as _sim

# ---------------------------------------------------------------------------
# Custom exceptions matching real serialem module
# ---------------------------------------------------------------------------

class SEMmoduleError(Exception):
    """Raised for connection / module-level errors."""
    pass

class SEMerror(Exception):
    """Raised when a SerialEM script command fails (used with StartTry/EndTry)."""
    pass

class SEMexited(Exception):
    """Raised when the script exits (Exit() without a non-zero code, or user STOP)."""
    pass


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

class _State:
    """Mutable singleton holding all mock microscope state."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Stage tilt
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
        self.probe_mode = 1  # 1 = nanoprobe, 0 = microprobe

        # Camera (dims, pixel sizes, mags, Low Dose, imaging states, buffers, files and navigator: see sim.py)
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
        self.ld_defocus_offsets = {"V": 0.0, "F": 0.0, "T": 0.0, "R": 0.0, "S": 0.0}

        # Navigator acquisition, files and montages
        self.nav_acquire_count = 0
        self.file_overwrite_allowed = False
        self.montage_param_set = 1

        # Persistent variables
        self.persistent_vars = {}
        self.script_vars = {"imageTickTime": time.time()}

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

        # Simulated microscope (optics, buffers, files, montages, navigator)
        _sim.reset()

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

def SuppressReports(flag=1):
    pass

def ErrorsToLog(flag=1):
    pass

def SetNewFileType(file_type, montage_type=0):
    pass

def ProgramTimeStamps(startup=0, options=0):
    pass

def SaveSettings():
    pass

def Exit(exc=0, text=""):
    """Ends the script; like a script run from SerialEM, raises SEMexited unless exc is non-zero."""
    if text:
        print(text)
    if not exc:
        raise SEMexited("Normal exit")

def KeyBreak(key="B"):
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
    return tuple(_sim.stage)

def MoveStageTo(x, y, z=None):
    _sim.stage[:2] = [float(x), float(y)]
    if z is not None:
        _sim.stage[2] = float(z)

def MoveStage(dx, dy, dz=0):
    _sim.stage += [float(dx), float(dy), float(dz)]

def ReportTiltAngle():
    return _state.tilt_angle

def TiltTo(angle, reset_is=0, x=0.0, y=0.0):
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

def ResetImageShift(backlash=0, relax=0.025):
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
    """Illuminated area in units of 100 µm (SPACEtomo multiplies by 100)."""
    return _sim.illuminatedArea() / 100

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
    """Returns (magnification, low mag flag, EFTEM/STEM flag)."""
    return (_sim_module.magnification(_sim.mag_index), int(_sim.mag_index < 16), 0)

def SetMag(mag, stem_direct=0):
    _sim.mag_index = min(_sim_module.MAGS, key=lambda index: abs(_sim_module.magnification(index) - mag))

def SetMagIndex(index):
    if index in _sim_module.MAGS:
        _sim.mag_index = int(index)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def SelectCamera(camera_id):
    _state.camera_id = int(camera_id)

def CameraProperties(camera=None, mag_index=None):
    """Returns: sizeX, sizeY, rotation/flip, detector pixel (µm), pixel size on specimen (nm), camera number."""
    pix_size = _sim_module.MAGS[mag_index] if mag_index in _sim_module.MAGS else _sim.pix_size
    return (*_sim_module.CAMERA_DIMS, 0, _sim_module.DETECTOR_PIX_SIZE, pix_size, camera or 1)

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
    """Pixel size in nm of the Low Dose area (or current mag), including binning."""
    return _sim_module.MAGS[_sim.areaMagIndex(area)] * _state.binnings.get(area, 1)

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

def SetSlitIn(state=1):
    _state.slit_in = int(state)

def SetSlitWidth(width):
    _state.slit_width = float(width)

def SetEnergyLoss(loss):
    _state.energy_loss = float(loss)

def RefineZLP(interval_min=0, trial_or_preview=0, retry=0, allow_fail=0):
    pass


# ---------------------------------------------------------------------------
# Apertures
# ---------------------------------------------------------------------------

def ReportApertureSize(aperture):
    """Aperture indexes as in SerialEM: 1 condenser (C2), 2 objective; 0 = retracted/not present."""
    return {1: _state.c_aperture, 2: _state.o_aperture}.get(aperture, 0)

def SetApertureSize(aperture, size):
    if aperture == 1:
        _state.c_aperture = int(size)
    elif aperture == 2:
        _state.o_aperture = int(size)


# ---------------------------------------------------------------------------
# Low Dose
# ---------------------------------------------------------------------------

def ReportLowDose():
    """Returns (on/off, area index 0-4 for V F T R S or -1 if undefined), never letters."""
    area = "VFTRS".index(_sim.ld_area) if _sim.low_dose and _sim.ld_area in ("V", "F", "T", "R", "S") else -1
    return (int(_sim.low_dose), area)

def SetLowDoseMode(mode):
    _sim.low_dose = bool(mode)
    if not mode:
        _sim.ld_area = None

def GoToLowDoseArea(area):
    if not _sim.low_dose:
        raise SEMerror("Low Dose is not on")
    _sim.goToArea(str(area))

def GetLowDoseAreaParams(area, var_name):
    """Stores low dose params of the area as one text line (like WriteLowDoseToString)."""
    params = ["VFTRS".index(area), _sim_module.LD_AREAS[area][0], _state.spot_size, _state.percent_c2 / 100.0, 0.0, 0,
              _state.slit_in, _state.slit_width, _state.energy_loss, 0, 0.0, 0.0, -999.0, -999.0,
              _state.beam_tilt[0], _state.beam_tilt[1], _state.probe_mode, 0, 0.0, 0.0, 100.0]
    _state.script_vars[var_name] = ("%d %d %d %f %f %d %d %f %f %d %f %f %f %f %f %f %d %d %f %f %f" % tuple(params))


# ---------------------------------------------------------------------------
# Imaging states
# ---------------------------------------------------------------------------

def GoToImagingState(state_name_or_number):
    return _sim.goToImagingState(state_name_or_number)

def ImagingStateProperties(state_name_or_number):
    """Returns: error, number, type, camera, magIndex, name."""
    name = str(state_name_or_number)
    for state_name, (number, state_type, camera, mag_index) in _sim_module.IMAGING_STATES.items():
        if name in (state_name, str(number)):
            if state_type >= 0:
                mag_index = _sim_module.LD_AREAS["VFTRSR"[state_type]][0]
            return (0, number, state_type, camera, mag_index, state_name)
    return 3 if name.isdigit() else 1


# ---------------------------------------------------------------------------
# Image acquisition
# ---------------------------------------------------------------------------


def R():
    """Record."""
    _acquire("R")

def V():
    """View."""
    _acquire("V")

def F():
    """Focus."""
    _acquire("F")

def T():
    """Trial."""
    _acquire("T")

def S(buffer="A", file_number=0):
    """Saves buffer to the current (or given) open file."""
    file = _sim.files[file_number - 1] if file_number else _sim.current_file
    if file is None:
        raise SEMerror("There is no open image file to save to")
    _sim.saveToFile(buffer, file)

def M(prescan=None):
    """Acquires the montage that was set up."""
    if not _sim.montage:
        raise SEMerror("No montage was set up")
    _sim.collectMontage()
    _state.script_vars["imageTickTime"] = _sim.buffers["A"].tick

def Search():
    """Search."""
    _acquire("S")

def G(action=1, trial=-1):
    """Autofocus."""
    _state.defocus = _state.target_defocus + np.random.uniform(-0.1, 0.1)
    return _state.defocus


# ---------------------------------------------------------------------------
# Eucentricity & alignment
# ---------------------------------------------------------------------------

def Eucentricity(mode=2, use_view=0):
    pass

def AlignTo(buffer, avoid_is=0, *args):
    _sim.align(buffer)

def AlignBetweenMags(buffer, *args):
    """Aligns A to the reference; afterwards the image is in A again (buffers rolled, as assumed by Buffer.alignTo)."""
    _sim.align(buffer)
    _sim.putInBuffer("A", _sim.buffers["A"].copy())

def ReportAlignShift():
    return tuple(_sim.align_shift)

def ClearAlignment(flag=0):
    _sim.align_shift = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Buffer operations
# ---------------------------------------------------------------------------

def Show(buffer_letter):
    pass

def Copy(src, dst):
    _sim.putInBuffer(dst, _buffer(src).copy())

def LoadOtherMap(nav_id, buffer="A"):
    """Loads the map of nav item nav_id (1-based) into buffer."""
    if not 0 < nav_id <= len(_sim.nav_items) or _sim.nav_items[nav_id - 1].item_type != 2:
        raise SEMerror(f"Navigator item {nav_id} is not a map")
    _sim.loadMap(nav_id, buffer)
    _state.script_vars["imageTickTime"] = _sim.buffers[buffer].tick

def PutImageInBuffer(img_array, buffer, sizeX, sizeY, source_buf="A"):
    source = _sim.buffers.get(source_buf)
    image = _sim_module.ImageBuffer(np.reshape(np.asarray(img_array), (sizeY, sizeX)), source.pix_size if source else 1.0,
                                    source.stage if source else None, None, source.mag_index if source else None)
    _sim.putInBuffer(buffer, image)
    _state.script_vars["imageTickTime"] = image.tick

def bufferImage(buffer_letter):
    """Returns image from buffer as numpy array."""
    return _buffer(buffer_letter).img

def ImageProperties(buffer="A"):
    """Returns: sizeX, sizeY, binning, exposure, pixelSize (nm), mode."""
    image = _buffer(buffer)
    return (image.img.shape[1], image.img.shape[0], image.binning, 1.0, image.pix_size, 1)

def ImageConditions(buffer="A"):
    """Returns: dose, magnification, ..."""
    image = _buffer(buffer)
    return (1.0, _sim_module.magnification(image.mag_index) if image.mag_index in _sim_module.MAGS else 0)

def ReportMeanCounts(buffer="A", option=0):
    return float(np.mean(_buffer(buffer).img))

def ReduceImage(buffer, factor):
    """Reduces image by factor into A."""
    source = _buffer(buffer)
    from skimage import transform as skt
    reduced = skt.rescale(source.img, 1.0 / factor, anti_aliasing=True, preserve_range=True)
    _sim.putInBuffer("A", _sim_module.ImageBuffer(reduced, source.pix_size * factor, source.stage, source.s2px / factor, source.mag_index))

def AddBufToStackWindow(buffer, *args):
    pass


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def BufImagePosToStagePos(buffer, flag, x, y):
    """Buffer image position to stage coordinates. Returns stage X, Y, Z."""
    image = _buffer(buffer)
    if image.stage is None:
        raise SEMerror(f"Image in buffer {buffer} has no stage position")
    return (*image.px2stage(x, y), image.stage[2])

def StagePosToBufImagePos(buffer, flag, sx, sy):
    """Stage coordinates to buffer image position."""
    image = _buffer(buffer)
    if image.stage is None:
        raise SEMerror(f"Image in buffer {buffer} has no stage position")
    return tuple(image.stage2px(sx, sy))


# ---------------------------------------------------------------------------
# Matrix transforms
# ---------------------------------------------------------------------------

def _identity_flat():
    return (1.0, 0.0, 0.0, 1.0)

def StageToSpecimenMatrix(mag_index=0):
    return (1.0, 0.0, 0.0, 1.0)

def SpecimenToStageMatrix(mag_index=0):
    return (1.0, 0.0, 0.0, 1.0)

def ISToSpecimenMatrix(mag_index=0):
    return _identity_flat()

def SpecimenToISMatrix(mag_index=0):
    return _identity_flat()

def CameraToSpecimenMatrix(mag_index=0):
    pix_size = _sim_module.MAGS.get(mag_index, _sim.pix_size) / 1000
    return (pix_size, 0.0, 0.0, -pix_size)

def SpecimenToCameraMatrix(mag_index=0):
    pix_size = _sim_module.MAGS.get(mag_index, _sim.pix_size) / 1000
    return (1 / pix_size, 0.0, 0.0, -1 / pix_size)

def ReportComaVsISmatrix():
    return _identity_flat()

def ReportTiltAxisOffset():
    return (_state.properties.get("TiltAxisOffset", 0.0),)


# ---------------------------------------------------------------------------
# Navigator
# ---------------------------------------------------------------------------

def OpenNavigator(path=None):
    _sim.nav_open = True
    _sim.nav_items, _sim.current_nav_index = [], 0
    if path:
        _sim.nav_file = _sim.path(path)
        _sim.writeNav()

def CloseNavigator():
    _sim.nav_open = False

def ReadNavFile(path):
    _sim.readNav(_sim.path(path))

def SaveNavigator(path=None):
    """Writes the mock's navigator items to the nav file."""
    if path:
        _sim.nav_file = _sim.path(path)
    if _sim.nav_file is None:
        raise SEMerror("Navigator has no file to save to")
    _sim.writeNav()


def ReportNavFile(parts=0):
    if _sim.nav_file is None:
        raise SEMerror("Navigator is not open with a file defined")
    if not parts:
        return str(_sim.nav_file)
    return (str(_sim.nav_file.with_suffix("")), _sim.nav_file.suffix, str(_sim.nav_file.parent) + "/", _sim.nav_file.stem)

def ReportIfNavOpen():
    """2 = open with file, 1 = open without file, 0 = not open."""
    if not _sim.nav_open:
        return 0
    return 2 if _sim.nav_file else 1

def ReportNumNavAcquire():
    return (_state.nav_acquire_count, 0)

def NavIndexWithLabel(label):
    return next((index + 1 for index, item in enumerate(_sim.nav_items) if item.label == label), 0)

def NavIndexWithNote(note_substr):
    return next((index + 1 for index, item in enumerate(_sim.nav_items) if note_substr in item.note), 0)

def _nav_item_report(index):
    if not index:
        raise SEMerror("There is no current Navigator item")
    item = _sim.nav_items[index - 1]
    return (index, *item.stage[:3], item.item_type)

def ReportNavItem():
    """Reports the current item: (index, x, y, z, type)."""
    return _nav_item_report(_sim.current_nav_index)

def ReportOtherItem(index):
    """Reports item index (negative counts from the end): (index, x, y, z, type)."""
    if index < 0:
        index = len(_sim.nav_items) + index + 1
    if not 0 < index <= len(_sim.nav_items):
        raise SEMerror("Index is out of range in statement: ReportOtherItem")
    return _nav_item_report(index)

def AddImagePosAsNavPoint(buffer, x, y, z=-999.0, group_id=0, no_redraw=0):
    """Adds a navigator point at the buffer image position. Returns 1-based index."""
    from SPACEtomo.modules.nav import NavItem
    image = _buffer(buffer)
    stage = image.px2stage(x, y)
    item = NavItem(len(_sim.nav_items) + 1, label=str(len(_sim.nav_items) + 1))
    item.createPoint([stage[0], stage[1], float(z) if z > -990 else image.stage[2]], _sim.newMapId(), group_id=int(group_id) or None)
    return _sim.addItem(item)

def DeleteNavigatorItem(index):
    _item(index)
    _sim.nav_items.pop(index - 1)
    for i, item in enumerate(_sim.nav_items):
        item.nav_index = i + 1

def ChangeItemLabel(index, label):
    _item(index).label = str(label)

def ChangeItemNote(index, note=""):
    _item(index).note = str(note)

def ChangeItemColor(index, color):
    _item(index).entries["Color"] = [str(int(color))]

def ChangeItemDraw(index, draw=None):
    item = _item(index)
    current = int(item.entries.get("Draw", ["1"])[0])
    item.entries["Draw"] = [str(int(not current) if draw is None else int(bool(draw)))]

def SetItemAcquire(index=0, state=1):
    _item(index or _sim.current_nav_index).entries["Acquire"] = [str(int(bool(state)))]

def SetSelectedNavItem(index, no_redraw=0):
    _item(index)
    _sim.current_nav_index = int(index)

def ShiftItemsByMicrons(dx, dy, registration=-1, as_marker_shift=0):
    for item in _sim.nav_items:
        item.stage[:2] += [float(dx), float(dy)]
        for key, delta in (("PtsX", float(dx)), ("PtsY", float(dy))):
            if key in item.entries:
                item.entries[key] = [str(float(val) + delta) for val in item.entries[key]]

def NewMap(flag=0, filename=""):
    """Makes a map from the image in A (or the montage just acquired). Returns 1-based nav index."""
    if "A" not in _sim.buffers:
        raise SEMerror("There is no image in buffer A to make a map from")
    return _sim.newMap(str(filename))

def GetUniqueNavID():
    return _sim.newMapId()

def RealignReloadedGrid(map_id, *args):
    pass


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def OpenNewFile(path):
    file = _sim.path(path)
    if file.exists() and not _state.file_overwrite_allowed:
        file.unlink()
    _sim.files.append(file)
    _sim.current_file = file

def OpenOldFile(path):
    file = _sim.path(path)
    if not file.exists():
        raise SEMerror(f"File {file} does not exist")
    if file not in _sim.files:
        _sim.files.append(file)
    _sim.current_file = file

def CloseFile(stop_if_none=""):
    if _sim.current_file is None:
        if stop_if_none:
            raise SEMerror("There is no open image file to close")
        return
    _sim.files.remove(_sim.current_file)
    _sim.current_file = _sim.files[-1] if _sim.files else None

def SwitchToFile(file_number):
    _sim.current_file = _sim.files[int(file_number) - 1]

def ReadFile(section, buffer="A"):
    """Reads section of the current file into buffer (no stage position known)."""
    if _sim.current_file is None:
        raise SEMerror("There is no open image file to read from")
    img, pix_size = _sim.readFromFile(_sim.current_file, int(section))
    _sim.buffers[buffer] = _sim_module.ImageBuffer(img, pix_size)
    _state.script_vars["imageTickTime"] = _sim.buffers[buffer].tick

def IsImageFileOpen(path):
    return int(_sim.path(path) in _sim.files)

def ReportFileNumber():
    return _sim.files.index(_sim.current_file) + 1 if _sim.current_file else 0

def ReportFileZsize():
    return 1

def ReportNumOpenFiles():
    return len(_sim.files)

def ReportOpenImageFile(file_number, parts=0):
    index = int(file_number) - 1
    if not 0 <= index < len(_sim.files):
        raise SEMerror(f"There is no open file number {file_number}")
    return str(_sim.files[index])

def AllowFileOverwrite(flag=1):
    _state.file_overwrite_allowed = bool(flag)

def SetFrameBaseName(*args):
    pass


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------

def SetupFullMontage(overlap, filename):
    """Sets up a montage of the whole grid at the current mag and opens the montage file."""
    _sim.montage = {"kind": "full", "file": _sim.path(filename), "overlap": float(overlap)}

def SetupPolygonMontage(nav_id, flag, filename):
    """Sets up a montage of the polygon item nav_id (1-based) and opens the montage file."""
    if _item(nav_id).item_type != 1:
        raise SEMerror(f"Navigator item {nav_id} is not a polygon")
    _sim.montage = {"kind": "polygon", "poly_index": int(nav_id), "file": _sim.path(filename), "view": _state.montage_param_set == 2}

def OpenNewMontage(sizeX, sizeY, filename):
    OpenNewFile(filename)

def SetMontageParams(stage_move, overlapX=-1, overlapY=-1, sizeX=-1, sizeY=-1, skip_corr=-1, binning=-1, max_is=-1, flexible=-1):
    pass

def SetMontPanelParams(*args):
    pass

def ParamSetToUseForMontage(param_set, defer=0):
    _state.montage_param_set = int(param_set)

def ReportNumMontagePieces():
    """Number of camera fields needed to cover the set up montage area."""
    if not _sim.montage:
        raise SEMerror("No montage was set up")
    if _sim.montage["kind"] == "full":
        extent = np.array([_sim.getSpecimen().region] * 2, dtype=float)
        pix_size = _sim.pix_size
    else:
        item = _item(_sim.montage["poly_index"])
        pts = np.array([[float(x) for x in item.entries["PtsX"]], [float(y) for y in item.entries["PtsY"]]])
        extent, pix_size = pts.max(axis=1) - pts.min(axis=1), _sim_module.MAGS[_sim_module.LD_AREAS["V"][0]]
    field = np.array(_sim_module.CAMERA_DIMS) * pix_size / 1000 * 0.9
    return int(np.prod(np.ceil(extent / field)))


# ---------------------------------------------------------------------------
# Autoloader
# ---------------------------------------------------------------------------

def ReportSlotStatus(slot):
    """Returns slot status. Simulates Thermo/FEI style: 1=occupied, 0=empty."""
    if 1 <= slot <= 12:
        return (1, f"G{slot:02d}")
    return (-1,)

def LoadCartridge(slot):
    """Loads a new grid: a new synthetic specimen is created on the next image."""
    if slot != _sim.loaded_slot:
        _sim.loaded_slot = int(slot)
        _sim.specimen = None
        _sim.maps = []

def UnloadCartridge():
    pass

def LongOperation(operation_type, *operations_and_intervals):
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

def AutoCorrPeakVectors(buffer, spacing_px, skip_trim=0, skip_display=0):
    """Returns (spacing, v1x, v1y, v2x, v2y, ...)."""
    spacing = spacing_px
    return (1.0, spacing, 0.0, 0.0, spacing, 0.0)


# ---------------------------------------------------------------------------
# Logging / UI
# ---------------------------------------------------------------------------

def SetStatusLine(line, text):
    pass

def ClearStatusLine(line=0):
    pass

def SetNextLogOutputStyle(style, color=0, green=0, blue=0):
    pass

def EchoBreakLines(text=""):
    """In mock mode, print to stdout so log messages are visible."""
    print(text)

def NoMessageBoxOnError(flag=1):
    pass

def OKBox(text):
    pass

def YesNoBox(text):
    """Always returns 1 (Yes) in mock mode."""
    return 1

def Pause(text=""):
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

def NavAcqAtEndUseParams(kind, file=""):
    pass

def SetNavAcqAtEndParams(option, value, *more_options_and_values):
    pass

def ReportIfNavAcquiring():
    return 0

def ReportNumNavAcquire():
    return (_state.nav_acquire_count, 0)

def ManageDewarsAndPumps(checks=0):
    pass


# ---------------------------------------------------------------------------
# Directory
# ---------------------------------------------------------------------------

def ReportDirectory():
    return str(_sim.directory)

def SetDirectory(path):
    _sim.directory = Path(path)

def UserSetDirectory(prompt=""):
    return str(Path.cwd())


# ---------------------------------------------------------------------------
# Simulated microscope helpers
# ---------------------------------------------------------------------------

def ReportMagIndex():
    return _sim.mag_index

def _acquire(area):
    _sim.acquire(area)
    _state.script_vars["imageTickTime"] = _sim.buffers["A"].tick

def L():
    """Search (Low Dose S area)."""
    _acquire("S")

def _buffer(letter):
    try:
        return _sim.getBuffer(letter)
    except KeyError:
        raise SEMerror(f"Buffer {letter} is empty") from None

def _item(index):
    if not 0 < index <= len(_sim.nav_items):
        raise SEMerror(f"Navigator item index {index} is out of range")
    return _sim.nav_items[index - 1]


# ---------------------------------------------------------------------------
# Version checking
# ---------------------------------------------------------------------------

def IsVersionAtLeast(version, date_string=""):
    """True if both the version and the build date are at least the given ones (as CMacCmd::IsVersionAtLeast)."""
    emulated = _COMMANDS[_version] if _COMMANDS else {"version_code": "40200", "build_date": "20250228"}
    date_ok = not date_string or not emulated["build_date"] or int(date_string) <= int(emulated["build_date"])
    return int(int(version) <= int(emulated["version_code"]) and date_ok)

def ReturnAllValuesAsTuples(flag=1):
    global _always_tuple
    _always_tuple = bool(flag)


# ---------------------------------------------------------------------------
# Utility: reset state for testing
# ---------------------------------------------------------------------------

def reset_mock_state():
    """Reset all mock state. Useful between test runs."""
    global _always_tuple
    _state.reset()
    _always_tuple = False


# ---------------------------------------------------------------------------
# SerialEM version emulation
# ---------------------------------------------------------------------------
# The real module is generated from SerialEM's MacroMasterList.h, so it differs between versions. Each command
# parses its arguments with PyArg_ParseTuple (keys I/D/S = int/number/string, lowercase = optional; wrong types
# or counts raise TypeError; SerialEM's own minimum argument count is not checked for Python calls), and reported
# values come back as floats or strings: None for nothing, a single value, or a tuple. set_version() wraps every mocked command
# accordingly and removes the commands the emulated version does not have (AttributeError, like the real module).
# The command table is exported from the SerialEM sources by .claude/skills/serialem/scripts/export_mock_table.py.

import json as _json
import functools as _functools
import os as _os

try:
    _COMMANDS = _json.loads(Path(__file__).with_name("sem_commands.json").read_text())["versions"]
except OSError:     # table not shipped: commands are not checked
    _COMMANDS = {}
_ALL_COMMANDS = set().union(*(v["commands"] for v in _COMMANDS.values()))
_RAW = {name: obj for name, obj in globals().items() if name in _ALL_COMMANDS and callable(obj)}
_version = None
_always_tuple = False


def _check_args(name, keys, args):
    """Raises TypeError like PyArg_ParseTuple with the command's argument keys."""

    n_required = next((i for i, key in enumerate(keys) if key.islower()), len(keys))
    if not n_required <= len(args) <= len(keys):
        if n_required == len(keys):
            expected = f"exactly {len(keys)}" if keys else "no"
        else:
            expected = f"from {n_required} to {len(keys)}"
        raise TypeError(f"{name}() takes {expected} arguments ({len(args)} given)")

    for i, (key, arg) in enumerate(zip(keys.upper(), args), start=1):
        if key == "I":
            # PyArg 'i' accepts anything with __index__ (int, bool, numpy ints), but no floats or strings
            if isinstance(arg, float) or not hasattr(type(arg), "__index__"):
                raise TypeError(f"{name}() argument {i} must be int, not {type(arg).__name__}")
        elif key == "D":
            if isinstance(arg, (str, bytes)) or not (hasattr(type(arg), "__float__") or hasattr(type(arg), "__index__")):
                raise TypeError(f"{name}() argument {i} must be real number, not {type(arg).__name__}")
        elif not isinstance(arg, str):
            raise TypeError(f"{name}() argument {i} must be str, not {type(arg).__name__}")


def _report(name, value):
    """Converts a mock return value into what the real module returns for the reported values."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    values = list(value) if isinstance(value, (tuple, list)) else [value]
    if not values:
        return None
    try:
        values = tuple(val if isinstance(val, str) else float(val) for val in values)
    except (TypeError, ValueError):
        raise TypeError(f"Mock {name}() returned {value!r}; SerialEM only reports numbers and strings") from None
    if len(values) == 1 and (isinstance(values[0], str) or not _always_tuple):
        return values[0]
    return values


def _emulate(name, func, min_args, keys):
    @_functools.wraps(func)
    def command(*args, **kwargs):
        if kwargs:
            raise TypeError(f"{name}() takes no keyword arguments")
        _check_args(name, keys, args)
        return _report(name, func(*args))
    command.__wrapped_command__ = func
    return command


def set_version(version):
    """Emulates the serialem module of the given SerialEM version ("4.2" or "4.3")."""

    global _version
    if not _COMMANDS:
        return
    if version not in _COMMANDS:
        raise ValueError(f"Unknown SerialEM version {version!r}, choose from {sorted(_COMMANDS)}")
    commands = _COMMANDS[version]["commands"]
    module_globals = globals()
    for name, func in _RAW.items():
        if name in commands:
            module_globals[name] = _emulate(name, func, *commands[name])
        else:
            module_globals.pop(name, None)
    _version = version


def get_version():
    """Returns the emulated SerialEM version."""
    return _version


def versions():
    """Returns all SerialEM versions the mock can emulate."""
    return sorted(_COMMANDS)


set_version(_os.environ.get("SPACETOMO_SEM_VERSION", "4.2"))
