"""Tests for SPACEtomo.modules.scope against the mock SerialEM module."""

import pytest
import serialem as sem

from SPACEtomo.modules import utils_sem as usem
from SPACEtomo.modules.scope import Microscope


@pytest.mark.parametrize("state, low_dose, area", [("MM1", True, "V"), ("WG", False, None)])
def test_changeImagingState_keeps_low_dose_area_as_letter(state, low_dose, area):
    """ReportLowDose reports the area as a number (0-4 for V F T R S, -1 if undefined), never as a letter."""

    microscope = Microscope()
    microscope.changeImagingState(state)

    assert microscope.low_dose is low_dose
    assert microscope.low_dose_area == area


def test_getLowDoseParams_parses_serialem_text(sem_version):
    """GetLowDoseAreaParams stores the parameters as one text line (SerialEM 4.3+)."""

    params = usem.getLowDoseParams("V")

    if sem_version == "4.2":
        assert params is None
        return
    assert params["low_dose_area"] == 0
    assert params["mag_index"] == sem._sim_module.LD_AREAS["V"][0]
    assert params["spot_size"] == sem._state.spot_size


# --- Imaging states and autoloader (values from SerialEM are floats) ---

def test_getImagingStates_with_serialem_floats():
    """ImagingStateProperties reports camera and mag index as floats; CameraProperties only accepts ints."""

    states = usem.getImagingStates()

    assert [state[1] for state in states] == ["WG", "IM", "MM1"]
    assert all(state[5] > 0 for state in states)


def test_getImagingStates_skips_pixel_size_of_unavailable_camera(monkeypatch):
    """Camera 0 means the state's camera is not available; SerialEM cannot report its pixel size."""

    real = sem.ImagingStateProperties
    monkeypatch.setattr(sem, "ImagingStateProperties", lambda name: (0.0, 1.0, -1.0, 0.0, 15.0, "Other") if name == "1" else real(name))

    states = usem.getImagingStates()

    assert states[0][1] == "Other" and states[0][5] is None


def test_checkAutoloader_jeol_cartridge_without_name(monkeypatch):
    """JEOL inventory reports cartridge ID, location, slot, type, angle, name (numbers as floats)."""

    inventory = {1: (101.0, 3.0, 1.0, 0.0, 0.0, "!NONAME!"), 2: (102.0, 1.0, 2.0, 0.0, 0.0, "MyGrid")}
    monkeypatch.setattr(sem, "ReportSlotStatus", lambda index: inventory.get(index, -1.0))

    microscope = Microscope()
    microscope.checkAutoloader()

    assert microscope.autoloader == {1: "Grid101", 2: "MyGrid"}
    assert microscope.loaded_grid == 1
