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
    assert params["mag_index"] == sem._state.mag_index
    assert params["spot_size"] == sem._state.spot_size
