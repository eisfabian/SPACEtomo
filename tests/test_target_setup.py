"""PACEArea.prepareForAcquisition reports when an area cannot be set up, so no targets file is written for it."""

from types import SimpleNamespace

import numpy as np
import pytest
import serialem as sem

from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.tgt import PACEArea


@pytest.fixture
def nav(tmp_path):
    sem.SetDirectory(str(tmp_path))
    nav = Navigator(tmp_path / "G01.nav")
    Buffer.nav = nav
    nav.newPoint(np.zeros(3), label="other", note="G01_L01.mrc")
    return nav


def _area(nav_id, map_pix_size, view_mag_index):
    area = PACEArea.__new__(PACEArea)
    area.nav_id = nav_id
    area.points = np.array([[100.0, 100.0]])
    area.geo_points = np.zeros((0, 2))
    area.pix_size = map_pix_size
    area.map_buffer = SimpleNamespace(pix_size=map_pix_size)
    area.imaging_params = SimpleNamespace(view_pix_size=50.0, cam_dims=np.array([1024, 1024]), view_mag_index=view_mag_index)
    return area


def test_missing_template_map_fails(nav, tmp_path):
    """Map of the targets is neither known from its buffer nor found by note."""

    assert _area(None, 50.0, 10).prepareForAcquisition(tmp_path / "G01_L02.mrc", nav) is False
    assert len(nav) == 1


def test_non_view_montage_without_view_mag_index_fails(nav, tmp_path):
    """Search montage on SerialEM 4.2 cannot be converted to View virtual maps."""

    assert _area(0, 100.0, None).prepareForAcquisition(tmp_path / "G01_L01.mrc", nav) is False
    assert len(nav) == 1
