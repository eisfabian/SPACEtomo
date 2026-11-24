"""Polygon montage setup for the MM map (GitHub issue #10: SetupPolygonMontage failing at high View magnification)."""

import numpy as np
import pytest
import serialem as sem

from SPACEtomo import config
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.scope import Microscope


class Stop(Exception):
    pass


@pytest.fixture
def polygon(tmp_path):
    sem.SetDirectory(str(tmp_path))
    nav = Navigator(tmp_path / "G01.nav")
    Buffer.nav = nav
    microscope = Microscope()
    microscope.changeImagingState("MM1", low_dose_expected=True)
    poly_id = nav.newPolygon([-10, -10, 10, 10, -10], [-5, 5, 5, -5, -5], 0.0, label="FP01")
    return microscope, poly_id


def test_polygon_fit_uses_configured_overlap(polygon, tmp_path, monkeypatch):
    """The fit decides the number of pieces; with the dialog's overlap factor (#O = 0) it does not match the overlap
    SPACEtomo sets afterwards, so the montage could cover less than the polygon."""

    microscope, poly_id = polygon
    calls = []

    def setup(*args):
        calls.append(args)
        raise Stop
    monkeypatch.setattr(sem, "SetupPolygonMontage", setup)

    with pytest.raises(Stop):
        microscope.collectPolygonMontage(poly_id, tmp_path / "G01_FP01.mrc", config.MM_montage_overlap)

    # Fit overlap (factor * long camera side) equals the acquired overlap (MM_montage_overlap * short camera side)
    assert calls[0][1] * max(microscope.camera_dims) == pytest.approx(int(config.MM_montage_overlap * min(microscope.camera_dims)))


def test_failed_setup_explains_minimum_overlap(polygon, tmp_path, monkeypatch, capsys):
    """At high View mag, the minimum overlap in microns from the Montage Setup dialog can exceed half a View image."""

    microscope, poly_id = polygon

    def setup(*args):
        raise sem.SEMerror("An error occurred setting up a polygon montage")
    monkeypatch.setattr(sem, "SetupPolygonMontage", setup)

    with pytest.raises(sem.SEMerror):
        microscope.collectPolygonMontage(poly_id, tmp_path / "G01_FP01.mrc", config.MM_montage_overlap)

    out = capsys.readouterr().out
    assert "ERROR:" in out and "minimum overlap" in out.lower() and "Montage Setup" in out
    assert sem.ReportIfNavOpen() and _try_depth() == 0      # StartTry/EndTry balanced


def _try_depth():
    from SPACEtomo.modules.dummy import serialem as mock
    return mock._state._try_depth


@pytest.mark.parametrize("max_block_is, stage_move", [(0, -1), (10, 2)])
def test_image_shift_in_blocks(polygon, tmp_path, monkeypatch, max_block_is, stage_move):
    """0 keeps the stage/image shift choice of the Montage Setup dialog, a positive value uses image shift in blocks up to that many microns."""

    microscope, poly_id = polygon
    calls = []

    def montage_params(*args):
        calls.append(args)
        raise Stop
    monkeypatch.setattr(sem, "SetMontageParams", montage_params)

    with pytest.raises(Stop):
        microscope.collectPolygonMontage(poly_id, tmp_path / "G01_FP01.mrc", config.MM_montage_overlap, max_block_is=max_block_is)

    assert calls[0][0] == stage_move
    if max_block_is:
        assert calls[0][7] == pytest.approx(max_block_is)
    else:
        assert len(calls[0]) < 8 or calls[0][7] == -1


def test_mm_map_uses_configured_block_image_shift(tmp_path, monkeypatch):
    from SPACEtomo.modules.mma import MMMAcquisition

    sem.SetDirectory(str(tmp_path))
    nav = Navigator(tmp_path / "G01.nav")
    Buffer.nav = nav
    microscope = Microscope()
    microscope.autoloader, microscope.loaded_grid = {1: "G01"}, 1
    monkeypatch.setattr(config, "MM_montage_IS_blocks", 7.5, raising=False)
    calls = []

    def collect(*args, **kwargs):
        calls.append(kwargs)
        raise Stop
    monkeypatch.setattr(microscope, "collectPolygonMontage", collect)

    with pytest.raises(Stop):
        MMMAcquisition(tmp_path, tmp_path / "G01_FP01.mrc", 0, microscope, nav).collectMap()

    assert calls[0]["max_block_is"] == 7.5
