"""Tests for the synthetic grid and the simulated microscope behind the mock serialem module."""

import numpy as np
import pytest

import serialem as sem

from SPACEtomo.modules.dummy.specimen import CATS, INTENSITY, Specimen, stageToPixelMatrix, toCounts


# --- Synthetic grid ---

def test_specimen_is_deterministic():
    first, second = Specimen(seed=7), Specimen(seed=7)

    assert [lamella.toDict() for lamella in first.lamellae] == [lamella.toDict() for lamella in second.lamellae]
    assert np.array_equal(first.render((0, 0), (300, 300), 4000), second.render((0, 0), (300, 300), 4000))


@pytest.mark.parametrize("vertical", [True, False])
def test_lamella_renders_at_its_world_position(vertical):
    """Slab at the lamella center, milled trenches before and after it along the long axis (column -> +x, row -> -y)."""

    specimen = Specimen(seed=3, n_lamellae=1, vertical=vertical)
    lamella = specimen.lamellae[0]
    view_center = lamella.center + [5.0, -8.0]
    img, seg = specimen.render(view_center, (600, 500), 100.0, seg=True)

    col, row = specimen._toPx(view_center, img.shape, 100.0, *lamella.center)
    assert seg[int(row), int(col)] == CATS["background"]
    assert img[int(row), int(col)] == pytest.approx(INTENSITY["background"])
    trench = lamella.center + (np.array([0, 0.75 * lamella.size[1]]) if vertical else np.array([0.75 * lamella.size[1], 0]))
    col, row = specimen._toPx(view_center, img.shape, 100.0, *trench)
    assert seg[int(row), int(col)] == CATS["white"]


def test_lamella_detail_contains_target_classes():
    specimen = Specimen(seed=3, n_lamellae=1)
    _, seg = specimen.render(specimen.lamellae[0].center, (1000, 1000), 30.0, detail=True, seg=True)

    for name in ("cell", "mitos", "coating", "white"):
        assert np.any(seg == CATS[name]), name


# --- Simulated microscope behind the mock serialem module ---

def _record_at(stage, state="IM"):
    sem.GoToImagingState(state)
    sem.MoveStageTo(*stage)
    sem.R()


def test_buffer_positions_map_to_stage_and_back():
    _record_at((100.0, -50.0))
    width, height, _, _, pix_size, _ = sem.ImageProperties("A")

    assert np.allclose(sem.BufImagePosToStagePos("A", 1, width / 2, height / 2)[:2], [100.0, -50.0])
    right_up = sem.BufImagePosToStagePos("A", 1, width / 2 + 100, height / 2 - 100)[:2]
    assert np.allclose(right_up, [100.0 + 100 * pix_size / 1000, -50.0 + 100 * pix_size / 1000])     # column +x, row -y
    assert np.allclose(sem.StagePosToBufImagePos("A", 1, *right_up), [width / 2 + 100, height / 2 - 100])


def test_new_images_roll_buffers():
    _record_at((0.0, 0.0))
    first = sem.bufferImage("A").copy()
    _record_at((50.0, 0.0))

    assert np.array_equal(sem.bufferImage("B"), first)


@pytest.mark.parametrize("true_shift", [(2.0, -3.0), (-4.0, 1.5)])
def test_AlignBetweenMags_reports_specimen_shift(true_shift):
    """Shift of the image content in A relative to a lower mag reference, reported in nm (x right, y up)."""

    from SPACEtomo.modules.dummy.sim import ImageBuffer, sim
    _record_at((0.0, 0.0))
    center = sim.getSpecimen().lamellae[0].center
    sim.buffers["O"] = ImageBuffer(toCounts(sim.specimen.render(center, (2048, 2048), 400.0)), 400.0, [*center, 0])
    sim.mag_index = 15
    sim.stage[:2] = center + true_shift - sim.specimen.offsets["IM"]     # IM images show the world with the IM offset
    sem.R()

    sem.AlignBetweenMags("O", -1, -1, -2, 0, -1, 1)

    assert np.allclose(np.array(sem.ReportAlignShift()[4:6]) / 1000, true_shift, atol=0.5)


def test_navigator_is_owned_by_the_mock(tmp_path):
    """Items made by SerialEM commands survive SaveNavigator/ReadNavFile; maps keep their geometry."""

    nav_file = tmp_path / "grid.nav"
    sem.OpenNavigator(str(nav_file))
    sem.SetDirectory(str(tmp_path))
    _record_at((20.0, 30.0))
    sem.OpenNewFile("map.mrc")
    sem.S()
    map_index = int(sem.NewMap(0, "map.mrc"))
    sem.CloseFile()
    point_index = int(sem.AddImagePosAsNavPoint("A", 0, 0))
    sem.SaveNavigator()

    sem.CloseNavigator()
    sem.ReadNavFile(str(nav_file))
    sem.LoadOtherMap(map_index, "B")

    assert "[Item = 1]" in nav_file.read_text() and "MapFile" in nav_file.read_text()
    assert np.allclose(sem.ReportOtherItem(map_index)[1:3], [20.0, 30.0])
    width, height = sem.ImageProperties("B")[:2]
    assert np.allclose(sem.BufImagePosToStagePos("B", 1, width / 2, height / 2)[:2], [20.0, 30.0], atol=0.01)
    assert np.allclose(sem.ReportOtherItem(point_index)[1:3], sem.BufImagePosToStagePos("B", 1, 0, 0)[:2], atol=0.01)


def test_stage_to_pixel_matrix_flips_y():
    assert np.allclose(stageToPixelMatrix(1000.0), [[1, 0], [0, -1]])


def test_dummy_refuses_real_serialem(monkeypatch):
    """DUMMY mode must never drive a microscope: installing the mock fails if the real module is loaded."""

    import sys
    from types import ModuleType
    from SPACEtomo.modules.dummy import install_mock_serialem
    monkeypatch.setitem(sys.modules, "serialem", ModuleType("serialem"))

    with pytest.raises(RuntimeError, match="real SerialEM module"):
        install_mock_serialem()


def test_open_new_file_does_not_overwrite(tmp_path):
    existing = tmp_path / "map.mrc"
    existing.write_bytes(b"data")

    with pytest.raises(sem.SEMerror, match="already exists"):
        sem.OpenNewFile(str(existing))
    assert existing.read_bytes() == b"data"

    sem.AllowFileOverwrite(1)
    sem.OpenNewFile(str(existing))
    assert not existing.exists()        # replaced when the new file is written


def test_image_properties_sets_buffer_tick_time():
    _record_at((0.0, 0.0))
    _record_at((50.0, 0.0))

    sem.ImageProperties("B")
    tick_b = sem.GetVariable("imageTickTime")
    sem.ImageProperties("A")

    assert tick_b < sem.GetVariable("imageTickTime")
