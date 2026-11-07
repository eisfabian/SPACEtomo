"""Tests for the synthetic grid and dummy buffer used by the end-to-end tests."""

import numpy as np
import pytest

from SPACEtomo.modules.dummy.buf import BufferDummy
from SPACEtomo.modules.dummy.specimen import CATS, INTENSITY, Specimen, stageToPixelMatrix, toCounts
from SPACEtomo.modules.mod_wg import BBox, Boxes
from SPACEtomo.modules.nav import Navigator


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


# --- Dummy buffer geometry and alignment ---

def _buffer(img, stage, pix_size, buffer="A"):
    BufferDummy.last_dummy_info = {"stage": np.array(stage, dtype=float), "pix_size": pix_size}
    buf = BufferDummy(buffer, img=img)
    buf.pix_size = pix_size
    return buf


def test_px2stage_and_stage2px_are_inverse():
    buf = _buffer(np.zeros((400, 600), dtype=np.int16), stage=(100.0, -50.0), pix_size=500.0)

    assert np.allclose(buf.px2stage(buf.shape / 2), [100.0, -50.0])
    assert np.allclose(buf.px2stage([400, 100]), [100.0 + 100 * 0.5, -50.0 + 100 * 0.5])     # right and up
    assert np.allclose(buf.stage2px(buf.px2stage([123.0, 321.0])), [123.0, 321.0])


@pytest.mark.parametrize("pix_size, shape, true_shift", [(10.0, (2048, 2048), (2.0, -3.0)), (400.0, (512, 512), (8.0, 4.0))])
def test_alignTo_recovers_specimen_shift(pix_size, shape, true_shift):
    """The content shift of an image relative to a (lower mag) reference is measured in µm (x right, y up)."""

    specimen = Specimen(seed=5, n_lamellae=1)
    center = specimen.lamellae[0].center
    ref = _buffer(toCounts(specimen.render(center, (2048, 2048), 400.0)), center, 400.0, buffer="O")
    img = _buffer(toCounts(specimen.render(center + true_shift, shape, pix_size)), center, pix_size)

    _, ss_shift = img.alignTo(ref)

    assert np.allclose(ss_shift, true_shift, atol=0.5)


def test_addNavBoxes_places_padded_polygon_at_box_position(tmp_path):
    nav = Navigator.__new__(Navigator)
    nav.file, nav.header, nav.items, nav.map_id_counter = tmp_path / "test.nav", "", [], 1
    BufferDummy.nav = nav
    buf = _buffer(np.zeros((1000, 1000), dtype=np.int16), stage=(0.0, 0.0), pix_size=400.0)
    buf.nav_id = None
    boxes = Boxes([[600, 200, 640, 300, 2, 0.9]], label_prefix="PP", pix_size=400.0, img_size=(1000, 1000))

    buf.addNavBoxes(boxes, label_prefix="PP", padding_factor=1.5)

    polygon = nav.items[-1]
    pts = np.array([polygon.entries["PtsX"], polygon.entries["PtsY"]], dtype=float)
    assert polygon.label == "PP01"
    assert np.allclose((pts.min(axis=1) + pts.max(axis=1)) / 2, buf.px2stage(BBox([600, 200, 640, 300], 2).center))
    assert np.allclose(pts.max(axis=1) - pts.min(axis=1), np.array([40, 100]) * 1.5 * 0.4)
    assert np.allclose(polygon.stage[:2], (pts.min(axis=1) + pts.max(axis=1)) / 2)


def test_stage_to_pixel_matrix_flips_y():
    assert np.allclose(stageToPixelMatrix(1000.0), [[1, 0], [0, -1]])
