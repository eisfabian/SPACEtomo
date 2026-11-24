"""Unit tests for SPACEtomo.modules.ima — IM alignment labelling of nav items.

Microscope, buffers and the WG model are replaced by minimal fakes so that only the
nav bookkeeping of IMAlignment.align() is exercised.
"""

import numpy as np
import pytest

from SPACEtomo.modules import ima
from SPACEtomo.modules.ima import IMAlignment
from SPACEtomo.modules.nav import NavItem, Navigator


SAMPLE_NAV = """\
AdocVersion = 2.00
LastSavedAs = test.nav

[Item = WG01]
Color = 2
StageXYZ = 0.0 0.0 0.0
NumPts = 5
Regis = 1
Type = 2
Note = G01.mrc
MapFile = /tmp/G01.mrc
MapID = 11
MapBinning = 1
MontBinning = 4
MapScaleMat = 0.5 0.0 0.0 0.5
PtsX = -50.0 -50.0 50.0 50.0 -50.0
PtsY = -50.0 50.0 50.0 -50.0 -50.0

[Item = PP01]
Color = 1
StageXYZ = 10.0 20.0 0.0
NumPts = 5
Regis = 1
Type = 1
Note = G01.mrc
MapID = 12
UserValue1 = 0.95
UserValue2 = {display_label}
PtsX = 8.0 8.0 12.0 12.0 8.0
PtsY = 18.0 22.0 22.0 18.0 18.0

"""


class FakeScope:
    def __init__(self):
        self.autoloader = {1: "G01"}
        self.loaded_grid = 1
        self.stage = np.array([10.0, 20.0, 0.0])
        self.camera_dims = np.array([100, 80])

    def changeImagingState(self, *args, **kwargs): pass
    def changeC2Aperture(self, *args, **kwargs): pass
    def moveStage(self, *args, **kwargs): pass
    def eucentricity(self): pass
    def record(self, *args, **kwargs): pass
    def getMatrices(self): return {"ss2s": np.eye(2)}


class FakeParams:
    IM_image_state = 1
    IM_pix_size = 10.0


class FakeBox:
    center = np.array([0.0, 0.0])
    cat = 2
    prob = 0.9


class FakeBoxes:
    pix_size = 10.0

    def __init__(self):
        self.boxes = [FakeBox()]

    def __len__(self): return len(self.boxes)
    def sortBy(self, key): pass


class FakeWGModel:
    pix_size = 10.0

    def findLamellae(self, map_name, save_future=None):
        return FakeBoxes()


def make_buffer_class(nav):
    """Fake Buffer whose addNavBoxes adds a slightly larger polygon around the ROI (so the IM polygon is kept)."""

    class FakeBuffer:
        pix_size = 10.0

        def __init__(self, buffer="A", file_path=None, nav_id=None):
            self.nav_id = nav_id

        def alignTo(self, ref_buffer, avoid_image_shift=False): return None, np.zeros(2)
        def clearAlignment(self): pass
        def saveImg(self, *args, **kwargs): return None
        def px2stage(self, coords): return np.array([10.0, 20.0])
        shape = np.array([100, 80])
        def stage2px(self, coords): return np.array([50.0, 40.0])
        def getCropImage(self, coords, fov): return np.zeros((int(fov[0]), int(fov[1])), dtype=np.int16)

        def addNavBoxes(self, boxes, labels=[], padding_factor=1, label_prefix=None):
            item = NavItem(len(nav) + 1, labels[0])
            item.entries = {"StageXYZ": ["10.0", "20.0", "0.0"], "Type": ["1"],
                            "PtsX": ["7.0", "7.0", "13.0", "13.0", "7.0"],
                            "PtsY": ["17.0", "23.0", "23.0", "17.0", "17.0"]}
            item.convertEntries()
            nav.items.append(item)

    return FakeBuffer


@pytest.fixture
def make_nav(tmp_path, monkeypatch):
    def _make(display_label):
        nav_file = tmp_path / "G01.nav"
        nav_file.write_text(SAMPLE_NAV.format(display_label=display_label))
        nav = Navigator.__new__(Navigator)
        nav.file = nav_file
        nav.header = ""
        nav.items = []
        nav.selected_item = None
        nav.map_id_counter = 1
        nav.readFromFile()

        # Keep everything in memory: push/pull would round-trip through the mock SerialEM
        monkeypatch.setattr(nav, "push", lambda: None)
        monkeypatch.setattr(nav, "pull", lambda: None)

        def fake_new_map(buffer=None, img_file="", label="", note="", **kwargs):
            item = NavItem(len(nav) + 1, label)
            item.entries = {"StageXYZ": ["10.0", "20.0", "0.0"], "Type": ["2"], "MapFile": [str(img_file)],
                            "PtsX": ["5.0", "5.0", "15.0", "15.0", "5.0"],
                            "PtsY": ["15.0", "25.0", "25.0", "15.0", "15.0"]}
            item.convertEntries()
            item.note = note
            nav.items.append(item)
            return len(nav) - 1
        monkeypatch.setattr(nav, "newMap", fake_new_map)

        monkeypatch.setattr(ima, "Buffer", make_buffer_class(nav))
        (tmp_path / f"G01_{display_label}_IMref.mrc").touch()
        return nav
    return _make


@pytest.mark.parametrize("display_label", ["PP01", "LamA"])
def test_align_keeps_pp_label_on_im_map_and_creates_fp(tmp_path, make_nav, display_label):
    """IM map must keep the PP nav label and the IM polygon must get the FP label, also with custom ROI names."""

    nav = make_nav(display_label)
    alignment = IMAlignment(tmp_path, tmp_path, FakeScope(), nav, FakeParams(), WG_model=FakeWGModel())
    alignment.align({"WG_distance_threshold": 0.1})

    im_ids = [i for i, item in enumerate(nav.items) if item.item_type == 2 and item.label != "WG01"]
    assert len(im_ids) == 1
    assert nav.items[im_ids[0]].label == "PP01"

    fp_ids = nav.searchByEntry("label", "FP", partial=True)
    assert len(fp_ids) == 1
    assert nav.items[fp_ids[0]].label == "FP01"
    assert nav.items[fp_ids[0]].entries["UserValue2"] == [display_label]


@pytest.mark.parametrize("display_label", ["PP01", "LamA"])
def test_align_rerun_detects_completed_alignment(tmp_path, make_nav, display_label):
    """A second IMAlignment on the same nav must find the imaged ROI and not redo the alignment."""

    nav = make_nav(display_label)
    IMAlignment(tmp_path, tmp_path, FakeScope(), nav, FakeParams(), WG_model=FakeWGModel()).align({"WG_distance_threshold": 0.1})

    rerun = IMAlignment(tmp_path, tmp_path, FakeScope(), nav, FakeParams(), WG_model=FakeWGModel())
    assert len(rerun.position_ids) == 1
    assert not rerun.checkNecessity()


# --- Realignment of a reloaded grid (real classes against the simulated microscope) ---

@pytest.fixture
def reloaded_grid(tmp_path):
    """Grid with a WG item and one ROI (PP1) at a lamella; returns a function that "reloads" the grid with a given displacement.

    with_im_map=False leaves PP1 as polygon like after a level 1 run, with_roi=False gives no ROI at all.
    """

    import serialem as sem
    from SPACEtomo.modules.buf import Buffer
    from SPACEtomo.modules.dummy.sim import sim
    from SPACEtomo.modules.scope import Microscope

    def _make(with_roi=True, with_im_map=True):
        sem.SetDirectory(str(tmp_path))
        nav = Navigator(tmp_path / "G01.nav")
        Buffer.nav = nav
        microscope = Microscope()

        wg_id = nav.newPoint(np.zeros(3), label="WG")         # stands in for the WG map (only has to stay in place; saves a K3 record)

        roi_stage = sim.getSpecimen().lamellae[0].center - sim.specimen.offsets["IM"]     # IM image centered on the lamella
        if with_roi and with_im_map:
            microscope.changeImagingState("IM", low_dose_expected=False)
            microscope.moveStage(roi_stage)
            microscope.record()
            nav.newMap(buffer=Buffer("A"), img_file=tmp_path / "G01_PP1_IM.mrc", label="PP1")
        elif with_roi:
            nav.newPolygon([roi_stage[0] + dx for dx in (-5, -5, 5, 5, -5)], [roi_stage[1] + dy for dy in (-5, 5, 5, -5, -5)], 0.0, label="PP1")
        nav.pull()

        microscope.changeImagingState("IM", low_dose_expected=False)
        microscope.moveStage((0.0, 0.0))

        def reload(displacement):
            """Specimen moved relative to the stage: what was imaged at stage s is now at s - displacement."""
            sim.specimen.offsets = {key: val + np.array(displacement) for key, val in sim.specimen.offsets.items()}

        return microscope, nav, wg_id, reload
    return _make


def _stages(nav):
    return {item.label: item.stage[:2].copy() for item in nav.items}


@pytest.mark.parametrize("displacement", [(-8.0, 5.0)])
def test_realignReloadedGrid_shifts_items_onto_specimen(reloaded_grid, displacement):
    """Displacement beyond the convergence threshold: the stage is moved once and the alignment is verified."""

    microscope, nav, wg_id, reload = reloaded_grid()
    before = _stages(nav)
    reload(displacement)

    shift = ima.realignReloadedGrid(microscope, nav, skip_item_ids=[wg_id])

    after = _stages(nav)
    assert np.allclose(shift, -np.array(displacement), atol=0.5)
    assert np.allclose(after["PP1"], before["PP1"] - displacement, atol=0.5)
    assert np.allclose(after["WG"], before["WG"])


def test_realignReloadedGrid_without_rois(reloaded_grid):
    """No ROI on the grid (e.g. no lamella found): keep the whole grid realignment instead of crashing."""

    microscope, nav, wg_id, reload = reloaded_grid(with_roi=False)
    before = _stages(nav)
    reload((3.0, -2.0))

    assert ima.realignReloadedGrid(microscope, nav, skip_item_ids=[wg_id]) is None
    assert all(np.allclose(_stages(nav)[label], stage) for label, stage in before.items())


def test_realignReloadedGrid_skips_rois_without_im_map(reloaded_grid):
    """After a level 1 run, PP items are polygons, which cannot be loaded as reference."""

    microscope, nav, wg_id, reload = reloaded_grid(with_im_map=False)
    before = _stages(nav)
    reload((3.0, -2.0))

    assert ima.realignReloadedGrid(microscope, nav, skip_item_ids=[wg_id]) is None
    assert all(np.allclose(_stages(nav)[label], stage) for label, stage in before.items())


def test_realignReloadedGrid_ignores_failed_alignment(reloaded_grid, monkeypatch):
    """A cross-correlation that does not converge (e.g. ROI lost or covered in ice) must not shift the items."""

    import serialem as sem

    microscope, nav, wg_id, reload = reloaded_grid()
    before = _stages(nav)
    monkeypatch.setattr(sem, "AlignTo", lambda *args: None)
    monkeypatch.setattr(sem, "ReportAlignShift", lambda: (1000.0, 1000.0, 0.0, 0.0, 10000.0, 10000.0))   # 10 µm off after every move

    assert ima.realignReloadedGrid(microscope, nav, skip_item_ids=[wg_id], max_iterations=2) is None
    assert all(np.allclose(_stages(nav)[label], stage) for label, stage in before.items())
    assert np.allclose(microscope.stage[:2], before["PP1"])


def test_realignReloadedGrid_in_debug_mode(reloaded_grid, monkeypatch):
    """DEBUG shows the cross-correlation (image rolled to B) and aligns again after copying B back to A."""

    from SPACEtomo import config
    monkeypatch.setattr(config, "DEBUG", True)

    microscope, nav, wg_id, reload = reloaded_grid()
    before = _stages(nav)
    reload((-8.0, 5.0))

    shift = ima.realignReloadedGrid(microscope, nav, skip_item_ids=[wg_id])

    assert np.allclose(shift, (8.0, -5.0), atol=0.5)
    assert np.allclose(_stages(nav)["PP1"], before["PP1"] + (8.0, -5.0), atol=0.5)


def test_realignReloadedGrid_survives_serialem_alignment_error(reloaded_grid, monkeypatch):
    """SerialEM aborts AlignTo when no correlation peak is found; the realignment has to give up without shifting items."""

    import serialem as sem

    microscope, nav, wg_id, reload = reloaded_grid()
    before = _stages(nav)
    reload((3.0, -2.0))

    def no_peak(*args):
        raise sem.SEMerror("No correlation peaks were found within the allowed range")
    monkeypatch.setattr(sem, "AlignTo", no_peak)

    assert ima.realignReloadedGrid(microscope, nav, skip_item_ids=[wg_id]) is None
    assert all(np.allclose(_stages(nav)[label], stage) for label, stage in before.items())


def test_align_retained_roi_moves_to_im_image(tmp_path, make_nav, monkeypatch):
    """No lamella detected on the IM image but high confidence on the grid map: the kept polygon has to move to where
    IM alignment centered the lamella (the IM image), not stay at the grid map position from before the alignment."""

    from SPACEtomo.modules import nav as nav_module
    monkeypatch.setattr(nav_module, "SERIALEM", False)       # Navigator kept in memory by make_nav, no SerialEM item to rename

    nav = make_nav("PP01")
    new_map = nav.newMap

    def new_map_after_alignment(**kwargs):
        nav_id = new_map(**kwargs)
        nav.items[nav_id].changeStage(np.array([13.0, 24.0]))       # IM image taken after alignToRef moved the stage
        return nav_id
    monkeypatch.setattr(nav, "newMap", new_map_after_alignment)

    class NoLamella(FakeWGModel):
        def findLamellae(self, map_name, save_future=None):
            boxes = FakeBoxes()
            boxes.boxes = []
            return boxes

    IMAlignment(tmp_path, tmp_path, FakeScope(), nav, FakeParams(), WG_model=NoLamella()).align({"WG_distance_threshold": 0.1})

    fp_ids = nav.searchByEntry("label", "FP", partial=True)
    assert len(fp_ids) == 1
    assert np.allclose(nav.items[fp_ids[0]].stage[:2], (13.0, 24.0))
