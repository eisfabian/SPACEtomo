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
