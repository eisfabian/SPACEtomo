"""Tests for SPACEtomo.run_prepTargets (target setup between PACEtomo runs at automation level 5)."""

import json

import pytest

from SPACEtomo import run_prepTargets as prep


def _touch(directory, *names):
    for name in names:
        (directory / name).touch()


def _points(directory, *names, empty=False):
    """Point files as written by target selection (maps without targets get an empty point file)."""
    for name in names:
        (directory / name).write_text(json.dumps({"points": [] if empty else [[100.0, 200.0]], "scores": [], "geo_points": []}))


@pytest.fixture
def dirs(tmp_path):
    map_dir, cur_dir = tmp_path / "SPACE_maps", tmp_path
    map_dir.mkdir()
    return map_dir, cur_dir


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(prep.time, "sleep", lambda seconds: calls.append(seconds) if len(calls) < 10 else pytest.fail("waits forever"))
    return calls


def test_findUnprocessedPointFiles_uses_target_area_names(dirs):
    """Maps with several target areas get one targets file per area (_A1, _A2, ...)."""

    map_dir, cur_dir = dirs
    _points(map_dir, "G01_PP1_points0.json", "G01_PP1_points1.json", "G01_PP2_points.json", "G01_PP3_points0.json", "G01_PP3_points1.json")
    _touch(cur_dir, "G01_PP1_A1_tgts.txt", "G01_PP1_A2_tgts.txt", "G01_PP2_tgts.txt", "G01_PP3_A1_tgts.txt")

    assert prep.findUnprocessedPointFiles(map_dir, cur_dir) == [map_dir / "G01_PP3_points1.json"]
    assert prep.getAreaName(map_dir / "G01_PP3_points1.json") == "G01_PP3_A2"
    assert prep.getAreaName(map_dir / "G01_PP2_points.json") == "G01_PP2"


def test_findUnprocessedPointFiles_ignores_maps_without_targets(dirs):
    """An empty point file marks a map as processed (no targets file is written for it), but only once it was inspected."""

    map_dir, cur_dir = dirs
    _points(map_dir, "G01_PP1_points.json", empty=True)

    assert prep.findUnprocessedPointFiles(map_dir, cur_dir) == []
    assert prep.findUnprocessedPointFiles(map_dir, cur_dir, wait_for_inspection=True) == [map_dir / "G01_PP1_points.json"]
    _touch(map_dir, "G01_PP1_inspected.txt")
    assert prep.findUnprocessedPointFiles(map_dir, cur_dir, wait_for_inspection=True) == []


def test_waitForPointFiles_while_only_segmentations_are_pending(dirs, sleeps):
    """Acquisition ran out of targets while a map is still being segmented (no map waiting for target selection)."""

    map_dir, cur_dir = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png", "G01_PP2.png")
    _points(map_dir, "G01_PP1_points.json")
    _touch(cur_dir, "G01_PP1_tgts.txt")
    steps = iter([lambda: _touch(map_dir, "G01_PP2_seg.png"), lambda: _points(map_dir, "G01_PP2_points0.json", "G01_PP2_points1.json")])

    ready = prep.waitForPointFiles(map_dir, cur_dir, update_function=lambda: next(steps, lambda: None)())

    assert ready == [map_dir / "G01_PP2_points0.json", map_dir / "G01_PP2_points1.json"]
    assert len(sleeps) == 2


def test_waitForPointFiles_returns_when_nothing_is_pending(dirs, sleeps):
    """Everything processed; a map without targets must not keep the acquisition waiting."""

    map_dir, cur_dir = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png", "G01_PP2.png", "G01_PP2_seg.png")
    _points(map_dir, "G01_PP1_points.json")
    _points(map_dir, "G01_PP2_points.json", empty=True)
    _touch(cur_dir, "G01_PP1_tgts.txt")

    assert prep.waitForPointFiles(map_dir, cur_dir) == []
    assert sleeps == []


def test_waitForPointFiles_waits_for_inspection(dirs, sleeps):
    """With MM_wait_for_inspection, targets are only set up after the operator confirmed the inspection."""

    map_dir, cur_dir = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png")
    _points(map_dir, "G01_PP1_points0.json")
    steps = iter([lambda: None, lambda: _touch(map_dir, "G01_PP1_inspected.txt")])

    ready = prep.waitForPointFiles(map_dir, cur_dir, wait_for_inspection=True, update_function=lambda: next(steps, lambda: None)())

    assert ready == [map_dir / "G01_PP1_points0.json"]
    assert len(sleeps) == 2


@pytest.mark.parametrize("empty_map", [False, True])
def test_main_waits_for_inspection_instead_of_ending_acquisition(dirs, sleeps, monkeypatch, empty_map):
    """No areas left to acquire and the next map still being inspected: set it up after inspection.

    empty_map adds a map without targets, which must not count as a map to set up.
    """

    from types import SimpleNamespace

    map_dir, cur_dir = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png", "G01_PP2.png", "G01_PP2_seg.png")
    _points(map_dir, "G01_PP1_points.json")
    _touch(cur_dir, "G01_PP1_tgts.txt")     # First map already set up and acquired
    _points(map_dir, "G01_PP2_points0.json")
    if empty_map:
        _touch(map_dir, "G01_PP3.png", "G01_PP3_seg.png")
        _points(map_dir, "G01_PP3_points.json", empty=True)

    class Nav:
        def searchByEntry(self, *args, **kwargs): return []
        def getIDfromNote(self, note): return 0

    set_up = []
    class Area:
        def __init__(self, point_file, *args): self.point_file = point_file
        def __len__(self): return 1
        def checkISLimit(self): pass
        def prepareForAcquisition(self, *args): set_up.append(self.point_file.name); return True
        def getTargetsInfo(self, area_name): return [], [], {}

    inspection = iter([lambda: None, lambda: _touch(map_dir, "G01_PP2_inspected.txt")])
    monkeypatch.setattr(prep.utils, "getCurDir", lambda: cur_dir)
    monkeypatch.setattr(prep.utils, "loadSettings", lambda file: {"external_map_dir": "", "MM_wait_for_inspection": True})
    monkeypatch.setattr(prep.utils, "writeTargets", lambda *args, **kwargs: None)
    monkeypatch.setattr(prep.usem, "prepareEnvironment", lambda cur, ext: (map_dir, False))
    monkeypatch.setattr(prep, "ImagingParams", lambda *args: None)
    monkeypatch.setattr(prep, "Navigator", lambda **kwargs: Nav())
    monkeypatch.setattr(prep, "WGModel", lambda *args: None)
    monkeypatch.setattr(prep.space_ext, "MMModel", lambda **kwargs: SimpleNamespace(setDimensions=lambda params: None))
    monkeypatch.setattr(prep.space_ext, "TgtParams", lambda **kwargs: None)
    monkeypatch.setattr(prep.space_ext, "updateQueue", lambda *args: next(inspection, lambda: None)())
    monkeypatch.setattr(prep, "Buffer", lambda **kwargs: None)
    monkeypatch.setattr(prep, "PACEArea", Area)

    prep.main()

    assert set_up == ["G01_PP2_points0.json"]


def test_waitForPointFiles_waits_for_manual_selection(dirs, sleeps):
    """Manual selection starts with an empty segmentation and point file; targets appear when the operator saves in the GUI."""

    map_dir, cur_dir = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png")
    _points(map_dir, "G01_PP1_points.json", empty=True)

    def select():
        (map_dir / "G01_PP1_points.json").unlink()       # GUI replaces previous point files on export
        _points(map_dir, "G01_PP1_points0.json")
        _touch(map_dir, "G01_PP1_inspected.txt")
    steps = iter([lambda: None, select])

    ready = prep.waitForPointFiles(map_dir, cur_dir, wait_for_inspection=True, update_function=lambda: next(steps, lambda: None)())

    assert ready == [map_dir / "G01_PP1_points0.json"]
    assert len(sleeps) == 2


@pytest.mark.parametrize("success", [True, False])
def test_main_writes_targets_file_only_after_successful_setup(dirs, sleeps, monkeypatch, success):
    """A targets file without acquisition item marks the area as done although it is never acquired."""

    from types import SimpleNamespace

    map_dir, cur_dir = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png")
    _points(map_dir, "G01_PP1_points.json")

    class Nav:
        def searchByEntry(self, *args, **kwargs): return [1]
        def getIDfromNote(self, note): return 0

    class Area:
        def __init__(self, point_file, *args): self.point_file = point_file
        def __len__(self): return 1
        def checkISLimit(self): pass
        def prepareForAcquisition(self, *args): return success
        def getTargetsInfo(self, area_name): return [], [], {}

    written = []
    monkeypatch.setattr(prep.utils, "getCurDir", lambda: cur_dir)
    monkeypatch.setattr(prep.utils, "loadSettings", lambda file: {"external_map_dir": "", "MM_wait_for_inspection": False})
    monkeypatch.setattr(prep.utils, "writeTargets", lambda file, *args, **kwargs: written.append(file.name))
    monkeypatch.setattr(prep.usem, "prepareEnvironment", lambda cur, ext: (map_dir, False))
    monkeypatch.setattr(prep, "ImagingParams", lambda *args: None)
    monkeypatch.setattr(prep, "Navigator", lambda **kwargs: Nav())
    monkeypatch.setattr(prep, "WGModel", lambda *args: None)
    monkeypatch.setattr(prep.space_ext, "MMModel", lambda **kwargs: SimpleNamespace(setDimensions=lambda params: None))
    monkeypatch.setattr(prep.space_ext, "TgtParams", lambda **kwargs: None)
    monkeypatch.setattr(prep.space_ext, "updateQueue", lambda *args: None)
    monkeypatch.setattr(prep, "Buffer", lambda **kwargs: None)
    monkeypatch.setattr(prep, "PACEArea", Area)

    prep.main()

    assert written == (["G01_PP1_tgts.txt"] if success else [])
    assert prep.getFailedFile(cur_dir, "G01_PP1").exists() is not success


def test_failed_area_does_not_stop_waiting_for_predictions(dirs, sleeps):
    """An area whose setup failed is not retried and must not count as ready, so the post action still waits for the next prediction."""

    map_dir, cur_dir = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png", "G01_PP2.png")
    _points(map_dir, "G01_PP1_points.json")
    prep.getFailedFile(cur_dir, "G01_PP1").touch()

    def predict():
        _touch(map_dir, "G01_PP2_seg.png")
        _points(map_dir, "G01_PP2_points.json")
    steps = iter([lambda: None, predict])

    assert prep.findUnprocessedPointFiles(map_dir, cur_dir) == []
    assert prep.waitForPointFiles(map_dir, cur_dir, update_function=lambda: next(steps, lambda: None)(), interval=0) == [map_dir / "G01_PP2_points.json"]
