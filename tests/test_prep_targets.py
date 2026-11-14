"""Tests for SPACEtomo.run_prepTargets (target setup between PACEtomo runs at automation level 5)."""

import pytest

from SPACEtomo import run_prepTargets as prep


def _touch(directory, *names):
    for name in names:
        (directory / name).touch()


@pytest.fixture
def dirs(tmp_path):
    map_dir, cur_dir = tmp_path / "SPACE_maps", tmp_path
    map_dir.mkdir()
    return map_dir, cur_dir


def test_findUnprocessedPointFiles_uses_target_area_names(dirs):
    """Maps with several target areas get one targets file per area (_A1, _A2, ...)."""

    map_dir, cur_dir = dirs
    _touch(map_dir, "G01_PP1_points0.json", "G01_PP1_points1.json", "G01_PP2_points.json", "G01_PP3_points0.json", "G01_PP3_points1.json")
    _touch(cur_dir, "G01_PP1_A1_tgts.txt", "G01_PP1_A2_tgts.txt", "G01_PP2_tgts.txt", "G01_PP3_A1_tgts.txt")

    assert prep.findUnprocessedPointFiles(map_dir, cur_dir) == [map_dir / "G01_PP3_points1.json"]
    assert prep.getAreaName(map_dir / "G01_PP3_points1.json") == "G01_PP3_A2"
    assert prep.getAreaName(map_dir / "G01_PP2_points.json") == "G01_PP2"


def test_waitForNewPointFiles_while_only_segmentations_are_pending(dirs, monkeypatch):
    """Acquisition ran out of targets while a map is still being segmented (no map waiting for target selection)."""

    map_dir, _ = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png", "G01_PP1_points.json", "G01_PP2.png")
    known = sorted(map_dir.glob("*_points*.json"))
    steps = iter([lambda: _touch(map_dir, "G01_PP2_seg.png"), lambda: _touch(map_dir, "G01_PP2_points0.json", "G01_PP2_points1.json")])
    sleeps = []
    monkeypatch.setattr(prep.time, "sleep", lambda seconds: sleeps.append(seconds) if len(sleeps) < 10 else pytest.fail("waits forever"))

    new_files = prep.waitForNewPointFiles(map_dir, known, update_function=lambda: next(steps, lambda: None)())

    assert new_files == [map_dir / "G01_PP2_points0.json", map_dir / "G01_PP2_points1.json"]
    assert len(sleeps) == 2


def test_waitForNewPointFiles_returns_when_nothing_is_pending(dirs, monkeypatch):
    map_dir, _ = dirs
    _touch(map_dir, "G01_PP1.png", "G01_PP1_seg.png", "G01_PP1_points.json")
    monkeypatch.setattr(prep.time, "sleep", lambda seconds: pytest.fail("should not wait"))

    assert prep.waitForNewPointFiles(map_dir, sorted(map_dir.glob("*_points*.json"))) == []
