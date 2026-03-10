"""Unit tests for SPACEtomo.modules.tgt — target area management and geometry."""

import json
from pathlib import Path

import numpy as np
import pytest

from SPACEtomo.modules.tgt import Targets, TargetArea
from SPACEtomo.modules.ext import calcScore_point, calcScore_cluster


@pytest.fixture(autouse=True)
def _reset_target_area_class_attrs():
    """Reset TargetArea class attributes between tests to avoid leaking state."""
    yield
    TargetArea.tgt_params = None
    TargetArea.rec_dims = None
    TargetArea.settings = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTgtParams:
    """Minimal stand-in for TgtParams with weight masks."""

    def __init__(self, rec_dims=(64, 64)):
        self.weight = np.ones(rec_dims, dtype=np.float32)
        self.edge_weights = []
        self.penalty = 0.3
        self.threshold = 0.1


def make_targets(map_dims=(512, 512), rec_dims=(64, 64)):
    """Create a Targets instance with fake params."""
    params = FakeTgtParams(rec_dims)
    return Targets(
        map_dir="/tmp",
        map_name="test",
        tgt_params=params,
        map_dims=np.array(map_dims),
    )


# ---------------------------------------------------------------------------
# TargetArea basics
# ---------------------------------------------------------------------------

class TestTargetArea:
    def test_empty(self):
        area = TargetArea()
        assert len(area) == 0
        assert not bool(area)
        assert area.center is None

    def test_add_point(self):
        area = TargetArea()
        area.addPoint(np.array([100.0, 200.0]), ld_area="R", score=0.5)
        assert len(area) == 1
        assert bool(area)
        np.testing.assert_array_equal(area.center, [100.0, 200.0])
        assert area.ld_areas[0] == "R"
        assert area.scores[0] == 0.5

    def test_add_multiple_points(self):
        area = TargetArea()
        area.addPoint(np.array([10.0, 20.0]))
        area.addPoint(np.array([30.0, 40.0]))
        area.addPoint(np.array([50.0, 60.0]))
        assert len(area) == 3
        # Center is first point (tracking target)
        np.testing.assert_array_equal(area.center, [10.0, 20.0])

    def test_add_geo_point(self):
        area = TargetArea()
        area.addPoint(np.array([100.0, 200.0]), geo=True)
        assert len(area) == 0  # geo points don't count as targets
        assert len(area.geo_points) == 1

    def test_remove_point(self):
        area = TargetArea()
        area.addPoint(np.array([10.0, 20.0]))
        area.addPoint(np.array([30.0, 40.0]))
        area.removePoint(0)
        assert len(area) == 1
        # Center updated to remaining first point
        np.testing.assert_array_equal(area.center, [30.0, 40.0])

    def test_remove_geo_point(self):
        area = TargetArea()
        area.addPoint(np.array([1.0, 2.0]), geo=True)
        area.addPoint(np.array([3.0, 4.0]), geo=True)
        area.removeGeoPoint(0)
        assert len(area.geo_points) == 1
        np.testing.assert_array_equal(area.geo_points[0], [3.0, 4.0])

    def test_update_point(self):
        area = TargetArea()
        area.addPoint(np.array([10.0, 20.0]))
        area.updatePoint(0, np.array([99.0, 88.0]))
        np.testing.assert_array_equal(area.points[0], [99.0, 88.0])
        np.testing.assert_array_equal(area.center, [99.0, 88.0])

    def test_make_track(self):
        area = TargetArea()
        area.addPoint(np.array([10.0, 20.0]), score=1.0)
        area.addPoint(np.array([30.0, 40.0]), score=2.0)
        area.addPoint(np.array([50.0, 60.0]), score=3.0)
        # Make point 2 (index 2) the tracking target
        area.makeTrack(2)
        np.testing.assert_array_equal(area.center, area.points[0])
        np.testing.assert_array_equal(area.points[0], [50.0, 60.0])
        assert area.scores[0] == 3.0

    def test_center_track(self):
        area = TargetArea()
        area.addPoint(np.array([10.0, 10.0]), score=1.0)
        area.addPoint(np.array([50.0, 50.0]), score=2.0)
        area.addPoint(np.array([90.0, 90.0]), score=3.0)
        # Center near (50,50) — should pick that as tracking target
        area.centerTrack(np.array([48.0, 52.0]))
        np.testing.assert_array_equal(area.center, [50.0, 50.0])

    def test_get_point_info(self):
        area = TargetArea()
        area.addPoint(np.array([0.0, 0.0]), score=0.8)
        area.addPoint(np.array([3.0, 4.0]), score=0.6)
        score, dist = area.getPointInfo(1)
        assert score == 0.6
        assert dist == pytest.approx(5.0)

    def test_get_geo_in_range(self):
        area = TargetArea()
        area.addPoint(np.array([100.0, 100.0]))  # center/tracking
        area.addPoint(np.array([105.0, 100.0]), geo=True)  # dist=5
        area.addPoint(np.array([200.0, 200.0]), geo=True)  # dist~141
        in_range = area.getGeoInRange(is_limit_px=50)
        assert len(in_range) == 1


# ---------------------------------------------------------------------------
# TargetArea JSON round-trip
# ---------------------------------------------------------------------------

class TestTargetAreaIO:
    def test_export_load_roundtrip(self, tmp_path):
        area = TargetArea()
        area.addPoint(np.array([100.0, 200.0]), ld_area="R", score=0.75)
        area.addPoint(np.array([300.0, 400.0]), ld_area="V", score=0.50)
        area.addPoint(np.array([50.0, 60.0]), geo=True)

        f = tmp_path / "points0.json"
        meta = {"pix_size": 1.5, "img_size": [512, 512]}
        area.exportToJson(f, settings={"startTilt": 0}, meta_data=meta)

        loaded = TargetArea(file=f)
        assert len(loaded) == 2
        np.testing.assert_array_almost_equal(loaded.points[0], [100.0, 200.0])
        assert loaded.ld_areas[0] == "R"
        assert loaded.scores[0] == pytest.approx(0.75)
        assert len(loaded.geo_points) == 1
        assert loaded.meta_data["pix_size"] == 1.5

    def test_export_with_geo_sets_measureGeo(self, tmp_path):
        area = TargetArea()
        area.addPoint(np.array([10.0, 20.0]))
        area.addPoint(np.array([5.0, 5.0]), geo=True)

        f = tmp_path / "points0.json"
        settings = {"startTilt": 0}
        area.exportToJson(f, settings=settings)

        with open(f) as fh:
            data = json.load(fh)
        assert data["settings"]["measureGeo"] is True

    def test_load_without_ld_areas_defaults_to_R(self, tmp_path):
        """Old point files may not have ld_areas."""
        f = tmp_path / "points0.json"
        data = {
            "points": [[100.0, 200.0], [300.0, 400.0]],
            "scores": [0.5, 0.6],
            "geo_points": [],
        }
        with open(f, "w") as fh:
            json.dump(data, fh)

        loaded = TargetArea(file=f)
        assert len(loaded) == 2
        assert all(ld == "R" for ld in loaded.ld_areas)


# ---------------------------------------------------------------------------
# Targets (collection of areas)
# ---------------------------------------------------------------------------

class TestTargets:
    def test_check_bounds_in(self):
        tgts = make_targets(map_dims=(512, 512), rec_dims=(64, 64))
        assert tgts.checkBounds(np.array([256.0, 256.0])) is True

    def test_check_bounds_edge(self):
        tgts = make_targets(map_dims=(512, 512), rec_dims=(64, 64))
        # Too close to left edge (x < rec_dims[0]/2 = 32)
        assert tgts.checkBounds(np.array([10.0, 256.0])) is False
        # Too close to right edge
        assert tgts.checkBounds(np.array([500.0, 256.0])) is False

    def test_add_target(self):
        tgts = make_targets()
        assert tgts.addTarget(np.array([256.0, 256.0])) is True
        assert len(tgts) == 1

    def test_add_target_out_of_bounds(self):
        tgts = make_targets()
        assert tgts.addTarget(np.array([10.0, 10.0])) is False
        assert len(tgts) == 0

    def test_add_duplicate_rejected(self):
        tgts = make_targets()
        tgts.addTarget(np.array([256.0, 256.0]))
        # Same coords — too close
        assert tgts.addTarget(np.array([256.0, 256.0])) is False
        assert len(tgts) == 1

    def test_add_target_new_area(self):
        tgts = make_targets()
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([400.0, 400.0]), new_area=True)
        assert len(tgts.areas) == 2
        assert len(tgts) == 2

    def test_add_geo_point(self):
        tgts = make_targets()
        tgts.addTarget(np.array([256.0, 256.0]))
        assert tgts.addGeoPoint(np.array([100.0, 100.0])) is True
        assert len(tgts.areas[0].geo_points) == 1

    def test_add_geo_point_without_targets(self):
        tgts = make_targets()
        assert tgts.addGeoPoint(np.array([256.0, 256.0])) is False

    def test_get_all_points(self):
        tgts = make_targets()
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([400.0, 400.0]), new_area=True)
        points, ids = tgts.getAllPoints()
        assert len(points) == 2
        assert len(ids) == 2
        assert ids[0][0] == 0  # area 0
        assert ids[1][0] == 1  # area 1

    def test_get_closest_area(self):
        tgts = make_targets()
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([400.0, 400.0]), new_area=True)
        # Point closer to area 0
        assert tgts.getClosestArea(np.array([120.0, 120.0])) == 0
        # Point closer to area 1
        assert tgts.getClosestArea(np.array([380.0, 380.0])) == 1

    def test_merge_areas(self):
        tgts = make_targets()
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([400.0, 400.0]), new_area=True)
        assert len(tgts.areas) == 2
        tgts.mergeAreas()
        assert len(tgts.areas) == 1
        assert len(tgts) == 2

    def test_move_point_to_area(self):
        tgts = make_targets()
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([200.0, 200.0]))
        tgts.addTarget(np.array([400.0, 400.0]), new_area=True)
        # Move point 1 from area 0 to area 1
        tgts.movePointToArea([0, 1], 1)
        assert len(tgts.areas[0]) == 1
        assert len(tgts.areas[1]) == 2

    def test_bool_empty(self):
        tgts = make_targets()
        assert not bool(tgts)

    def test_bool_with_points(self):
        tgts = make_targets()
        tgts.addTarget(np.array([256.0, 256.0]))
        assert bool(tgts)

    def test_reset_geo(self):
        tgts = make_targets()
        tgts.addTarget(np.array([256.0, 256.0]))
        tgts.addGeoPoint(np.array([100.0, 100.0]))
        tgts.resetGeo()
        assert len(tgts.areas[0].geo_points) == 0


# ---------------------------------------------------------------------------
# Targets export/load round-trip
# ---------------------------------------------------------------------------

class TestTargetsIO:
    def test_export_and_load(self, tmp_path):
        params = FakeTgtParams()
        tgts = Targets(
            map_dir=tmp_path,
            map_name="test",
            tgt_params=params,
            map_dims=np.array([512, 512]),
            map_pix_size=1.5,
        )
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([400.0, 400.0]), new_area=True)
        tgts.addGeoPoint(np.array([200.0, 200.0]))

        settings = {"startTilt": 0, "minTilt": -60}
        tgts.exportTargets(settings=settings)

        # Check files created
        point_files = sorted(tmp_path.glob("test_points*.json"))
        assert len(point_files) == 2

        # Load back
        tgts2 = Targets(
            map_dir=tmp_path,
            map_name="test",
            tgt_params=params,
            map_dims=np.array([512, 512]),
        )
        tgts2.loadAreas()
        assert len(tgts2.areas) == 2
        assert len(tgts2) == 2

    def test_export_empty_creates_placeholder(self, tmp_path):
        params = FakeTgtParams()
        tgts = Targets(
            map_dir=tmp_path,
            map_name="test",
            tgt_params=params,
            map_dims=[512, 512],  # list, not ndarray, for JSON serialization
        )
        tgts.exportTargets()

        f = tmp_path / "test_points.json"
        assert f.exists()
        with open(f) as fh:
            data = json.load(fh)
        assert data["points"] == []

    def test_export_overwrites_previous(self, tmp_path):
        params = FakeTgtParams()
        tgts = Targets(
            map_dir=tmp_path,
            map_name="test",
            tgt_params=params,
            map_dims=np.array([512, 512]),
        )
        # First export with 2 areas
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([400.0, 400.0]), new_area=True)
        tgts.exportTargets()
        assert len(list(tmp_path.glob("test_points*.json"))) == 2

        # Merge and re-export — old files should be deleted
        tgts.mergeAreas()
        tgts.exportTargets()
        assert len(list(tmp_path.glob("test_points*.json"))) == 1


# ---------------------------------------------------------------------------
# Scoring functions (from ext.py)
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_point_all_ones(self):
        """Full target mask with uniform weight → score = 1.0."""
        mask = np.ones((256, 256), dtype=np.float32)
        weight = np.ones((64, 64), dtype=np.float32)
        penalty = np.zeros((256, 256), dtype=np.float32)
        score = calcScore_point(
            np.array([128.0, 128.0]), mask, penalty, 0, weight
        )
        assert score == pytest.approx(1.0)

    def test_score_point_all_zeros(self):
        """Empty mask → score = 0."""
        mask = np.zeros((256, 256), dtype=np.float32)
        weight = np.ones((64, 64), dtype=np.float32)
        penalty = np.zeros((256, 256), dtype=np.float32)
        score = calcScore_point(
            np.array([128.0, 128.0]), mask, penalty, 0, weight
        )
        assert score == pytest.approx(0.0)

    def test_score_point_out_of_bounds(self):
        """Point too close to edge → score = 0."""
        mask = np.ones((256, 256), dtype=np.float32)
        weight = np.ones((64, 64), dtype=np.float32)
        penalty = np.zeros((256, 256), dtype=np.float32)
        score = calcScore_point(
            np.array([10.0, 10.0]), mask, penalty, 0, weight
        )
        assert score == 0

    def test_score_with_penalty(self):
        """Penalty mask reduces score."""
        mask = np.ones((256, 256), dtype=np.float32)
        weight = np.ones((64, 64), dtype=np.float32)
        penalty = np.ones((256, 256), dtype=np.float32)
        score_no_penalty = calcScore_point(
            np.array([128.0, 128.0]), mask, penalty, 0, weight
        )
        score_with_penalty = calcScore_point(
            np.array([128.0, 128.0]), mask, penalty, 0.5, weight
        )
        assert score_with_penalty < score_no_penalty

    def test_score_with_edge_weights(self):
        """Edge weight masks → max of edge scores returned."""
        mask = np.ones((256, 256), dtype=np.float32)
        weight = np.ones((64, 64), dtype=np.float32)
        penalty = np.zeros((256, 256), dtype=np.float32)
        edge1 = np.ones((64, 64), dtype=np.float32) * 0.5
        edge2 = np.ones((64, 64), dtype=np.float32) * 0.8
        score = calcScore_point(
            np.array([128.0, 128.0]), mask, penalty, 0, weight, [edge1, edge2]
        )
        # Max of (0.5, 0.8) applied to all-ones mask
        assert score == pytest.approx(0.8)

    def test_cluster_score_negative(self):
        """calcScore_cluster returns negative score (for minimization)."""
        mask = np.ones((256, 256), dtype=np.float32)
        weight = np.ones((64, 64), dtype=np.float32)
        penalty = np.zeros((256, 256), dtype=np.float32)
        points = np.array([[128.0, 128.0]])
        score = calcScore_cluster(
            np.zeros(2), points, mask, penalty, 0, weight
        )
        assert score < 0  # negated for minimization

    def test_cluster_offset_moves_score(self):
        """Non-zero offset shifts points, affecting score."""
        # Half the mask is ones, half is zeros
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[:, 128:] = 1.0
        weight = np.ones((64, 64), dtype=np.float32)
        penalty = np.zeros((256, 256), dtype=np.float32)
        # Point in the ones region
        points = np.array([[128.0, 192.0]])
        score_center = calcScore_cluster(
            np.zeros(2), points, mask, penalty, 0, weight
        )
        # Large offset pushing point into zeros region (offset * max(rec_dims) * 10 = -0.1 * 64 * 10 = -64 px)
        score_shifted = calcScore_cluster(
            np.array([0.0, -0.1]), points, mask, penalty, 0, weight
        )
        # Center score should be better (more negative)
        assert score_center < score_shifted


# ---------------------------------------------------------------------------
# Split areas
# ---------------------------------------------------------------------------

class TestSplitAreas:
    def test_split_creates_multiple_areas(self):
        tgts = make_targets(map_dims=(1024, 1024), rec_dims=(32, 32))
        # Add points in two clusters (spaced > rec_dims apart)
        for x in [100, 150, 200, 250]:
            tgts.addTarget(np.array([float(x), 100.0]))
        for x in [700, 750, 800, 850]:
            tgts.addTarget(np.array([float(x), 800.0]))
        assert len(tgts.areas) == 1
        assert len(tgts) == 8

        tgts.splitArea(area_num=2)
        assert len(tgts.areas) == 2
        assert len(tgts) == 8  # all points preserved

    def test_split_preserves_geo_points(self):
        tgts = make_targets(map_dims=(1024, 1024))
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([800.0, 800.0]))
        tgts.addGeoPoint(np.array([400.0, 400.0]))

        tgts.splitArea(area_num=2)
        # Geo points should be in all areas
        for area in tgts.areas:
            assert len(area.geo_points) >= 1

    def test_merge_then_split_roundtrip(self):
        tgts = make_targets(map_dims=(1024, 1024))
        tgts.addTarget(np.array([100.0, 100.0]))
        tgts.addTarget(np.array([800.0, 800.0]), new_area=True)
        assert len(tgts.areas) == 2

        tgts.mergeAreas()
        assert len(tgts.areas) == 1
        assert len(tgts) == 2

        tgts.splitArea(area_num=2)
        assert len(tgts.areas) == 2
        assert len(tgts) == 2
