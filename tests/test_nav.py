"""Unit tests for SPACEtomo.modules.nav — Navigator file parsing, writing, and item manipulation.

Navigator file format (SerialEM):
  - Header lines (e.g. "AdocVersion = 2.00")
  - Items delimited by "[Item = <label>]" followed by key-value entries, terminated by blank line
  - Entry format: "Key = value1 value2 ..."
  - Key entries: Type (0=point, 1=polygon, 2=map), StageXYZ, NumPts, PtsX, PtsY,
    Color, Note, MapFile, MapID, MapScaleMat, MapBinning, MontBinning, etc.
"""

from pathlib import Path
import numpy as np
import pytest

from SPACEtomo.modules.nav import NavItem, Navigator


# ---------------------------------------------------------------------------
# Sample nav file content
# ---------------------------------------------------------------------------

SAMPLE_NAV = """\
AdocVersion = 2.00
LastSavedAs = test.nav

[Item = WG01]
Color = 2
StageXYZ = 100.5 200.3 -5.0
NumPts = 5
Regis = 1
Type = 2
Note = G01.mrc
MapFile = /tmp/G01.mrc
MapID = 11
MapMontage = 1
MapSection = 0
MapBinning = 1
MontBinning = 4
MapMagInd = 8
MapCamera = 0
MapScaleMat = 0.5 0.0 0.0 0.5
MapWidthHeight = 4096 4096
PtsX = 96.5 96.5 104.5 104.5 96.5
PtsY = 196.3 204.3 204.3 196.3 196.3

[Item = PP01]
Color = 1
StageXYZ = 101.0 201.0 -5.0
NumPts = 4
Regis = 1
Type = 1
Note = lamella_1
MapID = 12
PtsX = 100.0 100.0 102.0 102.0
PtsY = 200.0 202.0 202.0 200.0

[Item = T01]
Color = 0
StageXYZ = 101.5 201.5 -5.0
NumPts = 1
Regis = 1
Type = 0
Note = target_1
MapID = 13
GroupID = 100
PtsX = 101.5
PtsY = 201.5

"""


@pytest.fixture
def nav_file(tmp_path):
    """Write sample nav to a temp file."""
    f = tmp_path / "test.nav"
    f.write_text(SAMPLE_NAV)
    return f


@pytest.fixture
def nav(nav_file):
    """Parse the sample nav file."""
    n = Navigator.__new__(Navigator)
    n.file = nav_file
    n.header = ""
    n.items = []
    n.selected_item = None
    n.readFromFile()
    n.setMapIDCounter()
    n.getSelectedItem()
    return n


# ---------------------------------------------------------------------------
# NavItem parsing
# ---------------------------------------------------------------------------

class TestNavItemParsing:
    def test_item_count(self, nav):
        assert len(nav.items) == 3

    def test_header_preserved(self, nav):
        assert "AdocVersion" in nav.header
        assert "LastSavedAs" in nav.header

    def test_map_item(self, nav):
        item = nav.items[0]
        assert item.label == "WG01"
        assert item.item_type == 2  # map
        assert item.note == "G01.mrc"
        assert item.map_file == Path("/tmp/G01.mrc")
        np.testing.assert_array_almost_equal(item.stage, [100.5, 200.3, -5.0], decimal=4)

    def test_polygon_item(self, nav):
        item = nav.items[1]
        assert item.label == "PP01"
        assert item.item_type == 1  # polygon
        assert item.note == "lamella_1"
        assert len(item.entries["PtsX"]) == 4

    def test_point_item(self, nav):
        item = nav.items[2]
        assert item.label == "T01"
        assert item.item_type == 0  # point
        assert item.note == "target_1"
        assert item.entries["GroupID"] == ["100"]


# ---------------------------------------------------------------------------
# NavItem write round-trip
# ---------------------------------------------------------------------------

class TestNavRoundTrip:
    def test_write_read_roundtrip(self, nav, tmp_path):
        """Write nav to file and re-read — items should match."""
        out_file = tmp_path / "roundtrip.nav"
        nav.writeToFile(out_file)

        nav2 = Navigator.__new__(Navigator)
        nav2.file = out_file
        nav2.header = ""
        nav2.items = []
        nav2.selected_item = None
        nav2.readFromFile()
        nav2.setMapIDCounter()

        assert len(nav2.items) == len(nav.items)
        for orig, loaded in zip(nav.items, nav2.items):
            assert orig.label == loaded.label
            assert orig.item_type == loaded.item_type
            assert orig.note == loaded.note
            np.testing.assert_array_almost_equal(orig.stage, loaded.stage)

    def test_empty_nav_roundtrip(self, tmp_path):
        """An empty nav file should round-trip cleanly."""
        nav = Navigator.__new__(Navigator)
        nav.file = tmp_path / "empty.nav"
        nav.header = "AdocVersion = 2.00\n"
        nav.items = []
        nav.selected_item = None
        nav.writeToFile()

        nav2 = Navigator.__new__(Navigator)
        nav2.file = nav.file
        nav2.header = ""
        nav2.items = []
        nav2.selected_item = None
        nav2.readFromFile()
        assert len(nav2.items) == 0
        assert "AdocVersion" in nav2.header

    def test_default_header_added(self, tmp_path):
        """Writing without a header should add AdocVersion."""
        nav = Navigator.__new__(Navigator)
        nav.file = tmp_path / "no_header.nav"
        nav.header = ""
        nav.items = []
        nav.writeToFile()

        content = nav.file.read_text()
        assert "AdocVersion = 2.00" in content


# ---------------------------------------------------------------------------
# NavItem creation
# ---------------------------------------------------------------------------

class TestNavItemCreation:
    def test_create_point(self):
        item = NavItem(0, "P01")
        item.createPoint(coords=[10.0, 20.0, -3.0], map_id=99, note="test point")
        assert item.item_type == 0
        np.testing.assert_array_almost_equal(item.stage, [10.0, 20.0, -3.0])
        assert item.note == "test point"
        assert item.entries["MapID"] == ["99"]

    def test_create_point_needs_3d(self):
        item = NavItem(0, "P01")
        with pytest.raises(ValueError, match="3D"):
            item.createPoint(coords=[10.0, 20.0], map_id=99)

    def test_create_polygon(self):
        pts_x = [0.0, 0.0, 10.0, 10.0]
        pts_y = [0.0, 10.0, 10.0, 0.0]
        item = NavItem(0, "Poly")
        item.createPolygon(pts_x, pts_y, z=-5.0, map_id=50, note="test poly")
        assert item.item_type == 1
        assert item.entries["NumPts"] == ["4"]
        # Center should be at (5, 5, -5)
        np.testing.assert_array_almost_equal(item.stage, [5.0, 5.0, -5.0])


# ---------------------------------------------------------------------------
# NavItem modification
# ---------------------------------------------------------------------------

class TestNavItemModification:
    def test_change_color(self, nav):
        item = nav.items[0]
        item.changeColor(3)
        assert item.entries["Color"] == ["3"]

    def test_change_label(self, nav):
        item = nav.items[0]
        item.changeLabel("NewLabel")
        assert item.label == "NewLabel"

    def test_change_note(self, nav):
        item = nav.items[0]
        item.changeNote("new note text")
        assert item.note == "new note text"

    def test_change_draw(self, nav):
        item = nav.items[1]
        item.changeDraw(False)
        assert item.entries["Drawn"] == ["0"]
        item.changeDraw(True)
        assert item.entries["Drawn"] == ["1"]

    def test_change_acquire(self, nav):
        item = nav.items[2]
        item.changeAcquire(1)
        assert item.entries["Acquire"] == ["1"]

    def test_change_stage_absolute(self, nav):
        item = nav.items[2]
        old_pts_x = float(item.entries["PtsX"][0])
        item.changeStage(np.array([50.0, 60.0, -2.0]))
        np.testing.assert_array_almost_equal(item.stage, [50.0, 60.0, -2.0])
        # Point should move with stage
        assert float(item.entries["PtsX"][0]) == pytest.approx(50.0)

    def test_change_stage_relative(self, nav):
        item = nav.items[1]  # polygon at (101, 201, -5)
        old_stage = item.stage.copy()
        item.changeStage(np.array([1.0, -1.0]), relative=True)
        np.testing.assert_array_almost_equal(item.stage[:2], old_stage[:2] + [1.0, -1.0])

    def test_change_z(self, nav):
        item = nav.items[0]
        item.changeZ(-10.0)
        assert item.stage[2] == pytest.approx(-10.0)

    def test_change_z_relative(self, nav):
        item = nav.items[0]
        old_z = item.stage[2]
        item.changeZ(2.0, relative=True)
        assert item.stage[2] == pytest.approx(old_z + 2.0)

    def test_scale_bounds(self, nav):
        item = nav.items[1]  # polygon
        old_pts_x = np.array(item.entries["PtsX"], dtype=float)
        old_center = item.stage[0]

        item.scaleBounds(2.0)

        new_pts_x = np.array(item.entries["PtsX"], dtype=float)
        # Distances from center should double
        old_rel = old_pts_x - old_center
        new_rel = new_pts_x - old_center
        np.testing.assert_array_almost_equal(new_rel, old_rel * 2.0)

    def test_scale_bounds_only_polygon(self, nav):
        item = nav.items[0]  # map, type 2
        # Should log error and return without changing
        item.scaleBounds(2.0)
        # No crash is the assertion

    def test_add_entry(self, nav):
        item = nav.items[0]
        item.addEntry("CustomKey", [1, 2, 3])
        assert item.entries["CustomKey"] == ["1", "2", "3"]

        item.addEntry("SingleVal", 42)
        assert item.entries["SingleVal"] == ["42"]


# ---------------------------------------------------------------------------
# NavItem geometry
# ---------------------------------------------------------------------------

class TestNavItemGeometry:
    def test_polygon_area(self):
        # 10x10 square
        pts_x = [0.0, 0.0, 10.0, 10.0]
        pts_y = [0.0, 10.0, 10.0, 0.0]
        item = NavItem(0, "Poly")
        item.createPolygon(pts_x, pts_y, z=0.0, map_id=1)
        assert item.area == pytest.approx(100.0)

    def test_point_area_is_zero(self):
        item = NavItem(0, "P")
        item.createPoint([0.0, 0.0, 0.0], map_id=1)
        assert item.area == 0

    def test_get_distance(self, nav):
        item = nav.items[0]  # at (100.5, 200.3)
        dist = item.getDistance([100.5, 200.3])
        assert dist == pytest.approx(0.0, abs=1e-4)

        dist = item.getDistance([103.5, 204.3])
        expected = np.sqrt(3.0**2 + 4.0**2)
        assert dist == pytest.approx(expected, abs=0.01)

    def test_get_vector(self, nav):
        item = nav.items[0]  # at (100.5, 200.3)
        vec = item.getVector([103.5, 204.3])
        np.testing.assert_array_almost_equal(vec, [3.0, 4.0], decimal=4)


# ---------------------------------------------------------------------------
# Navigator search
# ---------------------------------------------------------------------------

class TestNavigatorSearch:
    def test_get_id_from_note(self, nav):
        assert nav.getIDfromNote("G01.mrc") == 0
        assert nav.getIDfromNote("lamella_1") == 1
        assert nav.getIDfromNote("target_1") == 2
        assert nav.getIDfromNote("nonexistent", warn=False) is None

    def test_search_by_entry_exact(self, nav):
        results = nav.searchByEntry("Type", "1")
        assert results == [1]  # polygon

    def test_search_by_entry_partial(self, nav):
        results = nav.searchByEntry("Note", "lamella", partial=True)
        assert results == [1]

    def test_search_by_label(self, nav):
        results = nav.searchByEntry("label", "T01")
        assert results == [2]

    def test_search_by_entry_with_subset(self, nav):
        results = nav.searchByEntry("Type", "0", subset=[0, 2])
        assert results == [2]

    def test_search_by_coords(self, nav):
        # Search near the polygon center
        results = nav.searchByCoords([101.0, 201.0], margin=0.5)
        assert 1 in results

    def test_search_by_coords_no_match(self, nav):
        results = nav.searchByCoords([999.0, 999.0], margin=1.0)
        assert results == []

    def test_search_by_coords_with_subset(self, nav):
        results = nav.searchByCoords([101.0, 201.0], margin=5.0, subset=[0])
        # Item 0 is at (100.5, 200.3), within 5um of (101, 201)
        assert 0 in results
        assert 1 not in results  # excluded by subset


# ---------------------------------------------------------------------------
# Navigator item management
# ---------------------------------------------------------------------------

class TestNavigatorManagement:
    def test_map_id_counter(self, nav):
        # IDs in sample: 11, 12, 13 -> counter should be 14
        assert nav.map_id_counter == 14

    def test_new_point(self, nav, tmp_path):
        nav.file = tmp_path / "nav_with_point.nav"
        old_len = len(nav)
        nav_id = nav.newPoint([50.0, 60.0, -3.0], label="NewPt", note="added")
        assert len(nav) == old_len + 1
        assert nav.items[nav_id].label == "NewPt"
        assert nav.items[nav_id].item_type == 0

    def test_new_polygon(self, nav, tmp_path):
        nav.file = tmp_path / "nav_with_poly.nav"
        pts_x = [0.0, 0.0, 5.0, 5.0]
        pts_y = [0.0, 5.0, 5.0, 0.0]
        nav_id = nav.newPolygon(pts_x, pts_y, z=-5.0, label="NewPoly", note="test")
        assert nav.items[nav_id].item_type == 1
        assert nav.items[nav_id].label == "NewPoly"

    def test_new_point_group(self, nav, tmp_path):
        nav.file = tmp_path / "nav_group.nav"
        points = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
        old_len = len(nav)
        nav.newPointGroup(points, label_prefix="G", color_id=1, stage_z=-5.0)
        assert len(nav) == old_len + 3
        # Check labels
        assert nav.items[-3].label == "G1"
        assert nav.items[-2].label == "G2"
        assert nav.items[-1].label == "G3"

    def test_replace_item(self, nav, tmp_path):
        nav.file = tmp_path / "nav_replace.nav"
        new_item = NavItem(0, "Replacement")
        new_item.createPoint([0.0, 0.0, 0.0], map_id=999, note="replaced")
        nav.replaceItem(1, new_item)
        assert nav.items[1].label == "Replacement"
        assert nav.items[1].note == "replaced"

    def test_selected_item(self, nav):
        # Selected item is set during Navigator init
        assert nav.selected_item is not None
        assert nav.selected_item in nav.items

    def test_set_selected_item(self, nav):
        nav.setSelectedItem(2)
        assert nav.selected_item == nav.items[2]

    def test_len_and_iter(self, nav):
        assert len(nav) == 3
        labels = [item.label for item in nav]
        assert labels == ["WG01", "PP01", "T01"]


# ---------------------------------------------------------------------------
# Map item with downgrade on missing file
# ---------------------------------------------------------------------------

class TestMapDowngrade:
    def test_map_downgrades_to_polygon_if_file_missing(self, nav):
        item = nav.items[0]
        assert item.item_type == 2
        # map_file points to /tmp/G01.mrc which doesn't exist
        block = item.getBlock()
        assert "Type = 1" in block  # downgraded to polygon

    def test_map_stays_if_file_exists(self, nav, tmp_path):
        item = nav.items[0]
        mrc_file = tmp_path / "G01.mrc"
        # Create a minimal mrc
        from SPACEtomo.modules.utils import writeMrc
        writeMrc(mrc_file, np.zeros((10, 10), dtype=np.float32), 1.0)
        item.map_file = mrc_file
        block = item.getBlock()
        assert "Type = 2" in block


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_item_without_note(self, tmp_path):
        """Items without Note entry should have empty note."""
        nav_text = """\
AdocVersion = 2.00

[Item = NoNote]
Color = 0
StageXYZ = 0.0 0.0 0.0
NumPts = 1
Regis = 1
Type = 0
MapID = 1
PtsX = 0.0
PtsY = 0.0

"""
        f = tmp_path / "no_note.nav"
        f.write_text(nav_text)

        nav = Navigator.__new__(Navigator)
        nav.file = f
        nav.header = ""
        nav.items = []
        nav.selected_item = None
        nav.readFromFile()
        assert nav.items[0].note == ""

    def test_item_without_mapfile(self, tmp_path):
        """Non-map items don't have MapFile."""
        nav_text = """\
AdocVersion = 2.00

[Item = Pt]
Color = 0
StageXYZ = 5.0 10.0 0.0
NumPts = 1
Regis = 1
Type = 0
MapID = 1
PtsX = 5.0
PtsY = 10.0

"""
        f = tmp_path / "no_mapfile.nav"
        f.write_text(nav_text)

        nav = Navigator.__new__(Navigator)
        nav.file = f
        nav.header = ""
        nav.items = []
        nav.selected_item = None
        nav.readFromFile()
        assert nav.items[0].map_file is None

    def test_note_with_spaces(self, tmp_path):
        """Notes with spaces should round-trip correctly."""
        nav_text = """\
AdocVersion = 2.00

[Item = SpaceNote]
Color = 0
StageXYZ = 0.0 0.0 0.0
NumPts = 1
Regis = 1
Type = 0
Note = this is a multi word note
MapID = 1
PtsX = 0.0
PtsY = 0.0

"""
        f = tmp_path / "space_note.nav"
        f.write_text(nav_text)

        nav = Navigator.__new__(Navigator)
        nav.file = f
        nav.header = ""
        nav.items = []
        nav.selected_item = None
        nav.readFromFile()
        assert nav.items[0].note == "this is a multi word note"

        # Round-trip
        out = tmp_path / "space_note_out.nav"
        nav.writeToFile(out)

        nav2 = Navigator.__new__(Navigator)
        nav2.file = out
        nav2.header = ""
        nav2.items = []
        nav2.selected_item = None
        nav2.readFromFile()
        assert nav2.items[0].note == "this is a multi word note"
