"""Unit tests for SPACEtomo.modules.utils — pure functions with no SerialEM dependency."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# castString
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import castString


class TestCastString:
    def test_int(self):
        assert castString("42") == 42

    def test_negative_int(self):
        assert castString("-7") == -7

    def test_float(self):
        assert castString("3.14") == pytest.approx(3.14)

    def test_negative_float(self):
        assert castString("-0.5") == pytest.approx(-0.5)

    def test_bool_true(self):
        for val in ("True", "true", "yes", "Yes", "on"):
            assert castString(val) is True

    def test_bool_false(self):
        for val in ("False", "false", "no", "No", "off"):
            assert castString(val) is False

    def test_none(self):
        assert castString("None") is None
        assert castString("none") is None

    def test_empty_string(self):
        assert castString("") == ""

    def test_plain_string(self):
        assert castString("hello") == "hello"

    def test_quoted_string(self):
        assert castString("'hello'") == "hello"
        assert castString('"hello"') == "hello"

    def test_list_of_ints(self):
        assert castString("[1, 2, 3]") == [1, 2, 3]

    def test_list_of_strings(self):
        result = castString("['broken', 'gone']")
        assert result == ["broken", "gone"]

    def test_empty_list(self):
        assert castString("[]") == []

    def test_single_element_list(self):
        assert castString("[42]") == [42]

    def test_path(self, tmp_path):
        # Only returns Path if it exists on disk
        f = tmp_path / "test.txt"
        f.touch()
        result = castString(str(f))
        assert isinstance(result, Path)
        assert result == f

    def test_nonexistent_path_stays_string(self):
        result = castString("/nonexistent/path/xyz")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# convertToTaggedString / revertTaggedString  (JSON round-trip)
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import convertToTaggedString, revertTaggedString


class TestTaggedStrings:
    def test_numpy_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0])
        encoded = json.loads(json.dumps(arr, default=convertToTaggedString))
        decoded = revertTaggedString(encoded)
        np.testing.assert_array_equal(decoded, arr)

    def test_numpy_2d_roundtrip(self):
        arr = np.array([[1, 2], [3, 4]])
        encoded = json.loads(json.dumps(arr, default=convertToTaggedString))
        decoded = revertTaggedString(encoded)
        np.testing.assert_array_equal(decoded, arr)

    def test_path_roundtrip(self):
        p = Path("/some/path/file.txt")
        encoded = json.loads(json.dumps(p, default=convertToTaggedString))
        decoded = revertTaggedString(encoded)
        assert decoded == p

    def test_non_tagged_dict_passthrough(self):
        d = {"a": 1, "b": 2}
        assert revertTaggedString(d) == d

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            convertToTaggedString(object())


# ---------------------------------------------------------------------------
# findIndex
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import findIndex


class TestFindIndex:
    def test_found(self):
        lst = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert findIndex(lst, "name", "b") == 1

    def test_not_found(self):
        lst = [{"name": "a"}]
        assert findIndex(lst, "name", "z") is None

    def test_empty_list(self):
        assert findIndex([], "name", "a") is None


# ---------------------------------------------------------------------------
# guessMontageDims
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import guessMontageDims


class TestGuessMontageDims:
    def test_perfect_square(self):
        dims = guessMontageDims(16)
        assert dims[0] * dims[1] == 16
        np.testing.assert_array_equal(dims, [4, 4])

    def test_rectangular(self):
        dims = guessMontageDims(12)
        assert dims[0] * dims[1] == 12
        # Should pick factors closest to square root
        assert set(dims) == {3, 4}

    def test_prime(self):
        dims = guessMontageDims(7)
        assert dims[0] * dims[1] == 7
        assert 1 in dims and 7 in dims


# ---------------------------------------------------------------------------
# loadSettings / saveSettings round-trip
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import loadSettings, saveSettings


class TestSettings:
    def test_roundtrip(self, tmp_path):
        settings_file = tmp_path / "test_settings.ini"
        # saveSettings expects an ordered dict-like of variables
        from collections import OrderedDict
        variables = OrderedDict([
            ("automation_level", 4),
            ("grid_list", [1]),
            ("lamella", True),
            ("target_list", ["coating", "mitos"]),
            ("external_map_dir", ""),
        ])
        saveSettings(settings_file, variables, start="automation_level", end="external_map_dir")

        loaded = loadSettings(settings_file)
        assert loaded["automation_level"] == 4
        assert loaded["lamella"] is True
        assert loaded["target_list"] == ["coating", "mitos"]
        assert loaded["external_map_dir"] == ""

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            loadSettings(tmp_path / "nonexistent.ini")


# ---------------------------------------------------------------------------
# synonymKeys
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import synonymKeys


class TestSynonymKeys:
    def test_rename(self):
        d = {"old_key": 42, "keep": "yes"}
        result = synonymKeys(d, ["old_key"], ["new_key"])
        assert "new_key" in result
        assert "old_key" not in result
        assert result["new_key"] == 42
        assert result["keep"] == "yes"

    def test_missing_key_noop(self):
        d = {"a": 1}
        result = synonymKeys(d, ["missing"], ["new"])
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# hashFile
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import hashFile


class TestHashFile:
    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("hello")
        assert hashFile(f1) == hashFile(f2)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert hashFile(f1) != hashFile(f2)

    def test_nonexistent_returns_hash(self, tmp_path):
        result = hashFile(tmp_path / "nope.txt")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# alignCC
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import alignCC


class TestAlignCC:
    def test_no_shift(self):
        img = np.random.rand(64, 64).astype(np.float32)
        shift, score = alignCC(img, img)
        np.testing.assert_array_equal(shift, [0, 0])

    def test_known_shift(self):
        rng = np.random.RandomState(42)
        img = rng.rand(128, 128).astype(np.float32)
        # Shift by (5, 3) using roll
        shifted = np.roll(np.roll(img, 5, axis=1), 3, axis=0)
        shift, score = alignCC(img, shifted)
        assert abs(abs(shift[0]) - 5) <= 1
        assert abs(abs(shift[1]) - 3) <= 1

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError):
            alignCC(np.zeros((32, 32)), np.zeros((64, 64)))


# ---------------------------------------------------------------------------
# writeMrc round-trip
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import writeMrc


class TestWriteMrc:
    def test_roundtrip(self, tmp_path):
        import mrcfile
        img = np.random.rand(64, 64).astype(np.float32)
        f = tmp_path / "test.mrc"
        writeMrc(f, img, pix_size=1.5)

        with mrcfile.open(f) as mrc:
            np.testing.assert_array_almost_equal(mrc.data, img)
            assert mrc.voxel_size.x == pytest.approx(15.0)  # 1.5 * 10

    def test_float64_converted(self, tmp_path):
        import mrcfile
        img = np.random.rand(32, 32)  # float64 by default
        f = tmp_path / "test64.mrc"
        writeMrc(f, img, pix_size=1.0)

        with mrcfile.open(f) as mrc:
            assert mrc.data.dtype == np.float32


# ---------------------------------------------------------------------------
# writeTargets
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import writeTargets


class TestWriteTargets:
    def test_basic_output(self, tmp_path):
        f = tmp_path / "test_tgts.txt"
        targets = [
            {"tsX": 1.0, "tsY": 2.0},
            {"tsX": 3.0, "tsY": 4.0},
        ]
        writeTargets(f, targets)
        content = f.read_text()
        assert "_tgt = 001" in content
        assert "_tgt = 002" in content
        assert "tsX = 1.0" in content

    def test_with_settings(self, tmp_path):
        f = tmp_path / "test_tgts.txt"
        targets = [{"tsX": 0}]
        writeTargets(f, targets, settings={"startTilt": 0, "minTilt": -60})
        content = f.read_text()
        assert "_set startTilt = 0" in content
        assert "_set minTilt = -60" in content

    def test_with_geo_points(self, tmp_path):
        f = tmp_path / "test_tgts.txt"
        targets = [{"tsX": 0}]
        geo = [{"stageX": 1.0, "stageY": 2.0}]
        writeTargets(f, targets, geo_points=geo)
        content = f.read_text()
        assert "_geo = 1" in content


# ---------------------------------------------------------------------------
# cyclicRange
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import cyclicRange


class TestCyclicRange:
    def test_generates_values(self):
        gen = cyclicRange(0, 5, 1)
        values = [next(gen) for _ in range(12)]
        # Should cycle: 0,1,2,3,4, 0,1,2,3,4, 0,1
        assert values[:5] == [0, 1, 2, 3, 4]
        assert values[5:10] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# File check helpers
# ---------------------------------------------------------------------------

from SPACEtomo.modules.utils import hasWGMap, hasIMAlign, hasMMMap, hasTargetSetup, findInspectionStatus


class TestFileChecks:
    def test_hasWGMap(self, tmp_path):
        map_dir = tmp_path / "SPACE_maps"
        map_dir.mkdir()
        assert not hasWGMap(tmp_path, map_dir, "G01")
        (map_dir / "G01_wg.png").touch()
        assert hasWGMap(tmp_path, map_dir, "G01")

    def test_hasTargetSetup(self, tmp_path):
        assert not hasTargetSetup(tmp_path, tmp_path, "G01")
        (tmp_path / "G01_FP1_tgts.txt").touch()
        assert hasTargetSetup(tmp_path, tmp_path, "G01")

    def test_findInspectionStatus(self, tmp_path):
        status = findInspectionStatus("G01_FP1", tmp_path)
        assert status["wg_inspected"] is False
        assert status["mm_inspected"] is False
        assert status["points_ready"] is False

        (tmp_path / "G01_FP1_inspected.txt").touch()
        (tmp_path / "G01_FP1_points0.json").touch()
        status = findInspectionStatus("G01_FP1", tmp_path)
        assert status["mm_inspected"] is True
        assert status["points_ready"] is True
