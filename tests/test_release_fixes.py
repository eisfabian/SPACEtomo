"""Regression tests for the 1.3.1 bug fixes (no SerialEM needed, sem is replaced by small stubs)."""

import ast
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


class SEMerror(Exception):
    pass


class StubSEM:
    """Minimal stand-in for the serialem module that records calls."""

    SEMerror = SEMerror

    def __init__(self):
        self.calls = []
        self.try_level = 0

    def StartTry(self, *args):
        self.try_level += 1

    def EndTry(self):
        self.try_level -= 1


# --- Packaging ---

def _pyproject():
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_package_discovery_excludes_models_and_tests(tmp_path):
    from setuptools import find_namespace_packages, find_packages

    # Mimic a working checkout with local model backup config and test package
    for pkg in ("SPACEtomo", "SPACEtomo/modules", "tests"):
        (tmp_path / pkg).mkdir(parents=True)
        (tmp_path / pkg / "__init__.py").touch()
    (tmp_path / "SPACEtomo/models").mkdir()
    (tmp_path / "SPACEtomo/models/config.py").touch()

    find_cfg = _pyproject()["tool"]["setuptools"]["packages"]
    find_cfg = find_cfg.get("find", find_cfg)
    finder = find_namespace_packages if find_cfg.get("namespaces", True) else find_packages
    packages = finder(where=str(tmp_path), include=find_cfg.get("include", ["*"]), exclude=find_cfg.get("exclude", []))

    assert "SPACEtomo" in packages and "SPACEtomo.modules" in packages
    assert not [p for p in packages if p.startswith(("SPACEtomo.models", "tests"))]


def test_gui_icons_are_package_data():
    patterns = _pyproject()["tool"]["setuptools"].get("package-data", {}).get("SPACEtomo", [])
    icons = sorted((REPO / "SPACEtomo").glob("img/*.png"))
    assert icons
    for icon in icons:
        rel = icon.relative_to(REPO / "SPACEtomo")
        assert any(rel.match(pattern) for pattern in patterns), f"{rel} would be missing from the wheel"


# --- utils_sem ---

def test_getSessionDir_clears_persistent_vars_for_missing_dir(monkeypatch, tmp_path):
    import SPACEtomo.modules.utils_sem as usem

    class Exited(Exception):
        pass

    sem = StubSEM()
    sem.IsVariableDefined = lambda name: 1
    sem.GetVariable = lambda name: str(tmp_path / "gone")
    sem.ClearPersistentVars = lambda: sem.calls.append("ClearPersistentVars")
    def exit_():
        raise Exited()
    sem.Exit = exit_
    monkeypatch.setattr(usem, "sem", sem, raising=False)

    with pytest.raises(Exited):
        usem.getSessionDir()
    assert sem.calls == ["ClearPersistentVars"]


def test_getLowDoseParams_ends_try_on_failure(monkeypatch):
    import SPACEtomo.modules.utils_sem as usem

    sem = StubSEM()
    def fail(*args):
        raise SEMerror("unknown command")
    sem.GetLowDoseAreaParams = fail
    monkeypatch.setattr(usem, "sem", sem, raising=False)

    assert usem.getLowDoseParams("V") is None
    assert sem.try_level == 0


def test_getLowDoseParams_ends_try_on_success(monkeypatch):
    import SPACEtomo.modules.utils_sem as usem

    sem = StubSEM()
    sem.GetLowDoseAreaParams = lambda *args: None
    sem.GetVariable = lambda name: [0.0, 25.0] + [0.0] * 19
    monkeypatch.setattr(usem, "sem", sem, raising=False)

    assert usem.getLowDoseParams("V")["mag_index"] == 25.0
    assert sem.try_level == 0


# --- Apertures ---

def _microscope(autoloader_type):
    from SPACEtomo.modules.scope import Microscope
    mic = Microscope.__new__(Microscope)
    mic.autoloader_type = autoloader_type
    return mic


def _aperture_sem(accepts):
    """Stub whose aperture commands only accept ids of type accepts (like PyArg_ParseTuple)."""
    sem = StubSEM()
    def check(aperture):
        if not isinstance(aperture, accepts):
            raise TypeError(f"expected {accepts.__name__}, got {type(aperture).__name__}")
    def report(aperture):
        check(aperture)
        sem.calls.append(("Report", aperture))
        return 50.0
    def set_(aperture, size):
        check(aperture)
        sem.calls.append(("Set", aperture, size))
    sem.ReportApertureSize = report
    sem.SetApertureSize = set_
    return sem


@pytest.mark.parametrize("autoloader_type", ["thermo", "jeol"])
def test_apertures_use_integer_ids(monkeypatch, autoloader_type):
    import SPACEtomo.modules.scope as scope

    sem = _aperture_sem(int)
    monkeypatch.setattr(scope, "sem", sem, raising=False)
    mic = _microscope(autoloader_type)

    assert mic.condenser_aperture == 50
    assert mic.objective_aperture == 50
    mic.changeC2Aperture(70)
    mic.changeObjAperture(100)
    assert sem.calls == [("Report", 1), ("Report", 2), ("Set", 1, 70), ("Set", 2, 100)]
    assert sem.try_level == 0


@pytest.mark.parametrize("autoloader_type", ["thermo", "jeol"])
def test_apertures_fall_back_to_letters(monkeypatch, autoloader_type):
    import SPACEtomo.modules.scope as scope

    sem = _aperture_sem(str)
    monkeypatch.setattr(scope, "sem", sem, raising=False)
    mic = _microscope(autoloader_type)

    assert mic.condenser_aperture == 50
    assert mic.objective_aperture == 50
    mic.changeC2Aperture(70)
    mic.changeObjAperture(100)
    assert sem.calls == [("Report", "C"), ("Report", "O"), ("Set", "C", 70), ("Set", "O", 100)]
    assert sem.try_level == 0


# --- IM alignment ---

def test_im_alignment_of_last_roi_does_not_shift_all_items(monkeypatch, tmp_path):
    import SPACEtomo.modules.ima as ima

    class StopAfterAlignment(Exception):
        pass

    shifts = []
    nav = SimpleNamespace(
        items=[SimpleNamespace(label="PP1", entries={"Type": ["1"]}, stage=np.zeros(3), note="")],
        shiftItems=lambda shift, item_ids=[], skip_item_ids=[]: shifts.append(list(item_ids)),
        newMap=lambda **kwargs: (_ for _ in ()).throw(StopAfterAlignment()),
    )
    microscope = SimpleNamespace(autoloader={1: "G01"}, loaded_grid=1, changeImagingState=lambda *a, **k: None,
                                 changeC2Aperture=lambda *a, **k: None, moveStage=lambda *a, **k: None)

    ali = ima.IMAlignment.__new__(ima.IMAlignment)
    ali.cur_dir = tmp_path
    ali.microscope = microscope
    ali.nav = nav
    ali.imaging_params = SimpleNamespace(IM_image_state=1)
    ali.label_prefix = "PP"
    ali.position_ids = [0]      # Single ROI on grid
    ali.offsets = np.zeros((1, 2))
    ali.shifted = False
    (tmp_path / "G01_PP1_IMref.mrc").touch()     # v1.4 names IM references by display label

    monkeypatch.setattr(ali, "checkNecessity", lambda: True)
    monkeypatch.setattr(ali, "findEucentricity", lambda: None)
    monkeypatch.setattr(ali, "alignToRef", lambda ref_buffer: (None, np.array([1.0, 2.0])))
    monkeypatch.setattr(ima, "Buffer", lambda *args, **kwargs: None)

    with pytest.raises(StopAfterAlignment):
        ali.align(settings={})
    assert shifts == []
    assert np.allclose(ali.offsets[0], [1.0, 2.0])


# --- External monitoring ---

@pytest.mark.parametrize("script", ["SPACEtomo/run.py", "SPACEtomo/run_manual.py"])
def test_wait_callbacks_are_not_nested_lambdas(script):
    """A lambda returning a lambda would never be called by waitForFile (e.g. monitorExternal)."""
    tree = ast.parse((REPO / script).read_text())
    nested = [node.lineno for node in ast.walk(tree)
              if isinstance(node, ast.Lambda) and isinstance(node.body, ast.IfExp)
              and isinstance(node.body.orelse, ast.Lambda)]
    assert nested == []


# --- Post action target setup ---

def test_prepTargets_picks_up_new_point_files(monkeypatch, tmp_path):
    import SPACEtomo.run_prepTargets as prep

    map_dir = tmp_path
    (map_dir / "G01_L01_points0.json").touch()
    (tmp_path / "G01_L01_tgts.txt").touch()     # First map already set up

    class Nav:
        def searchByEntry(self, *args, **kwargs):
            return []
        def getIDfromNote(self, note):
            return 0

    seg_lists = iter([["G01_L02"], []])
    def monitorFiles(directory):
        seg_list = next(seg_lists)
        if not seg_list:    # Segmentation finished and point file appeared
            (map_dir / "G01_L02_points0.json").touch()
        return [], seg_list, []

    set_up = []
    class Area:
        def __init__(self, point_file, *args):
            self.point_file = point_file
        def __len__(self):
            return 1
        def checkISLimit(self):
            pass
        def prepareForAcquisition(self, *args):
            set_up.append(self.point_file.name)
        def getTargetsInfo(self, area_name):
            return [], [], {}

    monkeypatch.setattr(prep.utils, "getCurDir", lambda: tmp_path)
    monkeypatch.setattr(prep.utils, "loadSettings", lambda file: {"external_map_dir": "", "MM_wait_for_inspection": False})
    monkeypatch.setattr(prep.utils, "writeTargets", lambda *args, **kwargs: None)
    monkeypatch.setattr(prep.usem, "prepareEnvironment", lambda cur_dir, ext_dir: (map_dir, True))
    monkeypatch.setattr(prep, "ImagingParams", lambda *args: None)
    monkeypatch.setattr(prep, "Navigator", lambda **kwargs: Nav())
    monkeypatch.setattr(prep, "WGModel", lambda *args: None)
    monkeypatch.setattr(prep.space_ext, "MMModel", lambda: SimpleNamespace(setDimensions=lambda params: None))
    monkeypatch.setattr(prep.space_ext, "TgtParams", lambda **kwargs: None)
    monkeypatch.setattr(prep.space_ext, "monitorFiles", monitorFiles)
    monkeypatch.setattr(prep, "Buffer", lambda **kwargs: None)
    monkeypatch.setattr(prep, "PACEArea", Area)
    monkeypatch.setattr(prep.time, "sleep", lambda s: None)

    prep.main()
    assert set_up == ["G01_L02_points0.json"]


# --- Pillow compatibility of toNumpy ---

@pytest.mark.parametrize("mode", ["L", "RGB", "I;16", "F"])
def test_toNumpy_matches_np_array(mode):
    from PIL import Image
    from SPACEtomo.modules import utils

    base = np.random.default_rng(0).integers(0, 255, (37, 53), dtype=np.uint8)
    arr = {"L": base, "RGB": np.dstack([base] * 3), "I;16": base.astype(np.uint16) * 200, "F": base.astype(np.float32) / 7}[mode]
    img = Image.fromarray(arr)

    np.testing.assert_array_equal(utils.toNumpy(img), np.array(img))


def test_toNumpy_handles_setimage_requiring_extents(monkeypatch):
    """Pillow 13 made the extents argument of ImagingEncoder.setimage mandatory."""
    from PIL import Image
    from SPACEtomo.modules import utils

    img = Image.fromarray(np.arange(37 * 53, dtype=np.uint8).reshape(37, 53))
    expected = np.array(img)    # np.array uses the encoder internally, so compute before patching

    real_getencoder = Image._getencoder

    class StrictEncoder:
        def __init__(self, encoder):
            self._encoder = encoder

        def setimage(self, im, extents):
            return self._encoder.setimage(im, extents)

        def encode(self, bufsize):
            return self._encoder.encode(bufsize)

    monkeypatch.setattr(Image, "_getencoder", lambda *args: StrictEncoder(real_getencoder(*args)))

    np.testing.assert_array_equal(utils.toNumpy(img), expected)


# --- PyTorch cap must stay installable on new Python versions ---

@pytest.mark.parametrize("extra", ["predict", "ti"])
@pytest.mark.parametrize("package, capped, first_new", [("torch", "2.8.0", "2.9.0"), ("torchvision", "0.23.0", "0.24.0")])
@pytest.mark.parametrize("python_version", ["3.9", "3.11", "3.12", "3.13", "3.14", "3.15"])
def test_torch_cap_only_below_python_314(python_version, package, capped, first_new, extra):
    """K3 computers need torch<2.9, but Python 3.14 has no torch<2.9 wheels: an unconditional cap made pip fall back to SPACEtomo 1.3.0."""
    from packaging.requirements import Requirement

    reqs = [Requirement(r) for r in _pyproject()["project"]["optional-dependencies"][extra]]
    env = {"python_version": python_version, "python_full_version": python_version + ".0", "extra": extra}
    active = [r for r in reqs if r.name == package and (r.marker is None or r.marker.evaluate(env))]
    if not active and not any(r.name == package for r in reqs):
        pytest.skip(f"{package} not in {extra} extra")

    assert len(active) == 1
    if tuple(map(int, python_version.split("."))) < (3, 14):
        assert active[0].specifier.contains(capped) and not active[0].specifier.contains(first_new)
    else:
        assert active[0].specifier.contains(first_new)


def test_setup_py_predict_matches_pyproject():
    tree = ast.parse((REPO / "setup.py").read_text())
    setup_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setup")
    extras = ast.literal_eval(next(k.value for k in setup_call.keywords if k.arg == "extras_require"))
    torch_reqs = lambda reqs: sorted(r.replace(" ", "") for r in reqs if r.startswith("torch"))

    assert torch_reqs(extras["predict"]) == torch_reqs(_pyproject()["project"]["optional-dependencies"]["predict"])
