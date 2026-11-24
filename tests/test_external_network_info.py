"""External processing: the microscope PC gets the segmentation model's classes and pixel size from SPACEmonitor
(via the shared map folder) instead of needing the model itself."""

import json

import pytest

from SPACEtomo import config


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "seg_model"
    model_dir.mkdir()
    (model_dir / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "mitos": 1, "ice": 2}, "pixel_size": 2.5}))
    monkeypatch.setattr(config, "MM_model_folder", str(model_dir))
    return model_dir


def test_monitor_shares_model_info(local_model, tmp_path, monkeypatch):
    from SPACEtomo.modules import ext

    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    ext.shareModelInfo(map_dir)                                         # SPACEmonitor on the processing PC

    monkeypatch.setattr(config, "MM_model_folder", str(tmp_path / "not_on_this_pc"))
    model = ext.MMModel(info_dir=map_dir)                               # microscope PC without the model

    assert model.categories == {"background": 0, "mitos": 1, "ice": 2}
    assert model.pix_size == 2.5


def test_local_model_preferred_over_shared_info(local_model, tmp_path):
    from SPACEtomo.modules import ext

    (tmp_path / ext.MMModel.info_file).write_text(json.dumps({"labels": {"other": 0}, "pixel_size": 9.0}))
    model = ext.MMModel(info_dir=tmp_path)

    assert model.pix_size == 2.5 and "mitos" in model.categories


def test_missing_model_and_info_explains_monitor(tmp_path, monkeypatch):
    from SPACEtomo.modules import ext

    monkeypatch.setattr(config, "MM_model_folder", str(tmp_path / "not_on_this_pc"))
    with pytest.raises(FileNotFoundError) as err:
        ext.MMModel(info_dir=tmp_path)
    assert "SPACEmonitor" in str(err.value)


def test_settings_gui_without_model(tmp_path, monkeypatch):
    """The settings GUI runs on the microscope PC; without the model (and without shared info) it must still open."""

    pytest.importorskip("dearpygui")
    from SPACEtomo.modules.gui.set import SettingsGUI

    monkeypatch.setattr(config, "MM_model_folder", str(tmp_path / "not_on_this_pc"))
    gui = SettingsGUI.__new__(SettingsGUI)
    gui.session_dir = tmp_path
    assert gui.loadModel().categories == {}


def test_settings_gui_uses_shared_info(local_model, tmp_path, monkeypatch):
    pytest.importorskip("dearpygui")
    from SPACEtomo.modules import ext, utils
    from SPACEtomo.modules.gui.set import SettingsGUI

    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    ext.shareModelInfo(map_dir)
    monkeypatch.setattr(config, "MM_model_folder", str(tmp_path / "not_on_this_pc"))
    session = tmp_path / "session"
    session.mkdir()
    utils.saveSettings(session / "SPACEtomo_settings.ini", {"external_map_dir": str(map_dir)}, "external_map_dir")

    gui = SettingsGUI.__new__(SettingsGUI)
    gui.session_dir = session
    assert "mitos" in gui.loadModel().categories


def test_monitor_without_segmentation_model_keeps_running(tmp_path, monkeypatch):
    """A processing PC with only the lamella detection model can still serve level 1 runs."""
    from SPACEtomo.modules import ext

    monkeypatch.setattr(config, "MM_model_folder", str(tmp_path / "not_on_this_pc"))
    assert ext.shareModelInfo(tmp_path) is False
    assert not (tmp_path / ext.MMModel.info_file).exists()


@pytest.mark.parametrize("external, shared, level, ok", [
    ("", False, 4, False),          # no model, no external processing
    ("maps", False, 1, True),       # external level 1 only needs lamella detection on the processing PC
    ("maps", False, 2, False),      # SPACEmonitor not started (yet): refuse at the start instead of failing after the grid map
    ("maps", True, 4, True),        # model info shared by SPACEmonitor
])
def test_run_start_check(tmp_path, monkeypatch, external, shared, level, ok):
    from SPACEtomo.modules import ext

    monkeypatch.setattr(config, "NO_MM_MODEL", True)
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    if shared:
        (map_dir / ext.MMModel.info_file).write_text(json.dumps({"labels": {}, "pixel_size": 2.5}))

    assert ext.checkMMModel(str(tmp_path / external) if external else "", level) is ok
