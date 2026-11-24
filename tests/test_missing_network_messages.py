"""Missing model errors point to the Python environment (GitHub issue #10: SerialEM used another environment than the
one the models were imported into, because PathToPython was not set to it)."""

import sys

import pytest

from SPACEtomo import config


def _check(message, path):
    assert str(path) in message
    assert sys.prefix in message
    assert "PathToPython" in message and "SPACEmodel" in message


def test_missing_segmentation_model(tmp_path, monkeypatch):
    from SPACEtomo.modules.ext import MMModel

    monkeypatch.setattr(config, "MM_model_folder", str(tmp_path / "missing_model"))
    with pytest.raises(FileNotFoundError) as err:
        MMModel()
    _check(str(err.value), tmp_path / "missing_model")


def test_missing_detection_model(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    from SPACEtomo.modules.mod_wg import WGModel

    monkeypatch.setattr(config, "WG_model_file", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError) as err:
        WGModel(tmp_path)
    _check(str(err.value), tmp_path / "missing.pt")


def test_message_for_run_scripts():
    from SPACEtomo.modules import utils

    _check(utils.modelNotFoundMessage("Lamella segmentation model", "C:/models/seg"), "C:/models/seg")
