"""Lamella classes excluded in the settings are already excluded on the first detection, not only when boxes are reloaded."""

import json

import numpy as np
import pytest
from PIL import Image

from SPACEtomo import config
from SPACEtomo.modules.mod_wg import WGModel


class _Tensor:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)
    def to(self, device):
        return self
    def numpy(self):
        return self.values


class FakeYOLO:
    """Returns one "good" and one "gone" lamella in the first tile."""

    def __init__(self):
        self.calls = 0

    def predict(self, crop, device="cpu"):
        boxes = [[100, 100, 200, 150], [600, 600, 700, 650]] if self.calls == 0 else []
        cats = [config.WG_model_categories.index("good"), config.WG_model_categories.index("gone")] if self.calls == 0 else []
        self.calls += 1
        result = type("Result", (), {})()
        result.boxes = type("Boxes", (), {"xyxy": _Tensor(np.reshape(boxes, (-1, 4))), "cls": _Tensor(cats), "conf": _Tensor([0.9] * len(cats)),
                                          "__len__": lambda self: len(cats)})()
        return [result]


@pytest.fixture
def wg_model(tmp_path):
    Image.fromarray(np.zeros((config.WG_model_sidelen, config.WG_model_sidelen), dtype=np.uint8)).save(tmp_path / "G01_wg.png")
    model = WGModel(tmp_path, external=True)
    model.external = False
    model.model = FakeYOLO()
    return model


def test_first_detection_excludes_classes(wg_model, tmp_path):
    boxes = wg_model.findLamellae("G01", label_prefix="PP", exclude_cats=["gone"])

    assert [config.WG_model_categories[box.cat] for box in boxes.boxes] == ["good"]
    # Box file keeps all lamellae (GUI and reruns with other settings)
    assert len(json.loads((tmp_path / "G01_wg_boxes.json").read_text())["boxes"]) == 2


def test_rerun_gives_same_lamellae(wg_model):
    first = wg_model.findLamellae("G01", label_prefix="PP", exclude_cats=["gone"])
    rerun = wg_model.findLamellae("G01", label_prefix="PP", exclude_cats=["gone"])

    assert [box.label for box in first.boxes] == [box.label for box in rerun.boxes]
