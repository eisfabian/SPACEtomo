"""Navigator.push: new items address the right SerialEM item, and a failed write does not lose the navigator."""

import builtins

import numpy as np
import pytest
import serialem as sem

from SPACEtomo.modules import nav as nav_module
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.utils import writeMrc


@pytest.fixture
def nav(tmp_path):
    sem.SetDirectory(str(tmp_path))
    nav = Navigator(tmp_path / "G01.nav")
    Buffer.nav = nav
    return nav


def _add(nav, kind, label):
    if kind == "point":
        return nav.newPoint(np.zeros(3), label=label)
    if kind == "polygon":
        return nav.newPolygon([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], 0.0, label=label)
    img_file = nav.file.parent / f"{label}.mrc"
    writeMrc(img_file, np.zeros((64, 64), dtype=np.int16), 10.0)
    nav_id = nav.newMapFromImg(img_file=img_file, coords=(0, 0, 0), label=label)
    nav.push()
    return nav_id


@pytest.mark.parametrize("kind", ["point", "polygon", "map"])
def test_new_item_changes_go_to_its_serialem_item(nav, kind):
    """Changes applied through SerialEM right after creating an item (as for the first virtual map of a target area) must not hit the previous item."""

    nav.newPoint(np.zeros(3), label="first")
    nav_id = _add(nav, kind, "second")

    nav.items[nav_id].changeNote("A1_tgts.txt")
    nav.items[nav_id].changeAcquire(1)
    nav.pull()

    assert nav.items[0].note == ""
    assert "Acquire" not in nav.items[0].entries or nav.items[0].entries["Acquire"] == ["0"]
    assert nav.items[nav_id].note == "A1_tgts.txt"
    assert nav.items[nav_id].entries["Acquire"] == ["1"]


def test_failed_push_keeps_navigator(nav, monkeypatch):
    """Writing the nav file fails halfway (disk full, file locked): the old file stays intact and SerialEM keeps its navigator open."""

    nav.newPoint(np.zeros(3), label="first")
    nav.newPoint(np.ones(3), label="second")
    saved = nav.file.read_text()

    class FailingFile:
        def __init__(self, f):
            self.f = f
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.f.close()
        def write(self, text):
            self.f.write(text[:len(text) // 2])
            raise OSError(28, "No space left on device")

    def failing_open(file, mode="r", *args, **kwargs):
        f = builtins.open(file, mode, *args, **kwargs)
        return FailingFile(f) if "w" in mode else f

    monkeypatch.setattr(nav_module, "open", failing_open, raising=False)
    nav.newPoint(np.full(3, 2.0), label="third", update=False)

    with pytest.raises(OSError):
        nav.push()
    monkeypatch.undo()

    assert nav.file.read_text() == saved
    assert sem.ReportIfNavOpen() == 2
    assert sem.NavIndexWithLabel("second") == 2
