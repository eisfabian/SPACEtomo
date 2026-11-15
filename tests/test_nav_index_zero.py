"""A map at navigator index 0 (first item, e.g. the WG map or the first map in a manual run's navigator) is a valid map."""

import numpy as np
import pytest
import serialem as sem

from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.mod_wg import Boxes
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.scope import Microscope
from SPACEtomo.modules.tgt import PACEArea


@pytest.fixture
def first_map(tmp_path):
    """Navigator whose first item (index 0) is a map at stage Z 12 µm, loaded into a Buffer."""

    sem.SetDirectory(str(tmp_path))
    nav = Navigator(tmp_path / "G01.nav")
    Buffer.nav = nav
    microscope = Microscope()
    microscope.changeImagingState("WG", low_dose_expected=False)
    microscope.record()
    map_id = nav.newMap(buffer=Buffer("A"), img_file=tmp_path / "G01.mrc", note="Sec 1 - G01.mrc")
    nav.items[map_id].changeZ(12.0)
    nav.push()
    assert map_id == 0
    return nav, Buffer(nav_id=map_id)


def test_show_reloads_overwritten_map(first_map):
    """Another image ends up in the map's buffer: the map has to be reloaded from the navigator."""

    _, map_buffer = first_map
    map_img = np.array(sem.bufferImage(map_buffer.buf))
    sem.R()     # Image not taken through SPACEtomo, buffers were not rolled for the Buffer instances

    map_buffer.show()

    assert np.array_equal(sem.bufferImage(map_buffer.buf), map_img)


def test_addNavBoxes_uses_z_of_map(first_map):
    nav, map_buffer = first_map
    center = map_buffer.shape / 2
    boxes = Boxes([[center[0] - 50, center[1] - 50, center[0] + 50, center[1] + 50, 2, 0.9]], pix_size=map_buffer.pix_size, img_size=map_buffer.shape)

    map_buffer.addNavBoxes(boxes, label_prefix="PP")
    nav.pull()

    polygon = nav.items[nav.searchByEntry("label", "PP", partial=True)[0]]
    assert polygon.stage[2] == pytest.approx(12.0)


def test_findTemplateId_uses_map_at_index_0(first_map):
    """Map name of a section (G01_01) does not appear in the note, so the note search cannot replace the map's nav ID."""

    nav, map_buffer = first_map
    area = PACEArea.__new__(PACEArea)
    area.nav_id = map_buffer.nav_id

    assert area.findTemplateId(nav, "G01_01") == 0
