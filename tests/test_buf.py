"""Tests for SPACEtomo.modules.buf that don't need images from SerialEM."""

import numpy as np
import pytest

from SPACEtomo.modules.buf import Buffer


def _buffer(img):
    buffer = Buffer.__new__(Buffer)
    buffer.img = img
    return buffer


@pytest.mark.parametrize("coords", [(50, 50), (10, 90), (95, 5)])
def test_getCropImage_keeps_fov_and_centers_coords(coords):
    """Crops larger than the map (padded on both sides) or at its edge keep the fov and center the requested position."""

    img = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    fov = (150, 120)

    crop = _buffer(img).getCropImage(coords, fov)

    assert crop.shape == fov
    assert crop[fov[0] // 2, fov[1] // 2] == img[coords]
