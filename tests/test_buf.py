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


@pytest.mark.parametrize("dtype, factor", [(np.float32, 1.0), (np.float32, 0.5), (np.int16, 1.0)])
def test_rescaleImage_keeps_intensities_of_counts(dtype, factor):
    """Camera counts (any dtype) must not wrap around when rescaling; 8 bit conversion happens when saving."""

    img = np.tile(np.linspace(50, 4000, 200), (100, 1)).astype(dtype)
    buffer = _buffer(img)

    assert buffer.rescaleImage(factor)

    profile = buffer.img_mod[buffer.img_mod.shape[0] // 2]
    assert np.all(np.diff(profile) >= -1e-3), "intensities wrapped around"
    assert profile.max() == pytest.approx(img.max(), rel=0.05)
