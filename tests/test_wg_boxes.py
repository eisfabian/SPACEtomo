"""Tests for SPACEtomo.modules.mod_wg bounding boxes."""

import numpy as np
import pytest

from SPACEtomo.modules.mod_wg import BBox


@pytest.mark.parametrize("factor", [1.5, 0.5, 2.0])
def test_bbox_scale_keeps_center(factor):
    """Padding a lamella box must keep its center and scale its size by the factor."""

    box = BBox([100.0, 200.0, 150.0, 290.0], cat=2, prob=0.9)
    center, size = box.center.copy(), box.size.copy()

    box.scale(factor)

    assert np.allclose(box.center, center)
    assert np.allclose(box.size, size * factor)
