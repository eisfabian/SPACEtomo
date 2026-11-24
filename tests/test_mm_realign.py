"""Realignment of View to the ROI before the MM montage (GitHub issue #10).

The ROI polygon (FP) comes from lamella detection on the IM image and is more precise than the position detected on the
grid map (PP). Realignment has to center View on the FP position, using the IM image as reference.
"""

import numpy as np
import pytest
import serialem as sem

from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.mma import MMMAcquisition
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.scope import Microscope
from SPACEtomo.modules.utils import writeMrc


@pytest.fixture
def rois(tmp_path):
    """State after IM alignment of n ROIs whose grid map detections (PP) are pp_error µm off their lamellae.

    Returns (microscope, nav, FP ids, lamella stage positions as seen in View).
    """

    from types import SimpleNamespace
    from SPACEtomo.modules.dummy.sim import sim
    from SPACEtomo.modules.dummy.specimen import Specimen
    from SPACEtomo.modules.ima import IMAlignment

    def _make(pp_error=(6.0, -4.0), mm_offset=(0.0, 0.0), n=1):
        sem.SetDirectory(str(tmp_path))
        nav = Navigator(tmp_path / "G01.nav")
        Buffer.nav = nav
        microscope = Microscope()
        microscope.autoloader, microscope.loaded_grid = {1: "G01"}, 1

        specimen = sim.getSpecimen()
        if len(specimen.lamellae) < n:
            sim.specimen = specimen = Specimen(seed=specimen.seed, n_lamellae=n, offsets=specimen.offsets)
        specimen.offsets["MM"] = specimen.offsets["MM"] + np.array(mm_offset)
        offsets = specimen.offsets
        lamellae = [lamella.center for lamella in specimen.lamellae[:n]]

        # Grid map
        microscope.changeImagingState("WG", low_dose_expected=False)
        microscope.moveStage(np.mean(lamellae, axis=0) - offsets["WG"])
        microscope.record()
        wg_map = Buffer("A")

        fp_ids = []
        for i, lamella in enumerate(lamellae):
            label = f"PP{i + 1:02d}"
            # IMref: grid map crop centered on the PP position (as written by run.py)
            pp_stage = lamella - offsets["WG"] + np.array(pp_error)
            crop = wg_map.getCropImage(np.flip(wg_map.stage2px(pp_stage)[:2]), np.flip(microscope.camera_dims))
            writeMrc(tmp_path / f"G01_{label}_IMref.mrc", np.flip(crop, axis=0), wg_map.pix_size)

        for i, lamella in enumerate(lamellae):
            label = f"PP{i + 1:02d}"
            # IM map: IM alignment centered the IM image on the PP position
            microscope.changeImagingState("IM", low_dose_expected=False)
            microscope.moveStage(lamella - offsets["IM"] + np.array(pp_error))
            microscope.record()
            im_buffer = Buffer("A")
            nav.newMap(buffer=im_buffer, img_file=tmp_path / f"G01_{label}_IM.mrc", label=label)

            # FP polygon from lamella detection on the IM image, then the MM reference like IMAlignment.align
            fp_stage = lamella - offsets["IM"]
            fp_id = nav.newPolygon([fp_stage[0] + dx for dx in (-10, -10, 10, 10, -10)], [fp_stage[1] + dy for dy in (-5, 5, 5, -5, -5)], 0.0, label=f"FP{i + 1:02d}")
            nav.items[fp_id].entries["UserValue2"] = [label]
            nav.push()
            IMAlignment.saveMMReference(SimpleNamespace(nav=nav, microscope=microscope), im_buffer, f"FP{i + 1:02d}", tmp_path / f"G01_{label}_MMref.mrc")
            fp_ids.append(fp_id)

        microscope.changeImagingState("MM1", low_dose_expected=True)
        assert len(fp_ids) == n
        return microscope, nav, fp_ids, [lamella - offsets["MM"] for lamella in lamellae]
    return _make


def _realign(microscope, nav, fp_ids, i, tmp_path):
    """Realignment before the MM montage of ROI i, like run.py (shifts the remaining ROIs too)."""
    label = nav.items[fp_ids[i]].entries["UserValue2"][0]
    MMMAcquisition(tmp_path, tmp_path / f"G01_{label}.mrc", fp_ids[i], microscope, nav).realign(fp_ids[i:])


@pytest.mark.parametrize("mm_offset", [(0.0, 0.0), (5.0, 3.0)])
def test_realign_centers_view_on_im_detected_lamella(rois, tmp_path, mm_offset):
    """The FP position from the IM image must be kept; only the IM->View offset is corrected."""

    microscope, nav, fp_ids, lamellae_view = rois(mm_offset=mm_offset)

    _realign(microscope, nav, fp_ids, 0, tmp_path)

    assert np.linalg.norm(nav.items[fp_ids[0]].stage[:2] - lamellae_view[0]) < 2.0      # within the convergence threshold


def test_realign_of_following_rois_does_not_accumulate_offset(rois, tmp_path):
    """The IM->View offset found on the first ROI is applied to the remaining ROIs, which must still end up on their lamellae."""

    microscope, nav, fp_ids, lamellae_view = rois(pp_error=(0.0, 0.0), mm_offset=(5.0, 3.0), n=3)

    for i in range(len(fp_ids)):
        _realign(microscope, nav, fp_ids, i, tmp_path)
        assert np.linalg.norm(nav.items[fp_ids[i]].stage[:2] - lamellae_view[i]) < 2.0, f"ROI {i + 1}"


def test_realign_survives_failed_alignment(rois, tmp_path, monkeypatch):
    """SerialEM aborts AlignBetweenMags without a correlation peak; the montage is then collected at the FP position."""

    microscope, nav, fp_ids, _ = rois()
    fp_stage = nav.items[fp_ids[0]].stage[:2].copy()

    def no_peak(*args):
        raise sem.SEMerror("No correlation peaks were found within the allowed range")
    monkeypatch.setattr(sem, "AlignBetweenMags", no_peak)

    _realign(microscope, nav, fp_ids, 0, tmp_path)

    assert np.allclose(nav.items[fp_ids[0]].stage[:2], fp_stage)
    assert np.allclose(microscope.stage[:2], fp_stage)
