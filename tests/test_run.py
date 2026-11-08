"""End-to-end tests for SPACEtomo run.py using mock serialem.

These tests run the full SPACEtomo pipeline at different automation levels
using the dummy microscope and mock serialem module.

Marked as 'slow' since YOLO inference on CPU takes several minutes.
Run with: pytest tests/test_run.py -v
Skip slow: pytest tests/ -m "not slow"
"""

import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_settings


def _run_spacetomo(session_dir: Path, automation_level: int, capsys, manual_selection: bool = False):
    """Helper: configure and run SPACEtomo at a given automation level."""
    import serialem as sem

    # Update settings for this automation level
    grid_dir = session_dir / "G01"
    make_settings(session_dir / "SPACEtomo_settings.ini", session_dir,
                  automation_level=automation_level, manual_selection=manual_selection)
    shutil.copy(session_dir / "SPACEtomo_settings.ini", grid_dir / "SPACEtomo_settings.ini")

    # Reload run module so it picks up the current session dir
    import SPACEtomo.run as run_module
    importlib.reload(run_module)
    # Safety: ensure SES_DIR matches (reload should set it via getSessionDir)
    run_module.SES_DIR = session_dir

    # A finished run ends the script with sem.Exit(), which raises SEMexited like in SerialEM
    with pytest.raises(sem.SEMexited, match="Normal exit"):
        run_module.main()
    # Error exits raise the same exception, so make sure the run actually completed
    assert "SPACEtomo run completed" in capsys.readouterr().out


# --- Ground truth of the synthetic grid (written by the simulated microscope) ---

def _ground_truth(grid_dir):
    from SPACEtomo.modules.dummy.specimen import Specimen
    truth = json.loads((grid_dir / "dummy_ground_truth.json").read_text())
    specimen = Specimen(truth["seed"])
    assert [lamella.toDict() for lamella in specimen.lamellae] == truth["lamellae"]
    return truth, specimen


def _distance_to_lamellae(point, truth):
    return min(np.linalg.norm(np.array(point) - lamella["center"]) for lamella in truth["lamellae"])


def _nav_polygons(grid_dir, prefix):
    """Centers of the polygon/map items whose label starts with prefix."""
    from SPACEtomo.modules.nav import Navigator
    nav = Navigator.__new__(Navigator)
    nav.file, nav.header, nav.items = grid_dir / "G01.nav", "", []
    nav.readFromFile()
    centers = []
    for item in nav.items:
        if item.label.startswith(prefix) and "PtsX" in item.entries:
            pts = np.array([item.entries["PtsX"], item.entries["PtsY"]], dtype=float)
            centers.append((pts.min(axis=1) + pts.max(axis=1)) / 2)
    return centers


@pytest.mark.slow
def test_run_level1(session_dir, capsys):
    """Level 1: WG montage + lamella detection."""
    _run_spacetomo(session_dir, automation_level=1, capsys=capsys)

    grid_dir = session_dir / "G01"
    assert (grid_dir / "G01.mrc").exists(), "WG montage not created"
    assert (grid_dir / "G01.nav").exists(), "Navigator file not created"

    map_dir = grid_dir / "SPACE_maps"
    wg_files = list(map_dir.glob("G01_wg*"))
    assert len(wg_files) > 0, "WG map files not created"

    # Detected lamellae (PP polygons) are at true lamellae of the synthetic grid
    truth, _ = _ground_truth(grid_dir)
    pp_centers = _nav_polygons(grid_dir, "PP")
    assert pp_centers, "No lamellae detected"
    assert all(_distance_to_lamellae(center, truth) < 3 for center in pp_centers), f"PP polygons {pp_centers} not at lamellae {truth['lamellae']}"


@pytest.mark.slow
def test_run_level2(session_dir, capsys):
    """Level 2: WG + IM alignment + MM montage collection."""
    _run_spacetomo(session_dir, automation_level=2, capsys=capsys)

    grid_dir = session_dir / "G01"
    map_dir = grid_dir / "SPACE_maps"

    # Check WG outputs
    assert (grid_dir / "G01.mrc").exists()

    # Check IM outputs (at least one ROI found)
    im_files = list(grid_dir.glob("*_IM.mrc"))
    assert len(im_files) > 0, "No IM reference images created"

    # Check MM outputs
    mm_files = list(grid_dir.glob("G01_*.mrc"))
    # Should have WG + IM + at least one MM map
    assert len(mm_files) >= 2, f"Expected at least 2 MRC files, found {len(mm_files)}"

    # After IM alignment and MM realignment, each MM map covers the whole slab of a true lamella (despite simulated image offsets;
    # View images are shorter than the lamella, so the realignment can be off along the lamella axis by a few µm)
    truth, specimen = _ground_truth(grid_dir)
    assert truth["maps"], "No MM maps in ground truth"
    for mm_map in truth["maps"]:
        map_min = np.array(mm_map["world_center"]) - np.array(mm_map["extent"]) / 2
        map_max = np.array(mm_map["world_center"]) + np.array(mm_map["extent"]) / 2
        lamella = min(specimen.lamellae, key=lambda lamella: np.linalg.norm(lamella.center - mm_map["world_center"]))
        x_min, x_max, y_min, y_max = lamella.slab
        assert map_min[0] <= x_min and x_max <= map_max[0] and map_min[1] <= y_min and y_max <= map_max[1], f"MM map {mm_map} does not cover lamella slab {lamella.slab}"


@pytest.mark.slow
def test_run_level3(session_dir, capsys):
    """Level 3: WG + IM + MM + segmentation triggering."""
    _run_spacetomo(session_dir, automation_level=3, capsys=capsys, manual_selection=True)

    grid_dir = session_dir / "G01"
    map_dir = grid_dir / "SPACE_maps"

    # Check segmentation outputs
    seg_files = list(map_dir.glob("*_seg.png"))
    assert len(seg_files) > 0, "No segmentation files created"


@pytest.mark.slow
def test_run_level4(session_dir, capsys):
    """Level 4: Full pipeline through target setup."""
    _run_spacetomo(session_dir, automation_level=4, capsys=capsys)

    grid_dir = session_dir / "G01"
    map_dir = grid_dir / "SPACE_maps"

    # Check all stages completed
    assert (grid_dir / "G01.mrc").exists(), "WG montage not created"
    assert (grid_dir / "G01.nav").exists(), "Navigator not created"

    # Check segmentation
    seg_files = list(map_dir.glob("*_seg.png"))
    assert len(seg_files) > 0, "No segmentation files"

    # Check target selection
    point_files = list(map_dir.glob("*_points*.json"))
    assert len(point_files) > 0, "No point files created (target selection failed)"

    # Check target setup outputs
    tgt_files = list(grid_dir.glob("*_tgts.txt"))
    assert len(tgt_files) > 0, "No target files created"

    tgt_params = list(grid_dir.glob("tgt_params.json"))
    assert len(tgt_params) > 0, "No tgt_params.json created"

    # Targets (stage coords in the MM map frame) cover target classes of the true grid in their Record field of view
    from SPACEtomo.modules.dummy.specimen import CATS
    truth, specimen = _ground_truth(grid_dir)
    target_classes = [CATS[name] for name in ("mitos", "nucleus")]
    mic_params = json.loads((grid_dir / "mic_params.json").read_text())
    fov_nm = np.sqrt(np.prod(mic_params["cam_dims"]["$array"])) * mic_params["rec_pix_size"]     # square of the Record FOV area (rotation agnostic)
    for tgt_file in tgt_files:
        # Only target blocks ("_tgt = ..."), not geo points ("_geo = ...")
        blocks = [dict(line.split(" = ", 1) for line in block.splitlines()) for block in tgt_file.read_text().strip().split("\n\n")]
        for block in [block for block in blocks if "_tgt" in block]:
            point = (float(block["stageX"]), float(block["stageY"]))
            world = np.array(point) + truth["offsets"]["MM"]
            _, seg = specimen.render(world, (256, 256), fov_nm / 256, detail=True, seg=True)
            # Half the selection threshold (0.095), allowing for the residual map offset of up to ~2 µm
            assert np.isin(seg, target_classes).mean() > 0.05, f"Target {point} ({tgt_file.name}) does not cover target classes"
