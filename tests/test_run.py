"""End-to-end tests for SPACEtomo run.py using mock serialem.

These tests run the full SPACEtomo pipeline at different automation levels
using the dummy microscope and mock serialem module.

Marked as 'slow' since YOLO inference on CPU takes several minutes.
Run with: pytest tests/test_run.py -v
Skip slow: pytest tests/ -m "not slow"
"""

import importlib
import shutil
from pathlib import Path

import pytest

from tests.conftest import make_settings


def _run_spacetomo(session_dir: Path, automation_level: int, manual_selection: bool = False):
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

    run_module.main()


@pytest.mark.slow
def test_run_level1(session_dir):
    """Level 1: WG montage + lamella detection."""
    _run_spacetomo(session_dir, automation_level=1)

    grid_dir = session_dir / "G01"
    assert (grid_dir / "G01.mrc").exists(), "WG montage not created"
    assert (grid_dir / "G01.nav").exists(), "Navigator file not created"

    map_dir = grid_dir / "SPACE_maps"
    wg_files = list(map_dir.glob("G01_wg*"))
    assert len(wg_files) > 0, "WG map files not created"


@pytest.mark.slow
def test_run_level2(session_dir):
    """Level 2: WG + IM alignment + MM montage collection."""
    _run_spacetomo(session_dir, automation_level=2)

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


@pytest.mark.slow
def test_run_level3(session_dir):
    """Level 3: WG + IM + MM + segmentation triggering."""
    _run_spacetomo(session_dir, automation_level=3, manual_selection=True)

    grid_dir = session_dir / "G01"
    map_dir = grid_dir / "SPACE_maps"

    # Check segmentation outputs
    seg_files = list(map_dir.glob("*_seg.png"))
    assert len(seg_files) > 0, "No segmentation files created"


@pytest.mark.slow
def test_run_level4(session_dir):
    """Level 4: Full pipeline through target setup."""
    _run_spacetomo(session_dir, automation_level=4)

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
