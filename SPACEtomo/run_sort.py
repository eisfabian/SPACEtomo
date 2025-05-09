#!Python
# ===================================================================
# Purpose:      Sorts tilt series MRC stack.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2025/04/18
# Revision:     v1.3
# Last Change:  2025/04/18: created
# ===================================================================

import shutil
from pathlib import Path
import SPACEtomo.config as config

from SPACEtomo.modules.mrc import MRC
from SPACEtomo.modules.utils import log

def main(file_path: Path, key="TiltAngle"):
    """Sorts tilt series stack."""

    if not file_path or not file_path.exists():
        log(f"ERROR: File does not exist: {file_path}")
        return
    
    log(f"Sorting {file_path.name} by {key}...")
    # Back up mdoc
    shutil.copy(file_path.with_suffix(".mrc.mdoc"), file_path.stem + "_unsorted.mrc.mdoc")

    # Sort stack
    ts_stack = MRC(file_path)
    ts_stack.sortStack(key=key)
    ts_stack.updateMrc()
    #ts_stack.writeNewMrc()

    log(f"NOTE: {file_path.name} was sorted by {key}!")
