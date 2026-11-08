#!/usr/bin/env python
# ===================================================================
# ScriptName:   segment
# Purpose:      Stand-in for the lamella segmentation model in DUMMY mode: writes the ground truth segmentation
#               of a map of the synthetic grid (started by the processing queue like run_nnUNet.py).
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein / Claude
# Created:      2026/11/08
# Revision:     v1.4
# Last Change:  2026/11/08: created
# ===================================================================

import sys
import json
from pathlib import Path

# Use the package this script belongs to (also when another SPACEtomo copy is installed)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from SPACEtomo.modules.dummy.specimen import Specimen


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <map png in SPACE_maps>")
        sys.exit(1)

    map_png = Path(sys.argv[1])
    map_name = map_png.stem

    # Ground truth is written next to the map files (grid folder containing SPACE_maps)
    truth_file = map_png.parent.parent / "dummy_ground_truth.json"
    truth = json.loads(truth_file.read_text())
    mm_map = next(entry for entry in truth["maps"] if entry["name"] == map_name)

    # Render at map pixel size and scale to the png size (model pixel size)
    shape = np.round(np.flip(mm_map["extent"]) * 1000 / mm_map["pix_size"]).astype(int)
    _, seg = Specimen(truth["seed"]).render(np.array(mm_map["world_center"]), shape, mm_map["pix_size"], detail=True, seg=True)
    with Image.open(map_png) as png:
        size = png.size
    Image.fromarray(seg).resize(size, resample=Image.Resampling.NEAREST).save(map_png.parent / (map_name + "_seg.png"))
    print(f"Saved ground truth segmentation for {map_name}.")


if __name__ == "__main__":
    main()
