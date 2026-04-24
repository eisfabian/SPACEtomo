#!/usr/bin/env python
# ===================================================================
# ScriptName:   mma
# Purpose:      Handling of medium mag map acquisition.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2025/02/06
# Revision:     v1.3
# Last Change:  2025/02/12: added geo estimation
#               2025/02/06: outsourcing from run.py
# ===================================================================

import json
import mrcfile
import numpy as np
from pathlib import Path

from SPACEtomo.modules.scope import Microscope
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.geo import Geometry
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log
from SPACEtomo.modules.ext import MMModel
from SPACEtomo import config

if config.DUMMY:    # Shadow classes with dummy classes
    from SPACEtomo.modules.dummy.scope import MicroscopeDummy as Microscope
    from SPACEtomo.modules.dummy.buf import BufferDummy as Buffer

class MMMAcquisition:
    def __init__(self, map_dir: Path, map_file: Path, poly_id: int, microscope: Microscope, navigator: Navigator) -> None:
        """Medium mag map acquisition requires:
            - map_dir:      directory to save map files
            - map_file:     path to map file
            - poly_id:      nav_id of polygon to be acquired
            - microscope:   Microscope object
            - navigator:    Navigator object
        """
        
        self.microscope = microscope
        self.nav = navigator

        self.map_dir = map_dir
        self.cur_dir = map_file.parent
        self.map_file = map_file
        self.map_name = map_file.stem
        self.poly_id = poly_id

    def setupReacquisition(self, map_id: int):
        """Reads reacquisition json file and adjusts polygon accordingly."""

        with open((self.map_dir / (self.map_name + "_reacquire.json")), "r") as f:
            reacquire = json.load(f, object_hook=utils.revertTaggedString)

        if "restitch" in reacquire and reacquire["restitch"]:
            # Restitching of map
            log(f"NOTE: Map {self.map_name} was set for reloading and restitching.")
            
            # Collect only png files of old map
            map_file_list = list(self.map_dir.glob(f"**/{self.map_name}*.png"))
            map_file_list += list(self.map_dir.glob(f"**/{self.map_name}_reacquire.json"))
            map_file_list += list(self.map_dir.glob(f"**/{self.map_name}_inspected.txt"))

            # Mark NOT for reacquisition
            reacquire = False

        else:
            # Reacquisition of map
            log(f"NOTE: Map {self.map_name} was set for reacquisition.")

            # Find offset
            ## Get pixel size from mrc file
            with mrcfile.open(self.map_file) as mrc:
                pix_size = mrc.voxel_size.x / 10 # nm/px
            ## Binning between Montage and Map needs to be considered, because MapScaleMat is given in Map coordinates
            binning_factor = float(self.nav.items[map_id].entries["MontBinning"][0]) / float(self.nav.items[map_id].entries["MapBinning"][0])
            ## Get pixel to stage matrix
            px2stage_matrix = np.linalg.inv(np.array(self.nav.items[map_id].entries["MapScaleMat"], dtype=np.float32).reshape((2, 2)))
            ## Get stage offset from binned pixel coordinates
            stage_offset = px2stage_matrix @ (reacquire["center_offset"] / pix_size / binning_factor)
            log(f"DEBUG: Shifting ROI polygon by {stage_offset} and using padding factor {reacquire['padding']}...")

            # Adjust center and size of polygon
            self.nav.items[self.poly_id].changeStage(stage_offset, relative=True)
            self.nav.items[self.poly_id].scaleBounds(reacquire["padding"] / config.MM_padding_factor)

            # Rename mrc file and nav item
            counter = 0
            while (new_mrc_file := self.cur_dir / f"{self.map_name}_old{counter}.mrc").exists():
                counter += 1
            self.map_file.replace(new_mrc_file)
            if (self.cur_dir / f"{self.map_file.name}.mdoc").exists():
                (self.cur_dir / f"{self.map_file.name}.mdoc").replace(self.cur_dir / f"{new_mrc_file.name}.mdoc")
            self.nav.items[map_id].map_file = new_mrc_file
            self.nav.push() # Push nav before running change commands that use SerialEM commands
            self.nav.items[map_id].changeNote(new_mrc_file.name)
            self.nav.items[map_id].changeLabel("old")

            # Collect all files of old map
            map_file_list = self.map_dir.glob(f"**/{self.map_name}*")

            # Mark for reacquisition
            reacquire = True

        # Delete collected map files
        for file in map_file_list:
            if file.exists():
                if file.is_file():
                    log(f"DEBUG: Deleting {file}")
                    file.unlink()
                elif file.is_dir():
                    utils.rmDir(file)

        return reacquire
    
    def saveMap(self, map_id: int, pix_size: float, find_grid=False, save_future=None):
        """Saves map as image in SPACE_maps."""

        # Save montage as rescaled input image
        if not (self.map_dir / f"{self.map_name}.png").exists():
            log(f"NOTE: Saving map for {self.map_name}...")
            map_img = Buffer(nav_id=map_id)
            if find_grid:
                map_img.findGrid(spacing_nm=2500) # 2.5 micron spacing should work for R2/1, R1.2/1.3 or anything coarser

            if save_future is not None:
                save_future.result()
                save_future = None
            save_future = map_img.saveImg(self.map_dir / (self.map_name + ".png"), target_pix_size=pix_size)
            if save_future is not None:
                save_future.result() # Temporary, because montage collection is not threaded, TODO: multiprocessing instead
                save_future = None
            return save_future
        return None

    def collectMap(self):
        """Collects map and returns map_id."""

        # Check if map already exists in nav        
        if map_id := self.nav.getIDfromNote(self.map_file.name, warn=False):
            log(f"NOTE: Map {self.map_name} already exists.")

            # Check for reacquisition
            if (self.map_dir / (self.map_name + "_reacquire.json")).exists():
                if self.setupReacquisition(map_id):
                    map_id = None

        # Collect map if no nav item found
        if map_id is None:
            log(f"Collecting map for {self.map_name}...")

            # Delete potentially aborted montage file
            if self.map_file.exists():
                self.map_file.unlink()

            # Estimate sample geometry from View beam tilt pair
            #geo = self.geoFromView() # TODO: Work on reliability

            # Collect map
            map_id = self.microscope.collectPolygonMontage(self.poly_id, self.map_file, config.MM_montage_overlap)
            self.nav.pull()
            self.nav.items[map_id].changeLabel(f"L{self.map_name.split('_L')[-1]}")

        return map_id
    
    def geoFromView(self, beam_tilt_mrad=2):
        """Estimates geometry from View beam tilt pair."""

        # Estimate sample geometry from View beam tilt pair
        self.microscope.moveStage(self.nav.items[self.poly_id].stage)
        focus_offset = self.microscope.changeLowDoseArea("V")
        self.microscope.setDefocus(-10 - focus_offset, relative=True)
        view1, view2 = self.microscope.collectViewBeamTiltPair(beam_tilt_mrad=beam_tilt_mrad, file_path=self.cur_dir / (self.map_name + "_beamtilt_pair.mrc"))
        self.microscope.setDefocus(10 + focus_offset, relative=True)
        geo = Geometry(self.microscope)
        geo.fromBeamTiltPair(view1, view2, beam_tilt_mrad=beam_tilt_mrad)

        return geo

    def alignToRef(self, ref_buffer: Buffer, threshold=2, max_iterations=3):
        """Takes image and aligns to reference image iteratively using stage movement."""

        stage_initial = self.microscope.stage
        stage_shift = np.array([np.inf, np.inf])
        iterations = 0
        while np.linalg.norm(stage_shift) > threshold and iterations < max_iterations:
            self.microscope.record(view=True)
            im_buffer = Buffer(buffer="A")

            _, ss_shift = im_buffer.alignTo(ref_buffer, avoid_image_shift=True)
            stage_shift = self.microscope.getMatrices()["ss2s"] @ ss_shift

            if np.linalg.norm(stage_shift) <= threshold:
                return im_buffer, self.microscope.stage[:2] - stage_initial[:2]

            self.microscope.moveStage(-stage_shift, relative=True)
            iterations += 1

        log(f"WARNING: Alignment did not converge after {iterations} iterations. Resetting stage position...")
        self.microscope.moveStage(stage_initial)
        #self.microscope.record(view=True)
        #im_buffer = Buffer(buffer="A")
        return None, np.zeros(2)

    def realign(self, shift_items=[]):
        """Realigns View to IM reference."""

        # Check if map already exists in nav        
        if self.nav.getIDfromNote(self.map_file.name, warn=False):
            log(f"DEBUG: Skipping MM realignment since map already exists.")
            return

        # Move stage
        self.microscope.moveStage(self.nav.items[self.poly_id].stage)

        # Get grid name and map number
        grid_name = self.microscope.autoloader[self.microscope.loaded_grid]
        map_num = int(self.map_name.split(grid_name)[-1].split('_L')[-1])
        ref_file = self.cur_dir / f"{grid_name}_IM{map_num}.mrc"
        if ref_file.exists():
            log(f"Realigning to IM reference...")
            _, stage_shift = self.alignToRef(ref_buffer=Buffer(buffer="O", file_path=ref_file))
            self.nav.shiftItems(stage_shift, shift_items)
            log(f"NOTE: Adjusted polygon position by {stage_shift}.")
        else:
            log(f"ERROR: No reference image found for realignment! Please make sure [{ref_file}] exists.")
            log(f"WARNING: Skipping realignment for {self.map_name}.")
            return