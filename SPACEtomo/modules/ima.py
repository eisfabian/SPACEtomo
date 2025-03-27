#!/usr/bin/env python
# ===================================================================
# ScriptName:   ima
# Purpose:      Handling of intermediate mag alignments.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2025/01/10
# Revision:     v1.3
# Last Change:  2025/02/21: refactored to always have iterative IM reference alignment
#               2025/01/10: outsourcing from run.py
# ===================================================================

import json
import numpy as np
from copy import deepcopy

from SPACEtomo.modules.scope import Microscope, ImagingParams
from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.utils import log
from SPACEtomo.modules import utils
from SPACEtomo import config

if config.DUMMY:    # Shadow classes with dummy classes
    from SPACEtomo.modules.dummy.scope import MicroscopeDummy as Microscope
    from SPACEtomo.modules.dummy.buf import BufferDummy as Buffer

class IMAlignment:

    def __init__(self, cur_dir, map_dir, microscope: Microscope, navigator: Navigator, imaging_params: ImagingParams, label_prefix="PP", final_prefix="FP", WG_model=None) -> None:
        
        self.cur_dir = cur_dir
        self.map_dir = map_dir

        self.microscope = microscope
        self.nav = navigator
        self.imaging_params = imaging_params

        self.label_prefix = label_prefix
        self.final_prefix = final_prefix
        self.WG_model = WG_model
        self.is_lamella = True if self.WG_model else False

        self.position_ids = self.nav.searchByEntry("label", self.label_prefix, partial=True)
        self.eucentricity = False
        self.shifted = False
        self.save_future = None

        # Save offsets for each position
        self.offsets = np.zeros((len(self.position_ids), 2))

    def checkNecessity(self):
        """Checks if IM alignment is necessary."""

        return any([self.nav.items[nav_id].entries["Type"] != ["2"] for nav_id in self.position_ids]) or not self.nav.searchByEntry("label", self.final_prefix, partial=True)

    def collectImage(self, base_name, n):
        """Collects and saves IM image."""

        log(f"Collecting intermediate mag image...")

        im_file = self.cur_dir / (base_name + f"_IM{n + 1}.mrc")
        self.microscope.record(save=im_file)

        im_img = Buffer(buffer="A")

        return im_file, im_img
    
    def saveImage(self, im_file, im_buffer):
        """Saves image to map dir using model pixel size."""

        if not self.is_lamella:
            log(f"DEBUG: Skipping saving of IM image as no lamella detection is required.")
            return

        if self.save_future is not None:
            self.save_future.result()
            self.save_future = None
        im_img_file = self.map_dir / f"{im_file.stem}_wg.png"
        self.save_future = im_buffer.saveImg(im_img_file, self.WG_model.pix_size)

    def findLamella(self, p_id, im_file, im_buffer, distance_threshold):
        """Finds lamella and gets coordinate offset."""

        nav_id = self.position_ids[p_id]
        grid_name = self.microscope.autoloader[self.microscope.loaded_grid]

        # Find lamellae on image
        log(f"Finding lamella...")
        im_boxes = self.WG_model.findLamellae(im_file.stem, save_future=self.save_future)

        if im_boxes:
            if len(im_boxes) > 1:
                log(f"WARNING: Found more than one lamella. Using lamella closest to image center.")

            # Only keep box closest to center
            im_boxes.sortBy("center_dist")
            im_boxes.boxes = im_boxes.boxes[:1]

            log(f"NOTE: Detected lamella of class: {config.WG_model_categories[im_boxes.boxes[0].cat]} [{round(im_boxes.boxes[0].prob * 100)} %]")

            # Check if too close to previously detected lamella
            lamella_FP_ids = self.nav.searchByEntry("label", self.final_prefix, partial=True)     # Final polygons
            log(f"DEBUG: {self.final_prefix} ids: {lamella_FP_ids}")
            if self.nav.searchByCoords(im_buffer.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_buffer.pix_size)), margin=distance_threshold, subset=lamella_FP_ids):
                log(f"DEBUG: Distance: {im_buffer.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_buffer.pix_size))}")
                log(f"DEBUG: Search result: {self.nav.searchByCoords(im_buffer.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_buffer.pix_size)), margin=distance_threshold, subset=lamella_FP_ids)}")
                log(f"WARNING: Lamella seems to be among already detected lamellae and will be skipped.")
                return

            # Add box to nav via map buffer
            im_buffer.addNavBoxes(im_boxes, labels=[f"{self.final_prefix}{p_id + 1}"], padding_factor=config.MM_padding_factor if self.is_lamella else 1.0)                 # Add final lamella
            self.nav.pull()
            fp_id = self.nav.searchByEntry("label", f"{self.final_prefix}{p_id + 1}")[0]

            # Calculate shift between image center and detected lamella
            shift = self.nav.items[nav_id].getVector(im_buffer.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_buffer.pix_size)))
            log(f"DEBUG: Shift by image coords: {shift}")
            shift = self.nav.items[nav_id].getVector(self.nav.items[fp_id].stage)
            log(f"DEBUG: Shift by stage coords: {shift}")
            self.offsets[p_id] += shift

            # Shift items by vector between new box center stage coords and nav item stage coords
            # (already done during self.align())
            #if not self.shifted:
            #    # Get shift between image center and detected lamella
            #    #shift = self.nav.items[nav_id].getVector(im_buffer.px2stage(im_boxes.boxes[0].center * (im_boxes.pix_size / im_buffer.pix_size)))

            #    # Shift remaining PP nav items
            #    self.nav.shiftItems(self.offsets[p_id], items=self.position_ids[p_id + 1:])
            #    self.shifted = True

            #    # Save shifts
            #    self.offsets[p_id + 1:] += shift

            return fp_id
        else:
            return None

    def evaluateLamella(self, p_id, fp_nav_id, pp_item):
        """Evaluates which lamella polygon to retain."""

        if fp_nav_id is not None:
            label = self.nav.items[fp_nav_id].label
            note = self.nav.items[fp_nav_id].note

            # Retain larger polygon at position of final lamella
            log(f"DEBUG: WG lamella area: {pp_item.area} vs. IM lamella area: {self.nav.items[fp_nav_id].area}")
            if self.nav.items[fp_nav_id].area < pp_item.area:
                log(f"DEBUG: Keeping preliminary polygon as it is larger than IM detected polygon.")

                # Update stage coords
                log(f"DEBUG: Changing stage coords from [{pp_item.stage}] to [{self.nav.items[fp_nav_id].stage}]")
                pp_item.changeStage(self.nav.items[fp_nav_id].stage)

                # Replace final lamella with preliminary lamella
                self.nav.replaceItem(fp_nav_id, pp_item)

                # Only rename label and note after nav was pushed as it will update SerialEM directly
                self.nav.items[fp_nav_id].changeLabel(label)
                self.nav.items[fp_nav_id].changeNote(note)
        else:
            # Retain preliminary lamella if initial confidence was high (includes manually selected lamella)
            if float(pp_item.entries["UserValue1"][0]) >= 0.9:
                self.retainPreliminary(p_id, pp_item)
                log(f"WARNING: No lamella detected! Initially detected lamella was retained due to high confidence [{round(float(pp_item.entries['UserValue1'][0]) * 100)} %].")
            else:
                log(f"WARNING: No lamella detected. If you want to add lamellae manually, please select wait_for_inspection in the settings.")

    def retainPreliminary(self, p_id, pp_item, new_stage=None):
        """Retains preliminary polygon."""

        if new_stage is not None:
            # Update stage coords
            pp_item.changeStage(new_stage, relative=False)

        # Add initial polygon back to nav
        self.nav.items.append(pp_item) 
        pp_item.nav_index = len(self.nav) # Adjust nav_index
        self.nav.push()
        # Only rename label and note after nav was pushed as it will update SerialEM directly
        pp_item.changeLabel(f"{self.final_prefix}{p_id + 1}")
        pp_item.changeNote(f"{self.final_prefix}{p_id + 1}: " + pp_item.note.split(":")[-1])

    def findEucentricity(self):
        """Runs eucentricity routine and updates z coord of positions."""

        if not self.eucentricity:
            self.microscope.eucentricity()
            self.eucentricity = True

            # Update z for all polygon items
            for position_id in self.position_ids:
                self.nav.items[position_id].changeZ(self.microscope.stage[2])
            self.nav.push()

    def alignToRef(self, ref_buffer: Buffer, threshold=2, max_iterations=3):
        """Takes image and aligns to reference image iteratively using stage movement."""

        stage_initial = self.microscope.stage
        stage_shift = np.array([np.inf, np.inf])
        iterations = 0
        while np.linalg.norm(stage_shift) > threshold and iterations < max_iterations:
            self.microscope.record()
            im_buffer = Buffer(buffer="A")

            _, ss_shift = im_buffer.alignTo(ref_buffer, avoid_image_shift=True)
            stage_shift = self.microscope.getMatrices()["ss2s"] @ ss_shift

            if np.linalg.norm(stage_shift) <= threshold:
                log(f"DEBUG: Alignment converged {-stage_shift}.")
                im_buffer.clearAlignment()
                log(f"DEBUG: Pixel size after alignment: {im_buffer.pix_size}")
                return im_buffer, self.microscope.stage[:2] - stage_initial[:2]

            log(f"DEBUG: Moving stage by {-stage_shift}...")
            self.microscope.moveStage(-stage_shift, relative=True)
            iterations += 1

        log(f"WARNING: Alignment did not converge after {iterations} iterations. Resetting stage position...")
        self.microscope.moveStage(stage_initial)
        self.microscope.record()
        im_buffer = Buffer(buffer="A")
        return im_buffer, np.zeros(2)

    def align(self, settings):

        grid_name = self.microscope.autoloader[self.microscope.loaded_grid]

        if not self.checkNecessity():
            # Make sure IM pixel size is saved
            if not self.imaging_params.IM_pix_size:
                self.imaging_params.IM_pix_size = Buffer(nav_id=self.position_ids[0]).pix_size
                log(f"DEBUG: Updated IM pixel size to {self.imaging_params.IM_pix_size}.")

            log(f"WARNING: IM alignment was already done in a previous run! Skipping...")
            return
        
        self.microscope.changeImagingState(self.imaging_params.IM_image_state, low_dose_expected=False)
        self.microscope.changeC2Aperture(config.c2_apertures[1]) # Insert C2 aperture for IM state

        for p, position_id in enumerate(self.position_ids):
            # Skip if IM map already exists
            if self.nav.items[position_id].entries["Type"] == ["2"]:
                log(f"WARNING: ROI {p + 1} was already imaged at intermediate mag and will be skipped.")
                continue

            log(f"Moving to ROI {p + 1}...")
            self.microscope.moveStage(self.nav.items[position_id].stage)
            self.findEucentricity()

            ref_file = self.cur_dir / f"{grid_name}_IM{p + 1}_ref.mrc"
            if ref_file.exists():
                log(f"Realigning to WG reference...")
                im_buffer, stage_shift = self.alignToRef(ref_buffer=Buffer(buffer="O", file_path=ref_file))
                self.offsets[p] += stage_shift

                # Shift remaining PP nav items by alignment shift
                if not self.shifted:
                    self.nav.shiftItems(stage_shift, item_ids=self.position_ids[p + 1:])
                    self.shifted = True

                    # Save shifts
                    self.offsets[p + 1:] += stage_shift
            else:
                log(f"ERROR: No reference image found for IM alignment! Please make sure [{ref_file}] exists.")
                return
                    
            # Make new nav map
            im_file = self.cur_dir / f"{grid_name}_IM{p + 1}.mrc"
            map_id = self.nav.newMap(buffer=im_buffer, img_file=im_file, label=f"{self.label_prefix}{p + 1}", note=self.nav.items[position_id].note, coords=self.nav.items[position_id].stage) # Coords are only needed for DUMMY and ignored for normal map

            # Update IM pix_size
            if not self.imaging_params.IM_pix_size:
                self.imaging_params.IM_pix_size = im_buffer.pix_size
                log(f"DEBUG: Updated IM pixel size to {self.imaging_params.IM_pix_size}.")

            # Save image as PNG
            self.saveImage(im_file, im_buffer)

            # Retain preliminary polygon nav item in case no lamella is detected at IM
            pp_item = deepcopy(self.nav.items[position_id])
            pp_item.nav_index = None

            # Overwrite polygon with map
            self.nav.replaceItem(position_id, map_id)
            im_buffer.nav_id = position_id # Update nav id of buffer

            if self.is_lamella: # TODO: Check if this is necessary with MM alignment
                fp_nav_id = self.findLamella(p, im_file, im_buffer, distance_threshold=settings["WG_distance_threshold"])
                self.evaluateLamella(p, fp_nav_id, pp_item)
            else:
                self.retainPreliminary(p, pp_item, new_stage=self.nav.items[position_id].stage)

    def saveAlignment(self):
        """Saves alignment result to file."""

        ali_data = {"position_ids": self.position_ids, "offsets": self.offsets}
        with open(self.cur_dir / "IM_alignment.json", "w") as f:
            json.dump(ali_data, f, indent=4, default=utils.convertToTaggedString)