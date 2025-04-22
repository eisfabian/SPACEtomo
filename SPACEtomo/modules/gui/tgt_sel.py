#!/usr/bin/env python
# ===================================================================
# ScriptName:   tgt_sel
# Purpose:      User interface for selecting SPACEtomo targets
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/03/20
# Revision:     v1.3
# Last Change:  2025/04/12: added InfoBoxManager to stack popups
#               2025/04/08: fixed tooltip combination of hidden buttons
#               2025/03/14: added dense pattern, added threaded preloading of next map
#               2025/03/10: added polygon mode to exclude suggestions, added add suggestions button
#               2025/02/20: added plot legend
#               2025/02/10: added outline for reacquire settings
#               2025/01/30: added dragging of targets directly, removed drag points
#               2024/12/23: added reacquire settings
#               2024/12/20: added autoStartTilt option
#               2024/11/27: added reacquire button
#               2024/10/02: replaced text inputs with sliders for targeting settings
#               2024/09/25: added check to save acquisition settings
#               2024/09/04: small quality of life improvements after feedback
#               2024/08/22: adjust tilt axis position to tracking target
#               2024/08/20: fixed tilt axis plot
#               2024/08/09: Sped up loading by 30-40% using toNumpy, float32 and tiled textures, added menu to change target area of point
#               2024/08/08: all old functions reimplemented, added option to make target tracking target
#               2024/08/07: branched out tgt.py for Targets class, added adding and deleting targets
#               2024/07/31: rewrote map, segmentation and target loading
#               2024/07/30: started rewrite using gui module
#               2024/04/09: fixes after Rado Krios test, added default binning to speed up loading
#               2024/04/02: fixes after Krios 1 test, allowed click + D to delete and click + G to add geo point, added tilt axis
#               2024/03/27: added tooltips, added ask for save on close, added variable tooltip for targets on plot
#               2024/03/26: added geo points handling
#               2024/03/25: added dir as argument to be called from SPACEtomo script, added thicker outlines for targets, added None class to not show any overlay, added generic thumbnail, minor text fixes
#               2024/03/22: add table for map selection window
#               2024/03/21: Copied all manipulation functions and considered proper pixel size, added thumbnail creation and map selection window
#               2024/03/20: Copy most if inspection tab from SPACEtomo_TI
# ===================================================================

import os
os.environ["__GLVND_DISALLOW_PATCHING"] = "1"           # helps to minimize Segmentation fault crashes on Linux when deleting textures
import sys
import json
import time
from pathlib import Path
try:
    import dearpygui.dearpygui as dpg
except:
    print("ERROR: DearPyGUI module not installed! If you cannot install it, please run the GUI from an external machine.")
    sys.exit()
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import concurrent.futures

from SPACEtomo.modules import ext as space_ext
from SPACEtomo.modules.gui import gui
from SPACEtomo.modules.gui.thmb import MapWindow
from SPACEtomo.modules.gui.flm import FlmWindow
from SPACEtomo.modules.gui.menu import Menu
from SPACEtomo.modules.gui.info import InfoBoxManager, InfoBox, StatusLine, saveSnapshot
from SPACEtomo.modules.gui.plot import Plot, PlotBox, PlotPolygon
from SPACEtomo.modules.gui.map import MMap, Segmentation
from SPACEtomo.modules.tgt import Targets
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log
from SPACEtomo import __version__

import faulthandler
faulthandler.enable()

class TargetGUI:

    def mouseClick(self, sender, app_data):
        """Handle mouse clicks when plot is hovered within map boundaries."""

        # Get mouse coords in plot coord system
        mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # Delegate to sub-window if open
        if dpg.is_item_shown(self.lm_window.window):
            self.lm_window.mouseClick(mouse_coords)
            return

        # Check if click needs to be processed
        if not dpg.is_item_hovered(self.plot.plot) or not self.plot.withinBounds(mouse_coords) or self.inspected:
            return
        
        # Convert plot coords to image coords in px
        img_coords = self.loaded_map.microns2px(mouse_coords)

        # Left mouse button + Shift functions
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and (dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)):
            if self.targets.addTarget(img_coords, new_area=dpg.is_key_down(dpg.mvKey_T)):
                self.showTargets()
                self.showTargetAreaButtons()
                self.menu_right.showElements(["butsave"])

        # Right mouse button functions
        elif dpg.is_mouse_button_down(dpg.mvMouseButton_Right) or (dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and dpg.is_key_down(dpg.mvKey_E)):
            # Check for target in range to show menu
            if not self.showTargetMenu(img_coords=img_coords):
                # If no target in range, check for geo point or polygon in range to delete
                if not self.polygon_mode:
                    self.removeGeoPoint(img_coords=img_coords)
                else:
                    self.removePolygon(img_coords=img_coords)

        # Middle mouse button functions
        elif dpg.is_mouse_button_down(dpg.mvMouseButton_Middle) or (dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and dpg.is_key_down(dpg.mvKey_G)):
            if self.targets.addGeoPoint(img_coords):
                self.showTargets()
                self.menu_right.showElements(["butsave"])

        elif dpg.is_mouse_button_down(dpg.mvMouseButton_Left):

            # Get camera dims
            rec_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.rec_pix_size / self.loaded_map.pix_size).astype(int) 
            focus_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.focus_pix_size / self.loaded_map.pix_size).astype(int) 
            
            # Check if target is in range
            closest_point_id, in_range = self.targets.getClosestPoint(img_coords, rec_dims / 2)
            if in_range:
                self.drag_point = closest_point_id
                #log(f"DEBUG: Drag point {self.drag_point} selected at {self.targets.areas[self.drag_point[0]].points[self.drag_point[1]]}")
                self.drag_start = mouse_coords
                #log(f"DEBUG: Drag start at {self.drag_start}")

            else:
                # Check if geo point is in range
                closest_point_id, in_range = self.targets.getClosestGeoPoint(img_coords, focus_dims / 2)
                if in_range:
                    self.drag_point = closest_point_id
                    #log(f"DEBUG: Drag geo point {self.drag_point} selected at {self.targets.areas[0].geo_points[self.drag_point]}")
                    self.drag_start = mouse_coords
                    #log(f"DEBUG: Drag start at {self.drag_start}")

                # Check if suggestion is in range
                else:
                    closest_point_id, in_range = self.targets.getClosestSuggestion(img_coords, rec_dims / 2)
                    if in_range:
                        self.targets.addTarget(self.targets.suggestions[closest_point_id])
                        if self.hole_mode:
                            self.suggestHolePattern()
                        elif self.dense_mode:
                            self.suggestDensePattern()
                        self.showTargets()
                        self.showTargetAreaButtons()
                        self.menu_right.showElements(["butsave"])

                    # Add polygon if in polygon mode
                    else:
                        if self.polygon_mode:
                            if self.plot.boxes:
                                for b, box in reversed(list(enumerate(self.plot.boxes))):
                                    if isinstance(box, PlotPolygon) and box.open:
                                        box.addPoint(mouse_coords)
                                        return
                            
                            self.plot.boxes.append(PlotPolygon(mouse_coords, parent=self.plot.plot, color=gui.COLORS["geo"], thickness=0.1))
                            self.plot.boxes[-1].draw()

    def mouseDrag(self, sender, app_data):
        """Handles update on mouse drag."""

        if self.drag_point is not None and self.drag_start is not None:
            # Get mouse coords in plot coord system
            mouse_coords = np.array(dpg.get_plot_mouse_pos())
            mouse_coords = np.clip(mouse_coords, [0, 0], self.plot.bounds[:, 1])

            # Target has area_id and point_id, geo_point has only point_id
            if isinstance(self.drag_point, list):
                # Update target coords
                self.targets.areas[self.drag_point[0]].points[self.drag_point[1]] += self.loaded_map.microns2px(mouse_coords) - self.loaded_map.microns2px(self.drag_start)
                
                # Clip to map boundaries
                rec_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.rec_pix_size / self.loaded_map.pix_size).astype(int) 
                self.targets.areas[self.drag_point[0]].points[self.drag_point[1]] = np.clip(self.targets.areas[self.drag_point[0]].points[self.drag_point[1]], rec_dims // 2, self.loaded_map.img.shape - rec_dims // 2)

                # Move target overlay without redrawing all targets
                overlay_id = utils.findIndex(self.plot.overlays, "label", f"tgt_{self.drag_point[0]}_{self.drag_point[1]}")
                self.plot.shiftOverlay(overlay_id, mouse_coords - self.drag_start)
            else:
                # Update geo point coords
                for area in self.targets.areas:
                    area.geo_points[self.drag_point] += self.loaded_map.microns2px(mouse_coords) - self.loaded_map.microns2px(self.drag_start)

                # Move geo point overlay without redrawing all targets
                overlay_id = utils.findIndex(self.plot.overlays, "label", f"geo_0_{self.drag_point}")
                self.plot.shiftOverlay(overlay_id, mouse_coords - self.drag_start)

            # Update drag start
            self.drag_start = mouse_coords

    def mouseRelease(self, sender, app_data):
        """Handle mouse release and check if any drag points were moved."""

        # Delegate to sub-window if open
        if dpg.is_item_shown(self.lm_window.window):
            self.lm_window.mouseRelease()
            return

        # Only process when targets are editable
        if not self.targets or self.inspected:
            return

        # Reset drag tracking
        if self.drag_point is not None and self.drag_start is not None:
            # Update point within area (also updates area center if tracking target)
            if isinstance(self.drag_point, list):
                self.targets.areas[self.drag_point[0]].updatePoint(self.drag_point[1], self.targets.areas[self.drag_point[0]].points[self.drag_point[1]])

            self.drag_point = None
            self.drag_start = None
            self.showTargets()
            self.menu_right.showElements(["butsave"])

    def updateMapList(self, list_only=False):
        """Checks for new maps in folder."""

        prev_map_num = len(self.map_list)
        prev_tgt_num = sum(self.map_list_tgtnum)

        # Find all maps with segmentation (and existing map, which sometimes is saved later due to threading)
        seg_list = sorted(self.cur_dir.glob("*_seg.png"))
        self.map_list = [seg.name.split("_seg.png")[0] for seg in seg_list if (seg.parent / (seg.name.split("_seg.png")[0] + ".png")).exists()]

        if self.map_list and not list_only:        # distinction necessary for creating map list before GUI is set up
            # Update combo menu
            dpg.configure_item(self.menu_left.all_elements["map"], items=self.map_list, label=f"({len(self.map_list)} maps)")
            # Set value to loaded map
            if self.map_name:
                dpg.set_value(self.menu_left.all_elements["map"], self.map_name)
            else:
                dpg.set_value(self.menu_left.all_elements["map"], self.map_list[0])

        # Find selected targets per MM map
        self.map_list_tgtnum = []
        for m_name in self.map_list:
            point_files = sorted(self.cur_dir.glob(m_name + "_points*.json"))
            point_num = 0
            if len(point_files) > 0:
                for file in point_files:
                    # Load json data
                    with open(file, "r") as f:
                        point_data = json.load(f, object_hook=utils.revertTaggedString)
                        point_num += len(point_data["points"])
            self.map_list_tgtnum.append(point_num)

        # Check for changes
        if len(self.map_list) != prev_map_num or sum(self.map_list_tgtnum) != prev_tgt_num:
            return True
        else:
            return False
        
    def checkPointFiles(self):
        """Checks if new point file of loaded map was created."""

        # Don't check if no map is loaded or targets are already loaded (or selected) or if there is an active status (e.g. map is loading)
        if not self.map_name or (self.targets and len(self.targets) > 0) or self.checked_point_files or self.status.status:
            #log(f"DEBUG: NOT checking point files!")
            return
        
        log(f"DEBUG: Decided to check point files!")
        
        # Get all point files for MM map
        point_files = sorted(self.cur_dir.glob(self.map_name + "_points*.json"))
        if len(point_files) > 0:
            # Check if any points in point files
            point_num = 0
            for file in point_files:
                # Load json data
                with open(file, "r") as f:
                    point_data = json.load(f, object_hook=utils.revertTaggedString)
                    point_num += len(point_data["points"])

            # Arrange opening of prompt to load point files
            if point_num > 0:
                log(f"NOTE: Target coordinates for the loaded map were found!")
                # Need to open info box on next mouse move because info box requires split_frame function which can't be called from within main loop
                with dpg.handler_registry(tag="check_point_file"):
                    dpg.add_mouse_move_handler(callback=self.confirmLoadPointFiles)

    def confirmLoadPointFiles(self, sender=None, app_data=None, user_data=None):
        """Loads point file after user confirmation."""

        # Reset mouse move handler
        if dpg.does_item_exist("check_point_file"):
            dpg.delete_item("check_point_file")

        # Open box unless already called by box
        if not user_data:
            InfoBoxManager.push(InfoBox("New target coordinates", "Target coordinates for the currently loaded map were found. Do you want to load them?", callback=self.confirmLoadPointFiles, options=["Load", "Cancel"], options_data=[True, False], loading=False))
            self.checked_point_files = True
            return

        # Check for info box input
        if user_data and dpg.does_item_exist(user_data[0]):
            dpg.delete_item(user_data[0])
            dpg.split_frame()

        # Check user choice
        if user_data[1]:
            self.loadTargetsFile()

    def selectMap(self, sender=None, app_data=None, user_data=None, next_map=False):
        """Handles map selection from several buttons."""

        # Check for unsaved changes
        if dpg.is_item_shown(self.menu_right.all_elements["butsave"]):
            # Close map window if open
            self.map_window.hide()
            dpg.split_frame() # Close window before opening warning in next frame
            InfoBoxManager.push(InfoBox("WARNING", "There are unsaved changes to your targets. Please save or discard your targets before loading a new map!", callback=self.saveAndContinue, options=["Save", "Discard"], options_data=[True, False], loading=False))
            return
        
        # Check for valid user_data
        if user_data and user_data in self.map_list:
            self.map_name = user_data
        else:            
            self.map_name = dpg.get_value(self.menu_left.all_elements["map"])

        if next_map:
            map_id = self.map_list.index(self.map_name) + 1
            if map_id >= len(self.map_list):
                map_id = 0
                log("WARNING: Reached end of list of maps. Loading first map!")
            self.map_name = self.map_list[map_id]

        dpg.set_value(self.menu_left.all_elements["map"], self.map_name)

        # Close map window if open
        if dpg.is_item_visible(self.map_window.map_window):
            self.map_window.hide()
            # Split a frame to allow closing the map window if it was selected from there
            dpg.split_frame()

        # Update map list on every button click
        if self.updateMapList():
            # Update fill map table
            self.map_window.update(self.map_name, self.map_list, self.map_list_tgtnum)
        else:
            # Update only selected map
            self.map_window.updateMap(self.map_name)

        self.loadMap()

    def loadMap(self):
        """Loads map from file and all associated data."""

        file_path = self.cur_dir.joinpath(self.map_name + ".png")

        # Update GUI
        self.menu_left.hide()
        self.menu_right.hide()
        self.plot.updateLabel(file_path.name + " loading...")
        self.status.update(f"Loading {file_path.name}...", box=True)

        # Reset FLM window plots
        self.lm_window.clearAll()
        # Reset plot
        self.plot.clearAll()
        # Clear out texture references because textures have been deleted when plot was cleared
        self.target_overlays = {}

        # Reset reacquire settings
        dpg.set_value(self.menu_left.all_elements["center_offset_x"], 0)
        dpg.set_value(self.menu_left.all_elements["center_offset_y"], 0)
        dpg.set_value(self.menu_left.all_elements["padding"], 1.5)
        dpg.set_value(self.menu_left.all_elements["restitch"], False)

        # Check if map was preloaded
        if self.preloaded_data:
            # If currently loading a map, wait for it to finish
            if isinstance(self.preloaded_data, concurrent.futures.Future):
                self.preloaded_data = self.preloaded_data.result()
        if self.preloaded_data and self.preloaded_data["map"].file == file_path:
            log(f"DEBUG: Preloaded map was loaded.")
            self.loaded_map = self.preloaded_data["map"]
            map_textures = self.preloaded_data["textures"]
            bounds = self.preloaded_data["bounds"]
            self.preloaded_data = None

        else:
            # Discard wrong preloaded map to allow preloading next map
            if self.preloaded_data:
                # Delete preloaded textures
                for tex in self.preloaded_data["textures"]:
                    if dpg.does_item_exist(tex):
                        dpg.delete_item(tex)
                        dpg.split_frame(delay=10) # helps to reduce Segmentation fault crashes
                self.preloaded_data = None

            # Load map file
            self.loaded_map = MMap(file_path, pix_size=self.model.pix_size, status=self.status)
            self.loaded_map.checkBinning(default_binning=dpg.get_value(self.menu_left.all_elements["inpbin"]))
            self.status.update("Plotting map...", box=True)
            #map_texture = self.loaded_map.getTexture()
            map_textures, bounds = self.loaded_map.getTileTextures()

        # Plot map
        start = time.time()
        #self.plot.addImg(map_texture, [[0, self.loaded_map.dims_microns[0]], [0, self.loaded_map.dims_microns[1]]], self.loaded_map.binning, label="map")
        for i, (tex, bound) in enumerate(zip(map_textures, bounds)):
            self.plot.addImg(tex, bound, self.loaded_map.binning, label=f"map{i}")
        log(f"Map was plotted in {time.time() - start} s.")
        self.plot.resetZoom()
        self.plot.updateLabel(self.map_name)

        self.lm_window.loadEM(self.loaded_map, map_textures, bounds)

        # Load Segmentation
        self.status.update("Loading segmentation...", box=True)
        self.segmentation = Segmentation(self.loaded_map.file.parent / (self.loaded_map.file.stem + "_seg.png"), self.model, self.loaded_map.pix_size)

        # Show tilt axis
        self.showTiltAxis()

        # Check model pixel size and update if necessary
        if self.loaded_map.pix_size != self.model.pix_size:
            log(f"WARNING: Model pixel size ({self.model.pix_size}) does not match map pixel size ({self.loaded_map.pix_size}). Segmentation might be inaccurate!")
            self.model.pix_size = self.loaded_map.pix_size
            self.model.setDimensions(self.mic_params)

        # Pre-generate overlays to remove lag when selecting first target
        self.makeTargetOverlay()

        # Load targets
        self.status.update("Loading targets...", box=True)
        self.targets = Targets(map_dir=self.cur_dir, map_name=self.map_name, map_dims=self.loaded_map.img.shape, tgt_params=self.tgt_params, map_pix_size=self.loaded_map.pix_size)

        self.loadTargetsFile()

        # Reset modes
        self.polygon_mode = False
        self.dense_mode = False
        self.hole_mode = False

        # Reset checked for point files status
        self.checked_point_files = False

        # Update GUI
        #self.menu_left.unlockRows(["filters"])     # Not yet implemented
        self.menu_left.unlockRows(["reacquire"])
        dpg.bind_item_theme(self.menu_left.all_elements["butreacq"], None) # Reset active theme
        self.menu_left.lockRows(["reacquire_settings", "reacquire_center", "reacquire_padding", "reacquire_restitch", "reacquire_queue"])
        if self.segmentation.valid:         # Only show class selection if segmentation was performed
            self.menu_left.unlockRows(["class_list", "class_buttons"])
            self.menu_right.unlockRows(["settings1", "settings2", "settings3", "settings4", "settings5", "settings6"])
        self.menu_right.unlockRows(["settings4"])

        # Show toggle target suggestions button
        if "grid_vectors" in self.loaded_map.meta_data.keys():
            self.toggleHolePattern(force_off=True)
            self.menu_icon.showElements(["butholes"])
        self.toggleDensePattern(force_off=True)
        dpg.bind_item_theme(self.menu_icon.all_elements["butpolygon"], None)
        self.menu_icon.showElements(["butdense", "butpolygon"])
        if self.inspected:
            self.menu_icon.hideElements(["butholes", "butdense", "butpolygon"])

        self.menu_left.show()

        self.status.update()

    def preloadMap(self):
        """Preloads second map quietly."""

        # Get next map
        if not self.map_name:
            map_name = self.map_list[0]
        else:
            map_id = self.map_list.index(self.map_name) + 1
            if map_id >= len(self.map_list):
                return
            map_name = self.map_list[map_id]

        file_path = self.cur_dir.joinpath(map_name + ".png")

        log(f"DEBUG: Preloading map {file_path}...")

        # Load map file
        preloaded_map = MMap(file_path, pix_size=self.model.pix_size)
        preloaded_map.checkBinning(default_binning=dpg.get_value(self.menu_left.all_elements["inpbin"]))
        map_textures, bounds = preloaded_map.getTileTextures()

        return {"map": preloaded_map, "textures": map_textures, "bounds": bounds}

    def loadTargetsFile(self):
        """Loads targets from file and updates GUI."""

        self.targets.loadAreas()
        self.showTargets()

        # Fill settings input fields from loaded targets
        self.setAcquisitionSettings()

        # Get class scores for targets
        if self.segmentation.valid and self.targets and len(self.tgt_params.target_list) > 1:
            self.getTargetClassScores()

        # Check inspected
        if (self.cur_dir / (self.map_name + "_inspected.txt")).exists():
            self.inspected = True
            dpg.show_item("inspected")
        else:
            self.inspected = False
            dpg.hide_item("inspected")

        # Update GUI
        if not self.inspected:
            self.showTargetAreaButtons()
            self.menu_right.unlockRows(["areas", "delete", "acquisition1", "acquisition2", "acquisition3", "acquisition4", "acquisition5", "acquisition6", "buttons1", "buttons2"])
            self.menu_right.show()

            # Reset save button
            self.menu_right.hideElements(["butsave"])

    def loadOverlay(self, *args, cats=[]):
        """Plots overlay of selected categories."""

        # Update GUI
        self.menu_left.hide()
        self.menu_right.hide()
        self.status.update("Loading overlay...", box=True)

        if cats:
            selected_cats = cats
        else:
            # Get selected classes from checkboxes
            selected_cats = self.getClassSelection(self.menu_left)

        # Check if selected classes have changed
        if not all([cat in self.overlay for cat in selected_cats]):

            # Clear previous overlay
            self.plot.clearImg(self.plot.getImgByKeyword("seg") + [""])     # needs empty list entry, otherwise all items will be cleared in case of no hits

            # If no classes selected, don't load new overlay
            if selected_cats:

                # Get texture
                #mask_texture = self.segmentation.getMaskTexture(selected_cats)
                mask_textures, bounds = self.segmentation.getMaskTileTextures(selected_cats)

                # Plot map
                #start = time.time()
                #self.plot.addImg(mask_texture, [[0, self.loaded_map.dims_microns[0]], [0, self.loaded_map.dims_microns[1]]], self.segmentation.binning, label="seg")
                #log(f"Overlay plotted in {time.time() - start} s.")

                # Plot map
                start = time.time()
                for i, (tex, bound) in enumerate(zip(mask_textures, bounds)):
                    self.plot.addImg(tex, bound, self.loaded_map.binning, label=f"seg{i}")
                log(f"Overlay plotted in {time.time() - start} s.")

            # Update overlay classes
            self.overlay = selected_cats

        # Update GUI
        self.menu_left.show()
        if not self.inspected:
            self.menu_right.show()
        self.status.update()

    def showTiltAxis(self, point=None):
        """Plots tilt axis."""

        # Clear previous tilt axis
        self.plot.clearSeries(["tilt_axis"])

        # Get tilt axis rotation matrix
        theta = np.radians(self.mic_params.view_ta_rotation)
        rotM = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        axis_unit = rotM @ np.array([1, 0])

        if point is None:
            # Get centered tilt axis
            axis_length = min(abs(self.loaded_map.img.shape[0] / axis_unit[0]), abs(self.loaded_map.img.shape[1] / axis_unit[1]))               # total axis length
            axis_length1 = axis_length2 = axis_length / 2
            offset = np.array([self.loaded_map.img.shape[0] / 2, self.loaded_map.img.shape[1] / 2])                                             # Offset to center of map
        else:
            # Get tilt axis centered on point
            axis_length1 = min(abs(point[0] / axis_unit[0]), abs(point[1] / axis_unit[1]))                                                      # Split axis in lengths from point to edges of map
            axis_length2 = min(abs((self.loaded_map.img.shape[0] - point[0]) / axis_unit[0]), abs((self.loaded_map.img.shape[1] - point[1]) / axis_unit[1]))
            if abs(axis_unit[0]) > abs(axis_unit[1]):
                axis_length1 = -axis_length1
                axis_length2 = -axis_length2
            offset = np.array([self.loaded_map.img.shape[0] - self.targets.areas[0].points[0][0], self.targets.areas[0].points[0][1]])          # Apply offset coords from point considering axis inversion
        axis_points = (np.array([rotM @ np.array([-axis_length1, 0]), rotM @ np.array([axis_length2, 0])]) + offset) * self.loaded_map.pix_size / 1000

        # Add custom series to plot.series list
        self.plot.addCustomPlot(self.plot.series, label="tilt_axis", plot=dpg.add_line_series(list(axis_points[:, 1]), list(axis_points[:, 0]), parent=self.plot.x_axis), theme="axis_theme")

    def filterMap(self):
        """Loads map filtered with LISC."""

        # TODO
        log("WARNING: Map filtering is not yet implemented.")

    def showReacquireSettings(self):
        """Shows reacquire settings for map and shows outline."""

        # Toggle settings menu
        if dpg.is_item_shown(self.menu_left.all_rows["reacquire_settings"]):
            dpg.bind_item_theme(self.menu_left.all_elements["butreacq"], None)
            self.menu_left.lockRows(["reacquire_settings", "reacquire_center", "reacquire_padding", "reacquire_restitch", "reacquire_queue"])
            self.plot.clearBoxes()
        else:
            dpg.bind_item_theme(self.menu_left.all_elements["butreacq"], "active_btn_theme")
            self.menu_left.unlockRows(["reacquire_settings", "reacquire_center", "reacquire_padding", "reacquire_restitch", "reacquire_queue"])
            self.plot.boxes.append(PlotBox(np.zeros(2), parent=self.plot.plot, thickness=0.1))
            self.plot.boxes[-1].draw()
            self.updateReacquireBounds()

    def updateReacquireBounds(self):
        """Updates bounds of reacquired map."""

        p1 = self.plot.bounds[[0, 1], [0, 0]]
        p2 = self.plot.bounds[[0, 1], [1, 1]]
        size = p2 - p1
        center_offset = np.array([dpg.get_value(self.menu_left.all_elements["center_offset_x"]) / 100, dpg.get_value(self.menu_left.all_elements["center_offset_y"]) / 100])
        padding = dpg.get_value(self.menu_left.all_elements["padding"])

        p1 += center_offset * size * np.array([1, -1])
        p2 += center_offset * size * np.array([1, -1])
        p1 -= (padding / 1.5 - 1) * size / 2
        p2 += (padding / 1.5 - 1) * size / 2

        self.plot.boxes[-1].updateP1(p1)
        self.plot.boxes[-1].updateP2(p2)

    def reacquireMap(self):
        """Marks map to be reacquired by next SPACEtomo run."""

        # Get settings
        center_offset = np.array([dpg.get_value(self.menu_left.all_elements["center_offset_x"]) / 100, dpg.get_value(self.menu_left.all_elements["center_offset_y"]) / 100])
        center_offset_nm = center_offset * np.flip(self.loaded_map.img.shape) * self.loaded_map.pix_size
        padding = dpg.get_value(self.menu_left.all_elements["padding"])
        restitch = dpg.get_value(self.menu_left.all_elements["restitch"])

        # Create reacquire file
        with open(self.cur_dir / (self.map_name + "_reacquire.json"), "w") as f:
            json.dump({"center_offset": center_offset_nm, "padding": padding, "restitch": restitch}, f, indent=4, default=utils.convertToTaggedString)

        log(f"DEBUG: Reacquire map {self.map_name} on next SPACEtomo run!")
        InfoBoxManager.push(InfoBox("INFO", "This map will be reacquired on the next SPACEtomo run. If you want to reacquire now, please stop the SPACEtomo run and start it again.", loading=False))

        self.clearTargets()
        self.markInspected(None, None, None, keep_map=True)

    def getReacquireCenter(self, sender, app_data, user_data):
        """Sets center offset for map reacquisition to the current tracking target if it exists."""

        # Get coords of first target as center for new map
        if self.targets and len(self.targets) > 0:
            center = self.targets.areas[0].points[0] / np.array(self.loaded_map.img.shape) - np.array((0.5, 0.5))
            log(f"DEBUG: New center of reacquired map will be shifted by {center}")

            dpg.set_value(self.menu_left.all_elements["center_offset_x"], int(round(center[1] * 100)))
            dpg.set_value(self.menu_left.all_elements["center_offset_y"], int(round(center[0] * 100)))

            self.updateReacquireBounds()
                          
        else:
            log(f"ERROR: Please select a tracking target to use as center of the reacquired map.")
            InfoBoxManager.push(InfoBox("ERROR", "Please select a tracking target to use as center of the reacquired map!", loading=False))

    def applyClassSelection(self, sender, app_data, user_data):
        """Applies checked classes to target list."""

        # Check for user data
        if user_data:
            menu, element = user_data

        # Set target_list as default
        if not element:
            element = self.menu_right.all_elements["target_list"]

        # Get selected classes from checkboxes
        selected_cats = self.getClassSelection(menu)

        # Fill input
        dpg.set_value(element, ",".join(selected_cats))


    def getClassSelection(self, menu):
        """Gets selected classes from checkboxes."""

        selected_cats = []
        for cat in self.model.categories:
            if dpg.get_value(menu.all_elements[cat]):
                selected_cats.append(cat)

        return selected_cats

    def showTargets(self):
        """Updates target overlays on plot."""

        # Skip rest if currently no targets
        if not self.targets and not self.targets.suggestions:
            return
        
        # Clear targets from plot
        self.clearTargets(plot_only=True)
        self.plot.clearSeries(skip_labels=["tilt_axis"])

        # Adjust plot label
        self.plot.updateLabel(f"{self.map_name} [{len(self.targets)} targets]")

        # Generate target overlay texture
        if not self.target_overlays or not dpg.does_item_exist(self.target_overlays["target"]):
            self.makeTargetOverlay()

        # Get map dims
        dims = np.flip(np.array(self.loaded_map.img.shape))

        # Add empty scatter series for legend
        if self.targets:
            self.plot.addSeries([], [], label="Tracking target", theme=f"limit_scatter_theme")
            if len(self.targets) > 1:
                self.plot.addSeries([], [], label="Target", theme=f"scatter_theme0")
            if len(self.targets.areas[0].geo_points) > 0:
                self.plot.addSeries([], [], label="Geo points", theme=f"geo_scatter_theme")
            self.plot.addSeries([], [], label="Beam size", theme=f"beam_scatter_theme")
            if len(self.targets.suggestions) > 0:
                self.plot.addSeries([], [], label="Target suggestions", theme=f"suggestion_scatter_theme")

        tgt_counter = 0
        geo_counter = 0
        for t, target_area in enumerate(self.targets.areas):
            if len(target_area.points) == 0: continue
            # Transform coords to plot
            points = np.array([self.loaded_map.px2microns(point) for point in target_area.points])

            # Scatter plot of targets to mark areas only if more than one area exists
            if len(self.targets.areas) > 1:
                self.plot.addSeries(points[:, 0], points[:, 1], label=f"Target Area {t + 1}", theme=f"scatter_theme{t % len(gui.THEME_COLORS)}")

            # Show points above image shift limit
            distances = np.linalg.norm(points - points[0], axis=1)
            points_beyond_limit = points[distances > self.mic_params.IS_limit]
            if len(points_beyond_limit) > 0:
                self.plot.addSeries(points_beyond_limit[:, 0], points_beyond_limit[:, 1], label=f"Out of IS range", theme=f"limit_scatter_theme")

            # Go over all target points
            scaled_overlay_dims = self.target_overlays["tgtdims"] * self.loaded_map.pix_size / 1000
            for p, (x, y) in enumerate(points):
                # Show graphical overlays
                bounds = ((x - scaled_overlay_dims[1] / 2, x + scaled_overlay_dims[1] / 2), (y - scaled_overlay_dims[0] / 2, y + scaled_overlay_dims[0] / 2))
                if p == 0:
                    self.plot.addOverlay(self.target_overlays["track"], bounds, label=f"tgt_{t}_{p}")
                else:
                    self.plot.addOverlay(self.target_overlays["target"], bounds, label=f"tgt_{t}_{p}")

                tgt_counter += 1

            if tgt_counter > 0 and len(target_area.geo_points) > 0:
                # Transform geo coords to plot
                points = np.array([self.loaded_map.px2microns(point) for point in target_area.geo_points])

                # Go over all geo points to draw overlays only for first area
                if t == 0:
                    scaled_overlay_dims = self.target_overlays["geodims"] * self.loaded_map.pix_size / 1000
                    for p, (x, y) in enumerate(points):
                        # Show graphical overlays
                        bounds = ((x - scaled_overlay_dims[1] / 2, x + scaled_overlay_dims[1] / 2), (y - scaled_overlay_dims[0] / 2, y + scaled_overlay_dims[0] / 2))
                        self.plot.addOverlay(self.target_overlays["geo"], bounds, label=f"geo_{t}_{p}")

                        geo_counter += 1

        # Show target suggestions
        if self.targets.suggestions:
            # Transform geo coords to plot
            points = np.array([self.loaded_map.px2microns(point) for point in self.targets.suggestions])

            scaled_overlay_dims = self.target_overlays["tgtdims"] * self.loaded_map.pix_size / 1000
            for p, (x, y) in enumerate(points):
                # Show graphical overlays
                bounds = ((x - scaled_overlay_dims[1] / 2, x + scaled_overlay_dims[1] / 2), (y - scaled_overlay_dims[0] / 2, y + scaled_overlay_dims[0] / 2))
                self.plot.addOverlay(self.target_overlays["suggestion"], bounds, label=f"sug_{p}")

            # Show add suggestions button
            self.menu_right.unlockRows(["suggestions"])
        else:
            # Hide add suggestions button
            self.menu_right.lockRows(["suggestions"])

        # Show tilt axis centered on tracking target
        self.showTiltAxis(self.targets.areas[0].points[0] if self.targets else None)

    def targetTooltip(self):
        """Configures plot tooltip to contain target information."""

        mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # Convert plot coords to image coords in px
        if self.loaded_map and self.targets:
            img_coords = self.loaded_map.microns2px(mouse_coords)

            # Get camera dims
            rec_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.rec_pix_size / self.loaded_map.pix_size).astype(int) 
            focus_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.focus_pix_size / self.loaded_map.pix_size).astype(int) 

            # Check if coords are too close to existing point
            closest_point_id, in_range = self.targets.getClosestPoint(img_coords, rec_dims / 2)

            if in_range:
                dpg.set_value("tt_heading", "Target information:")
                info, color = self.assembleTargetInfo(closest_point_id)
                dpg.set_value("tt_text", info)
                dpg.configure_item("tt_text", color=color)

                # Change cursor to crosshair
                self.showCrosshair()
                return
            else:
                closest_point_id, in_range = self.targets.getClosestGeoPoint(img_coords, focus_dims / 2)
                if in_range:
                    # Change cursor to crosshair
                    self.showCrosshair()
                    return

        dpg.set_value("tt_heading", "Target manipulation:")
        dpg.set_value("tt_text", "- Drag target to reposition\n- Shift + left click to add target\n- Right click to open target editing\n- Middle click to add geo point\n- Right click to delete geo point")
        dpg.configure_item("tt_text", color=(255, 255, 255, 255))

        # Remove crosshair cursor if present
        self.plot.clearOverlays(["temp_cursor"], delete_textures=False)

    def showCrosshair(self):
        """Shows crosshair cursor on plot."""

        # NOTE: Had to switch from showing icon as image at abs position to drawing it on plot because global coordinates were not consistent between MacOS and Windows.

        plot_mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # Calculate cursor scale
        cursor_size = 20 # px
        plot_size_px = dpg.get_item_rect_size(self.plot.plot) # px
        x_axis_range = dpg.get_axis_limits(self.plot.x_axis) # microns
        scale = (max(x_axis_range) - min(x_axis_range)) / plot_size_px[0] * cursor_size / 2

        # Create image at cursor position
        if not self.plot.getOverlaysByKeyword("temp_cursor"):
            self.plot.addOverlay(gui.makeIconShift(), bounds=((plot_mouse_coords[0] - scale, plot_mouse_coords[0] + scale), (plot_mouse_coords[1] - scale, plot_mouse_coords[1] + scale)), label="temp_cursor")
        else:
            # Update cursor position
            dpg.configure_item(self.plot.overlays[utils.findIndex(self.plot.overlays, "label", "temp_cursor")]["plot"], bounds_min=plot_mouse_coords - scale, bounds_max=plot_mouse_coords + scale)

    def showTargetMenu(self, img_coords):
        """Configures target menu when right clicking on point."""

        # Get camera dims
        rec_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.rec_pix_size / self.loaded_map.pix_size).astype(int) 

        # Check if coords are too close to existing point
        closest_point_id, in_range = self.targets.getClosestPoint(img_coords, rec_dims / 2)

        if in_range:
            dpg.set_value(self.menu_tgt.all_elements["heading_txt"], "Target information:")

            info, color = self.assembleTargetInfo(closest_point_id)

            dpg.set_value(self.menu_tgt.all_elements["info_txt"], info)
            dpg.configure_item(self.menu_tgt.all_elements["info_txt"], color=color)

            # Configure buttons
            dpg.configure_item(self.menu_tgt.all_elements["btn_del"], user_data=closest_point_id)
            if closest_point_id[1] > 0:
                # Only show option when target is not a tracking target
                dpg.configure_item(self.menu_tgt.all_elements["btn_trk"], user_data=closest_point_id)
                self.menu_tgt.showElements(["btn_trk"])
            else:
                self.menu_tgt.hideElements(["btn_trk"])

            # Configure area selection
            if len(self.targets.areas) > 1:
                dpg.configure_item(self.menu_tgt.all_elements["area"], items=[area + 1 for area in range(len(self.targets.areas))], default_value=closest_point_id[0] + 1, user_data=closest_point_id)
                self.menu_tgt.unlockRows(["area"])
            else:
                self.menu_tgt.lockRows(["area"])

            # Configure optimization button (only show when segmentation was loaded)
            if self.segmentation.valid:
                dpg.configure_item(self.menu_tgt.all_elements["btn_opt"], user_data=closest_point_id)
                self.menu_tgt.showElements(["btn_opt"])
            else:
                self.menu_tgt.hideElements(["btn_opt"])

            # Unlock rows
            self.menu_tgt.unlockRows(["info", "buttons"])

            # Get UI mouse coords for window placement
            mouse_coords_global = dpg.get_mouse_pos(local=False)
            dpg.set_item_pos("win_tgt", mouse_coords_global)
            dpg.show_item("win_tgt")

            return True
        return False

    def assembleTargetInfo(self, closest_point_id):
        """Creates string with target info for display."""

        # Set default color
        color = (255, 255, 255, 255)

        # Show area information if more than one target area
        area = f"Area: {closest_point_id[0] + 1}\n" if len(self.targets.areas) > 1 else ""

        # Get score and distance from tracking target
        final_score, dist = self.targets.areas[closest_point_id[0]].getPointInfo(closest_point_id[1])
        score_text = round(final_score, 2) if final_score < 100 else "manual"
        dist *= self.loaded_map.pix_size / 1000

        # Get class scores if present (and if menu is in advanced mode)
        class_scores = ""
        if self.menu_left.show_advanced and self.targets.areas[closest_point_id[0]].class_scores:
            for cat, score in self.targets.areas[closest_point_id[0]].class_scores.items():
                if score[closest_point_id[1]] > 0:
                    # Format class scores to show percentage of final score
                    class_scores += f"- {cat}:{' ' * (max([len(cat_name) for cat_name in self.tgt_params.target_list]) + 1 - len(cat))}{round(score[closest_point_id[1]], 2)}\n"
        
        # Format and set info text
        if dist >= self.mic_params.IS_limit - 0.5:
            warning_is = "[!]"
            color = gui.COLORS["error"]
        else:
            warning_is = ""

        # Show geo points in range if more than one target area
        geo_in_range = ""
        if len(self.targets.areas) > 1:
            num_geo_in_range = len(self.targets.areas[closest_point_id[0]].getGeoInRange(self.mic_params.IS_limit * 1000 / self.loaded_map.pix_size))
            geo_in_range = f"\nGeo points: {num_geo_in_range}"
            if num_geo_in_range < 3:
                geo_in_range += "[!]"
                color = gui.COLORS["geo"] if color != gui.COLORS["error"] else gui.COLORS["error"]

        return f"{area}Target: {closest_point_id[1] + 1}\nScore: {score_text}\n{class_scores}IS: {round(dist, 2)}{warning_is} µm{geo_in_range}", color

    def showTargetAreaButtons(self):
        """Determines which target area actions are available."""

        # Split areas button
        if len(self.targets) > 1:
            self.menu_right.showElements(["butsplit"])
        else:
            self.menu_right.hideElements(["butsplit"])

        # Redistribute targets and merge buttons
        if len(self.targets.areas) > 1:
            self.menu_right.showElements(["butdist", "butmerge"])
        else:
            self.menu_right.hideElements(["butdist", "butmerge"])

    def openClassSelection(self, sender, app_data, user_data):
        """Opens selection window for classes."""

        # Get current text input
        selected_cats = [value.strip() for value in dpg.get_value(app_data[1]).split(",")]

        # Adjust checkboxes accordingly
        for cat in self.model.categories:
            if cat in selected_cats:
                dpg.set_value(self.menu_sel.all_elements[cat], True)
            else:
                dpg.set_value(self.menu_sel.all_elements[cat], False)
            # Add text field id from app_data
            dpg.configure_item(self.menu_sel.all_elements[cat], user_data=[self.menu_sel, app_data[1]])

        # Get UI mouse coords for window placement
        mouse_coords_global = dpg.get_mouse_pos(local=False)
        dpg.set_item_pos("win_sel", mouse_coords_global)
        dpg.show_item("win_sel")

    def runTargetSelection(self):
        """Calls the target selection procedure."""

        # Check for existing point files and delete them
        point_files = sorted(self.cur_dir.glob(self.map_name + "_points*.json"))
        for file in point_files:
            # Delete file
            file.unlink()

        # Delete previous targets
        self.clearTargets()

        # Update tgt params from GUI settings
        self.tgt_params.target_list = [cat.strip() for cat in dpg.get_value(self.menu_right.all_elements["target_list"]).split(",")]
        self.tgt_params.penalty_list = [cat.strip() for cat in dpg.get_value(self.menu_right.all_elements["avoid_list"]).split(",")]
        self.tgt_params.parseLists(self.model)
        self.tgt_params.checkLists(self.model)

        self.tgt_params.sparse = dpg.get_value(self.menu_right.all_elements["sparse_targets"])
        self.tgt_params.edge = dpg.get_value(self.menu_right.all_elements["target_edge"])
        self.tgt_params.penalty = dpg.get_value(self.menu_right.all_elements["penalty_weight"])
        self.tgt_params.threshold = dpg.get_value(self.menu_right.all_elements["target_score_threshold"])
        self.tgt_params.max_tilt = dpg.get_value(self.menu_right.all_elements["max_tilt"])
        self.tgt_params.extra_track = dpg.get_value(self.menu_right.all_elements["extra_tracking"])

        self.mic_params.IS_limit = dpg.get_value(self.menu_right.all_elements["IS_limit"])

        # Load overlay
        self.status.update("Loading overlay...", box=True)
        self.loadOverlay(cats=self.tgt_params.target_list)

        # Update GUI
        self.menu_left.hide()
        self.menu_right.hide()

        # Run target selection
        self.status.update("Selecting targets...\n(See console output for details!)", box=True)
        space_ext.runTargetSelection(self.cur_dir, self.map_name, self.tgt_params, self.mic_params, self.model, save_final_plot=False)

        # Plot targets
        self.targets = Targets(map_dir=self.cur_dir, map_name=self.map_name, map_dims=self.loaded_map.img.shape, tgt_params=self.tgt_params, map_pix_size=self.loaded_map.pix_size)
        self.targets.loadAreas()
        self.showTargets()
        self.showTargetAreaButtons()

        # Get class specific scores
        self.getTargetClassScores()

        # Update GUI
        self.menu_left.show()
        self.menu_right.hideElements(["butsave"])   # Only activate save button when targets were changed
        if not self.inspected:
            self.menu_right.show()
        self.status.update()

    def suggestHolePattern(self):
        """Suggests target pattern based on grid vectors."""

        # Show detected grid pattern
        if self.loaded_map and "grid_vectors" in self.loaded_map.meta_data.keys():
            if not self.targets:
                dpg.bind_item_theme(self.menu_icon.all_elements["butholes"], None)
                self.hole_mode = False
                InfoBoxManager.push(InfoBox("ERROR", "Please select one target to center grid on!", loading=False))
                return
            
            # Reset existing suggestions
            if len(self.targets.suggestions):
                self.plot.clearOverlays(labels=[f"sug_{p}" for p in range(len(self.targets.suggestions))], delete_textures=False)
                self.targets.suggestions = []

            # Load grid vectors
            grid_vectors = np.array(self.loaded_map.meta_data["grid_vectors"]) / 1000 # microns
            grid_vectors[:, 1] *= -1 # Invert the y axis

            # Calculate the number of points needed to cover the area
            num_points_x = int(np.ceil(self.loaded_map.dims_microns[0] / np.linalg.norm(grid_vectors[0])))
            num_points_y = int(np.ceil(self.loaded_map.dims_microns[1] / np.linalg.norm(grid_vectors[1])))

            # Generate grid of points
            i, j = np.meshgrid(np.arange(num_points_x), np.arange(num_points_y), indexing='ij')
            points = (grid_vectors[0] * i[..., np.newaxis] + grid_vectors[1] * j[..., np.newaxis]).reshape(-1, 2)

            # Shift points to center the grid on the map
            points += self.loaded_map.dims_microns / 2 - np.mean(points, axis=0)

            # Find offset
            center_coords = self.loaded_map.px2microns(self.targets.areas[0].center)
            # Find the closest point
            closest_point = points[np.argmin(np.linalg.norm(points - center_coords, axis=1))]
            # Calculate the offset
            offset = center_coords - closest_point
            # Shift points
            points += offset

            # Remove points outside of map
            points = points[np.all((points >= 0.05 * self.loaded_map.dims_microns) & 
                                   (points < 0.95 * self.loaded_map.dims_microns), axis=1)]

            # Remove points in same location as existing targets
            target_points, _ = self.targets.getAllPoints()
            for target in target_points:
                points = np.array([point for point in points if np.linalg.norm(point - self.loaded_map.px2microns(target)) > np.linalg.norm(grid_vectors[0]) / 2])

            # Remove points outside of polygons if polygon mode is active
            if self.polygon_mode:
                # Check if point is in any of the closed polygons
                points = np.array([point for point in points 
                                   if any(isinstance(polygon, PlotPolygon) 
                                          and not polygon.open 
                                          and polygon.within(point) 
                                          for polygon in self.plot.boxes)
                                    ])

            #self.plot.addSeries(points[:, 0], points[:, 1], label="Suggested pattern")
            self.targets.suggestions = [self.loaded_map.microns2px(point) for point in points]

            self.showTargets()

    def toggleHolePattern(self, sender=None, app_data=None, user_data=None, force_off=False):
        """Toggles showing hole target pattern on map."""

        if not self.loaded_map:
            return

        if self.hole_mode or force_off:
            log(f"DEBUG: Toggling hole mode OFF.")
            if self.targets is not None and len(self.targets.suggestions):
                self.plot.clearOverlays(labels=[f"sug_{p}" for p in range(len(self.targets.suggestions))], delete_textures=False)
                self.targets.suggestions = []
                self.menu_right.lockRows(["suggestions"])
            self.hole_mode = False
            dpg.bind_item_theme(self.menu_icon.all_elements["butholes"], None)
        else:
            log(f"DEBUG: Toggling hole mode ON.")
            if self.dense_mode:
                self.toggleDensePattern()
            dpg.bind_item_theme(self.menu_icon.all_elements["butholes"], "active_btn_theme")
            self.hole_mode = True
            self.suggestHolePattern()

    def suggestDensePattern(self):
        """Suggests dense target pattern based on beam size."""

        max_tilt = dpg.get_value(self.menu_right.all_elements["max_tilt"])
        beam_diameter = float(self.mic_params.rec_beam_diameter)
        beam_margin = dpg.get_value(self.menu_right.all_elements["beam_margin"]) / 100 * beam_diameter
        cam_dims = self.model.cam_dims * self.loaded_map.pix_size / 1000

        # Flip cam_dims if tilt axis angle rotated too much
        ta_rot = np.radians(self.mic_params.view_ta_rotation)
        if np.cos(ta_rot) / np.sin(ta_rot) > cam_dims[0] / cam_dims[1]:
            log(f"DEBUG: Flipping camera dimensions due to tilt axis rotation.")
            cam_dims = np.flip(cam_dims)

        # Calculate grid vectors for hexagonal arrangement based on beam diameter
        rotM_pattern = np.array([[np.cos(np.radians(60)), -np.sin(np.radians(60))], 
                                [np.sin(np.radians(60)), np.cos(np.radians(60))]])

        x1 = np.array([beam_diameter, 0])
        x2 = rotM_pattern @ x1
        x2[1] = beam_diameter / np.cos(np.radians(max_tilt)) # Use full beam diameter for easier adjustment to camera dimensions

        # Adjust to allow overlap up to camera dimensions (5 % beam diameter margin)
        x1[0] -= (x1[0] - cam_dims[1]) / 2 - beam_margin
        x2[0] -= (x1[0] - cam_dims[1]) / 2 - beam_margin
        x2[1] -= (x2[1] - cam_dims[0]) / 2 - beam_margin

        # Apply view rotation matrix (needs to be rotated by additional 90 degrees TODO: check if correct also for other mic params?)
        view_rotM = self.mic_params.view_rotM
        x1 = np.array([[0, -1], [1, 0]]) @ view_rotM @ x1
        x2 = np.array([[0, -1], [1, 0]]) @ view_rotM @ x2

        if abs(self.mic_params.view_ta_rotation) > 45:
            x1, x2 = x2, x1

        # Calculate the number of rows and columns
        dims_microns = self.loaded_map.dims_microns
        max_rows = abs(int(dims_microns[0] / (x1[0] + x2[0]) * 2))
        max_cols = abs(int(dims_microns[1] / (x1[1] + x2[1]) * 2))

        # Generate points
        i, j = np.meshgrid(np.arange(max_rows), np.arange(max_cols), indexing='ij')
        points = (i[..., np.newaxis] * x1 + j[..., np.newaxis] * x2 - 
                max_rows // 2 * x1 - max_cols // 2 * x2 + dims_microns / 2).reshape(-1, 2)

        # Shift to align with tracking target
        if self.targets:
            # Find offset
            center_coords = self.loaded_map.px2microns(self.targets.areas[0].center)
            # Find the closest point
            closest_point = points[np.argmin(np.linalg.norm(points - center_coords, axis=1))]
            # Calculate the offset
            offset = center_coords - closest_point
            # Shift points
            points += offset
        
        # Remove points outside of map
        points = points[np.all((points >= 0.05 * self.loaded_map.dims_microns) & 
                               (points < 0.95 * self.loaded_map.dims_microns), axis=1)]
        
        # Remove points in same location as existing targets
        target_points, _ = self.targets.getAllPoints()
        for target in target_points:
            points = np.array([point for point in points if np.linalg.norm(point - self.loaded_map.px2microns(target)) > min(cam_dims)])

        # Remove points outside of polygons if polygon mode is active
        if self.polygon_mode:
            # Check if point is in any of the closed polygons
            points = np.array([point for point in points 
                                if any(isinstance(polygon, PlotPolygon) 
                                        and not polygon.open 
                                        and polygon.within(point) 
                                        for polygon in self.plot.boxes)
                                ])

        self.targets.suggestions = [self.loaded_map.microns2px(point) for point in points]

        self.showTargets()

    def toggleDensePattern(self, sender=None, app_data=None, user_data=None, force_off=False):
        """Toggles showing dense target pattern on map."""

        if not self.loaded_map:
            return
        
        if self.dense_mode or force_off:
            log(f"DEBUG: Toggling dense mode OFF.")
            if self.targets is not None and len(self.targets.suggestions):
                self.plot.clearOverlays(labels=[f"sug_{p}" for p in range(len(self.targets.suggestions))], delete_textures=False)
                self.targets.suggestions = []
                self.menu_right.lockRows(["suggestions"])
            self.dense_mode = False
            dpg.bind_item_theme(self.menu_icon.all_elements["butdense"], None)
        else:
            log(f"DEBUG: Toggling dense mode ON.")
            if self.hole_mode:
                self.toggleHolePattern()
            dpg.bind_item_theme(self.menu_icon.all_elements["butdense"], "active_btn_theme")
            self.dense_mode = True
            self.suggestDensePattern()

    def addSuggestions(self):
        """Adds suggested targets to target list."""                
        
        for point in self.targets.suggestions:
            self.targets.addTarget(point)

        # Clear suggestions
        self.toggleHolePattern(force_off=True)
        self.toggleDensePattern(force_off=True)
        self.showTargets()
        self.showTargetAreaButtons()
        self.menu_right.showElements(["butsave"])

    def togglePolygonMode(self):
        """Toggles polygon mode for target selection."""

        if not self.loaded_map:
            return
        
        # Toggle mode
        self.polygon_mode = not self.polygon_mode
        dpg.bind_item_theme(self.menu_icon.all_elements["butpolygon"], "active_btn_theme" if self.polygon_mode else None)

        # If turning off
        if not self.polygon_mode:
            # Close any open polygons
            for polygon in self.plot.boxes:
                if isinstance(polygon, PlotPolygon):
                    if polygon.open:
                        polygon.close()
                    #polygon.drawLabel("closed")

            # Reset suggestions by reloading them
            if self.hole_mode:
                self.suggestHolePattern()
            elif self.dense_mode:
                self.suggestDensePattern()

        # If turning on
        else:
            # Toggle suggestions off
            self.toggleHolePattern(force_off=True)
            self.toggleDensePattern(force_off=True)

    def updateMaxTilt(self):
        new_max_tilt = dpg.get_value(self.menu_right.all_elements["max_tilt"])

        if abs(new_max_tilt - self.tgt_params.max_tilt) > 5:
            self.tgt_params.max_tilt = new_max_tilt

            # Reset target overlays
            for overlay in self.target_overlays:
                if dpg.does_item_exist(overlay):
                    dpg.delete_item(overlay)
            self.target_overlays = {}

            if self.dense_mode:
                # Update target suggestions
                self.suggestDensePattern()
            else:
                # Just re-plot targets
                self.showTargets()

    def updateBeamMargin(self):
        """Updates beam margin from slider."""

        if self.dense_mode:
            # Update target suggestions
            self.suggestDensePattern()

    def updateISLimit(self):
        """Updates IS limit from slider."""

        if self.mic_params.IS_limit != dpg.get_value(self.menu_right.all_elements["IS_limit"]):
            self.mic_params.IS_limit = dpg.get_value(self.menu_right.all_elements["IS_limit"])

            # Update beyond IS limit warnings
            self.showTargets()        

    def getTargetClassScores(self):
        """Gets class scores for cats in target_list."""

        for cat in self.tgt_params.target_list:
            for area in self.targets.areas:
                area.getClassScores(cat, self.segmentation.getMask([cat], unbinned=True))

    def changeArea(self, sender, app_data, user_data):
        """Changes target area of single target."""

        new_area = int(dpg.get_value(sender)) - 1
        self.targets.movePointToArea(user_data, new_area_id=new_area)

        dpg.hide_item("win_tgt")
        self.showTargets()
        self.showTargetAreaButtons()
        # Enable save button
        self.menu_right.showElements(["butsave"])
        
    def makeTrackDialogue(self, sender, app_data, user_data):
        """Creates dialogue for make tracking target."""

        dpg.hide_item("win_tgt")
        dpg.split_frame()
        InfoBoxManager.push(InfoBox("Make tracking target", "Do you want to make this target the tracking target of the current area or create a new acquisition area?", loading=False, callback=self.makeTrack, options=["Current area", "New area", "Cancel"], options_data=[[user_data, "old"], [user_data, "new"], False]))

    def makeTrack(self, sender, app_data, user_data):
        """Makes selected target a tracking target of current or new area."""

        # Check for info box input
        if user_data and dpg.does_item_exist(user_data[0]):
            #dpg.delete_item(user_data[0])
            #dpg.split_frame()
            InfoBoxManager.pop()
        
        if not user_data[1]:
            return
        
        area_id, point_id = user_data[1][0]
        new_area = True if user_data[1][1] == "new" else False

        if new_area:
            # Get coords
            coords = self.targets.areas[area_id].points[point_id]

            # Remove point from old area
            self.targets.areas[area_id].removePoint(point_id)
            
            # If no points left in area, remove area
            if len(self.targets.areas[area_id]) == 0:
                self.targets.areas.pop(area_id)
            
            # Create new area with point
            self.targets.addTarget(coords, new_area=True)
            log("NOTE: Created new acquisition area!")
        else:
            # Move point to beginning of area
            self.targets.areas[area_id].makeTrack(point_id)
            log("NOTE: Made new tracking target!")

        self.showTargets()
        self.showTargetAreaButtons()
        # Enable save button
        self.menu_right.showElements(["butsave"])

    def optimizeTarget(self, sender, app_data, user_data):
        """Translates target according to loaded target mask."""
        
        # Check if segmentation is loaded
        if not self.overlay:
            log("ERROR: Please load class overlay first to optimize positioning of target.")
            return

        if user_data:
            area_id, point_id = user_data
            self.targets.areas[area_id].optimizeLocally(point_id, self.segmentation.getMask(self.overlay, unbinned=True))

            dpg.hide_item("win_tgt")
            self.showTargets()
            # Enable save button
            self.menu_right.showElements(["butsave"])

    def removeTarget(self, sender, app_data, user_data):
        """Deletes a single target."""

        if user_data:
            area_id, point_id = user_data
            self.targets.areas[area_id].removePoint(point_id)
            if len(self.targets.areas[area_id]) == 0:
                self.targets.areas.pop(area_id)
            dpg.hide_item("win_tgt")

            if self.targets:
                self.showTargets()
            else:
                self.clearTargets()
            self.showTargetAreaButtons()
            # Enable save button
            self.menu_right.showElements(["butsave"])

    def removeGeoPoint(self, img_coords):
        """Deletes closest geo point from all target areas. (Assumes geo_points are kept synced for all target areas.)"""

        # Get camera dims
        focus_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.focus_pix_size / self.loaded_map.pix_size).astype(int) 

        # If no target it range check for geo points
        closest_point_id, in_range = self.targets.getClosestGeoPoint(img_coords, focus_dims / 2)
        if in_range:
            # Remove geo point from all areas
            self.targets.removeGeoPoint(closest_point_id)
            self.showTargets()
            self.menu_right.showElements(["butsave"])

    def removePolygon(self, sender=None, app_data=None, user_data=None, img_coords=None):
        """Deletes closest polygon from plot."""

        # Find polygon and open confirmation box on first call (no user_data yet)
        if not user_data:
            plot_coords = self.loaded_map.px2microns(img_coords)
            inside_polygons = []
            for p, polygon in enumerate(self.plot.boxes):
                if isinstance(polygon, PlotPolygon) and polygon.within(plot_coords):
                    inside_polygons.append((polygon.area, p))
            # Find smallest area polygon
            if inside_polygons: 
                _, p = sorted(inside_polygons, key=lambda x: x[0])[0]
                self.plot.boxes[p].updateColor(gui.COLORS["error"])
                InfoBoxManager.push(InfoBox("Delete polygon", "Do you want to delete this polygon?", loading=False, callback=self.removePolygon, options=["Delete", "Cancel"], options_data=[[True, p], [False, p]]))
            return
            
        if user_data and dpg.does_item_exist(user_data[0]):
            #dpg.delete_item(user_data[0])
            #dpg.split_frame()
            InfoBoxManager.pop()

            # Get info from user_data
            remove, p = user_data[1]
            
            # Cancel
            if not remove:
                # Reset color
                self.plot.boxes[p].updateColor(gui.COLORS["geo"])
                return
            
        # Delete polygon
        self.plot.boxes[p].remove()
        self.plot.boxes.pop(p)

    def mergeAreas(self):
        """Merge all target areas."""

        self.targets.mergeAreas()

        self.showTargets()
        self.showTargetAreaButtons()
        # Enable save button
        self.menu_right.showElements(["butsave"])

    def splitAreas(self):
        """Splits targets into target areas using k-means clustering."""

        # Number of areas is one more than currently, unless number of areas is already the same as number of targets
        area_num = min(len(self.targets.areas) + 1, len(self.targets))
        self.targets.splitArea(area_num=area_num)

        self.showTargets()
        self.showTargetAreaButtons()
        # Enable save button
        self.menu_right.showElements(["butsave"])

    def splitTargets(self):
        """Splits all targets among current tracking targets."""

        self.targets.splitAreaManual()

        self.showTargets()
        self.showTargetAreaButtons()
        # Enable save button
        self.menu_right.showElements(["butsave"])      

    def clearTargets(self, *args, plot_only=False):
        """Deletes all targets (or just clear from plot.)"""

        # Clear targets from plot
        self.plot.clearOverlays(delete_textures=not plot_only)
        self.plot.clearDragPoints()
        self.plot.clearSeries(skip_labels=["tilt_axis"])#labels=self.plot.getSeriesByKeyword("geo") + self.plot.getSeriesByKeyword("tgt") + [""])
        if not plot_only:
            # Release target overlay textures
            self.target_overlays = {}
            # Reinitialize targets object
            self.targets = Targets(map_dir=self.cur_dir, map_name=self.map_name, map_dims=self.loaded_map.img.shape, tgt_params=self.tgt_params, map_pix_size=self.loaded_map.pix_size)
            log("NOTE: Deleted targets!")
            self.showTargetAreaButtons()

            # Reset suggestion buttons
            self.toggleHolePattern(force_off=True)
            self.toggleDensePattern(force_off=True)

            # Enable save button
            self.menu_right.showElements(["butsave"])

    def clearGeoPoints(self):
        """Deletes all geo points."""

        # Clear all items with "geo" labels (needs empty list entry, otherwise all items will be cleared in case of no hits)
        self.plot.clearOverlays(labels=self.plot.getOverlaysByKeyword("geo") + [""], delete_textures=False)
        self.plot.clearDragPoints(labels=self.plot.getDragPointsByKeyword("geo") + [""])
        self.plot.clearSeries(labels=self.plot.getSeriesByKeyword("geo") + [""])

        self.targets.resetGeo()
        log("NOTE: Deleted geo points!")

        # Enable save button
        self.menu_right.showElements(["butsave"])

    def saveTargets(self):
        """Exports targets as json file."""

        # Get acquisition settings
        settings = self.getAcquisitionSettings()

        self.targets.exportTargets(settings)
        self.menu_right.hideElements(["butsave"])
        log("NOTE: Saved targets!")

        # Also check FLM and save
        self.lm_window.saveLM(None, None, None)

    def markInspected(self, sender, app_data, user_data, keep_map=False):
        """Creates inspected.txt file and locks down editing."""

        # Check for geo_points
        if self.targets and len(self.targets.areas[0].points) and not len(self.targets.areas[0].geo_points) and not user_data:
            log(f"WARNING: No geo points were selected!")
            InfoBoxManager.push(InfoBox("WARNING", "No geo points were selected to measure the sample geometry. If you go ahead, the manual input for pretilt and rotation will be used!", loading=False, callback=self.markInspected, options=["Continue", "Cancel"], options_data=[True, False]))
            return
        # Close geo_points confirmation info box
        if user_data and dpg.does_item_exist(user_data[0]):
            #dpg.delete_item(user_data[0])
            #dpg.split_frame()
            InfoBoxManager.pop()
            # Cancel
            if not user_data[1]:
                return

        # Save targets if unsaved changes
        if dpg.is_item_shown(self.menu_right.all_elements["butsave"]):
            self.saveTargets()

        # Create empty inspected file
        (self.cur_dir / (self.map_name + "_inspected.txt")).touch()

        self.inspected = True

        # Update GUI
        self.menu_right.hide()
        dpg.show_item("inspected")

        # Update map window to reflected inspected status
        self.map_window.update(self.map_name, self.map_list, self.map_list_tgtnum)

        log("NOTE: Targets were marked as inspected!")

        # If there are not inspected maps left, load next map
        if not keep_map and not self.checkAllInspected():
            log("\n\nLoading next map...")
            self.selectMap(next_map=True)

    def markAllInspected(self, sender, app_data, user_data):
        """Marks all maps as inspected!"""

        for map_name in self.map_list:
            (self.cur_dir / (map_name + "_inspected.txt")).touch(exist_ok=True)

        self.inspected = True

        self.checkAllInspected()

    def checkAllInspected(self):
        """Checks if all maps have been marked inspected and prompts GUI close."""

        # Check if all maps were inspected
        inspected = []
        for map_name in self.map_list:
            inspected.append((self.cur_dir / (map_name + "_inspected.txt")).exists())
        if all(inspected):
            log("NOTE: All MM maps were inspected!")
            if self.auto_close:
                # Confirm closing
                InfoBoxManager.push(InfoBox("FINISHED?", "All available MM maps were inspected. Are MM maps still being acquired? If not, you can close this GUI.", loading=False, callback=self.closeAllInspected, options=["Wait", "Close"], options_data=[False, True]))
                return True
        else:
            return False

    def closeAllInspected(self, sender, app_data, user_data):
        """Closes GUI if all MM maps were inspected and user confirmed."""

        # Close GUI
        if user_data and user_data[1]:
            dpg.stop_dearpygui()
        else:
            # Close confirmation info box
            if dpg.does_item_exist(user_data[0]):
                #dpg.delete_item(user_data[0])
                #dpg.split_frame()
                InfoBoxManager.pop()

    def getAcquisitionSettings(self):
        """Gets acquisition settings from input fields."""

        settings = {}
        # Only save settings if override checkbox was checked
        if dpg.get_value(self.menu_right.all_elements["acq_save"]):
            settings["startTilt"] = dpg.get_value(self.menu_right.all_elements["acq_start_tilt"])
            settings["minTilt"] = dpg.get_value(self.menu_right.all_elements["acq_min_tilt"])
            settings["maxTilt"] = dpg.get_value(self.menu_right.all_elements["acq_max_tilt"])
            settings["step"] = dpg.get_value(self.menu_right.all_elements["acq_step_tilt"])
            settings["pretilt"] = dpg.get_value(self.menu_right.all_elements["acq_pretilt"])
            settings["rotation"] = dpg.get_value(self.menu_right.all_elements["acq_rotation"])

        settings["autoStartTilt"] = dpg.get_value(self.menu_right.all_elements["acq_autostart"])

        return settings
    
    def setAcquisitionSettings(self):
        """Sets acquisition settings in case they were loaded from file."""

        if self.targets.settings:
            if "startTilt" in self.targets.settings.keys():
                dpg.set_value(self.menu_right.all_elements["acq_start_tilt"], self.targets.settings["startTilt"])
                dpg.set_value(self.menu_right.all_elements["acq_min_tilt"], self.targets.settings["minTilt"])
                dpg.set_value(self.menu_right.all_elements["acq_max_tilt"], self.targets.settings["maxTilt"])
                dpg.set_value(self.menu_right.all_elements["acq_step_tilt"], self.targets.settings["step"])
            if "pretilt" in self.targets.settings.keys():
                dpg.set_value(self.menu_right.all_elements["acq_pretilt"], self.targets.settings["pretilt"])
                dpg.set_value(self.menu_right.all_elements["acq_rotation"], self.targets.settings["rotation"])
            if "autoStartTilt" in self.targets.settings.keys():
                dpg.set_value(self.menu_right.all_elements["acq_autostart"], self.targets.settings["autoStartTilt"])

    def checkRunConditions(self):
        """Checks if mic and tgt params could be loaded."""

        if self.mic_params is None or self.model is None:
            if not self.checked_run_conditions:
                log("ERROR: No microscope parameter file found. Please launch the GUI in a SPACE_maps folder!")
                # Change exit callback to avoid askForSave method
                dpg.set_exit_callback(dpg.stop_dearpygui)
                InfoBoxManager.push(InfoBox("ERROR", "No microscope parameter file found.\nPlease launch the GUI in a SPACE_maps folder!", loading=False, callback=dpg.stop_dearpygui))
            self.checked_run_conditions = True
            return

        # Create popup if no maps have been found yet
        if not self.map_list:
            if not self.checked_map_files:
                InfoBoxManager.push(InfoBox("NOTE", "No MM map segmentations are available yet.\n\nReload after a few minutes to check for MM map segmentations.", loading=False, callback=self.manualMapListReload, options=["Reload", "Close"], options_data=[True, False]))
            self.checked_map_files = True
            return

        # Reset mouse_move_handler
        if dpg.does_item_exist("check_run_condition"):
            dpg.delete_item("check_run_condition")

        # Set trigger help mouse_move_handler
        with dpg.handler_registry(tag="trigger_help"):
            dpg.add_mouse_move_handler(callback=self.triggerShowHelp)

    def manualMapListReload(self, sender, app_data, user_data):
        """Forces map list reload when prompted from info box."""

        # Close info box
        if dpg.does_item_exist(user_data[0]):
            #dpg.delete_item(user_data[0])
            #dpg.split_frame()
            InfoBoxManager.pop()

        if user_data[1]:
            self.updateMapList()
            # Reopen prompt if still no maps found
            if not self.map_list:
                self.checked_map_files = False
        else:
            dpg.stop_dearpygui()

    def makeTargetOverlay(self):
        """Generates target overlay textures."""

        # Return early if target overlays already exist
        if self.target_overlays:
            return

        # TGT
        # Get camera dims
        rec_beam_diameter_px = self.mic_params.rec_beam_diameter * 1000 / self.loaded_map.pix_size
        rec_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.rec_pix_size / self.loaded_map.pix_size).astype(int) 

        # Create canvas with size of stretched beam diameter
        tgt_overlay = np.zeros([int(rec_beam_diameter_px), int(rec_beam_diameter_px / np.cos(np.radians(self.tgt_params.max_tilt)))])
        canvas = Image.fromarray(tgt_overlay).convert('RGB')
        draw = ImageDraw.Draw(canvas)

        # Draw beam
        draw.ellipse((0, 0, tgt_overlay.shape[1] - 1, tgt_overlay.shape[0] - 1), outline="#ffd700", width=20)
        #draw.ellipse(((tgt_overlay.shape[1] - tgt_overlay.shape[0]) / 2, 0, (tgt_overlay.shape[1] + tgt_overlay.shape[0]) / 2 - 1, tgt_overlay.shape[0] - 1), outline="#ffd700", width=20)

        # Rotate tilt axis
        canvas = canvas.rotate(-self.mic_params.view_ta_rotation, expand=True)

        # Draw camera outline
        rect = ((canvas.width - rec_dims[1]) // 2, (canvas.height - rec_dims[0]) // 2, (canvas.width + rec_dims[1]) // 2, (canvas.height + rec_dims[0]) // 2)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(rect, outline="#578abf", width=20)

        # Convert to array
        tgt_overlay = np.array(canvas).astype(float) / 255

        # Draw camera outline for tracking target
        draw.rectangle(rect, outline="#c92b27", width=20)

        # Convert to array
        trk_overlay = np.array(canvas).astype(float) / 255

        # Draw in grey scale for target suggestions
        sug_overlay = np.zeros((canvas.height, canvas.width, 3))
        mask = np.array(canvas.convert('L')) > 0
        sug_overlay[mask] = [1, 1, 1]

        # GEO
        # Get camera dims
        focus_beam_diameter_px = self.mic_params.focus_beam_diameter * 1000 / self.loaded_map.pix_size
        focus_dims = (self.mic_params.cam_dims[[1, 0]] * self.mic_params.focus_pix_size / self.loaded_map.pix_size).astype(int) 

        # Create canvas for geo with non-stretched beam diameter
        geo_overlay = np.zeros([int(focus_beam_diameter_px), int(focus_beam_diameter_px)])
        canvas = Image.fromarray(geo_overlay).convert('RGB')
        draw = ImageDraw.Draw(canvas)

        # Draw beam and camera dims
        draw.ellipse((0, 0, geo_overlay.shape[1] - 1, geo_overlay.shape[0] - 1), outline="#ee8844", width=20)
        rect = ((canvas.width - focus_dims[1]) // 2, (canvas.height - focus_dims[0]) // 2, (canvas.width + focus_dims[1]) // 2, (canvas.height + focus_dims[0]) // 2)
        draw.rectangle(rect, outline="#ee8844", width=20)

        # Convert to array
        geo_overlay = np.array(canvas).astype(float) / 255

        # Make textures
        self.target_overlays["tgtdims"] = np.array(tgt_overlay.shape)[:2]
        alpha = np.zeros(tgt_overlay.shape[:2])
        alpha[np.sum(tgt_overlay, axis=-1) > 0] = 1
        tgt_overlay_image = np.ravel(np.dstack([tgt_overlay, alpha]))
        trk_overlay_image = np.ravel(np.dstack([trk_overlay, alpha]))
        sug_overlay_image = np.ravel(np.dstack([sug_overlay, alpha * 0.25]))

        self.target_overlays["geodims"] = np.array(geo_overlay.shape)[:2]
        alpha = np.zeros(geo_overlay.shape[:2])
        alpha[np.sum(geo_overlay, axis=-1) > 0] = 1
        geo_overlay_image = np.ravel(np.dstack([geo_overlay, alpha]))

        with dpg.texture_registry():
            self.target_overlays["target"] = dpg.add_static_texture(width=int(self.target_overlays["tgtdims"][1]), height=int(self.target_overlays["tgtdims"][0]), default_value=tgt_overlay_image)
            self.target_overlays["track"] = dpg.add_static_texture(width=int(self.target_overlays["tgtdims"][1]), height=int(self.target_overlays["tgtdims"][0]), default_value=trk_overlay_image)
            self.target_overlays["geo"] = dpg.add_static_texture(width=int(self.target_overlays["geodims"][1]), height=int(self.target_overlays["geodims"][0]), default_value=geo_overlay_image)
            self.target_overlays["suggestion"] = dpg.add_static_texture(width=int(self.target_overlays["tgtdims"][1]), height=int(self.target_overlays["tgtdims"][0]), default_value=sug_overlay_image)

    def savePlot(self, sender, app_data, user_data):
        """Gets frame buffer to save plot to file. (Does not work on MacOS.)"""

        # Check if map was opened
        if not self.map_name:
            log(f"ERROR: Please load a map before saving a snapshot!")
            InfoBoxManager.push(InfoBox("WARNING", "Please load a map before saving a snapshot!"))
            return
        
        # Get name for snapshot
        counter = 1
        while (snapshot_file_path := self.cur_dir / f"{self.map_name}_snapshot{counter}.png").exists():
            counter += 1
        
        saveSnapshot(self.plot.plot, snapshot_file_path)

    def askForSave(self):
        """Opens popup if there are unsaved changes."""

        # Check for unsaved changes
        if dpg.is_item_shown(self.menu_right.all_elements["butsave"]):
            InfoBoxManager.push(InfoBox("WARNING", "There are unsaved changes to your targets!", loading=False, callback=self.saveAndClose, options=["Save", "Discard"], options_data=[True, False]))
        else:

            # Check inspected status and give warning when some but not all targets have been inspected
            inspected = []
            for map_name in self.map_list:
                inspected.append((self.cur_dir / (map_name + "_inspected.txt")).exists())
            if any(inspected) and not all(inspected):
                InfoBoxManager.push(InfoBox("WARNING", "Not all targets have been marked as inspected. Are you sure you want to exit?", loading=False, callback=self.confirmClose, options=["Exit", "Cancel"], options_data=[True, False]))
            else:
                dpg.stop_dearpygui()

    def saveAndClose(self, sender, app_data, user_data):
        """Saves targets and closes GUI."""

        # Save targets if user clicked Save
        if user_data[1]:
            self.saveTargets()

        # Change exit callback to avoid infinite loop
        dpg.set_exit_callback(dpg.stop_dearpygui)
        dpg.stop_dearpygui()

    def saveAndContinue(self, sender, app_data, user_data):
        """Saves targets and continues with next map."""

        # Save targets if user clicked Save
        if user_data[1]:
            self.saveTargets()
        else:
            # Discard by hiding save button
            self.menu_right.hideElements(["butsave"])

        # Close info box
        if dpg.does_item_exist(user_data[0]):
            #dpg.delete_item(user_data[0])
            #dpg.split_frame()
            InfoBoxManager.pop()

    def confirmClose(self, sender, app_data, user_data):
        """Closes GUI or keeps it open."""

        if user_data[1]:
            # Change exit callback to avoid infinite loop
            dpg.set_exit_callback(dpg.stop_dearpygui)
            dpg.stop_dearpygui()
        else:
            # Close info box
            if dpg.does_item_exist(user_data[0]):
                #dpg.delete_item(user_data[0])
                #dpg.split_frame()
                InfoBoxManager.pop()

    def toggleAdvanced(self):
        """Shows advanced menu options."""

        self.menu_left.toggleAdvanced()
        self.menu_right.toggleAdvanced()

    def triggerShowHelp(self):
        """Triggers show help once."""

        if dpg.does_item_exist("trigger_help"):
            dpg.delete_item("trigger_help")

        # Show help
        dpg.split_frame()
        self.showHelp()

    def openWGMap(self):
        """Opens WG map GUI."""

        # Find WG map
        map_canditates = sorted(self.cur_dir.glob("**/*_wg.png"), key=lambda x: len(str(x)))
        map_path = str(map_canditates[0]) if map_canditates else ""

        utils.guiProcess("grid", file_path=map_path)

    @staticmethod
    def showHelp():
        """Shows popup with shortcuts."""

        message = "Load an MM map from the list to\ninspect or select targets and geo points.\n\n[Confirm inspection] when you are done!\n\n\n"
        message += "Controls:\n\n"
        message += "Shift + left click      Add new target\n"
        message += "Right click on target   Open menu\n"
        message += "Middle click            Add geo point\n"
        message += "\n\n"
        message += "Keyboard shortcuts:\n\n"
        message += "A     Toggle advanced settings\n"
        message += "H     Show help\n"
        message += "N     Load next map\n"
        message += "D     Toggle dense pattern\n"
        message += "P     Toggle polygon mode\n"
        message += "R     CLEM registration\n"
        message += "T     Toggle CLEM overlay\n"
        message += "Space Show available maps\n"

        InfoBoxManager.push(InfoBox("Help", message, loading=False))

#################################################################################


    def __init__(self, path="", auto_close=False) -> None:
        log(f"\n########################################\nRunning SPACEtomo Target Selection GUI\n########################################\n")

        # Automatically close GUI after map was inspected
        self.auto_close = auto_close

        # Keep list of popup blocking windows
        self.blocking_windows = []

        # Make logo
        self.logo_dims = gui.makeLogo()

        self.configureHandlers()
        self.configureThemes()

        # Create plot object to hold all entities
        self.plot = Plot("plot")

        # Initialise globals
        self.menu_left = None
        self.menu_right = None
        self.status = None

        self.drag_point = None # Keep track of target being dragged
        self.drag_start = None # Keep track of drag start position

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.preloaded_data = None      # Map data to be preloaded

        # Modes
        self.polygon_mode = False   # Mode to draw polygons
        self.brush_mode = False     # TODO: Implement brush mode
        self.hole_mode = False      # Mode to overlay hole pattern as target suggestions
        self.dense_mode = False     # Mode to overlay dense pattern as target suggestions
        
        # Check given path
        if path:
            path = Path(path)
            if not path.is_dir():
                path = path.parent
            self.cur_dir = Path(path)
        else:
            self.cur_dir = Path.cwd()

        # Maps and targets
        self.map_list = []                          # List of segmentation files found in folder
        self.map_name = ""                          # Name of currently loaded map
        self.loaded_map = None                      # MMap object
        self.overlay = []                           # Classes currently shown in overlay
        self.targets = None                         # Targets object containing points, scores and geo_points for all areas 
        self.target_overlays = {}                   # Textures for target overlays
        self.map_list_tgtnum = []                   # List of target numbers for each map
        self.inspected = False                      # Inspection state of current map

        # Parameters
        self.mic_params = None                      # Mic parameter object
        self.model = None                           # Segmentation model object
        self.tgt_params = None                      # Target parameter object
        self.loadParameterFiles()

        # One-time calls
        self.checked_run_conditions = False           # Needed to show error popup only once
        self.checked_point_files = False            # Needed to only prompt user once to load found point files
        self.checked_map_files = False              # Needed to only prompt user once to reload maps

        # Settings
        self.preload_maps = True

        # Developer settings
        self.thumbnail_size = (100, 100)            # Dims of thumbnails for map window

    def loadParameterFiles(self):
        """Loads parameters saved by SPACEtomo."""

        # Find all maps with segmentation
        self.updateMapList(list_only=True)

        # Check if mic_params exist
        if self.cur_dir.joinpath("mic_params.json").exists():
            log("NOTE: Microscope parameters found.")
            # Instantiate mic params from settings
            self.mic_params = space_ext.MicParams_ext(self.cur_dir)        
            # Load model
            self.model = space_ext.MMModel()
            self.model.setDimensions(self.mic_params)

            # Check if tgt_params exist
            if self.cur_dir.joinpath("tgt_params.json").exists():
                log("NOTE: Target parameters found.")
                # Instantiate tgt params from settings
                self.tgt_params = space_ext.TgtParams(file_dir=self.cur_dir, MM_model=self.model)
            else:
                # Use defaults
                self.tgt_params = space_ext.TgtParams(["lamella"], ["black", "white", "crack", "ice"], self.model, False, False, 0.3, 0.5, 60, self.mic_params, False, 10)

    def configureHandlers(self):
        """Sets up dearpygui registries and handlers."""

        # Create file dialogues

        # Create event handlers
        with dpg.handler_registry(tag="mouse_handler"):
            dpg.add_mouse_click_handler(callback=self.mouseClick)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self.mouseRelease)
            #m_release_left = dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=tgt_tgtUpdate)
            dpg.add_mouse_move_handler(tag="check_run_condition", callback=self.checkRunConditions)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self.mouseDrag)

            # Shortcuts
            dpg.add_key_press_handler(dpg.mvKey_H, callback=self.showHelp)
            dpg.add_key_press_handler(dpg.mvKey_N, callback=lambda: self.selectMap(next_map=True))
            dpg.add_key_press_handler(dpg.mvKey_A, callback=self.toggleAdvanced)
            dpg.add_key_press_handler(dpg.mvKey_D, callback=self.toggleDensePattern)
            dpg.add_key_press_handler(dpg.mvKey_P, callback=self.togglePolygonMode)
            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=lambda: self.map_window.show())
            dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda: self.lm_window.show() if not dpg.is_item_shown(self.lm_window.window) else self.lm_window.hide())
            dpg.add_key_press_handler(dpg.mvKey_T, callback=lambda: self.lm_window.toggleOverlay())

        with dpg.item_handler_registry(tag="point_tooltip_handler"):
            dpg.add_item_hover_handler(callback=self.targetTooltip)

        with dpg.item_handler_registry(tag="class_input_handler"):
            dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=self.openClassSelection)

        dpg.set_viewport_resize_callback(callback=lambda: gui.window_size_change(self.logo_dims))

    @staticmethod
    def configureThemes():
        """Sets up dearpygui themes."""

        gui.configureGlobalTheme()

        # Color themes for target areas plots
        for c, color in enumerate(gui.THEME_COLORS):
            with dpg.theme(tag="scatter_theme" + str(c)):
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Diamond, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 10, category=dpg.mvThemeCat_Plots)

        # Theme for targets outside IS limits
        with dpg.theme(tag="limit_scatter_theme"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, gui.COLORS["error"], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, gui.COLORS["error"], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Asterisk, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7, category=dpg.mvThemeCat_Plots)

        # Theme for geo points legend
        with dpg.theme(tag="geo_scatter_theme"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, gui.COLORS["geo"], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, gui.COLORS["geo"], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7, category=dpg.mvThemeCat_Plots)

        # Theme for beam legend
        with dpg.theme(tag="beam_scatter_theme"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, gui.COLORS["heading"], category=dpg.mvThemeCat_Plots)

        # Theme for target suggestion legend
        with dpg.theme(tag="suggestion_scatter_theme"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, gui.COLORS["subtle"], category=dpg.mvThemeCat_Plots)

        # Theme for tilt axis plot
        with dpg.theme(tag="axis_theme"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 64), category=dpg.mvThemeCat_Plots)

    def show(self):
        """Structures and launches main window of GUI."""

        # Setup window
        dpg.create_viewport(title="SPACEtomo Target Selection", disable_close=True, small_icon=str(Path(__file__).parent / "logo.ico"), large_icon=str(Path(__file__).parent / "logo.ico"))
        dpg.setup_dearpygui()

        # Create main window
        with dpg.window(label="GUI", tag="GUI", no_scrollbar=True, no_scroll_with_mouse=True):

            with dpg.table(header_row=False):
                dpg.add_table_column(init_width_or_weight=200, width_fixed=True)
                dpg.add_table_column()
                dpg.add_table_column(init_width_or_weight=200, width_fixed=True)

                with dpg.table_row():
                    with dpg.table_cell(tag="tblleft"):
                        dpg.add_text(default_value="Load your MM map", tag="l1", color=gui.COLORS["heading"])
                        with dpg.tooltip("l1", delay=0.5):
                            dpg.add_text("Select and load a map generated by SPACEtomo.")

                        self.menu_left = Menu()

                        self.menu_left.newRow(tag="mapcombo", separator=False, locked=False)
                        self.menu_left.addCombo(tag="map", label=f"({len(self.map_list)} maps)", combo_list=self.map_list)
                        self.updateMapList()
                        
                        self.menu_left.newRow(tag="load", horizontal=True, separator=False, locked=False)
                        self.menu_left.addButton(tag="butload", label="Load map", callback=self.selectMap, tooltip="Load selected map.")
                        self.menu_left.addButton(tag="butnext", label="Next", callback=lambda: self.selectMap(next_map=True), show=bool(self.map_list), tooltip="Load next map.")
                        self.menu_left.addButton(tag="butgrid", label="[]", callback=lambda: self.map_window.show(), tooltip="Show overview over all maps.")

                        self.menu_left.newRow(tag="advload", separator=False, locked=False, advanced=True)
                        self.menu_left.addInput(tag="inpbin", label="Binning", value=1, tooltip="Bin map when loading.")

                        self.menu_left.newRow(tag="windows", horizontal=True, separator=False, locked=False)
                        self.menu_left.addButton(tag="butlamgui", label="Grid map", callback=self.openWGMap, tooltip="Open whole grid map GUI.")
                        self.menu_left.addButton(tag="butflm", label="CLEM", callback=lambda: self.lm_window.show(), tooltip="Open correlative light microscopy image registration GUI.")

                        self.menu_left.newRow(tag="filters", separator=False, locked=True, advanced=False)
                        self.menu_left.addButton(tag="butfilter", label="Filter map", callback=self.filterMap)

                        self.menu_left.newRow(tag="reacquire", separator=False, locked=True, advanced=False)
                        self.menu_left.addButton(tag="butreacq", label="Reacquire", callback=self.showReacquireSettings, tooltip="Show settings to reacquire map.")#lambda: self.menu_left.unlockRows(["reacquire_settings", "reacquire_center", "reacquire_padding", "reacquire_restitch", "reacquire_queue"]), tooltip="Show settings to reacquire the map.")

                        self.menu_left.newRow(tag="reacquire_settings", separator=False, locked=True, advanced=False)
                        self.menu_left.addText(tag="reacquire_heading", value="Reacquisition settings:", color=gui.COLORS["heading"])
                        self.menu_left.addText(tag="center_offset", value="Center offset [%]")

                        self.menu_left.newRow(tag="reacquire_center", horizontal=True, separator=False, locked=True, advanced=False)
                        self.menu_left.addSlider(tag="center_offset_x", label="X", value=0, value_range=[-50, 50], callback=self.updateReacquireBounds)
                        self.menu_left.addSlider(tag="center_offset_y", label="Y", value=0, value_range=[-50, 50], callback=self.updateReacquireBounds)
                        self.menu_left.addButton(tag="butcenter_offset", label="Auto", callback=self.getReacquireCenter, theme="small_btn_theme", tooltip="Use current tracking target as new map center.")

                        self.menu_left.newRow(tag="reacquire_padding", separator=False, locked=True, advanced=False)
                        self.menu_left.addSlider(tag="padding", label="Padding [times]", value=1.5, value_range=[1, 2], callback=self.updateReacquireBounds, tooltip="Padding factor for area around original bounding box. (Default: 1.5x)\nDoes not represent absolute bounds of map!")

                        self.menu_left.newRow(tag="reacquire_restitch", separator=False, locked=True, advanced=False)
                        self.menu_left.addCheckbox(tag="restitch", label="Just restitch map", value=False, tooltip="Only restitch existing map without reacquiring.")

                        self.menu_left.newRow(tag="reacquire_queue", horizontal=True, separator=False, locked=True, advanced=False)
                        self.menu_left.addButton(tag="butreacquirequeue", label="Queue reacquisition", callback=self.reacquireMap, tooltip="This will require you to restart the SPACEtomo run!")

                        self.menu_left.newRow(tag="class_list", separator=False, locked=True)
                        self.menu_left.addText(tag="class_heading", value="\nClasses", color=gui.COLORS["heading"], tooltip="Choose a class to be displayed as overlay.")
                        if self.model is not None:
                            for key in self.model.categories.keys():
                                self.menu_left.addCheckbox(tag=key, label=key, value=False)

                        self.menu_left.newRow(tag="class_buttons", horizontal=True, separator=False, locked=True)
                        self.menu_left.addButton(tag="clsmask", label="Create overlay", callback=self.loadOverlay, tooltip="Create overlay of selected classes. (This can take a few seconds.)")
                        self.menu_left.addButton(tag="clsapply", label="Apply", callback=self.applyClassSelection, user_data=[self.menu_left, None], tooltip="Apply selected classes to target list.")

                        self.status = StatusLine()

                    with dpg.table_cell(tag="tblplot"):

                        self.menu_icon = Menu(outline=False)
                        self.menu_icon.newRow(tag="icon", horizontal=True, separator=False, locked=False)
                        self.menu_icon.addText(tag="icon_heading", value="MM map", color=gui.COLORS["heading"])
                        self.menu_icon.addImageButton("butresetzoom", gui.makeIconResetZoom(), callback=self.plot.resetZoom, tooltip="Reset zoom")
                        self.menu_icon.addImageButton("butsnapshot", gui.makeIconSnapshot(), callback=self.savePlot, tooltip="Save snapshot")
                        self.menu_icon.addImageButton("butholes", gui.makeIconHoles(), callback=self.toggleHolePattern, tooltip="Show hole pattern", show=False)
                        self.menu_icon.addImageButton("butdense", gui.makeIconDense(), callback=self.toggleDensePattern, tooltip="Show dense pattern", show=False)
                        self.menu_icon.addImageButton("butpolygon", gui.makeIconPolygon(), callback=self.togglePolygonMode, tooltip="Draw polygon", show=False)

                        self.plot.makePlot(x_axis_label="x [µm]", y_axis_label="y [µm]", width=-1, height=-1, equal_aspects=True, no_menus=True, crosshairs=True, pan_button=dpg.mvMouseButton_Right, no_box_select=True)

                    with dpg.table_cell(tag="tblright"):
                        dpg.add_text(default_value="Target selection", tag="r1", color=gui.COLORS["heading"])
                        with dpg.tooltip("r1", delay=0.5):
                            dpg.add_text("Select targets based on segmentation.")

                        if self.mic_params is not None:
                            self.menu_right = Menu()

                            self.menu_right.newRow(tag="settings1", separator=False)
                            self.menu_right.addText(tag="target_list_label", value="Target classes:")
                            self.menu_right.addInput(tag="target_list", label="", value=",".join(self.tgt_params.target_list), width=-1, tooltip="List of target classes (comma separated). For exhaustive acquisition use \"lamella\".")
                            dpg.bind_item_handler_registry(self.menu_right.all_elements["target_list"], "class_input_handler")

                            self.menu_right.newRow(tag="settings2", separator=False)
                            self.menu_right.addText(tag="avoid_list_label", value="Avoid classes:")
                            self.menu_right.addInput(tag="avoid_list", label="", value=",".join(self.tgt_params.penalty_list), width=-1, tooltip="List of classes to avoid (comma separated).")
                            dpg.bind_item_handler_registry(self.menu_right.all_elements["avoid_list"], "class_input_handler")

                            self.menu_right.newRow(tag="settings3", separator=False)
                            self.menu_right.addText(tag="settings_heading", value="\nTargeting options:")
                            self.menu_right.addSlider(tag="target_score_threshold", label="Score threshold", value=float(self.tgt_params.threshold), value_range=[0, 1], width=75, tooltip="Score threshold [0-1] below targets will be excluded.")
                            self.menu_right.addSlider(tag="penalty_weight", label="Penalty weight", value=float(self.tgt_params.penalty), value_range=[0, 1], width=75, tooltip="Relative weight of avoided classes to target classes.")
                            self.menu_right.newRow(tag="settings4", separator=False)
                            self.menu_right.addSlider(tag="max_tilt", label="Max. tilt angle", value=int(self.tgt_params.max_tilt), value_range=[0, 80], width=75, callback=self.updateMaxTilt, tooltip="Maximum tilt angle [degrees] to consider electron beam exposure.")
                            self.menu_right.addSlider(tag="beam_margin", label="[%] Beam margin", value=5, value_range=[0, 50], width=75, callback=self.updateBeamMargin, advanced=True, tooltip="Margin around target area to avoid exposure [% of beam diameter].")
                            self.menu_right.addSlider(tag="IS_limit", label="Image shift limit", value=int(self.mic_params.IS_limit), value_range=[5, 20], width=75, callback=self.updateISLimit, tooltip="Image shift limit [µm] for PACEtomo acquisition. If targets are further apart, target area will be split.")
                            self.menu_right.newRow(tag="settings5", separator=False)
                            self.menu_right.addCheckbox(tag="sparse_targets", label="Sparse targets", value=self.tgt_params.sparse, tooltip="Target positions will be initialized only on target classes and refined independently (instead of grid based target target setup to minimize exposure overlap).")
                            self.menu_right.addCheckbox(tag="target_edge", label="Target edge", value=self.tgt_params.edge, tooltip="Targets will be centered on edge of segmented target instead of maximising coverage.")
                            self.menu_right.addCheckbox(tag="extra_tracking", label="Extra tracking", value=self.tgt_params.sparse, tooltip="An extra target will be placed centrally for tracking.")

                            self.menu_right.newRow(tag="settings6", separator=False)
                            self.menu_right.addButton(tag="butselect", label="Auto select targets", callback=self.runTargetSelection, tooltip="Run target selection using current settings.")
                            self.menu_right.addText(tag="rsp1", value="")

                            self.menu_right.newRow(tag="suggestions", separator=False)
                            self.menu_right.addButton(tag="butaddsug", label="Add target suggestions", callback=self.addSuggestions, tooltip="Add all displayed suggestions as targets.")

                            self.menu_right.newRow(tag="areas", separator=False)
                            self.menu_right.addButton(tag="butsplit", label="Split target areas", callback=self.splitAreas, tooltip="Split targets into separate target areas using k-means clustering.")
                            self.menu_right.addButton(tag="butdist", label="Redistribute targets", callback=self.splitTargets, tooltip="Distribute targets to their closest tracking target.")
                            self.menu_right.addButton(tag="butmerge", label="Merge target areas", callback=self.mergeAreas, tooltip="Merge all target areas.")

                            self.menu_right.newRow(tag="delete", separator=False)
                            self.menu_right.addButton(tag="butdelete", label="Delete targets", callback=self.clearTargets, tooltip="Delete all targets and setup from scratch. (This cannot be undone!)")
                            self.menu_right.addButton(tag="butdeletegeo", label="Delete geo points", callback=self.clearGeoPoints)

                            self.menu_right.newRow(tag="acquisition1", separator=False)
                            self.menu_right.addText(tag="acq_heading", value="\nAcquisition settings", color=gui.COLORS["heading"])
                            self.menu_right.addText(tag="tilt_label1", value="Tilt angles [deg]:")

                            self.menu_right.newRow(tag="acquisition2", horizontal=True, separator=False)
                            self.menu_right.addInput(tag="acq_start_tilt", label="Start  ", value=0, width=34, callback=lambda: dpg.set_value(self.menu_right.all_elements["acq_save"], True))
                            self.menu_right.addInput(tag="acq_step_tilt", label="Step   ", value=3, width=34, callback=lambda: dpg.set_value(self.menu_right.all_elements["acq_save"], True))
                            
                            self.menu_right.newRow(tag="acquisition3", horizontal=True, separator=False)
                            self.menu_right.addInput(tag="acq_min_tilt", label="Min    ", value=-60, width=34, callback=lambda: dpg.set_value(self.menu_right.all_elements["acq_save"], True))
                            self.menu_right.addInput(tag="acq_max_tilt", label="Max    ", value=60, width=34, callback=lambda: dpg.set_value(self.menu_right.all_elements["acq_save"], True))

                            self.menu_right.newRow(tag="acquisition4", separator=False)
                            self.menu_right.addText(tag="geo_label1", value="Fallback geometry [deg]:")

                            self.menu_right.newRow(tag="acquisition5", horizontal=True, separator=False)
                            self.menu_right.addInput(tag="acq_pretilt", label="Pretilt", value=0, width=34, callback=lambda: dpg.set_value(self.menu_right.all_elements["acq_save"], True))
                            self.menu_right.addInput(tag="acq_rotation", label="Rotation", value=0, width=34, callback=lambda: dpg.set_value(self.menu_right.all_elements["acq_save"], True))

                            self.menu_right.newRow(tag="acquisition6", separator=False)
                            self.menu_right.addCheckbox(tag="acq_save", label="Override script settings")
                            self.menu_right.addCheckbox(tag="acq_autostart", label="Auto adjust start tilt", tooltip="Automatically calculate start tilt based on pretilt when running PACEtomo.")

                            self.menu_right.newRow(tag="buttons1", separator=False)
                            self.menu_right.addText(tag="save_heading", value="\n\nSave targets", color=gui.COLORS["heading"])

                            self.menu_right.newRow(tag="buttons2", separator=False)
                            self.menu_right.addButton(tag="butsave", label="Save", callback=self.saveTargets, show=False)
                            self.menu_right.addButton(tag="butins", label="Confirm inspection", callback=self.markInspected, theme="large_btn_theme", tooltip="Mark targets as inspected. (No more changes can be made.)")
                            self.menu_right.addButton(tag="butinsall", label="Confirm all maps", callback=self.markAllInspected, tooltip="Mark all maps as inspected.")

                            self.menu_right.hide()

                            dpg.add_text(tag="inspected", default_value="\nTargets for this map were\nalready marked as inspected.\n(Editing disabled)", color=gui.COLORS["heading"], show=False)
                        else:
                            dpg.add_text(tag="rerr", default_value="No microscope parameters\nfile found!", color=gui.COLORS["error"])

            # Create plot tooltip
            with dpg.tooltip("tblplot", delay=0.5, hide_on_activity=True):
                dpg.add_text(default_value="", color=gui.COLORS["heading"], tag="tt_heading")
                dpg.add_text(default_value="", tag="tt_text")
            dpg.bind_item_handler_registry("plot", "point_tooltip_handler")

            # Show logo
            dpg.add_image("logo", pos=(10, dpg.get_viewport_client_height() - 40 - self.logo_dims[0]), tag="logo_img")
            #dpg.add_text(default_value="SPACEtomo", pos=(10 + self.logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() - 40 - self.logo_dims[0] / 2), tag="logo_text")
            dpg.add_text(default_value="v" + __version__, pos=(10 + self.logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() + 5 - self.logo_dims[0] / 2), tag="version_text")

        # Make window for map thumbnails
        self.map_window = MapWindow(self.cur_dir, self.map_name, self.map_list, self.map_list_tgtnum, self.selectMap, self.thumbnail_size, self.executor)
        self.map_window.makeMapTable()
        self.blocking_windows.append(self.map_window.map_window) # Add to blocking windows to keep track of open popups

        # LM window
        self.lm_window = FlmWindow(self.plot)

        # Make window for target editing menu
        with dpg.window(label="Target", tag="win_tgt", no_scrollbar=True, no_scroll_with_mouse=True, popup=True, show=False) as win_tgt:
            self.menu_tgt = Menu()
            self.menu_tgt.newRow(tag="heading", separator=False, locked=False)
            self.menu_tgt.addText(tag="heading_txt", value="Target", color=gui.COLORS["heading"])
            self.menu_tgt.newRow(tag="info", separator=False)
            self.menu_tgt.addText(tag="info_txt")
            
            self.menu_tgt.newRow(tag="area", horizontal=True, separator=False)
            self.menu_tgt.addCombo(tag="area", label="Target area", callback=self.changeArea, width=30)

            self.menu_tgt.newRow(tag="buttons", separator=False)
            self.menu_tgt.addButton(tag="btn_del", label="Delete", callback=self.removeTarget)
            self.menu_tgt.addButton(tag="btn_trk", label="Make tracking target", callback=self.makeTrackDialogue)
            self.menu_tgt.addButton(tag="btn_opt", label="Optimize position", callback=self.optimizeTarget)
        self.blocking_windows.append(win_tgt) # Add to blocking windows to keep track of open popups

        # Make window for class list selection
        with dpg.window(label="Selection", tag="win_sel", no_scrollbar=True, no_scroll_with_mouse=True, popup=True, show=False) as win_sel:
            self.menu_sel = Menu()
            self.menu_sel.newRow(tag="heading", separator=False, locked=False)
            self.menu_sel.addText(tag="heading", value="Classes", color=gui.COLORS["heading"])
            if self.model is not None:
                for k, key in enumerate(self.model.categories.keys()):
                    if k % 3 == 0:
                        self.menu_sel.newRow(tag=str(k), horizontal=True, separator=False, locked=False)
                    self.menu_sel.addCheckbox(tag=key, label=key, value=False, callback=self.applyClassSelection)
        self.blocking_windows.append(win_sel) # Add to blocking windows to keep track of open popups

        InfoBoxManager.blocking_windows = self.blocking_windows

        dpg.bind_theme("global_theme")

        dpg.set_exit_callback(self.askForSave)

        dpg.set_primary_window("GUI", True)
        dpg.show_viewport()

        # Render loop
        next_update = time.time() + 1
        while dpg.is_dearpygui_running():

            # Recheck folder for segmentation every minute
            now = time.time()
            if now > next_update:
                next_update = now + 1

                # Check if info box needs to be shown
                InfoBoxManager.unblock()

                # Check if new maps (or thumbnails) were finished
                if self.updateMapList() or (any([future.done() for future in self.map_window.futures])):
                    self.map_window.update(self.map_name, self.map_list, self.map_list_tgtnum)

                # Check if new target selection was finished
                self.checkPointFiles()

                # Preload map in background
                if self.preload_maps and self.map_list:
                    # If preloading is in progress, check if it is done
                    if isinstance(self.preloaded_data, concurrent.futures.Future):
                        if self.preloaded_data.done():
                            self.preloaded_data = self.preloaded_data.result()
                    # If no preloading is done or in progress (and no status is set, e.g. map is loading), start new one
                    elif self.preloaded_data is None and not self.status.status:
                        self.preloaded_data = self.executor.submit(self.preloadMap)

            dpg.render_dearpygui_frame()
