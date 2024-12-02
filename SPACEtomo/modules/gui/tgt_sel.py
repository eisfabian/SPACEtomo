#!/usr/bin/env python
# ===================================================================
# ScriptName:   gui_tgt
# Purpose:      User interface for selecting SPACEtomo targets
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/03/20
# Revision:     v1.2
# Last Change:  2024/11/27: added reacquire button
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

from SPACEtomo.modules import ext as space_ext
from SPACEtomo.modules.gui import gui
from SPACEtomo.modules.gui.thmb import MapWindow
from SPACEtomo.modules.gui.flm import FlmWindow
from SPACEtomo.modules.gui.menu import Menu
from SPACEtomo.modules.gui.plot import Plot
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

        # Left mouse button functions
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and (dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)):
            if self.targets.addTarget(img_coords, new_area=dpg.is_key_down(dpg.mvKey_T)):
                self.showTargets()
                dpg.show_item(self.menu_right.all_elements["butsave"])

        # Right mouse button functions
        elif dpg.is_mouse_button_down(dpg.mvMouseButton_Right) or (dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and dpg.is_key_down(dpg.mvKey_E)):
            self.showTargetMenu(img_coords)  

        # Middle mouse button functions
        elif dpg.is_mouse_button_down(dpg.mvMouseButton_Middle) or (dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and dpg.is_key_down(dpg.mvKey_G)):
            if self.targets.addGeoPoint(img_coords):
                self.showTargets()
                dpg.show_item(self.menu_right.all_elements["butsave"])

    def mouseRelease(self, sender, app_data):
        """Handle mouse release and check if any drag points were moved."""

        # Delegate to sub-window if open
        if dpg.is_item_shown(self.lm_window.window):
            self.lm_window.mouseRelease()
            return

        # Only process when targets are editable
        if not self.targets or self.inspected:
            return

        # Check all drag points for changes
        update = False
        for point in self.plot.drag_points:
            update = update or self.checkDragPoint(point["plot"])

        if update:
            self.plot.clearSeries(["temp_drag"])
            self.showTargets()
            dpg.show_item(self.menu_right.all_elements["butsave"])

    def dragPointUpdate(self, sender, app_data, user_data):
        """Handles dragging of points and updates scatter point to mouse position."""

        # Don't allow changes on inspected targets
        if self.inspected: return

        # Update scatter indicator for drag point (drag points are always plotted behind images)
        coords = dpg.get_value(sender)[:2]
        self.plot.clearSeries(["temp_drag"])
        self.plot.addSeries([coords[0]], [coords[1]], label="temp_drag", theme="drag_scatter_theme")

    def checkDragPoint(self, point):
        """Compares drag points to targets coords and updates targets if they have changed."""

        # Get coords from drag point value
        coords = np.array(dpg.get_value(point)[:2])
        
        # Get area and point IDs from user data embedded in drag point
        user_data = dpg.get_item_user_data(point).split("_")
        point_type = user_data[0]
        point_id = np.array(user_data[1:], dtype=int)
        
        # Transform points to plot points for comparison
        if point_type == "pt":
            old_coords = self.loaded_map.px2microns(self.targets.areas[point_id[0]].points[point_id[1]])
        elif point_type == "geo":
            old_coords = self.loaded_map.px2microns(self.targets.areas[point_id[0]].geo_points[point_id[1]])
        else:
            log(f"ERROR: Invalid drag point! [{user_data}]")
            return
        
        # Go to next points if coords have not changed
        if np.all(coords == old_coords):
            return False
        else:
            # Get camera dims in microns
            rec_dims = np.array(self.tgt_params.weight.shape) * self.loaded_map.pix_size / 10000

            # Clip coords to map size
            coords[0] = np.clip(coords[0], rec_dims[1] / 2, self.loaded_map.dims_microns[0] - rec_dims[1] / 2)
            coords[1] = np.clip(coords[1], rec_dims[0] / 2, self.loaded_map.dims_microns[1] - rec_dims[0] / 2)

            # Update coords if they have changed
            if point_type == "pt":
                self.targets.areas[point_id[0]].points[point_id[1]] = self.loaded_map.microns2px(coords)
            elif point_type =="geo":
                self.targets.areas[point_id[0]].geo_points[point_id[1]] = self.loaded_map.microns2px(coords)

            return True

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

        # Find selected targets per lamella
        self.map_list_tgtnum = []
        for m_name in self.map_list:
            point_files = sorted(self.cur_dir.glob(m_name + "_points*.json"))
            point_num = 0
            if len(point_files) > 0:
                for file in point_files:
                    # Load json data
                    with open(file, "r") as f:
                        point_data = json.load(f, object_hook=utils.revertArray)
                        point_num += len(point_data["points"])
            self.map_list_tgtnum.append(point_num)

        # Check for changes
        if len(self.map_list) != prev_map_num or sum(self.map_list_tgtnum) != prev_tgt_num:
            return True
        else:
            return False
        
    def checkPointFiles(self):
        """Checks if new point file of loaded map was created."""

        # Don't check if no map is loaded or targets are already loaded (or selected)
        if not self.map_name or (self.targets and len(self.targets) > 0) or self.checked_point_files:
            return
        
        # Get all point files for lamella map
        point_files = sorted(self.cur_dir.glob(self.map_name + "_points*.json"))
        if len(point_files) > 0:
            # Check if any points in point files
            point_num = 0
            for file in point_files:
                # Load json data
                with open(file, "r") as f:
                    point_data = json.load(f, object_hook=utils.revertArray)
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
            gui.showInfoBox("New target coordinates", "Target coordinates for the currently loaded map were found. Do you want to load them?", callback=self.confirmLoadPointFiles, options=["Load", "Cancel"], options_data=[True, False])
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
        self.map_window.hide()

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
        self.status.update("Loading map...", box=True)

        # Reset FLM window plots
        self.lm_window.clearAll()
        # Reset plot
        self.plot.clearAll()
        # Clear out texture references because textures have been deleted when plot was cleared
        self.target_overlays = {}
        self.makeTargetOverlay() # Pre-generate overlays to remove lag when selecting first target

        # Load map file
        self.loaded_map = MMap(file_path, pix_size=self.model.pix_size * 10, status=self.status)
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

        # Load targets
        self.status.update("Loading targets...", box=True)
        self.targets = Targets(map_dir=self.cur_dir, map_name=self.map_name, map_dims=self.loaded_map.img.shape, tgt_params=self.tgt_params, map_pix_size=self.loaded_map.pix_size)

        self.loadTargetsFile()

        # Reset checked for point files status
        self.checked_point_files = False

        # Update GUI
        #self.menu_left.unlockRows(["filters"])     # Not yet implemented
        self.menu_left.unlockRows(["reacquire"])
        if self.segmentation.valid:         # Only show class selection if segmentation was performed
            self.menu_left.unlockRows(["class_list", "class_buttons"])
            self.menu_right.unlockRows(["settings1", "settings2", "settings3", "settings4"])

        self.menu_left.show()

        self.status.update()

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
            dpg.hide_item(self.menu_right.all_elements["butsave"])

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
        axis_points = (np.array([rotM @ np.array([-axis_length1, 0]), rotM @ np.array([axis_length2, 0])]) + offset) * self.loaded_map.pix_size / 10000

        # Add custom series to plot.series list
        self.plot.addCustomPlot(self.plot.series, label="tilt_axis", plot=dpg.add_line_series(list(axis_points[:, 1]), list(axis_points[:, 0]), parent=self.plot.x_axis), theme="axis_theme")

    def filterMap(self):
        """Loads map filtered with LISC."""

        # TODO
        log("Filter map")

    def reacquireMap(self):
        """Marks map to be reacquired by changing file names."""

        mrc_file = self.cur_dir.parent / (self.map_name + ".mrc")
        if mrc_file.exists():
            counter = 0
            while (new_mrc_file := self.cur_dir.parent / (self.map_name + f"_old{counter}.mrc")).exists():
                counter += 1
            new_map_name = self.map_name + f"_old{counter}"
            # Rename mrc file and all generated files
            mrc_file.replace(new_mrc_file)
            if (self.cur_dir / (self.map_name + ".png")).exists():
                (self.cur_dir / (self.map_name + ".png")).replace(self.cur_dir / (new_map_name + ".png"))
            if (self.cur_dir / (self.map_name + "_seg.png")).exists():
                (self.cur_dir / (self.map_name + "_seg.png")).replace(self.cur_dir / (new_map_name + "_seg.png"))
            if (self.cur_dir / "thumbnails" / (self.map_name + ".png")).exists():
                (self.cur_dir / "thumbnails" / (self.map_name + ".png")).replace(self.cur_dir / "thumbnails" / (new_map_name + ".png"))
            # Provide empty point file
            if not list(self.cur_dir.glob(new_map_name + "_points*.json")):
                Targets(map_dir=self.cur_dir, map_name=new_map_name, map_dims=self.loaded_map.img.shape, tgt_params=self.tgt_params, map_pix_size=self.loaded_map.pix_size).exportTargets()
            # Mark new name as inspected and original name (to not stall current run)
            (self.cur_dir / (self.map_name + "_inspected.txt")).touch(exist_ok=True)
            (self.cur_dir / (new_map_name + "_inspected.txt")).touch(exist_ok=True)

            self.map_name = new_map_name

            log(f"DEBUG: Reacquire map {mrc_file} on next SPACEtomo run!")
            gui.showInfoBox("INFO", "This map will be reacquired on the next SPACEtomo run. If you want to reacquire now, please stop the SPACEtomo run and start it again.")
        else:
            log(f"ERROR: Cannot find existing map MRC file. Reacquisition could not be scheduled.")
            gui.showInfoBox("ERROR", "Cannot find existing map MRC file. Reacquisition could not be scheduled!")

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
        if not self.targets:
            return
        
        # Clear targets from plot
        self.clearTargets(plot_only=True)
        self.plot.clearSeries(["temp_drag"])

        # Generate target overlay texture
        if not self.target_overlays or not dpg.does_item_exist(self.target_overlays["target"]):
            self.makeTargetOverlay()

        # Get map dims
        dims = np.flip(np.array(self.loaded_map.img.shape))

        tgt_counter = 0
        geo_counter = 0
        for t, target_area in enumerate(self.targets.areas):
            if len(target_area.points) == 0: continue
            # Transform coords to plot
            points = np.array([self.loaded_map.px2microns(point) for point in target_area.points])

            # Scatter plot of targets
            self.plot.addSeries(points[:, 0], points[:, 1], label=f"tgtscatter_{t}", theme=f"scatter_theme{t}")

            # Go over all target points
            scaled_overlay_dims = self.target_overlays["tgtdims"] * self.loaded_map.pix_size / 10000
            for p, (x, y) in enumerate(points):
                # Add draggable point
                self.plot.addDragPoint(x, y, label=f"tgt_{tgt_counter + 1}", callback=self.dragPointUpdate, user_data=f"pt_{t}_{p}", color=gui.THEME_COLORS[t % len(gui.THEME_COLORS)])
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

                # Scatter plot of geo points
                self.plot.addSeries(points[:, 0], points[:, 1], label=f"geoscatter_{t}", theme=f"scatter_theme{t}")

                # Go over all geo points
                scaled_overlay_dims = self.target_overlays["geodims"] * self.loaded_map.pix_size / 10000
                for p, (x, y) in enumerate(points):
                    # Add draggable point
                    self.plot.addDragPoint(x, y, label=f"geo_{geo_counter}", callback=self.dragPointUpdate, user_data=f"geo_{t}_{p}", color=gui.THEME_COLORS[4])
                    # Show graphical overlays
                    bounds = ((x - scaled_overlay_dims[1] / 2, x + scaled_overlay_dims[1] / 2), (y - scaled_overlay_dims[0] / 2, y + scaled_overlay_dims[0] / 2))
                    self.plot.addOverlay(self.target_overlays["geo"], bounds, label=f"geo_{t}_{p}")

                    geo_counter += 1

        # Show tilt axis centered on tracking target
        self.showTiltAxis(self.targets.areas[0].points[0])

    def targetTooltip(self):
        """Configures plot tooltip to contain target information."""

        mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # Convert plot coords to image coords in px
        if self.loaded_map and self.targets:
            img_coords = self.loaded_map.microns2px(mouse_coords)

            # Get camera dims
            rec_dims = np.array(self.tgt_params.weight.shape)

            # Check if coords are too close to existing point
            closest_point_id, in_range = self.targets.getClosestPoint(img_coords, np.min(rec_dims))

            if in_range:
                dpg.set_value("tt_heading", "Target information:")

                """
                # Set point information
                area = f"Area: {closest_point_id[0] + 1}\n" if len(self.targets.areas) > 1 else ""
                score, dist = self.targets.areas[closest_point_id[0]].getPointInfo(closest_point_id[1])
                score = round(score, 2) if score < 100 else "manual"
                dist *= self.loaded_map.pix_size / 10000
                info = f"{area}Target: {closest_point_id[1] + 1}\nScore: {score}\nIS: {round(dist, 2)} µm"
                """
                info = self.assembleTargetInfo(closest_point_id)

                dpg.set_value("tt_text", info)
                return

        dpg.set_value("tt_heading", "Target manipulation:")
        dpg.set_value("tt_text", "- Drag target to reposition\n- Shift + left click to add target\n- Right click to open target editing\n- Middle click to add geo point")

    def showTargetMenu(self, img_coords):
        """Configures target menu when right clicking on point."""

        # Get camera dims
        rec_dims = np.array(self.tgt_params.weight.shape)

        # Check if coords are too close to existing point
        closest_point_id, in_range = self.targets.getClosestPoint(img_coords, np.min(rec_dims))

        if in_range:
            dpg.set_value(self.menu_tgt.all_elements["heading_txt"], "Target information:")

            info = self.assembleTargetInfo(closest_point_id)
            """
            # Set area information if more than one target area
            area = f"Area: {closest_point_id[0] + 1}\n" if len(self.targets.areas) > 1 else ""

            # Get score and distance from tracking target
            final_score, dist = self.targets.areas[closest_point_id[0]].getPointInfo(closest_point_id[1])
            score_text = round(final_score, 2) if final_score < 100 else "manual"
            dist *= self.loaded_map.pix_size / 10000

            # Get class scores if present
            class_scores = ""
            if self.targets.areas[closest_point_id[0]].class_scores:
                for cat, score in self.targets.areas[closest_point_id[0]].class_scores.items():
                    if score[closest_point_id[1]] > 0:
                        # Format class scores to show percentage of final score
                        class_scores += f"- {cat}:{' ' * (max([len(cat_name) for cat_name in self.tgt_params.target_list]) + 1 - len(cat))}{round(score[closest_point_id[1]], 2)}\n"
            
            # Format and set info text
            info = f"{area}Target: {closest_point_id[1] + 1}\nScore: {score_text}\n{class_scores}IS: {round(dist, 2)} µm"
            """
            dpg.set_value(self.menu_tgt.all_elements["info_txt"], info)

            # Configure buttons
            dpg.configure_item(self.menu_tgt.all_elements["btn_del"], user_data=closest_point_id)
            if closest_point_id[1] > 0:
                # Only show option when target is not a tracking target
                dpg.configure_item(self.menu_tgt.all_elements["btn_trk"], user_data=closest_point_id)
                dpg.show_item(self.menu_tgt.all_elements["btn_trk"])
            else:
                dpg.hide_item(self.menu_tgt.all_elements["btn_trk"])

            # Configure area selection
            if len(self.targets.areas) > 1:
                dpg.configure_item(self.menu_tgt.all_elements["area"], items=[area + 1 for area in range(len(self.targets.areas))], default_value=closest_point_id[0] + 1, user_data=closest_point_id)
                self.menu_tgt.unlockRows(["area"])
            else:
                self.menu_tgt.lockRows(["area"])

            # Configure optimization button (only show when segmentation was loaded)
            if self.segmentation.valid:
                dpg.configure_item(self.menu_tgt.all_elements["btn_opt"], user_data=closest_point_id)
                dpg.show_item(self.menu_tgt.all_elements["btn_opt"])
            else:
                dpg.hide_item(self.menu_tgt.all_elements["btn_opt"])

            # Unlock rows
            self.menu_tgt.unlockRows(["info", "buttons"])

            # Get UI mouse coords for window placement
            mouse_coords_global = dpg.get_mouse_pos(local=False)
            dpg.set_item_pos("win_tgt", mouse_coords_global)
            dpg.show_item("win_tgt")

    def assembleTargetInfo(self, closest_point_id):
        """Creates string with target info for display."""

        # Set area information if more than one target area
        area = f"Area: {closest_point_id[0] + 1}\n" if len(self.targets.areas) > 1 else ""

        # Get score and distance from tracking target
        final_score, dist = self.targets.areas[closest_point_id[0]].getPointInfo(closest_point_id[1])
        score_text = round(final_score, 2) if final_score < 100 else "manual"
        dist *= self.loaded_map.pix_size / 10000

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
        else:
            warning_is = ""
        return f"{area}Target: {closest_point_id[1] + 1}\nScore: {score_text}\n{class_scores}IS: {round(dist, 2)}{warning_is} µm"

    def showTargetAreaButtons(self):
        """Determines which target area actions are available."""

        # Split areas button
        if len(self.targets) > 0:
            dpg.show_item(self.menu_right.all_elements["butsplit"])
        else:
            dpg.hide_item(self.menu_right.all_elements["butsplit"])

        # Redistribute targets and merge buttons
        if len(self.targets.areas) > 1:
            dpg.show_item(self.menu_right.all_elements["butdist"])
            dpg.show_item(self.menu_right.all_elements["butmerge"])
        else:
            dpg.hide_item(self.menu_right.all_elements["butdist"])
            dpg.hide_item(self.menu_right.all_elements["butmerge"])

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
        dpg.hide_item(self.menu_right.all_elements["butsave"])     # Only activate save button when targets were changed
        if not self.inspected:
            self.menu_right.show()
        self.status.update()

    def updateMaxTilt(self):
        new_max_tilt = dpg.get_value(self.menu_right.all_elements['max_tilt'])

        if abs(new_max_tilt - self.tgt_params.max_tilt) > 5:
            self.tgt_params.max_tilt = new_max_tilt

            # Reset target overlays
            for overlay in self.target_overlays:
                if dpg.does_item_exist(overlay):
                    dpg.delete_item(overlay)
            self.target_overlays = {}

            # Re-plot targets
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
        dpg.show_item(self.menu_right.all_elements["butsave"])
        
    def makeTrackDialogue(self, sender, app_data, user_data):
        """Creates dialogue for make tracking target."""

        dpg.hide_item("win_tgt")
        dpg.split_frame()
        gui.showInfoBox("Make tracking target", "Do you want to make this target the tracking target of the current area or create a new acquisition area?", callback=self.makeTrack, options=["Current area", "New area", "Cancel"], options_data=[[user_data, "old"], [user_data, "new"], False])

    def makeTrack(self, sender, app_data, user_data):
        """Makes selected target a tracking target of current or new area."""

        # Check for info box input
        if user_data and dpg.does_item_exist(user_data[0]):
            dpg.delete_item(user_data[0])
            dpg.split_frame()
        
        if not user_data[1]:
            return
        
        area_id, point_id = user_data[1][0]
        new_area = True if user_data[1][1] == "new" else False

        if new_area:
            # Get coords
            coords = self.targets.areas[area_id].points[point_id]

            # Remove point from old area
            self.targets.areas[area_id].removePoint(point_id)
            
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
        dpg.show_item(self.menu_right.all_elements["butsave"])

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
            dpg.show_item(self.menu_right.all_elements["butsave"])

    def removeTarget(self, sender, app_data, user_data):
        """Deletes a single target."""

        if user_data:
            area_id, point_id = user_data
            self.targets.areas[area_id].removePoint(point_id)
            if len(self.targets.areas[area_id]) == 0:
                self.targets.areas.pop(area_id)
            dpg.hide_item("win_tgt")

            self.showTargets()
            self.showTargetAreaButtons()
            # Enable save button
            dpg.show_item(self.menu_right.all_elements["butsave"])

    def mergeAreas(self):
        """Merge all target areas."""

        self.targets.mergeAreas()

        self.showTargets()
        self.showTargetAreaButtons()
        # Enable save button
        dpg.show_item(self.menu_right.all_elements["butsave"])

    def splitAreas(self):
        """Splits targets into target areas using k-means clustering."""

        self.targets.splitArea(area_num=2)

        self.showTargets()
        self.showTargetAreaButtons()
        # Enable save button
        dpg.show_item(self.menu_right.all_elements["butsave"])

    def splitTargets(self):
        """Splits all targets among current tracking targets."""

        self.targets.splitAreaManual()

        self.showTargets()
        self.showTargetAreaButtons()
        # Enable save button
        dpg.show_item(self.menu_right.all_elements["butsave"])        

    def clearTargets(self, *args, plot_only=False):
        """Deletes all targets (or just clear from plot.)"""

        # Clear targets from plot
        self.plot.clearOverlays(delete_textures=not plot_only)
        self.plot.clearDragPoints()
        self.plot.clearSeries(labels=self.plot.getSeriesByKeyword("geo") + self.plot.getSeriesByKeyword("tgt") + [""])
        if not plot_only:
            # Release target overlay textures
            self.target_overlays = {}
            # Reinitialize targets object
            self.targets = Targets(map_dir=self.cur_dir, map_name=self.map_name, map_dims=self.loaded_map.img.shape, tgt_params=self.tgt_params, map_pix_size=self.loaded_map.pix_size)
            log("NOTE: Deleted targets!")

            # Enable save button
            dpg.show_item(self.menu_right.all_elements["butsave"])

    def clearGeoPoints(self):
        """Deletes all geo points."""

        # Clear all items with "geo" labels (needs empty list entry, otherwise all items will be cleared in case of no hits)
        self.plot.clearOverlays(labels=self.plot.getOverlaysByKeyword("geo") + [""], delete_textures=False)
        self.plot.clearDragPoints(labels=self.plot.getDragPointsByKeyword("geo") + [""])
        self.plot.clearSeries(labels=self.plot.getSeriesByKeyword("geo") + [""])

        self.targets.resetGeo()
        log("NOTE: Deleted geo points!")

        # Enable save button
        dpg.show_item(self.menu_right.all_elements["butsave"])

    def saveTargets(self):
        """Exports targets as json file."""

        # Get acquisition settings
        settings = self.getAcquisitionSettings()

        self.targets.exportTargets(settings)
        dpg.hide_item(self.menu_right.all_elements["butsave"])
        log("NOTE: Saved targets!")

    def markInspected(self, sender, app_data, user_data):
        """Creates inspected.txt file and locks down editing."""

        # Check for geo_points
        if self.targets and len(self.targets.areas[0].points) and not len(self.targets.areas[0].geo_points) and not user_data:
            log(f"WARNING: No geo points were selected!")
            gui.showInfoBox("WARNING", "No geo points were selected to measure the sample geometry. If you go ahead, the manual input for pretilt and rotation will be used!", callback=self.markInspected, options=["Continue", "Cancel"], options_data=[True, False])
            return
        # Close geo_points confirmation info box
        if user_data and dpg.does_item_exist(user_data[0]):
            dpg.delete_item(user_data[0])
            dpg.split_frame()
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
        if not self.checkAllInspected():
            log("\n\nLoading next map...")
            self.selectMap(next_map=True)

    def markAllInspected(self, sender, app_data, user_data):
        """Marks all maps as inspected!"""

        for map_name in self.map_list:
            (self.cur_dir / (map_name + "_inspected.txt")).touch(exist_ok=True)

        self.inspected = True

        self.checkAllInspected()

    def checkAllInspected(self):
        """Checks if all lamella have been marked inspected and prompts GUI close."""

        # Check if all maps were inspected
        inspected = []
        for map_name in self.map_list:
            inspected.append((self.cur_dir / (map_name + "_inspected.txt")).exists())
        if all(inspected):
            log("NOTE: All lamella maps were inspected!")
            if self.auto_close:
                # Confirm closing
                gui.showInfoBox("FINISHED?", "All available lamellae were inspected. Are lamella maps still being acquired? If not, you can close this GUI.", callback=self.closeAllInspected, options=["Wait", "Close"], options_data=[False, True])
                return True
        else:
            return False

    def closeAllInspected(self, sender, app_data, user_data):
        """Closes GUI if all lamellae were inspected and user confirmed."""

        # Close GUI
        if user_data and user_data[1]:
            dpg.stop_dearpygui()
        else:
            # Close confirmation info box
            if dpg.does_item_exist(user_data[0]):
                dpg.delete_item(user_data[0])
                dpg.split_frame()        

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

        return settings
    
    def setAcquisitionSettings(self):
        """Sets acquisition settings in case they were loaded from file."""

        if self.targets.settings:
            dpg.set_value(self.menu_right.all_elements["acq_start_tilt"], self.targets.settings["startTilt"])
            dpg.set_value(self.menu_right.all_elements["acq_min_tilt"], self.targets.settings["minTilt"])
            dpg.set_value(self.menu_right.all_elements["acq_max_tilt"], self.targets.settings["maxTilt"])
            dpg.set_value(self.menu_right.all_elements["acq_step_tilt"], self.targets.settings["step"])
            dpg.set_value(self.menu_right.all_elements["acq_pretilt"], self.targets.settings["pretilt"])
            dpg.set_value(self.menu_right.all_elements["acq_rotation"], self.targets.settings["rotation"])

    def checkRunConditions(self):
        """Checks if mic and tgt params could be loaded."""

        if self.mic_params is None or self.model is None:
            if not self.checked_run_conditions:
                log("ERROR: No microscope parameter file found. Please launch the GUI in a SPACE_maps folder!")
                # Change exit callback to avoid askForSave method
                dpg.set_exit_callback(dpg.stop_dearpygui)
                gui.showInfoBox("ERROR", "No microscope parameter file found.\nPlease launch the GUI in a SPACE_maps folder!", callback=dpg.stop_dearpygui)
            self.checked_run_conditions = True
            return

        # Create popup if no maps have been found yet
        if not self.map_list:
            if not self.checked_map_files:
                gui.showInfoBox("NOTE", "No lamella map segmentations are available yet.\n\nReload after a few minutes to check for lamella map segmentations.", callback=self.manualMapListReload, options=["Reload", "Close"], options_data=[True, False])
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
            dpg.delete_item(user_data[0])
            dpg.split_frame()

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

        # Get camera dims
        rec_dims = np.array(self.tgt_params.weight.shape)

        # TGT
        # Create canvas with size of stretched beam diameter
        tgt_overlay = np.zeros([int(self.model.beam_diameter), int(self.model.beam_diameter / np.cos(np.radians(self.tgt_params.max_tilt)))])
        canvas = Image.fromarray(tgt_overlay).convert('RGB')
        draw = ImageDraw.Draw(canvas)

        # Draw beam
        draw.ellipse((0, 0, tgt_overlay.shape[1] - 1, tgt_overlay.shape[0] - 1), outline="#ffd700", width=20)

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

        # GEO
        # Create canvas for geo with non-stretched beam diameter
        geo_overlay = np.zeros([int(self.model.beam_diameter), int(self.model.beam_diameter)])
        canvas = Image.fromarray(geo_overlay).convert('RGB')
        draw = ImageDraw.Draw(canvas)

        # Draw beam and camera dims
        draw.ellipse((0, 0, geo_overlay.shape[1] - 1, geo_overlay.shape[0] - 1), outline="#ee8844", width=20)
        rect = ((canvas.width - rec_dims[1]) // 2, (canvas.height - rec_dims[0]) // 2, (canvas.width + rec_dims[1]) // 2, (canvas.height + rec_dims[0]) // 2)
        draw.rectangle(rect, outline="#ee8844", width=20)

        # Convert to array
        geo_overlay = np.array(canvas).astype(float) / 255

        # Make textures
        self.target_overlays["tgtdims"] = np.array(tgt_overlay.shape)[:2]
        alpha = np.zeros(tgt_overlay.shape[:2])
        alpha[np.sum(tgt_overlay, axis=-1) > 0] = 1
        tgt_overlay_image = np.ravel(np.dstack([tgt_overlay, alpha]))
        trk_overlay_image = np.ravel(np.dstack([trk_overlay, alpha]))

        self.target_overlays["geodims"] = np.array(geo_overlay.shape)[:2]
        alpha = np.zeros(geo_overlay.shape[:2])
        alpha[np.sum(geo_overlay, axis=-1) > 0] = 1
        geo_overlay_image = np.ravel(np.dstack([geo_overlay, alpha]))

        with dpg.texture_registry():
            self.target_overlays["target"] = dpg.add_static_texture(width=int(self.target_overlays["tgtdims"][1]), height=int(self.target_overlays["tgtdims"][0]), default_value=tgt_overlay_image)
            self.target_overlays["track"] = dpg.add_static_texture(width=int(self.target_overlays["tgtdims"][1]), height=int(self.target_overlays["tgtdims"][0]), default_value=trk_overlay_image)
            self.target_overlays["geo"] = dpg.add_static_texture(width=int(self.target_overlays["geodims"][1]), height=int(self.target_overlays["geodims"][0]), default_value=geo_overlay_image)

    def askForSave(self):
        """Opens popup if there are unsaved changes."""

        # Check for unsaved changes
        if dpg.is_item_shown(self.menu_right.all_elements["butsave"]):
            gui.showInfoBox("WARNING", "There are unsaved changes to your targets!", callback=self.saveAndClose, options=["Save", "Discard"], options_data=[True, False])
        else:

            # Check inspected status and give warning when some but not all targets have been inspected
            inspected = []
            for map_name in self.map_list:
                inspected.append((self.cur_dir / (map_name + "_inspected.txt")).exists())
            if any(inspected) and not all(inspected):
                gui.showInfoBox("WARNING", "Not all targets have been marked as inspected. Are you sure you want to exit?", callback=self.confirmClose, options=["Exit", "Cancel"], options_data=[True, False])
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

    def confirmClose(self, sender, app_data, user_data):
        """Closes GUI or keeps it open."""

        if user_data[1]:
            # Change exit callback to avoid infinite loop
            dpg.set_exit_callback(dpg.stop_dearpygui)
            dpg.stop_dearpygui()
        else:
            # Close info box
            if dpg.does_item_exist(user_data[0]):
                dpg.delete_item(user_data[0])
                dpg.split_frame()

    def toggleAdvanced(self):
        """Shows advanced menu options."""

        self.menu_left.toggleAdvanced()
        self.menu_right.toggleAdvanced()

    def triggerShowHelp(self):
        """Triggers show help once."""

        if dpg.does_item_exist("trigger_help"):
            dpg.delete_item("trigger_help")

        # Show help
        self.showHelp()

    @staticmethod
    def showHelp():
        """Shows popup with shortcuts."""

        message = "Load a lamella map from the list to\ninspect or select targets and geo points.\n\n[Confirm inspection] when you are done!\n\n\n"
        message += "Controls:\n\n"
        message += "Shift + left click      Add new target\n"
        message += "Right click on target   Open menu\n"
        message += "Middle click            Add geo point\n"
        message += "\n\n"
        message += "Keyboard shortcuts:\n\n"
        message += "A     Toggle advanced settings\n"
        message += "H     Show help\n"
        message += "N     Load next map\n"
        message += "R     CLEM registration\n"
        message += "Space Show available maps\n"

        gui.showInfoBox("Help", message)

#################################################################################


    def __init__(self, path="", auto_close=False) -> None:
        log(f"\n########################################\nRunning SPACEtomo Target Selection GUI\n########################################\n")

        # Automatically close GUI after map was inspected
        self.auto_close = auto_close

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
        self.use_thumbnails = False

        # Developer settings
        self.thumbnail_size = (100, 100)            # Dims of thumbnails for map window

    def loadParameterFiles(self):
        """Loads parameters saved by SPACEtomo."""

        # Find all maps with segmentation
        thumbnail_size = (100, 100)
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

            # Shortcuts
            dpg.add_key_press_handler(dpg.mvKey_H, callback=self.showHelp)
            dpg.add_key_press_handler(dpg.mvKey_N, callback=lambda: self.selectMap(next_map=True))
            dpg.add_key_press_handler(dpg.mvKey_A, callback=self.toggleAdvanced)
            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=lambda: self.map_window.show())
            dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda: self.lm_window.show() if not dpg.is_item_shown(self.lm_window.window) else self.lm_window.hide())

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

        # Theme for temporary drag point
        with dpg.theme(tag="drag_scatter_theme"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (200, 0, 0, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7, category=dpg.mvThemeCat_Plots)

        # Theme for tilt axis plot
        with dpg.theme(tag="axis_theme"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 64), category=dpg.mvThemeCat_Plots)

    def show(self):
        """Structures and launches main window of GUI."""

        # Setup window
        dpg.create_viewport(title="SPACEtomo Target Selection", disable_close=True)
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

                        self.menu_left = Menu()

                        self.menu_left.newRow(tag="mapcombo", separator=False, locked=False)
                        self.menu_left.addCombo(tag="map", label=f"({len(self.map_list)} maps)", combo_list=self.map_list)
                        self.updateMapList()
                        
                        self.menu_left.newRow(tag="load", horizontal=True, separator=False, locked=False)
                        self.menu_left.addButton(tag="butload", label="Load map", callback=self.selectMap)
                        self.menu_left.addButton(tag="butnext", label="Next", callback=lambda: self.selectMap(next_map=True), show=bool(self.map_list))
                        self.menu_left.addButton(tag="butgrid", label="[]", callback=lambda: self.map_window.show())

                        self.menu_left.newRow(tag="advload", separator=False, locked=False, advanced=True)
                        self.menu_left.addInput(tag="inpbin", label="Binning", value=1)

                        self.menu_left.newRow(tag="windows", horizontal=True, separator=False, locked=False)
                        self.menu_left.addButton(tag="butlamgui", label="Grid map", callback=lambda: utils.guiProcess("lamella"))
                        self.menu_left.addButton(tag="butflm", label="CLEM", callback=lambda: self.lm_window.show())

                        self.menu_left.newRow(tag="filters", separator=False, locked=True, advanced=False)
                        self.menu_left.addButton(tag="butfilter", label="Filter map", callback=self.filterMap)

                        self.menu_left.newRow(tag="reacquire", separator=False, locked=True, advanced=True)
                        self.menu_left.addButton(tag="butreacq", label="Reacquire", callback=self.reacquireMap)

                        self.menu_left.newRow(tag="class_list", separator=False, locked=True)
                        self.menu_left.addText(tag="class_heading", value="\nClasses", color=gui.COLORS["heading"])
                        if self.model is not None:
                            for key in self.model.categories.keys():
                                self.menu_left.addCheckbox(tag=key, label=key, value=False)

                        self.menu_left.newRow(tag="class_buttons", horizontal=True, separator=False, locked=True)
                        self.menu_left.addButton(tag="clsmask", label="Create overlay", callback=self.loadOverlay)
                        self.menu_left.addButton(tag="clsapply", label="Apply", callback=self.applyClassSelection, user_data=[self.menu_left, None])

                        self.status = gui.StatusLine()

                    with dpg.table_cell(tag="tblplot"):
                        self.plot.makePlot(x_axis_label="x [µm]", y_axis_label="y [µm]", width=-1, height=-1, equal_aspects=True, no_menus=True, crosshairs=True, pan_button=dpg.mvMouseButton_Right, no_box_select=True)

                    with dpg.table_cell(tag="tblright"):
                        dpg.add_text(default_value="Target selection", tag="r1", color=gui.COLORS["heading"])

                        if self.mic_params is not None:
                            self.menu_right = Menu()

                            self.menu_right.newRow(tag="settings1", separator=False)
                            self.menu_right.addText(tag="target_list_label", value="Target classes:")
                            self.menu_right.addInput(tag="target_list", label="", value=",".join(self.tgt_params.target_list), width=-1)
                            dpg.bind_item_handler_registry(self.menu_right.all_elements["target_list"], "class_input_handler")

                            self.menu_right.newRow(tag="settings2", separator=False)
                            self.menu_right.addText(tag="avoid_list_label", value="Avoid classes:")
                            self.menu_right.addInput(tag="avoid_list", label="", value=",".join(self.tgt_params.penalty_list), width=-1)
                            dpg.bind_item_handler_registry(self.menu_right.all_elements["avoid_list"], "class_input_handler")

                            self.menu_right.newRow(tag="settings3", separator=False)
                            self.menu_right.addText(tag="settings_heading", value="\nTargeting options:")
                            #self.menu_right.addInput(tag="target_score_threshold", label="Score threshold", value=self.tgt_params.threshold)
                            self.menu_right.addSlider(tag="target_score_threshold", label="Score threshold", value=float(self.tgt_params.threshold), value_range=[0, 1], width=75)
                            #self.menu_right.addInput(tag="penalty_weight", label="Penalty weight", value=self.tgt_params.penalty)
                            self.menu_right.addSlider(tag="penalty_weight", label="Penalty weight", value=float(self.tgt_params.penalty), value_range=[0, 1], width=75)
                            #self.menu_right.addInput(tag="max_tilt", label="Max. tilt angle", value=self.tgt_params.max_tilt, callback=self.updateMaxTilt)
                            self.menu_right.addSlider(tag="max_tilt", label="Max. tilt angle", value=int(self.tgt_params.max_tilt), value_range=[0, 80], width=75, callback=self.updateMaxTilt)
                            #self.menu_right.addInput(tag="IS_limit", label="Image shift limit", value=self.mic_params.IS_limit)
                            self.menu_right.addSlider(tag="IS_limit", label="Image shift limit", value=int(self.mic_params.IS_limit), value_range=[5, 20], width=75)
                            self.menu_right.addCheckbox(tag="sparse_targets", label="Sparse targets", value=self.tgt_params.sparse)
                            self.menu_right.addCheckbox(tag="target_edge", label="Target edge", value=self.tgt_params.edge)
                            self.menu_right.addCheckbox(tag="extra_tracking", label="Extra tracking", value=self.tgt_params.sparse)

                            self.menu_right.newRow(tag="settings4", separator=False)
                            self.menu_right.addButton(tag="butselect", label="Auto select targets", callback=self.runTargetSelection)
                            self.menu_right.addText(tag="rsp1", value="")

                            self.menu_right.newRow(tag="areas", separator=False)
                            self.menu_right.addButton(tag="butsplit", label="Split target areas", callback=self.splitAreas)
                            self.menu_right.addButton(tag="butdist", label="Redistribute targets", callback=self.splitTargets)
                            self.menu_right.addButton(tag="butmerge", label="Merge target areas", callback=self.mergeAreas)

                            self.menu_right.newRow(tag="delete", separator=False)
                            self.menu_right.addButton(tag="butdelete", label="Delete targets", callback=self.clearTargets)
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

                            self.menu_right.newRow(tag="acquisition6", horizontal=True, separator=False)
                            self.menu_right.addCheckbox(tag="acq_save", label="Override script settings")

                            self.menu_right.newRow(tag="buttons1", separator=False)
                            self.menu_right.addText(tag="save_heading", value="\n\nSave targets", color=gui.COLORS["heading"])

                            self.menu_right.newRow(tag="buttons2", separator=False)
                            self.menu_right.addButton(tag="butsave", label="Save", callback=self.saveTargets, show=False)
                            self.menu_right.addButton(tag="butins", label="Confirm inspection", callback=self.markInspected, theme="large_btn_theme")
                            self.menu_right.addButton(tag="butins", label="Confirm all maps", callback=self.markAllInspected)

                            self.menu_right.hide()

                            dpg.add_text(tag="inspected", default_value="\nTargets for this map were\nalready marked as inspected.\n(Editing disabled)", color=gui.COLORS["heading"], show=False)
                        else:
                            dpg.add_text(tag="rerr", default_value="No microscope parameters\nfile found!", color=gui.COLORS["error"])


            # Create tooltips
            with dpg.tooltip("l1", delay=0.5):
                dpg.add_text("Select and load a map generated by SPACEtomo.")
            with dpg.tooltip(self.menu_left.all_elements["class_heading"], delay=0.5):
                dpg.add_text("Choose a class to be displayed as overlay.")
            with dpg.tooltip(self.menu_left.all_elements["butreacq"], delay=0.5):
                dpg.add_text("This will require you to restart the SPACEtomo run!")
            with dpg.tooltip(self.menu_left.all_elements["clsmask"], delay=0.5):
                dpg.add_text("Create overlay of selected classes. (This can take a few seconds.)")
            with dpg.tooltip(self.menu_left.all_elements["clsapply"], delay=0.5):
                dpg.add_text("Apply selected classes to target list.")

            with dpg.tooltip("tblplot", delay=0.5, hide_on_activity=True):
                dpg.add_text(default_value="", color=gui.COLORS["heading"], tag="tt_heading")
                dpg.add_text(default_value="", tag="tt_text")
            dpg.bind_item_handler_registry("plot", "point_tooltip_handler")

            if self.mic_params is not None:
                with dpg.tooltip("r1", delay=0.5):
                    dpg.add_text("Select targets based on segmentation.")
                with dpg.tooltip(self.menu_right.all_elements["target_list"], delay=0.5):
                    dpg.add_text("List of target classes (comma separated). For exhaustive acquisition use \"lamella\".")
                with dpg.tooltip(self.menu_right.all_elements["avoid_list"], delay=0.5):
                    dpg.add_text("List of classes to avoid (comma separated).")
                with dpg.tooltip(self.menu_right.all_elements["target_score_threshold"], delay=0.5):
                    dpg.add_text("Score threshold [0-1] below targets will be excluded.")
                with dpg.tooltip(self.menu_right.all_elements["penalty_weight"], delay=0.5):
                    dpg.add_text("Relative weight of avoided classes to target classes.")
                with dpg.tooltip(self.menu_right.all_elements["max_tilt"], delay=0.5):
                    dpg.add_text("Maximum tilt angle [degrees] to consider electron beam exposure.")
                with dpg.tooltip(self.menu_right.all_elements["IS_limit"], delay=0.5):
                    dpg.add_text("Image shift limit [µm] for PACEtomo acquisition. If targets are further apart, target area will be split.")
                with dpg.tooltip(self.menu_right.all_elements["sparse_targets"], delay=0.5):
                    dpg.add_text("Target positions will be initialized only on target classes and refined independently (instead of grid based target target setup to minimize exposure overlap).")
                with dpg.tooltip(self.menu_right.all_elements["target_edge"], delay=0.5):
                    dpg.add_text("Targets will be centered on edge of segmented target instead of maximising coverage.")
                with dpg.tooltip(self.menu_right.all_elements["extra_tracking"], delay=0.5):
                    dpg.add_text("An extra target will be placed centrally for tracking.")

                with dpg.tooltip(self.menu_right.all_elements["butdelete"], delay=0.5):
                    dpg.add_text("Delete all targets and setup from scratch. (This cannot be undone!)")
                with dpg.tooltip(self.menu_right.all_elements["butins"], delay=0.5):
                    dpg.add_text("Mark targets as inspected. (No more changes can be made.)")

            # Show logo
            dpg.add_image("logo", pos=(10, dpg.get_viewport_client_height() - 40 - self.logo_dims[0]), tag="logo_img")
            dpg.add_text(default_value="SPACEtomo", pos=(10 + self.logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() - 40 - self.logo_dims[0] / 2), tag="logo_text")
            dpg.add_text(default_value="v" + __version__, pos=(10 + self.logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() - 27 - self.logo_dims[0] / 2), tag="version_text")

        # Make window for map thumbnails
        self.map_window = MapWindow(self.cur_dir, self.map_name, self.map_list, self.map_list_tgtnum, self.selectMap, self.use_thumbnails, self.thumbnail_size)
        self.map_window.makeMapTable()

        # LM window
        self.lm_window = FlmWindow(self.plot)

        # Make window for target editing menu
        with dpg.window(label="Target", tag="win_tgt", no_scrollbar=True, no_scroll_with_mouse=True, popup=True, show=False):
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

        # Make window for class list selection
        with dpg.window(label="Selection", tag="win_sel", no_scrollbar=True, no_scroll_with_mouse=True, popup=True, show=False):
            self.menu_sel = Menu()
            self.menu_sel.newRow(tag="heading", separator=False, locked=False)
            self.menu_sel.addText(tag="heading", value="Classes", color=gui.COLORS["heading"])
            if self.model is not None:
                for k, key in enumerate(self.model.categories.keys()):
                    if k % 3 == 0:
                        self.menu_sel.newRow(tag=str(k), horizontal=True, separator=False, locked=False)
                    self.menu_sel.addCheckbox(tag=key, label=key, value=False, callback=self.applyClassSelection)

        dpg.bind_theme("global_theme")

        dpg.set_exit_callback(self.askForSave)

        dpg.set_primary_window("GUI", True)
        dpg.show_viewport()

        # Render loop
        next_update = time.time() + 30
        while dpg.is_dearpygui_running():

            # Recheck folder for segmentation every minute
            now = time.time()
            if now > next_update:
                next_update = now + 30
                # Check if new maps were finished
                if self.updateMapList():
                    self.map_window.update(self.map_name, self.map_list, self.map_list_tgtnum)
                # Check if new target selection was finished
                self.checkPointFiles()

            dpg.render_dearpygui_frame()
