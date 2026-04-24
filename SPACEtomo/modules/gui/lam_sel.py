#!/usr/bin/env python
# ===================================================================
# ScriptName:   gui_lam
# Purpose:      User interface for training SPACEtomo segmentation models using nnU-Netv2
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/04/26
# Revision:     v1.3
# Last Change:  2025/04/12: added InfoBoxManager to stack popups, fixed tooltips of hidden buttons showing up
#               2025/03/13: renamed lamella to ROI
#               2025/03/06: added grid boxes, added class legend
#               2025/01/30: fixed lamella list and find maps button not shown when inspected
#               2024/12/20: added box drag cursors
#               2024/09/02: fixed lock editing after inspected, fixed save button, added askForSave on exit
#               2024/08/09: changed to pathlib, adjusted to Plot class changes
#               2042/07/25: added clipping of plot boxes to map bounds
#               2024/07/18: reworked GUI to class
#               2024/06/07: added lamella table row highlighting, added save coords in YOLO format
#               2024/06/05: introduced Menu class, added shortcuts for most buttons
#               2024/06/04: added pix size check for export, added rescaling of mrc file
#               2024/06/03: added include empty tiles option, added save json option
#               2024/05/31: added status line, fixed tiles bounds, added next button, added hide/showElements
#               2024/05/30: added rescaling, saving as png, fixed export tiles
#               2024/05/29: outsourced plot clearing
#               2024/05/23: added lamella labels on plot, added padding when exporting edge tiles, fixed plotTiles
#               2024/05/10: added reading and writing of YOLO format labels, added saving of tiles with labels, added map class
#               2024/05/02: added lamella color from config, added reset zoom, delegated plot generation, call lamella detection
#               2024/04/30: added lamella focus buttons, outline scaling, list sorting
#               2024/04/26: Copy most of SPACEtomo_tgt
# ===================================================================

import os
import sys
os.environ["__GLVND_DISALLOW_PATCHING"] = "1"           # helps to minimize Segmentation fault crashes on Linux when deleting textures
import time
from pathlib import Path
try:
    import dearpygui.dearpygui as dpg
except:
    print("ERROR: DearPyGUI module not installed! If you cannot install it, please run the GUI from an external machine.")
    sys.exit()
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

import SPACEtomo.config as config
from SPACEtomo.modules.mod_wg import WGModel, Boxes
from SPACEtomo.modules.gui import gui
from SPACEtomo.modules.gui.menu import Menu
from SPACEtomo.modules.gui.info import InfoBoxManager, InfoBox, StatusLine, saveSnapshot
from SPACEtomo.modules.gui.plot import Plot, PlotBox
from SPACEtomo.modules.gui.map import MMap
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log
from SPACEtomo import __version__


# Check if IMOD is available (needed for mrc file import)
if "IMOD" in os.environ["PATH"]:
    IMOD = True
else:
    IMOD = False

### FUNCTIONS ###
class RegionBox(PlotBox):
    """Class adding ROI information to PlotBox."""

    def __init__(self, coords, parent, color=gui.COLORS["error"], roi_info=None) -> None:
        super().__init__(coords, parent, color)

        self.label = None
        if roi_info is not None:
            self.addInfo(roi_info)

    def addInfo(self, roi_info):
        if self.label is None:
            self.label = roi_info["label"]
        self.cat = roi_info["cat"]
        self.prob = roi_info["prob"]
        self.coords = (self.p1 + self.p2) / 2

class GridGUI:

    def mouseClick(self, sender, app_data):
        """Handles all mouse clicks on plot."""

        mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # Check if click needs to be processed
        if not dpg.is_item_hovered(self.plot.plot) or not self.plot.withinBounds(mouse_coords) or self.inspected:
            return

        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and (dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)) and len(self.plot.img) > 0:
            # Clip box dims to map bounds
            mouse_coords = np.array((np.clip(mouse_coords[0], 0, self.loaded_map.img.shape[0] * self.loaded_map.pix_size / 1000), np.clip(mouse_coords[1], 0, self.loaded_map.img.shape[1] * self.loaded_map.pix_size / 1000)))

            if len(self.plot.boxes) > 0:
                for b, box in enumerate(self.plot.boxes):
                    if box.within(mouse_coords, margin=5):
                        self.mouse_box = box
                        self.mouse_box.updateP1(mouse_coords)
                        self.plot.boxes.pop(b)
                        break
            if self.mouse_box is None:
                self.mouse_box = RegionBox(mouse_coords, parent=self.plot.plot)
                self.scaleOutlines()

        elif dpg.is_mouse_button_down(dpg.mvMouseButton_Right) and len(self.plot.boxes) > 0:
            # Call ROI menu by right clicking
            for b, box in enumerate(self.plot.boxes):
                dpg.unhighlight_table_row("tblroi", b)
                if box.within(mouse_coords):
                    # Highlight selected ROI in table
                    dpg.highlight_table_row("tblroi", b, gui.COLORS["subtle"])
                    self.selected_roi = b
                    # Show ROI window
                    self.showROIInfo(sender, app_data, b)

        elif dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and len(self.plot.boxes) > 0:
            # Start dragging to update ROI box
            for b, box in enumerate(self.plot.boxes):
                dpg.unhighlight_table_row("tblroi", b)
                if box.within(mouse_coords, margin=min(box.width, box.height) * 0.1):
                    # Highlight selected ROI in table
                    dpg.highlight_table_row("tblroi", b, gui.COLORS["subtle"])
                    self.selected_roi = b
                    self.drag_start = mouse_coords
                    box.startUpdate(mouse_coords)

                    # Show drag cursor
                    # NOTE: Had to switch from showing icon as image at abs position to drawing it on plot because global coordinates were not consistent between MacOS and Windows.

                    cursor_texture = None
                    if all(mode in box.update_mode for mode in ["top", "bottom", "left", "right"]):
                        cursor_texture = gui.makeIconShift()
                    elif all(mode in box.update_mode for mode in ["top", "left"]) or all(mode in box.update_mode for mode in ["bottom", "right"]):
                        cursor_texture = gui.makeIconShiftULDR()
                    elif all(mode in box.update_mode for mode in ["top", "right"]) or all(mode in box.update_mode for mode in ["bottom", "left"]):
                        cursor_texture = gui.makeIconShiftURDL()
                    elif any(mode in box.update_mode for mode in ["left", "right"]):
                        cursor_texture = gui.makeIconShiftLR()
                    elif any(mode in box.update_mode for mode in ["top", "bottom"]):
                        cursor_texture = gui.makeIconShiftUD()

                    if cursor_texture:
                        # Calculate cursor scale
                        cursor_size = 20 # px
                        plot_size_px = dpg.get_item_rect_size(self.plot.plot) # px
                        x_axis_range = dpg.get_axis_limits(self.plot.x_axis) # microns
                        scale = (max(x_axis_range) - min(x_axis_range)) / plot_size_px[0] * cursor_size / 2

                        # Add cursor as plot overlay
                        self.plot.addOverlay(cursor_texture, bounds=((mouse_coords[0] - scale, mouse_coords[0] + scale), (mouse_coords[1] - scale, mouse_coords[1] + scale)), label="temp_cursor")
                    break

            # If no box was selected, check grid boxes
            if hasattr(self.plot, "boxes_grid") and self.plot.boxes_grid:
                for b, box in enumerate(self.plot.boxes_grid):
                    if box.within(mouse_coords, margin=min(box.width, box.height) * 0.1):
                        self.plot.boxes.append(box)
                        self.plot.boxes_grid.pop(b)

                        # Add ROI data
                        if len(self.plot.boxes) > 1:
                            index = int(self.plot.boxes[-2].label.split("L")[-1]) + 1
                        else:
                            index = 1
                        roi_info = {"label": "L" + str(index).zfill(2), "cat": config.WG_model_categories.index("good"), "prob": 1}
                        self.plot.boxes[-1].addInfo(roi_info)
                        self.plot.boxes[-1].updateColor(config.WG_model_gui_colors[config.WG_model_categories.index("good")])
                        self.plot.boxes[-1].drawLabel(self.plot.boxes[-1].label)
                        self.plot.sortList("boxes", "label")

                        # Plot grid of other boxes
                        #self.plotGrid()

                        # Update table
                        self.makeROITable()

                        dpg.show_item(self.menu.all_elements["btn_save"])
                        break

    def mouseDrag(self, sender, app_data):
        """Handles dragging of boxes."""

        mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # Update box boundary when drawing
        if self.mouse_box is not None:
            self.mouse_box.updateP2(mouse_coords)
        
        # Update box drag position
        elif self.drag_start is not None:
            self.plot.boxes[self.selected_roi].update(mouse_coords)

            # Update drag cursor
            cursor_item = self.plot.overlays[utils.findIndex(self.plot.overlays, "label", "temp_cursor")]["plot"]
            if dpg.does_item_exist(cursor_item):
                # Calculate cursor scale
                cursor_size = 20 # px
                plot_size_px = dpg.get_item_rect_size(self.plot.plot) # px
                x_axis_range = dpg.get_axis_limits(self.plot.x_axis) # microns
                scale = (max(x_axis_range) - min(x_axis_range)) / plot_size_px[0] * cursor_size / 2

                # Change bounds of cursor
                dpg.configure_item(cursor_item, bounds_min=mouse_coords - scale, bounds_max=mouse_coords + scale)

    def mouseRelease(self, sender, app_data):
        """Handles updates on mouse release."""

        mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # End drawing of ROI box
        if self.mouse_box is not None:
            # Clip box dims to map bounds
            mouse_coords = np.array((np.clip(mouse_coords[0], 0, self.loaded_map.img.shape[0] * self.loaded_map.pix_size / 1000), np.clip(mouse_coords[1], 0, self.loaded_map.img.shape[1] * self.loaded_map.pix_size / 1000)))

            self.mouse_box.updateP2(mouse_coords)
            self.plot.boxes.append(self.mouse_box)
            self.mouse_box = None

            # Add ROI data
            if len(self.plot.boxes) > 1:
                index = int(self.plot.boxes[-2].label.split("L")[-1]) + 1
            else:
                index = 1
            roi_info = {"label": "L" + str(index).zfill(2), "cat": config.WG_model_categories.index("good"), "prob": 1}
            self.plot.boxes[-1].addInfo(roi_info)
            self.plot.boxes[-1].updateColor(config.WG_model_gui_colors[config.WG_model_categories.index("good")])
            self.plot.boxes[-1].drawLabel(self.plot.boxes[-1].label)
            self.plot.sortList("boxes", "label")

            # Plot grid of other boxes
            self.plotGrid()

            # Update table
            self.makeROITable()

            dpg.show_item(self.menu.all_elements["btn_save"])

        # End dragging of ROI box
        elif self.drag_start is not None:
            self.drag_start = None
            self.plot.boxes[self.selected_roi].endUpdate()
            dpg.show_item(self.menu.all_elements["btn_save"])

            # Reset drag cursor
            self.plot.clearOverlays(["temp_cursor"], delete_textures=False)

            # Update grid box dimensions
            self.plotGrid()

    def mouseScroll(self, sender, app_data):
        """Catches scroll on plot to scale outline of drawn boxes."""

        if dpg.is_item_hovered(self.plot.plot) and len(self.plot.boxes) > 0:
            self.scaleOutlines()

    def loadMap(self, sender, app_data, user_data, tile_size=None, stitched=None, quantile=None):
        """Loads map file and detected lamellae or ROIs from file."""

        # Remove auto load trigger if exist
        if dpg.does_item_exist("auto_load_map"):
            dpg.delete_item("auto_load_map")

        # Get file selection
        selected_files = sorted([file for file in app_data["selections"].values()])
        file_path = Path(selected_files[0])

        # Have to manually overwrite args, because buttons seems to fill all args with None
        if stitched is None:
            stitched = False
        if quantile is None:
            quantile = dpg.get_value(self.menu.all_elements["inp_quantile"])

        #if not file_path.endswith((".png", ".mrc", ".map")):
        if not file_path.suffix.lower() in [".png", ".mrc", ".map"]:
            log("ERROR: Can only read .png and .mrc files!")
            InfoBoxManager.push(InfoBox("ERROR", "Can only read .png and .mrc files!"))
            return
        
        # Update GUI
        self.menu.hide()
        self.plot.updateLabel(file_path.name + " loading...")
        self.status.update("Loading map...")

        if tile_size is None:
            tile_size = (config.WG_model_sidelen, config.WG_model_sidelen)

        # Reset plot
        self.plot.clearAll()

        # Reset grid boxes if present
        if hasattr(self.plot, "boxes_grid") and self.plot.boxes_grid:
            for box in self.plot.boxes_grid:
                box.remove()
            self.plot.boxes_grid = []

        pix_size_png = config.WG_model_pix_size # nm/px, will be overwritten by mrc header pixel size when loading map
        dpg.set_item_label(self.plot.x_axis, "x [µm]")
        dpg.set_item_label(self.plot.y_axis, "y [µm]")

        # Load map file
        self.loaded_map = MMap(file_path, pix_size_png, stitched, tile_size, quantile=quantile, status=self.status)

        # Determine binning
        self.loaded_map.checkBinning()

        self.status.update("Plotting map...", box=True)

        map_texture = self.loaded_map.getTexture()
        dims_plot = np.flip(np.array(self.loaded_map.img.shape) * self.loaded_map.pix_size / 1000)

        # Load map to plot
        self.plot.addImg(map_texture, [[0, dims_plot[0]], [0, dims_plot[1]]], self.loaded_map.binning, label="map")

        # Fit axes
        self.plot.resetZoom()
        self.plot.updateLabel(self.loaded_map.file.name)

        self.status.update("Loading regions of interest...")

        # Load ROIs from file
        self.loadBboxes_yolo(fallback=self.loadBboxes_json)

        # Show ROI list
        self.selected_roi = None
        self.makeROITable()

        # Show tiles for export
        self.plotTiles()

        # Show legend
        for c, cat in enumerate(config.WG_model_categories):
            self.plot.addSeries([], [], label=cat, theme=f"cat_theme{c}")

        # Update GUI
        self.menu.unlockRows(["roilist", "detect", "inspect", "rescale"])
        if self.loaded_map.file.suffix.lower() in [".mrc", ".map"]:
            self.menu.unlockRows(["mrcscale"])
        dpg.set_value(self.menu.all_elements["inp_pixsize"], self.loaded_map.pix_size)

        # Check if map was already inspected
        #if os.path.exists(os.path.splitext(self.loaded_map.file)[0] + "_inspected.txt"):
        if (self.loaded_map.file.parent / (self.loaded_map.file.stem + "_inspected.txt")).exists():
            self.markInspected()
            return

        self.menu.show()
        self.status.update()

        # Check if more files are available for load next button
        # Add file_name to user_data in case rescale reloads map in png (sender also overwrites user_data because it was manually loaded)
        btn_next = self.menu.all_elements["btn_next"]
        if dpg.get_item_user_data(btn_next) is None or sender is not None:
            dpg.configure_item(btn_next, user_data=self.loaded_map.file)
            dpg.configure_item("key_next", user_data=self.loaded_map.file)
        else:
            # If new map has same extension as old map, update user_data, but keep same if map was just converted from mrc to png
            if Path(dpg.get_item_user_data(btn_next)).suffix == self.loaded_map.file.suffix:
                dpg.configure_item(btn_next, user_data=self.loaded_map.file)
                dpg.configure_item("key_next", user_data=self.loaded_map.file)

        #file_list = sorted(glob.glob(os.path.join(os.path.dirname(dpg.get_item_user_data(btn_next)), "*" + os.path.splitext(dpg.get_item_user_data(btn_next))[-1])))
        next_file = Path(dpg.get_item_user_data(btn_next))
        file_list = sorted(next_file.parent.glob("*" + next_file.suffix))

        cur_id = file_list.index(dpg.get_item_user_data(btn_next))
        if cur_id + 1 < len(file_list):
            dpg.show_item(btn_next)


    def loadNextMap(self, sender, app_data, user_data):
        """Determines next map file and loads it."""

        if self.loaded_map is None:
            return
        
        # Load extension from button if possible
        if user_data is not None:
            extension = Path(user_data).suffix
            map_file = user_data
        else:
            extension = self.loaded_map.file.suffix
            map_file = self.loaded_map.file

        # Make list of files with same extension
        #file_list = sorted(glob.glob(os.path.join(os.path.dirname(self.loaded_map.file), "*" + extension)))
        file_list = sorted(self.loaded_map.file.parent.glob("*" + extension))

        # Get ID of current map in list
        cur_id = file_list.index(map_file)

        # Load next map and update button
        if cur_id + 1 < len(file_list):
            self.loadMap(None, {"selections": {file_list[cur_id + 1].name: file_list[cur_id + 1]}}, None)
        if cur_id + 2 >= len(file_list):
            dpg.hide_item(self.menu.all_elements["btn_next"])

    def rescaleMap(self):
        """Rescales map to new pixel size and saves it."""

        # Update GUI
        self.menu.hide()

        # Get pix size input
        rescale_pix_size = dpg.get_value(self.menu.all_elements["inp_pixsize"]) # nm/px

        # Rescale map
        log("Rescaling map...")
        old_dims = np.array(self.loaded_map.img.shape)
        self.loaded_map.rescale(rescale_pix_size, padding=(config.WG_model_sidelen, config.WG_model_sidelen))

        # Save only if any changes (rescaling or padding) were made
        if not np.all(self.loaded_map.img.shape == old_dims):
            self.status.update("Saving map...")
            # Save rescaled map
            file_path = self.loaded_map.file.parent / (self.loaded_map.file.stem.split("_wg")[0] + "_" + str(int(self.loaded_map.pix_size)) + "nmpx_wg.png")
            log(f"Saving map as {file_path.name}...")
            Image.fromarray(np.uint8(self.loaded_map.img * 255)).save(file_path)

            # Load rescaled map
            self.loadMap(None, {"selections": {file_path.name: file_path}}, None, tile_size=self.loaded_map.tile_size, stitched=self.loaded_map.stitched)

    def loadBboxes_json(self, fallback=None):
        """Loads bounding boxes from json file."""

        bbox_file = self.loaded_map.file.parent / (self.loaded_map.file.stem + "_boxes.json")
        if not bbox_file.exists():
            log("No regions of interest found.")
            # Check fallback
            if fallback is not None:
                log("Trying fallback...")
                fallback()
        else:
            # Read bboxes
            bboxes = Boxes(bbox_file)

            dims_img = np.flip(self.loaded_map.img.shape)

            prev_box_num = len(self.plot.boxes)
            for b, bbox in enumerate(bboxes.boxes):        
                p1 = np.array([bbox.x_min, dims_img[1] - bbox.y_min]) * self.loaded_map.pix_size / 1000
                p2 = np.array([bbox.x_max, dims_img[1] - bbox.y_max]) * self.loaded_map.pix_size / 1000

                self.plot.boxes.append(RegionBox(p1, parent=self.plot.plot, color=config.WG_model_gui_colors[bbox.cat]))
                self.plot.boxes[-1].updateP2(p2)

                # Add ROI data
                roi_info = {"label": "L" + str(prev_box_num + b + 1).zfill(2), "cat": bbox.cat, "prob": bbox.prob}
                self.plot.boxes[-1].addInfo(roi_info)
                self.plot.boxes[-1].drawLabel(self.plot.boxes[-1].label)

            log(f"{len(self.plot.boxes)} regions of interest found.")
            self.menu.unlockRows(["export", "inspect"])

    def loadBboxes_yolo(self, fallback=None):
        """Loads bounding boxes from YOLO label file."""

        # Check for label file in same folder
        label_file = self.loaded_map.file.parent / (self.loaded_map.file.stem + ".txt")
        if not label_file.exists():
            # Also check for label file in typical YOLO folder structure
            label_file = self.loaded_map.file.parent.parent / "labels" / (self.loaded_map.file.stem + ".txt")
            if not label_file.exists():
                log("No regions of interest found.")
                # Check fallback
                if fallback is not None:
                    log("Trying fallback...")
                    fallback()
                    return

        # Read labels
        with open(label_file, "r") as f:
            lines = f.readlines()
        labels = []
        for label in lines:
            labels.append([float(val) for val in label.split()])

        dims_plot = dpg.get_item_configuration(self.plot.img[utils.findIndex(self.plot.img, "label", "map")]["plot"])["bounds_max"]
        #print(dims_plot)

        prev_box_num = len(self.plot.boxes)
        for l, label in enumerate(labels):
            p1 = np.array([label[1] - label[3] / 2, 1 - label[2] - label[4] / 2]) * dims_plot
            p2 = np.array([label[1] + label[3] / 2, 1 - label[2] + label[4] / 2]) * dims_plot
            
            self.plot.boxes.append(RegionBox(p1, parent=self.plot.plot, color=config.WG_model_gui_colors[int(label[0])]))
            self.plot.boxes[-1].updateP2(p2)

            # Add ROI data
            roi_info = {"label": "L" + str(prev_box_num + l + 1).zfill(2), "cat": int(label[0]), "prob": 1}
            self.plot.boxes[-1].addInfo(roi_info)
            self.plot.boxes[-1].drawLabel(self.plot.boxes[-1].label)

        log(f"{len(self.plot.boxes)} regions of interest found.")
        self.menu.unlockRows(["export"])
        
    def makeROITable(self):
        """Generates ROI list table."""

        # Delete table
        if dpg.does_item_exist("tblroi"):
            dpg.delete_item("tblroi")
        if len(self.plot.boxes) > 0:
            # Make new table
            with dpg.table(label="Regions of Interest", tag="tblroi", parent=self.menu.all_rows["roilist"], height=400, scrollY=True, policy=dpg.mvTable_SizingFixedFit):
                dpg.add_table_column(label="Name")
                dpg.add_table_column(label="Class")
                dpg.add_table_column(label="")
                #dpg.add_table_column(label="Coords")
                #dpg.add_table_column(label="Area")

                for b, box in enumerate(self.plot.boxes):
                    with dpg.table_row():
                        with dpg.table_cell():
                            dpg.add_button(label=box.label, callback=self.focusROI, user_data=box)
                        with dpg.table_cell():
                            dpg.add_text(default_value=config.WG_model_categories[box.cat] + " " + str(round(box.prob, 2)), color=config.WG_model_gui_colors[box.cat])
                        with dpg.table_cell():
                            if not self.inspected:
                                dpg.add_image_button(gui.makeIconEdit(), callback=self.showROIInfo, user_data=b, tag=f"btn_roi_edit_{b}")
                        #with dpg.table_cell():
                        #    dpg.add_text(default_value=str(round(lamella["area"])))

                    #dpg.bind_item_handler_registry("tbllamcat" + lamella.label, "tbl_click_handler")
            
            with dpg.tooltip("tblroi", delay=0.5, hide_on_activity=True):
                dpg.add_text("List of ROIs in map.\nClick on name to zoom.")

    def plotTiles(self):
        """Plots tile bounds on map."""

        self.plot.clearSeries()
        x_vals = np.array(self.loaded_map.tile_bounds[0]) * self.loaded_map.pix_size / 1000 #scaling
        try:
            self.plot.series.append({"label": "tiles_v", "plot": dpg.add_inf_line_series(x_vals, parent=self.plot.x_axis, show=False)})
        except AttributeError:      # Backward compatibility with dearpyguy<2.0
            log(f"WARNING: Consider updating DearPyGUI to version >=2.0!")
            self.plot.series.append({"label": "tiles_v", "plot": dpg.add_vline_series(x_vals, parent=self.plot.x_axis, show=False)})
        dpg.bind_item_theme(self.plot.series[-1]["plot"], "plot_tiletheme")
        
        y_vals = (-1 * np.array(self.loaded_map.tile_bounds[1]) + self.loaded_map.img.shape[0]) * self.loaded_map.pix_size / 1000 #scaling
        try:
            self.plot.series.append({"label": "tiles_h", "plot": dpg.add_inf_line_series(y_vals, horizontal=True, parent=self.plot.x_axis, show=False)})
        except AttributeError:      # Backward compatibility with dearpyguy<2.0
            self.plot.series.append({"label": "tiles_h", "plot": dpg.add_hline_series(y_vals, parent=self.plot.x_axis, show=False)})
        dpg.bind_item_theme(self.plot.series[-1]["plot"], "plot_tiletheme")

    def plotGrid(self):
        """Plots grid pattern on map."""

        # Reset grid boxes if present
        if hasattr(self.plot, "boxes_grid") and self.plot.boxes_grid:
            for box in self.plot.boxes_grid:
                box.remove()
        self.plot.boxes_grid = []

        # Show detected grid pattern
        if "grid_vectors" in self.loaded_map.meta_data.keys() and self.plot.boxes:
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
            center_coords = self.plot.boxes[0].center
            # Find the closest point
            closest_point = points[np.argmin(np.linalg.norm(points - center_coords, axis=1))]
            # Calculate the offset
            offset = center_coords - closest_point
            # Shift points
            points += offset

            # Remove points outside of map
            points = points[np.all((points >= 0.05 * self.loaded_map.dims_microns) & (points < 0.95 * self.loaded_map.dims_microns), axis=1)]

            # Remove points in same location as existing boxes
            for box in self.plot.boxes:
                points = np.array([point for point in points if np.linalg.norm(point - box.center) > np.linalg.norm(grid_vectors[0]) / 2])
                    
            #self.plot.addSeries(x_vals=points[:, 0], y_vals=points[:, 1], label="grid")

            # Plot boxes
            for point in points:
                p1 = point - np.array([self.plot.boxes[0].width, -self.plot.boxes[0].height]) / 2
                p2 = point + np.array([self.plot.boxes[0].width, -self.plot.boxes[0].height]) / 2
                self.plot.boxes_grid.append(RegionBox(p1, parent=self.plot.plot, color=gui.COLORS["subtle"]))
                self.plot.boxes_grid[-1].updateP2(p2)

            self.scaleOutlines()

            # Show toggle grid button
            dpg.bind_item_theme(self.menu_icon.all_elements["butgrid"], "active_btn_theme")
            self.menu_icon.showElements(["butgrid"])

    def toggleTiles(self):
        """Toggles showing tile bounds on map."""

        if len(self.plot.series) > 0:
            for series in self.plot.series:
                if dpg.is_item_shown(series["plot"]):
                    dpg.hide_item(series["plot"])
                else:
                    dpg.show_item(series["plot"])

    def toggleGrid(self):
        """Toggles showing grid pattern on map."""

        if hasattr(self.plot, "boxes_grid") and self.plot.boxes_grid:
            dpg.bind_item_theme(self.menu_icon.all_elements["butgrid"], None)
            for box in self.plot.boxes_grid:
                box.remove()
            self.plot.boxes_grid = []
        else:
            self.plotGrid()

    def toggleAdvanced(self):
        """Toggles showing advanced menu options."""

        self.menu.toggleAdvanced()

    def showROIInfo(self, sender, app_data, user_data, update_pos=True):
        """Shows info window for ROI."""

        # Focus plot on ROI when called from table
        if app_data is None:
            self.focusROI(sender, app_data, self.plot.boxes[user_data])
        
        # Get mouse position for info window
        mouse_coords_global = dpg.get_mouse_pos(local=False)

        dpg.set_value("roi_label", self.plot.boxes[user_data].label)
        dpg.set_value("roi_cat", config.WG_model_categories[self.plot.boxes[user_data].cat])
        dpg.configure_item("roi_cat", user_data=user_data)
        dpg.configure_item("roi_btndel", user_data=user_data)
        dpg.configure_item("roi_btnup", user_data=user_data)
        dpg.configure_item("roi_btndown", user_data=user_data)
        if update_pos:
            dpg.set_item_pos("ROI", mouse_coords_global)
        dpg.show_item("ROI")

    def deleteROI(self, sender, app_data, user_data):
        """Deletes ROI from list and plot."""

        log(f"Deleted ROI: {self.plot.boxes[user_data].label} at {np.round((self.plot.boxes[user_data].p1 + self.plot.boxes[user_data].p2) / 2)}")
        self.plot.boxes[user_data].remove()
        self.plot.boxes.pop(user_data)
        # Reset ROI menu
        dpg.hide_item("ROI")
        dpg.configure_item("roi_cat", user_data=None)
        self.selected_roi -= 1
        # Update table
        self.makeROITable()
        # Replot grid of boxes
        self.plotGrid()
        dpg.show_item(self.menu.all_elements["btn_save"])

    def catROI(self, sender, app_data, user_data):
        """Re-categorizes ROI."""

        # Deal with call from shortcuts
        if user_data is None:
            user_data = dpg.get_item_user_data("roi_cat")
        if user_data is not None:
            self.plot.boxes[user_data].cat = config.WG_model_categories.index(app_data)
            self.plot.boxes[user_data].prob = 1
            self.plot.boxes[user_data].updateColor(config.WG_model_gui_colors[config.WG_model_categories.index(app_data)])
            # Reset ROI menu
            dpg.hide_item("ROI")
            dpg.configure_item("roi_cat", user_data=None)
            # Update table
            self.makeROITable()
            dpg.show_item(self.menu.all_elements["btn_save"])

    def reorderROIUp(self, sender, app_data, user_data):
        """Moves ROI up in list and renames it."""

        if user_data is not None:
            box_id = user_data
            new_box_id = user_data - 1
            if new_box_id < 0: 
                new_box_id = len(self.plot.boxes) - 1

            self.plot.boxes[new_box_id], self.plot.boxes[box_id] = self.plot.boxes[box_id], self.plot.boxes[new_box_id]
            self.relabelROIs()
            self.makeROITable()
            self.showROIInfo(None, None, user_data=new_box_id, update_pos=False)
            dpg.show_item(self.menu.all_elements["btn_save"])

    def reorderROIDown(self, sender, app_data, user_data):
        """Moves ROI down in list and renames it."""

        if user_data is not None:
            box_id = user_data
            new_box_id = user_data + 1
            if new_box_id >= len(self.plot.boxes): 
                new_box_id = 0

            self.plot.boxes[new_box_id], self.plot.boxes[box_id] = self.plot.boxes[box_id], self.plot.boxes[new_box_id]
            self.relabelROIs()
            self.makeROITable()
            self.showROIInfo(None, None, user_data=new_box_id, update_pos=False)
            dpg.show_item(self.menu.all_elements["btn_save"])

    def relabelROIs(self):
        """Relabels ROIs by order."""

        for b, box in enumerate(self.plot.boxes):
            box.label = "L" + str(b + 1).zfill(2)
            box.drawLabel(box.label)

    def saveROIs(self, sender, app_data, user_data, map_file=None, map_pix_size=None):
        """Saves detected ROIs to file."""

        if map_file is None:
            map_file = self.loaded_map.file
        if map_pix_size is None:
            map_pix_size = self.loaded_map.pix_size
            map_y = self.loaded_map.img.shape[0]
        else:
            map_y = int(self.loaded_map.img.shape[0] * self.loaded_map.pix_size / map_pix_size)

        bboxes = []
        for plot_box in self.plot.boxes:
            bbox = np.zeros(6)
            # Convert plot coords to pixel coords
            bbox[0], bbox[1] = plot_box.p1 / map_pix_size * 1000
            bbox[1] = map_y - bbox[1]
            bbox[2], bbox[3] = plot_box.p2 / map_pix_size * 1000
            bbox[3] = map_y - bbox[3]
            # Add class
            bbox[4] = plot_box.cat
            bbox[5] = plot_box.prob

            bboxes.append(bbox)
        bboxes = np.array(bboxes)

        # Save bboxes
        box_file = map_file.parent / (map_file.stem + "_boxes.json")
        bboxes = Boxes(bboxes, pix_size=map_pix_size, img_size=self.loaded_map.img.shape)
        bboxes.saveFile(box_file)

        #with open(os.path.splitext(map_file)[0].split("_wg")[0] + "_boxes.json", "w+") as f:
        #with open(box_file, "w+") as f:
        #    json.dump(bboxes, f, indent=4, default=utils.convertArray)
        
        log(f"Saved Regions of Interest to: {box_file}")
        dpg.hide_item(self.menu.all_elements["btn_save"])

    def saveLabel(self):        # in YOLO format
        """Saves detected ROIs in YOLO format to file."""

        # Get bboxes
        labels = []
        for box in self.plot.boxes:
            c = box.cat
            # Scale coords to relative img coords
            x, y = box.coords * 1000 / self.loaded_map.pix_size / np.array(self.loaded_map.img.shape)
            w, h = np.abs((box.p2 - box.p1)) * 1000 / self.loaded_map.pix_size / np.array(self.loaded_map.img.shape)
            # Invert y axis
            y = 1 - y
            labels.append([c, x, y, w, h])
        # Write txt file
        label_file = self.loaded_map.file.parent / (self.loaded_map.file.stem + ".txt")
        #with open(os.path.splitext(self.loaded_map.file)[0] + ".txt", "w+") as f: 
        with open(label_file, "w+") as f: 
            for label in labels:
                f.write(" ".join(str(val) for val in label) + "\n")   
        log(f"Saved Regions of Interest in YOLO format to: {label_file}") 

    def focusROI(self, sender, app_data, user_data):
        """Focuses plot on selected ROI."""

        plot_box = user_data
        self.selected_roi = self.plot.boxes.index(plot_box)

        # Highlight ROI in table
        for i in range(len(self.plot.boxes)):
            dpg.unhighlight_table_row("tblroi", i)
        dpg.highlight_table_row("tblroi", self.selected_roi, gui.COLORS["subtle"])

        # Get FOV for plot
        margin = 50

        min_x = min([plot_box.p1[0], plot_box.p2[0]]) - margin
        max_x = max([plot_box.p1[0], plot_box.p2[0]]) + margin
        min_y = min([plot_box.p1[1], plot_box.p2[1]]) - margin
        max_y = max([plot_box.p1[1], plot_box.p2[1]]) + margin

        # Get old limits to determine aspect ratio
        x_lim = dpg.get_axis_limits(self.plot.x_axis)
        y_lim = dpg.get_axis_limits(self.plot.y_axis)
        aspect = (x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0])

        # Lock axes to center ROI
        dpg.set_axis_limits(self.plot.y_axis, min_y, max_y)
        dpg.set_axis_limits(self.plot.x_axis, min_x + margin - (max_y - min_y) * aspect / 2, min_x + margin + (max_y - min_y) * aspect / 2)

        # Unlock the axes one frame later so axis limits are applied first
        dpg.set_frame_callback(dpg.get_frame_count() + 1, callback=self.unlockAxes)

    def nextROI(self):
        """Focuses on next ROI in list."""

        if len(self.plot.boxes) > 0:
            if self.selected_roi is None:
                self.selected_roi = 0
            else:
                self.selected_roi += 1
            if self.selected_roi >= len(self.plot.boxes):
                self.selected_roi = 0
            box = self.plot.boxes[self.selected_roi]
            self.focusROI(None, None, box)

    def prevROI(self):
        """Focuses on previous ROI in list."""

        if len(self.plot.boxes) > 0:
            if self.selected_roi is None:
                self.selected_roi = 0
            else:
                self.selected_roi -= 1
            if self.selected_roi < 0:
                self.selected_roi = len(self.plot.boxes) - 1
            box = self.plot.boxes[self.selected_roi]
            self.focusROI(None, None, box)

    def unlockAxes(self):
        """Unlocks axes after focusing on ROI."""

        dpg.set_axis_limits_auto(self.plot.x_axis)
        dpg.set_axis_limits_auto(self.plot.y_axis)
        # Scale box outlines to new zoom level
        self.scaleOutlines()

    def scaleOutlines(self, fraction=200):
        """Scales outline of drawn boxes to zoom level."""
        
        x_lim = dpg.get_axis_limits(self.plot.x_axis)
        width = x_lim[1] - x_lim[0]
        thickness = width / fraction

        for box in self.plot.boxes:
            box.updateThickness(thickness)

        # Also mouse box
        if self.mouse_box is not None:
            self.mouse_box.updateThickness(thickness)

        # Also scale grid boxes
        if hasattr(self.plot, "boxes_grid") and self.plot.boxes_grid:
            for box in self.plot.boxes_grid:
                box.updateThickness(thickness)

    def plotHover(self):
        """Checks for boxes to be highlighted."""

        mouse_coords = np.array(dpg.get_plot_mouse_pos())
        
        for b, box in enumerate(self.plot.boxes):
            if box.within(mouse_coords, margin=min(box.width, box.height) * 0.1):
                self.scaleOutlines()
                box.highlight()                    
                return

        # If no box was selected, check grid boxes
        if hasattr(self.plot, "boxes_grid") and self.plot.boxes_grid:
            for b, box in enumerate(self.plot.boxes_grid):
                if box.within(mouse_coords, margin=min(box.width, box.height) * 0.1):
                    self.scaleOutlines()
                    box.highlight()                    
                    return
                
        # If no box was selected, reset outlines
        self.scaleOutlines()

    def callLamellaDetection(self, sender, app_data, user_data):
        """Calls lamella detection model on loaded map."""

        # Check for info box input
        if user_data is not None and dpg.does_item_exist(user_data[0]):
            dpg.delete_item(user_data[0])
            dpg.split_frame()
            # Deal with info box input
            if len(user_data) > 1 and user_data[1] == 0:
                dpg.set_value(self.menu.all_elements["inp_pixsize"], config.WG_model_pix_size)
                self.rescaleMap()        
            else:
                self.menu.show()
                return
        
        # Update GUI
        self.menu.hide()

        # Check if map has correct pixel size (within 10% of model pixel size)
        if abs(self.loaded_map.pix_size - config.WG_model_pix_size) > 0.01 * config.WG_model_pix_size:
            log("WARNING: Loaded map does not have a suitable pixel size for lamella detection model. Rescaling...")
            InfoBoxManager.push(InfoBox("WARNING", "The loaded map does not have a suitable pixel size for the lamella detection model.\nThe map will be rescaled to " + str(config.WG_model_pix_size) + " nm/px.", self.callLamellaDetection, ["OK", "Cancel"]))
            return

        # Reset boxes
        self.plot.clearBoxes()

        # Reset grid boxes
        if hasattr(self.plot, "boxes_grid") and self.plot.boxes_grid:
            for box in self.plot.boxes_grid:
                box.remove()
            self.plot.boxes_grid = []

        log("Searching for lamellae...")
        self.status.update("Searching lamellae...", box=True)
        WG_model = WGModel(self.loaded_map.file.parent)
        #WG_model.findLamellae(self.loaded_map.file.parent, self.loaded_map.file.name, suffix="")
        WG_model.findLamellae(self.loaded_map.file.name, suffix="")
        self.loadBboxes_json()

        # Show ROI list
        self.makeROITable()

        # Update GUI
        self.menu.show()
        self.status.update()
        dpg.set_value(self.menu.all_elements["inp_pixsize"], self.loaded_map.pix_size)

    def splitTilesForExport(self, sender, app_data, user_data):
        """Splits map into tiles for export."""

        # Check for info box input
        if user_data is not None and dpg.does_item_exist(user_data[0]):
            dpg.delete_item(user_data[0])
            dpg.split_frame()
            if len(user_data) > 1 and user_data[1] == 0:
                # Rescale
                dpg.set_value(self.menu.all_elements["inp_pixsize"], config.WG_model_pix_size)
                map_file = self.loaded_map.file.parent / (self.loaded_map.file.stem + "_" + str(config.WG_model_pix_size) + "nmpx_wg.png")
                self.saveROIs(None, None, None, map_file=map_file, map_pix_size=config.WG_model_pix_size)
                self.rescaleMap()
            elif len(user_data) > 1 and user_data[1] == 2:
                # Cancel
                return
        else:
            # Check if map has correct pixel size (within 10% of model pixel size)
            if abs(self.loaded_map.pix_size - config.WG_model_pix_size) > 0.01 * config.WG_model_pix_size:
                log("WARNING: Loaded map does not have a suitable pixel size for lamella detection model. Rescaling?")
                InfoBoxManager.push(InfoBox("WARNING", "The loaded map does not have a suitable pixel size for the lamella detection model.\nThe map should be rescaled to " + str(config.WG_model_pix_size) + " nm/px.", self.splitTilesForExport, ["Rescale", "Export anyways", "Cancel"]))
                return
            
        # Check checkboxes
        save_empty_tiles = dpg.get_value(self.menu.all_elements["chk_empty"])
        use_model_tile_size = dpg.get_value(self.menu.all_elements["chk_modeltiles"])
        use_padding = dpg.get_value(self.menu.all_elements["chk_padding"])

        # Change map tile size and bounds
        if use_model_tile_size:
            self.loaded_map.changeTiling(np.array((config.WG_model_sidelen, config.WG_model_sidelen)))
            self.plotTiles()
        
        self.menu.hide()
        self.status.update("Exporting tiles...")
        # Get bboxes
        bboxes = []
        for box in self.plot.boxes:
            c = box.cat
            # Scale coords to img coords
            x, y = box.coords * 1000 / self.loaded_map.pix_size
            w, h = np.abs((box.p2 - box.p1)) * 1000 / self.loaded_map.pix_size# + np.array([10, 10])
            # Invert y axis
            y = self.loaded_map.img.shape[0] - y
            bboxes.append([c, x, y, w, h])

        # Go through all tiles (+ 1 for partial tiles)
        total_tile_count = 0
        total_roi_count = 0
        for i in range(self.loaded_map.tile_num[0] + 1):
            for j in range(self.loaded_map.tile_num[1] + 1):
                labels = []
                tile_img, tile_bounds = self.loaded_map.returnTile(i, j, padding=use_padding)
                if np.any(tile_img):
                    for bbox in bboxes:
                        # Convert to lower and upper bounds [[xmin, xmax], [ymin, ymax]]
                        bounds = [[bbox[2] - bbox[4] / 2, bbox[2] + bbox[4] / 2], [bbox[1] - bbox[3] / 2, bbox[1] + bbox[3] / 2]]
                        overlap, overlap_bounds = self.overlapBoxes(bounds, tile_bounds)
                        if overlap:
                            # Get bounds in coords within tile image (subtract lower tile bounds)
                            rel_bounds = [np.array(overlap_bounds[0]) - tile_bounds[0][0], np.array(overlap_bounds[1]) - tile_bounds[1][0]]
                            # Get relative coordinates for YOLO [xc, yc, w, h]
                            #yolo_bbox = [(bbox[1] - tile_bounds[1][0]) / loaded_map.tile_size[1], (bbox[2] - tile_bounds[0][0]) / loaded_map.tile_size[0], bbox[3] / loaded_map.tile_size[1], bbox[4] / loaded_map.tile_size[0]]
                            yolo_bbox = [np.sum(rel_bounds[1]) / 2 / self.loaded_map.tile_size[1], np.sum(rel_bounds[0]) / 2 / self.loaded_map.tile_size[0], np.diff(rel_bounds[1])[0] / self.loaded_map.tile_size[1], np.diff(rel_bounds[0])[0] / self.loaded_map.tile_size[0]]
                            # Add label
                            labels.append([bbox[0]] + yolo_bbox)

                    if len(labels) > 0 or save_empty_tiles:
                        self.status.update("Exporting tile [" + str(i) + ", " + str(j) + "]...")
                        #if not os.path.exists(os.path.join(os.path.dirname(self.loaded_map.file), "YOLO_dataset")): 
                        #    os.makedirs(os.path.join(os.path.dirname(self.loaded_map.file), "YOLO_dataset"))
                        if not (self.loaded_map.file.parent / "YOLO_dataset").exists(): 
                            (self.loaded_map.file.parent / "YOLO_dataset").mkdir()
                        #self.exportTile(os.path.join(os.path.dirname(self.loaded_map.file), "YOLO_dataset", os.path.splitext(os.path.basename(self.loaded_map.file))[0] + "_" + str(i) + "_" + str(j) + ".png"), tile_img, labels)
                        self.exportTile(self.loaded_map.file.parent / "YOLO_dataset" / (self.loaded_map.file.stem + f"_{i}_{j}.png"), tile_img, labels)
                        total_tile_count += 1
                        total_roi_count += len(labels)

        # Update GUI
        self.menu.show()
        self.status.update()
        dpg.set_value(self.menu.all_elements["inp_pixsize"], self.loaded_map.pix_size)

        log(f"Successfully exported {total_tile_count} tiles containing {total_roi_count} ROIs to {self.loaded_map.file.parent / 'YOLO_dataset'}")
        InfoBoxManager.push(InfoBox("INFO", "Successfully exported " + str(total_tile_count) + " tiles containing " + str(total_roi_count) + " ROIs!"))

    @staticmethod
    def overlapBoxes(bounds1, bounds2):
        """Checks if two boxes overlap and returns overlapping bounds."""

        if bounds1[0][0] > bounds2[0][1] or bounds1[0][1] < bounds2[0][0] or bounds1[1][0] > bounds2[1][1] or bounds1[1][1] < bounds2[1][0]:
            return False, []
        else: 
            x_min = max(bounds1[0][0], bounds2[0][0])
            x_max = min(bounds1[0][1], bounds2[0][1])
            y_min = max(bounds1[1][0], bounds2[1][0])
            y_max = min(bounds1[1][1], bounds2[1][1])
            return True, [[x_min, x_max], [y_min, y_max]]

    @staticmethod
    def exportTile(name, img, labels):
        """Exports tile with labels to file."""

        if img.dtype == np.uint8:
            Image.fromarray(img).save(name)
        else:
            Image.fromarray((img * 255).astype(np.uint8)).save(name)
        with open(name.parent / (name.stem + ".txt"), "w+") as f: 
            for label in labels:
                f.write(" ".join(str(val) for val in label) + "\n")
        log(f"Exported {len(labels)} ROIs: {name}")

    def datasetComposition(self):
        """Shows composition of dataset in current directory."""
        
        if self.loaded_map is not None and (self.loaded_map.file.parent / "YOLO_dataset").exists():
            dataset_dir = self.loaded_map.file.parent / "YOLO_dataset"
        else:
            dataset_dir = Path.cwd()

        labelfiles = sorted(dataset_dir.glob("**/*.txt"))

        total = 0
        categories = np.zeros(len(config.WG_model_categories))
        for file in labelfiles:
            with open(file) as f:
                lines = f.readlines()

            total += len(lines)
            for line in lines:
                categories[int(line.split()[0])] += 1

        log(f"Dataset composition in {dataset_dir}:")
        message = "Number of images: " + str(len(labelfiles)) + "\nTotal ROIs:   " + str(total) + "\n\nCategories: "
        if total > 0:
            categories = sorted(zip(categories, config.WG_model_categories), reverse=True)
            for cat in categories:
                percent = str(round(cat[0] / total * 100))
                message += "\n" + cat[1] + ":" + " " * (20 - len(cat[1]) - len(str(cat[0]))) + str(int(cat[0])) + " " * (6 - len(percent)) + "(" + percent + "%)"
        else:
            message += "None"
        log(message)
        InfoBoxManager.push(InfoBox("Dataset composition", message))

    def markInspected(self):
        """Creates inspected.txt file and locks down editing."""

        # Save targets if unsaved changes
        if dpg.is_item_shown(self.menu.all_elements["btn_save"]):
            self.saveROIs(None, None, None)

        (self.loaded_map.file.parent / (self.loaded_map.file.stem + "_inspected.txt")).touch()
        self.inspected = True

        # Remake ROI table without edit buttons
        self.makeROITable()

        self.menu.lockRows(["detect", "inspect", "rescale", "mrcscale", "export"])
        self.menu.show()
        self.status.update("\nRegions of Interest were\nmarked as inspected.\n(Editing disabled)", color=gui.COLORS["heading"])

        if self.auto_close:
            log(f"NOTE: Finished inspecting Regions of Interest and closed GUI.")
            dpg.stop_dearpygui()

    def savePlot(self, sender, app_data, user_data):
        """Gets frame buffer to save plot to file. (Does not work on MacOS.)"""

        # Check if map was opened
        if not self.loaded_map:
            log(f"ERROR: Please load a map before saving a snapshot!")
            InfoBoxManager.push(InfoBox("WARNING", "Please load a map before saving a snapshot!"))
            return
        
        # Get name for snapshot
        counter = 1
        while (snapshot_file_path := self.loaded_map.file.parent / f"{self.loaded_map.file.stem}_snapshot{counter}.png").exists():
            counter += 1
        
        saveSnapshot(self.plot.plot, snapshot_file_path)

    def askForSave(self):
        """Opens popup if there are unsaved changes."""

        if dpg.is_item_shown(self.menu.all_elements["btn_save"]):
            InfoBoxManager.push(InfoBox("WARNING", "There are unsaved changes to your Regions of Interest!", callback=self.saveAndClose, options=["Save", "Discard"], options_data=[True, False]))
        else:
            dpg.stop_dearpygui()

    def saveAndClose(self, sender, app_data, user_data):
        """Saves targets and closes GUI."""

        # Save targets if user clicked Save
        if user_data[1]:
            self.saveROIs(None, None, None)

        # Change exit callback to avoid infinite loop
        dpg.set_exit_callback(dpg.stop_dearpygui)
        dpg.stop_dearpygui()

    @staticmethod
    def showHelp():
        message = "Keyboard shortcuts:\n\n"
        message += "A     Toggle advanced settings\n"
        message += "C     Show current dataset composition\n"
        message += "D     Detect lamellae\n"
        message += "E     Export tiles\n"
        message += "F     Open map file dialogue\n"
        message += "G     Toggle show grid pattern\n"
        message += "H     Show help\n"
        message += "N     Load next map\n"
        message += "R     Rescale map to pixel size\n"
        message += "S     Save ROIs as SPACE .json file\n"
        message += "T     Toggle tile boundaries on plot\n"
        message += "Y     Save ROIs as YOLO .txt file\n"
        message += "Down  Show next ROI on plot\n"
        message += "Up    Show previous ROI on plot\n"

        InfoBoxManager.push(InfoBox("Help", message))

### END FUNCTIONS ###

    def __init__(self, file=None, auto_close=False) -> None:
        log("\n########################################\nRunning SPACEtomo Region Selection GUI\n########################################\n")

        if file:
            self.file = Path(file)
        else:
            self.file = None

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
        self.menu = None
        self.status = None
        self.loaded_map = None
        self.selected_roi = None
        self.mouse_box = None
        self.drag_start = None

        self.inspected = False              # Inspection state of current map

    def configureHandlers(self):
        """Sets up dearpygui registries and handlers."""

        # Create file dialogues
        gui.fileNav("mapfile", self.loadMap, extensions=[".png", ".mrc", ".map"])

        dpg.set_viewport_resize_callback(callback=lambda: gui.window_size_change(self.logo_dims))

        # Create event handlers
        with dpg.handler_registry() as mouse_handler:
            dpg.add_mouse_click_handler(callback=self.mouseClick)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self.mouseRelease)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self.mouseDrag)
            dpg.add_mouse_wheel_handler(callback=self.mouseScroll)

            # If map given, auto load on mouse movement
            if self.file:
                dpg.add_mouse_move_handler(tag="auto_load_map", callback=lambda: self.loadMap(None, {"selections": {self.file.name: self.file}}, None))

            # Shortcuts
            dpg.add_key_press_handler(dpg.mvKey_H, callback=self.showHelp)
            dpg.add_key_press_handler(dpg.mvKey_F, callback=lambda: dpg.show_item("mapfile"))
            dpg.add_key_press_handler(dpg.mvKey_N, callback=self.loadNextMap, tag="key_next")
            dpg.add_key_press_handler(dpg.mvKey_T, callback=self.toggleTiles)
            dpg.add_key_press_handler(dpg.mvKey_G, callback=self.toggleGrid)
            dpg.add_key_press_handler(dpg.mvKey_A, callback=self.toggleAdvanced)
            dpg.add_key_press_handler(dpg.mvKey_D, callback=self.callLamellaDetection)
            dpg.add_key_press_handler(dpg.mvKey_S, callback=self.saveROIs)
            dpg.add_key_press_handler(dpg.mvKey_Y, callback=self.saveLabel)
            dpg.add_key_press_handler(dpg.mvKey_R, callback=self.rescaleMap)
            dpg.add_key_press_handler(dpg.mvKey_E, callback=self.splitTilesForExport)
            dpg.add_key_press_handler(dpg.mvKey_C, callback=self.datasetComposition)
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=self.nextROI)
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=self.prevROI)

            # Setup shortcuts for class selection (loop does not work for callback lambda assignment)
            if len(config.WG_model_categories) > 0: dpg.add_key_press_handler(dpg.mvKey_NumPad0, callback=lambda: self.catROI(None, config.WG_model_categories[0], None))
            if len(config.WG_model_categories) > 1: dpg.add_key_press_handler(dpg.mvKey_NumPad1, callback=lambda: self.catROI(None, config.WG_model_categories[1], None))
            if len(config.WG_model_categories) > 2: dpg.add_key_press_handler(dpg.mvKey_NumPad2, callback=lambda: self.catROI(None, config.WG_model_categories[2], None))
            if len(config.WG_model_categories) > 3: dpg.add_key_press_handler(dpg.mvKey_NumPad3, callback=lambda: self.catROI(None, config.WG_model_categories[3], None))
            if len(config.WG_model_categories) > 4: dpg.add_key_press_handler(dpg.mvKey_NumPad4, callback=lambda: self.catROI(None, config.WG_model_categories[4], None))
            if len(config.WG_model_categories) > 5: dpg.add_key_press_handler(dpg.mvKey_NumPad5, callback=lambda: self.catROI(None, config.WG_model_categories[5], None))
            if len(config.WG_model_categories) > 6: dpg.add_key_press_handler(dpg.mvKey_NumPad6, callback=lambda: self.catROI(None, config.WG_model_categories[6], None))
            if len(config.WG_model_categories) > 7: dpg.add_key_press_handler(dpg.mvKey_NumPad7, callback=lambda: self.catROI(None, config.WG_model_categories[7], None))
            if len(config.WG_model_categories) > 8: dpg.add_key_press_handler(dpg.mvKey_NumPad8, callback=lambda: self.catROI(None, config.WG_model_categories[8], None))
            if len(config.WG_model_categories) > 9: dpg.add_key_press_handler(dpg.mvKey_NumPad9, callback=lambda: self.catROI(None, config.WG_model_categories[9], None))

        # Create item handlers
        with dpg.item_handler_registry(tag="plot_hover_handler") as item_handler:
            dpg.add_item_hover_handler(callback=self.plotHover)

    @staticmethod
    def configureThemes():
        """Sets up dearpygui themes."""

        gui.configureGlobalTheme()

        # Color themes for ROI classes
        for c, color in enumerate(config.WG_model_gui_colors):
            with dpg.theme(tag=f"cat_theme{c}"):
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)

        with dpg.theme(tag="plot_tiletheme"):
            try:
                with dpg.theme_component(dpg.mvInfLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, gui.COLORS["heading"], category=dpg.mvThemeCat_Plots)   
            except AttributeError:      # Backward compatibility with dearpyguy<2.0
                with dpg.theme_component(dpg.mvHLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, gui.COLORS["heading"], category=dpg.mvThemeCat_Plots) 
                with dpg.theme_component(dpg.mvVLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, gui.COLORS["heading"], category=dpg.mvThemeCat_Plots)

    def show(self):
        """Structures and launches main window of GUI."""

        # Setup window
        dpg.create_viewport(title="SPACEtomo Region Selection", disable_close=True, small_icon=str(Path(__file__).parent / "logo.ico"), large_icon=str(Path(__file__).parent / "logo.ico"))
        dpg.setup_dearpygui()

        # Create main window
        with dpg.window(label="GUI", tag="GUI", no_scrollbar=True, no_scroll_with_mouse=True):

            with dpg.table(header_row=False):
                dpg.add_table_column(init_width_or_weight=200, width_fixed=True)
                dpg.add_table_column()

                with dpg.table_row():
                    with dpg.table_cell(tag="tblleft"):
                        dpg.add_text(default_value="Load your WG map", tag="l1", color=gui.COLORS["heading"])

                        self.menu = Menu()
                        self.menu.newRow(tag="load", horizontal=True, locked=False)
                        if self.file:        # If file name is given, make load map button
                            self.menu.addButton(tag="btn_load", label="Load map", callback=lambda: self.loadMap(None, {"selections": {self.file.name: self.file}}, None))
                        else:
                            self.menu.addButton(tag="btn_load", label="Find map", callback=lambda: dpg.show_item("mapfile"))
                        self.menu.addButton(tag="btn_next", label="Next", callback=self.loadNextMap, show=False)

                        self.menu.newRow(tag="roilist")

                        self.menu.newRow(tag="detect", horizontal=True, advanced=True)
                        self.menu.addButton(tag="btn_detect", label="Detect lamellae", callback=self.callLamellaDetection)

                        self.menu.newRow(tag="rescale", horizontal=True, advanced=True)
                        self.menu.addInput("inp_pixsize", label="[nm/px]", value=100)
                        self.menu.addButton(tag="btn_rescale", label="Rescale", callback=self.rescaleMap)

                        self.menu.newRow(tag="inspect")
                        self.menu.addButton(tag="btn_save", label="Save", callback=self.saveROIs, show=False)
                        self.menu.addButton(tag="btn_inspect", label="Confirm inspection", callback=self.markInspected, theme="large_btn_theme")

                        self.menu.newRow(tag="export")
                        self.menu.addCheckbox(tag="chk_empty", label="Include empty tiles", value=False, advanced=True)
                        self.menu.addCheckbox(tag="chk_modeltiles", label="Use model tile size", value=True, advanced=True)
                        self.menu.addCheckbox(tag="chk_padding", label="Use padding", value=True, advanced=True)
                        self.menu.addButton(tag="btn_exptiles", label="Export tiles", callback=self.splitTilesForExport)
                        self.menu.addButton(tag="btn_datacomp", label="Dataset composition", callback=self.datasetComposition, advanced=True)

                        self.menu.newRow(tag="mrcscale", horizontal=True, advanced=True)
                        self.menu.addInput(tag="inp_quantile", label="quantile", value=0.01)
                        self.menu.addButton(tag="btn_rescalemrc", label="Rescale mrc", callback=lambda: self.loadMap(None, {"selections": {self.loaded_map.file.name: self.loaded_map.file}}, None, quantile=dpg.get_value(self.menu.all_elements["inp_quantile"])))

                        self.status = StatusLine()

                    with dpg.table_cell(tag="tblplot"):

                        self.menu_icon = Menu(outline=False)
                        self.menu_icon.newRow(tag="icon", horizontal=True, separator=False, locked=False)
                        self.menu_icon.addText(tag="icon_heading", value="MM map", color=gui.COLORS["heading"])
                        self.menu_icon.addImageButton("butresetzoom", gui.makeIconResetZoom(), callback=self.plot.resetZoom, tooltip="Reset zoom")
                        self.menu_icon.addImageButton("butsnapshot", gui.makeIconSnapshot(), callback=self.savePlot, tooltip="Save snapshot")
                        self.menu_icon.addImageButton("butgrid", gui.makeIconGrid(), callback=self.toggleGrid, tooltip="Show detected grid pattern", show=False)

                        self.plot.makePlot(x_axis_label="x [µm]", y_axis_label="y [µm]", width=-1, height=-1, equal_aspects=True, no_menus=True, crosshairs=True, pan_button=dpg.mvMouseButton_Right, no_box_select=True)
                        dpg.bind_item_handler_registry(self.plot.plot, "plot_hover_handler")
            # Create tooltips
            with dpg.tooltip("l1", delay=0.5):
                dpg.add_text("Select an .mrc or .png whole grid montage.\nIf IMOD is available, the piece coordinates will be read from the mrc header.")
            with dpg.tooltip(self.menu.all_elements["btn_detect"], delay=0.5):
                dpg.add_text("Use SPACEtomo detection model to find lamellae.")
            with dpg.tooltip(self.menu.all_elements["btn_save"], delay=0.5):
                dpg.add_text("Save ROI coordinates for SPACEtomo.")
            with dpg.tooltip(self.menu.all_elements["btn_rescale"], delay=0.5):
                dpg.add_text("Rescale mrc map to desired pixel size. If a png file is loaded, the pixel size of the SPACEtomo model is assumed.")
            with dpg.tooltip(self.menu.all_elements["btn_exptiles"], delay=0.5):
                dpg.add_text("Export map tiles and ROI boxes for YOLO training.")

            with dpg.tooltip("tblplot", delay=0.5, hide_on_activity=True):
                #dpg.add_text(default_value="", color=gui.COLORS["heading"], tag="tt_heading")
                dpg.add_text(default_value="- Shift + left click + drag to add ROI box\n- Right click to edit ROI", tag="tt_text")

            # Show logo
            dpg.add_image("logo", pos=(10, dpg.get_viewport_client_height() - 40 - self.logo_dims[0]), tag="logo_img")
            #dpg.add_text(default_value="SPACEtomo", pos=(10 + self.logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() - 40 - self.logo_dims[0] / 2), tag="logo_text")
            dpg.add_text(default_value="v" + __version__, pos=(10 + self.logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() + 5 - self.logo_dims[0] / 2), tag="version_text")

        # Create ROI menu
        with dpg.window(label="Region of Interest", tag="ROI", no_scrollbar=True, no_scroll_with_mouse=True, popup=True, show=False) as win_roi:
            dpg.add_text(default_value="Region of Interest", color=gui.COLORS["heading"], tag="roi_label")
            dpg.add_radio_button(config.WG_model_categories, callback=self.catROI, user_data=None, tag="roi_cat")
            dpg.add_button(label="Delete", callback=self.deleteROI, user_data=None, tag="roi_btndel")
            with dpg.group(horizontal=True):
                dpg.add_button(arrow=True, direction=dpg.mvDir_Up, callback=self.reorderROIUp, user_data=None, tag="roi_btnup")
                dpg.add_button(arrow=True, direction=dpg.mvDir_Down, callback=self.reorderROIDown, user_data=None, tag="roi_btndown")

                dpg.bind_item_theme("roi_btnup", "small_btn_theme")
                dpg.bind_item_theme("roi_btndown", "small_btn_theme")
                with dpg.tooltip("roi_btnup", delay=0.5):
                    dpg.add_text("Move ROI up in list order.")
                with dpg.tooltip("roi_btndown", delay=0.5):
                    dpg.add_text("Move ROI down in list order.")
        self.blocking_windows.append(win_roi) # Add to blocking windows to keep track of open popups

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

            dpg.render_dearpygui_frame()
