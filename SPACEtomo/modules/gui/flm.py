#!/usr/bin/env python
# ===================================================================
# ScriptName:   flm
# Purpose:      User interface utility to generate window to correlate light microscope images
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/11/11
# Revision:     v1.2
# Last Change:  2024/11/20: added pre rotation/flip
#               2024/11/19: added auto thresholding, simplified dim handling, added black background, added basic multi channel tif handling
#               2024/11/18: added clearing of maps, added histograms
#               2024/11/14: added thresholding update by slider
#               2024/11/13: added dragging of registration points
#               2024/11/12: implemented transform and show channels
#               2024/11/11: copied from thmb 
# ===================================================================

import os
os.environ["__GLVND_DISALLOW_PATCHING"] = "1"           # helps to minimize Segmentation fault crashes on Linux when deleting textures
import sys
from pathlib import Path
from skimage import exposure
from scipy.ndimage import affine_transform
try:
    import dearpygui.dearpygui as dpg
except:
    print("ERROR: DearPyGUI module not installed! If you cannot install it, please run the GUI from an external machine.")
    sys.exit()
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

from SPACEtomo.modules.gui import gui
from SPACEtomo.modules.gui.plot import Plot
from SPACEtomo.modules.gui.menu import Menu
from SPACEtomo.modules.gui.map import MMap
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log

class FlmWindow:
    """Window showing 2 plots for image registration."""

    def __init__(self, main_plot) -> None:
        # Reference to main plot to show transformed overlay
        self.main_plot = main_plot

        self.window = dpg.add_window(label="CLEM image registration", tag="flm_window", no_collapse=True, show=False, width=dpg.get_viewport_client_width(), height=dpg.get_viewport_client_height())
        self.layout_table = None
        self.lm_menu = None
        self.status = None

        # EM map
        self.loaded_map = None
        self.em_map_textures = []
        self.em_map_bounds = []
        self.em_plot = None

        # LM maps
        self.lm_maps = []
        self.lm_plot = None

        # Registration points
        self.em_points = []
        self.lm_points = []

        # State tracking
        self.active_slider = None
        self.flipped = [False, False]
        self.rotated = 0

        self.createLayoutTable()

        # Create map file dialogue
        gui.fileNav("flm_file", self.loadLM, extensions=[".png", ".tif"])

    def createLayoutTable(self):
        """Creates layout of window and instantiates menu and plots."""

        # Instantiate plots
        if self.em_plot is None:
            self.em_plot = Plot(tag="em_plot")
        if self.lm_plot is None:
            self.lm_plot = Plot(tag="lm_plot")

        # Create layout
        if self.layout_table is None:
            with dpg.table(header_row=False, parent=self.window) as self.layout_table:
                # Add table columns [menu, em plot, lm plot]
                dpg.add_table_column(init_width_or_weight=200, width_fixed=True)
                dpg.add_table_column()
                dpg.add_table_column()

                with dpg.table_row():
                    with dpg.table_cell(tag="flm_tblleft"):
                        dpg.add_text(default_value="Load your LM map", tag="flm_l1", color=gui.COLORS["heading"])

                        # Create menu
                        self.lm_menu = Menu()
                        self.lm_menu.newRow(tag="flm_row_load", locked=False)
                        self.lm_menu.addButton(tag="flm_butload", label="Find LM map", callback=lambda: dpg.show_item("flm_file"))
                        self.lm_menu.newRow(tag="flm_row_del", horizontal=True, locked=False)
                        self.lm_menu.addButton(tag="flm_delmap", label="Clear LM maps", callback=lambda: self.clearAll(keep_em=True), show=False)
                        self.lm_menu.addButton(tag="flm_delpts", label="Clear points", callback=self.clearPoints, show=False)
                        self.lm_menu.newRow(tag="flm_row_trans", locked=False)
                        self.lm_menu.addButton(tag="flm_transform", label="Transform", callback=self.showTransform, show=False)

                        # Create line for status updates
                        self.status = gui.StatusLine()

                    with dpg.table_cell(tag="flm_emplot"):
                        dpg.add_text(default_value="EM map", tag="flm_em1", color=gui.COLORS["heading"])
                        self.em_plot.makePlot(x_axis_label="x [µm]", y_axis_label="y [µm]", width=-1, height=-1, equal_aspects=True, no_menus=True, crosshairs=True, pan_button=dpg.mvMouseButton_Right, no_box_select=True)

                    with dpg.tooltip("flm_emplot", delay=0.5, hide_on_activity=True):
                        dpg.add_text(default_value="Left click to add registration point!", color=gui.COLORS["heading"])

                    with dpg.table_cell(tag="flm_lmplot"):
                        with dpg.group(tag="flm_btn_rotflip", horizontal=True):
                            dpg.add_text(default_value="LM map", tag="flm_lm1", color=gui.COLORS["heading"])
                            dpg.add_image_button(gui.makeIconRotation(), callback=self.rotateMap, user_data=False)
                            dpg.add_image_button(gui.makeIconRotation(ccw=True), callback=self.rotateMap, user_data=True)
                            dpg.add_image_button(gui.makeIconFlip(), callback=self.flipMap, user_data=False)
                            dpg.add_image_button(gui.makeIconFlip(horizontal=True), callback=self.flipMap, user_data=True)
                        dpg.bind_item_theme("flm_btn_rotflip", "small_btn_theme")
                        self.lm_plot.makePlot(x_axis_label="x [px]", y_axis_label="y [px]", width=-1, height=-1, equal_aspects=True, no_menus=True, crosshairs=True, pan_button=dpg.mvMouseButton_Right, no_box_select=True)

                    with dpg.tooltip("flm_lmplot", delay=0.5, hide_on_activity=True):
                        dpg.add_text(default_value="Left click to add registration point!", color=gui.COLORS["heading"])

    def loadEM(self, loaded_map, textures, bounds):
        """Plots loaded map."""

        self.loaded_map = loaded_map
        self.em_map_textures = textures
        self.em_map_bounds = bounds
        # Plot map
        for i, (tex, bound) in enumerate(zip(self.em_map_textures, self.em_map_bounds)):
            self.em_plot.addImg(tex, bound, self.loaded_map.binning, label=f"map{i}")
        self.em_plot.resetZoom()

    def loadLM(self, sender, app_data, user_data):
        """Loads LM map from file and plots it."""

        selected_files = sorted([file for file in app_data["selections"].values()])
        file_path = Path(selected_files[0])

        # Load map file
        new_map = MMap(file_path, pix_size=10000, status=self.status)
        if np.all(new_map.img == 0):
            log(f"ERROR: No map loaded.")
            return
        self.lm_maps.append(new_map)
        new_map.checkBinning()

        # Adjust flip/rotation
        if self.rotated > 0:
            new_map.img_bin = np.rot90(new_map.img_bin, k=self.rotated)
        if self.flipped[0]:
            new_map.img_bin = np.flip(new_map.img_bin, axis=0)
        if self.flipped[1]:
            new_map.img_bin = np.flip(new_map.img_bin, axis=1)

        # Make sure map is RGBA with transparency
        new_map.padDims(dims=4)

        # Add channel attribute: white by default
        new_map.channel = "W"

        # Add histogram and default thresholds
        hist, edges = np.histogram(new_map.img_bin[:, :, :3], 255)
        new_map.histogram = hist
        new_map.thresholds = np.array([0, 1])

        # Get texture and dimensions
        texture = new_map.getTexture()#mod=use_mod)
        dims_plot = np.flip(np.array(new_map.img.shape[:2]) * new_map.pix_size / 10000)

        # Load black background
        if not self.lm_plot.getOverlaysByKeyword("background"):
            bg_tex_img = np.zeros((1, 1, 4))
            bg_tex_img[:, :, 3] = 1
            bg_dims = np.flip(np.array(bg_tex_img.shape[:2])).tolist()
            with dpg.texture_registry():
                bg_tex = dpg.add_static_texture(width=bg_dims[0], height=bg_dims[1], default_value=np.ravel(bg_tex_img))
            self.lm_plot.addOverlay(bg_tex, [[0, dims_plot[0]], [0, dims_plot[1]]], label="background")

        # Load map to plot
        self.lm_plot.addImg(texture, [[0, dims_plot[0]], [0, dims_plot[1]]], new_map.binning, label=f"lm_map_{len(self.lm_maps) - 1}")

        # Fit axes
        self.lm_plot.resetZoom()

        # Update UI
        dpg.show_item(self.lm_menu.all_elements["flm_delmap"])
        dpg.show_item(self.lm_menu.all_elements["flm_transform"])

        # Update channel table
        self.makeChannelTable()

    def rotateMap(self, sender, app_data, user_data):
        """Rotates all LM maps by 90 deg and transforms registration points."""

        if not self.lm_maps:
            return
        
        ccw = user_data

        for m, lm_map in enumerate(self.lm_maps):
            # Rotate numpy array
            lm_map.img_bin = np.rot90(lm_map.img_bin, k=1 + int(not ccw) * 2)
            # Update dims for px2micron conversion
            lm_map.dims_microns = np.flip(lm_map.dims_microns)
            # Plot new image
            self.changeColor(None, lm_map.channel, m)
            # Save state
            self.rotated += 1 + int(not ccw) * 2
            self.rotated = self.rotated % 4

        # Rotate registration points
        if len(self.lm_points) > 0:
            width = self.lm_maps[0].img_bin.shape[0]
            height = self.lm_maps[0].img_bin.shape[1]

            # Get appropriate rotation matrix
            if ccw:
                rot_mat = np.array([[0, -1, width], [1, 0, 0], [0, 0, 1]])
            else:
                rot_mat = np.array([[0, 1, 0], [-1, 0, height], [0, 0, 1]])
            # Add 3rd dimension for matrix multiplication, transform, remove 3rd dimension for output
            self.lm_points = (np.column_stack([self.lm_points, np.ones(len(self.lm_points))]) @ rot_mat.T)[:, :2].tolist()

            self.plotPoints()

    def flipMap(self, sender, app_data, user_data):
        """Flips all LM maps and transforms registration points."""

        if not self.lm_maps:
            return
        
        horizontal = user_data

        for m, lm_map in enumerate(self.lm_maps):
            lm_map.img_bin = np.flip(lm_map.img_bin, axis=int(horizontal))
            self.changeColor(None, lm_map.channel, m)
            if not horizontal:
                self.flipped[0] = not self.flipped[0]
            else:
                self.flipped[1] = not self.flipped[1]

        # Flip registration points
        if not horizontal:
            self.lm_points = [[lm_map.img_bin.shape[0] - point[0], point[1]] for point in self.lm_points]
        else:
            self.lm_points = [[point[0], lm_map.img_bin.shape[1] - point[1]] for point in self.lm_points]
        self.plotPoints()

    def makeChannelTable(self):
        """Creates table for loaded LM files aka channels."""

        # Delete table
        if dpg.does_item_exist("flm_tblch"):
            dpg.delete_item("flm_tblch")
        if len(self.lm_maps) > 0:
            # Make new table
            with dpg.table(label="Channels", tag="flm_tblch", parent="flm_tblleft", height=500, scrollY=True, policy=dpg.mvTable_SizingFixedFit):
           
                for m, map in enumerate(self.lm_maps):
                    dpg.add_table_column(label=f"Ch{m + 1}", tag=f"flm_ch_name_{m}")
                    with dpg.tooltip(f"flm_ch_name_{m}"):
                        dpg.add_text(map.file.name)

                with dpg.table_row():
                    for m, map in enumerate(self.lm_maps):
                        with dpg.table_cell():
                            dpg.add_radio_button(items=["R", "G", "B", "W"], default_value=map.channel, callback=self.changeColor, user_data=m)

                with dpg.table_row():
                    for m, map in enumerate(self.lm_maps):
                        with dpg.table_cell():
                            dpg.add_checkbox(tag=f"flm_chkbox_{m}", label="", default_value=True, callback=self.changeShow, user_data=m)
                            with dpg.tooltip(f"flm_chkbox_{m}"):
                                dpg.add_text("Show")

                with dpg.table_row():
                    for m, map in enumerate(self.lm_maps):
                        with dpg.table_cell():
                            dpg.add_slider_float(tag=f"flm_thrmax_{m}", label="", default_value=map.thresholds[1], min_value=0, max_value=1, callback=self.activateSlider, user_data=m, vertical=True)
                            with dpg.tooltip(f"flm_thrmax_{m}"):
                                dpg.add_text("Max threshold")

                with dpg.table_row():
                    for m, map in enumerate(self.lm_maps):
                        with dpg.table_cell():
                            dpg.add_slider_float(tag=f"flm_thrmin_{m}", label="", default_value=map.thresholds[0], min_value=0, max_value=1, callback=self.activateSlider, user_data=m, vertical=True)
                            with dpg.tooltip(f"flm_thrmin_{m}"):
                                dpg.add_text("Min threshold")

                with dpg.table_row():
                    # Define theme for histogram plots
                    if not dpg.does_item_exist("theme_histogram"):
                        with dpg.theme(tag="theme_histogram"):
                            with dpg.theme_component(dpg.mvPlot):
                                dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 0, 0, category=dpg.mvThemeCat_Plots)

                    # Plot histograms
                    for m, map in enumerate(self.lm_maps):
                        with dpg.table_cell(tag=f"flm_tblplot_{m}"):
                            # Get histogram and bin edges
                            hist = map.histogram
                            edges = np.arange(len(hist) + 1)

                            with dpg.plot(tag=f"flm_hist_plot_{m}", no_box_select=True, no_menus=True, no_title=True, no_mouse_pos=True, width=30, height=80):
                                dpg.add_plot_axis(dpg.mvXAxis, tag=f"flm_hist_x_{m}", no_gridlines=True, no_tick_labels=True, no_tick_marks=True)
                                with dpg.plot_axis(dpg.mvYAxis, tag=f"flm_hist_y_{m}", no_gridlines=True, no_tick_labels=True, no_tick_marks=True) as yaxis:
                                    dpg.add_bar_series(x=hist[1:-1], y=edges[1:len(hist) - 1], horizontal=True)

                                    y_vals = map.thresholds * len(hist)
                                    try:
                                        dpg.add_inf_line_series(tag=f"flm_hist_thr_{m}", x=y_vals)
                                    except AttributeError:      # Backward compatibility with dearpyguy<2.0
                                        log(f"WARNING: Consider updating DearPyGUI to version >=2.0!")
                                        dpg.add_hline_series(tag=f"flm_hist_thr_{m}", x=y_vals)

                                dpg.set_axis_limits(f"flm_hist_x_{m}", 0, np.max(hist[1:-1]))
                                dpg.set_axis_limits(f"flm_hist_y_{m}", 0, np.max(edges))
                            dpg.bind_item_theme(f"flm_hist_plot_{m}", "theme_histogram")

                            with dpg.tooltip(f"flm_tblplot_{m}", delay=0.5):
                                dpg.add_text("Please use the sliders above to adjust the thresholds!")

                with dpg.table_row():
                    for m, map in enumerate(self.lm_maps):
                        with dpg.table_cell():
                            dpg.add_button(tag=f"flm_btnauto_{m}", label="Auto", callback=self.autoThresholding, user_data=m)
                            dpg.bind_item_theme(f"flm_btnauto_{m}", "small_btn_theme")
            
    def changeColor(self, sender, app_data, user_data):
        """Changes color of channel."""

        map_id = user_data
        
        # Either use thresholded image or use unprocessed image
        if self.lm_maps[map_id].thresholds[0] > 0 or self.lm_maps[map_id].thresholds[1] < 1:
            img = self.changeThresholding(map_id=map_id, get=True)
        else:
            img = self.lm_maps[map_id].img_bin

        # Get plot
        plot_id = utils.findIndex(self.lm_plot.img, "label", f"lm_map_{map_id}")

        # Get color channel index
        color = ["R", "G", "B", "W"].index(app_data)

        # If "W" plot original image
        if color == 3:
            new_img = img
        # Else plot image to chosen channel
        else:
            new_img = np.zeros((img.shape[0], img.shape[1], 4))
            # Average all channels (possibly subtract other channels instead?)
            img = np.average(img[:, :, :3], axis=2)

            # Set channel and alpha to pixel values
            new_img[:, :, color] = img
            new_img[:, :, 3] = img# if map_id > 0 else 255

        # Add modified image
        self.lm_maps[map_id].img_mod = new_img
        texture = self.lm_maps[map_id].getTexture(mod=True)

        # Update background in case dims have changed
        dims_plot = np.flip(np.array(self.lm_maps[map_id].img_mod.shape[:2]) * self.lm_maps[map_id].pix_size / 10000)
        self.lm_plot.updateOverlay(utils.findIndex(self.lm_plot.overlays, "label", "background"), texture=None, bounds=[[0, dims_plot[0]], [0, dims_plot[1]]])

        # Just replace texture to keep plotting order the same
        self.lm_plot.updateImg(plot_id, texture, bounds=[[0, dims_plot[0]], [0, dims_plot[1]]])

        # Update channel attribute
        self.lm_maps[map_id].channel = app_data

        # Rescale lm_map in EM plot
        if utils.findIndex(self.em_plot.img, "label", f"lm_map_{map_id}"):
            self.showTransform(subset=[map_id])

    def changeShow(self, sender, app_data, user_data):
        """Changes if channel is plotted."""

        # Toggle show in LM plot
        dpg.configure_item(self.lm_plot.img[utils.findIndex(self.lm_plot.img, "label", f"lm_map_{user_data}")]["plot"], show=app_data)

        # Toggle show in EM plot
        if m := utils.findIndex(self.em_plot.img, "label", f"lm_map_{user_data}"):
            dpg.configure_item(self.em_plot.img[m]["plot"], show=app_data)

        # Toggle show in main plot
        if m := utils.findIndex(self.main_plot.img, "label", f"lm_map_{user_data}"):
            dpg.configure_item(self.main_plot.img[m]["plot"], show=app_data)

    def changeThresholding(self, map_id, thresholds=None, get=False):
        """Updates thresholding of channel map."""

        # Get thresholds
        if thresholds is None:
            min_threshold = np.min(self.lm_maps[map_id].img_bin) + dpg.get_value(f"flm_thrmin_{map_id}") * (np.max(self.lm_maps[map_id].img_bin) - np.min(self.lm_maps[map_id].img_bin))
            max_threshold = np.min(self.lm_maps[map_id].img_bin) + dpg.get_value(f"flm_thrmax_{map_id}") * (np.max(self.lm_maps[map_id].img_bin) - np.min(self.lm_maps[map_id].img_bin))
            min_threshold, max_threshold = min(min_threshold, max_threshold), max(min_threshold, max_threshold)
        else:
            min_threshold, max_threshold = thresholds * (np.max(self.lm_maps[map_id].img_bin) - np.min(self.lm_maps[map_id].img_bin)) + np.min(self.lm_maps[map_id].img_bin)
            # Update sliders
            dpg.set_value(f"flm_thrmin_{map_id}", thresholds[0])
            dpg.set_value(f"flm_thrmax_{map_id}", thresholds[1])

        # Update map attribute
        self.lm_maps[map_id].thresholds = np.array([dpg.get_value(f"flm_thrmin_{map_id}"), dpg.get_value(f"flm_thrmax_{map_id}")])

        # Update histogram indicator
        dpg.configure_item(f"flm_hist_thr_{map_id}", x=[min_threshold, max_threshold])

        # Go via color if not coming from color and color is not "W" (to ensure threshold is always adjusted first and then color)
        if not get and self.lm_maps[map_id].channel != "W":
            self.changeColor(sender=None, app_data=self.lm_maps[map_id].channel, user_data=map_id)
            return

        # Rescale lm_map image and update texture
        img = exposure.rescale_intensity(self.lm_maps[map_id].img_bin, in_range=(min_threshold, max_threshold))

        # Only return image if get
        if get:
            return img
        
        # Update map
        self.lm_maps[map_id].img_mod = img
        texture = self.lm_maps[map_id].getTexture(mod=True)
        self.lm_plot.updateImg(map_id, texture)

        # Rescale lm_map in EM plot
        if utils.findIndex(self.em_plot.img, "label", f"lm_map_{map_id}"):
            self.showTransform(subset=[map_id])

    def autoThresholding(self, sender=None, app_data=None, user_data=None, map_id=None):
        """Automatically adjusts threshold using algorithm from ImageJ."""

        if user_data is not None:
            map_id = user_data

        outlier_limit = 0.1             # exclude bins with more than that fraction of total pixels
        auto_threshold = 1 / 255 / 2    # significant number of pixels as fraction of total pixels
        hist = self.lm_maps[map_id].histogram
        pix_count = np.sum(hist)

        lower_threshold = 0
        for i in range(len(hist)):
            if hist[i] > outlier_limit * pix_count:
                continue
            if hist[i] > auto_threshold * pix_count:
                lower_threshold = i / len(hist)
                break
        
        upper_threshold = 1
        for i in range(len(hist) - 1, -1, -1):
            if hist[i] > outlier_limit * pix_count:
                continue
            if hist[i] > auto_threshold * pix_count:
                upper_threshold = i / len(hist)
                break

        self.changeThresholding(map_id, thresholds=np.array([lower_threshold, upper_threshold]))

    def mouseClick(self, mouse_coords):
        """Handle mouse clicks when plot is hovered within map boundaries."""

        # Get mouse coords in plot coord system
        mouse_coords = np.array(dpg.get_plot_mouse_pos())

        # Add EM registration point
        if dpg.is_item_hovered(self.em_plot.plot):
            if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                if not self.loaded_map:
                    log(f"WARNING: Please load a lamella map first in the main window!")
                    return
                
                img_coords = self.loaded_map.microns2px(mouse_coords)
                self.em_points.append(img_coords)
                self.plotPoints()

                dpg.show_item(self.lm_menu.all_elements["flm_delpts"])

        # Add LM registration point
        elif dpg.is_item_hovered(self.lm_plot.plot):
            if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                if not self.lm_maps:
                    log(f"WARNING: Please load a LM map first!")
                    return

                img_coords = self.lm_maps[0].microns2px(mouse_coords)
                self.lm_points.append(img_coords)
                self.plotPoints()

                dpg.show_item(self.lm_menu.all_elements["flm_delpts"])

    def mouseRelease(self):
        """Update points after mouse release."""

        # Check all drag points for changes
        update = False
        for point in self.em_plot.drag_points:
            update = update or self.checkDragPoint(point["plot"])
        for point in self.lm_plot.drag_points:
            update = update or self.checkDragPoint(point["plot"])

        if update:
            self.em_plot.clearSeries(["temp_drag"])
            self.lm_plot.clearSeries(["temp_drag"])
            self.plotPoints()

        # Check sliders
        if self.active_slider is not None:
            self.changeThresholding(self.active_slider)
            self.active_slider = None

    def activateSlider(self, sender, app_data, user_data):
        """Mark slider as activated to call update on release."""

        self.active_slider = user_data

    def dragPointUpdate(self, sender, app_data, user_data):
        """Updates temporary drag point indicator (since drag points are drawn behind images)."""

        # Get which plot from user_data
        if "em" in user_data:
            plot = self.em_plot
        elif "lm" in user_data:
            plot = self.lm_plot
        else:
            raise ValueError(f"Unknown plot for drag point [{user_data}]!")

        # Update scatter indicator for drag point (drag points are always plotted behind images)
        coords = dpg.get_value(sender)[:2]
        plot.clearSeries(["temp_drag"])
        plot.addSeries([coords[0]], [coords[1]], label="temp_drag", theme="drag_scatter_theme")

    def checkDragPoint(self, drag_point):
        """Compares drag points to registration point coords and updates registration points if they have changed."""

        # Get coords from drag point value
        coords = np.array(dpg.get_value(drag_point)[:2])
        
        # Get area and point IDs from user data embedded in drag point
        user_data = dpg.get_item_user_data(drag_point).split("_")

        # Get which plot from user_data
        if "em" in user_data:
            points = self.em_points
            map = self.loaded_map
        elif "lm" in user_data:
            points = self.lm_points
            map = self.lm_maps[0]
        else:
            log(f"ERROR: Invalid drag point! [{user_data}]")
            return
        
        # Get point id from user_data
        point_id = int(user_data[1])
        
        # Transform points to plot points for comparison
        old_coords = map.px2microns(points[point_id])

        # Go to next points if coords have not changed
        if np.all(coords == old_coords):
            return False
        else:
            # Clip coords to map size
            coords[0] = np.clip(coords[0], 0, map.dims_microns[0])
            coords[1] = np.clip(coords[1], 0, map.dims_microns[1])

            # Update coords if they have changed
            points[point_id] = map.microns2px(coords)

            return True

    def plotPoints(self):
        """Shows scatter plot of selected registration points."""

        # EM scatter plot
        if len(self.em_points) > 0:
            # Clear old plot
            self.em_plot.clearSeries()
            self.em_plot.clearDragPoints()
            self.em_plot.clearAnnotations()

            points = np.array([self.loaded_map.px2microns(point) for point in self.em_points])
            self.em_plot.addSeries(points[:, 0], points[:, 1], label=f"em_scatter", theme=f"scatter_theme4")
            for p, point in enumerate(points):
                self.em_plot.addDragPoint(point[0], point[1], label=f"Point {p + 1}", callback=self.dragPointUpdate, user_data=f"em_{p}", color=gui.THEME_COLORS[4])
                self.em_plot.addAnnotation(f"{p + 1}", point[0], point[1], color=gui.THEME_COLORS[4])

        # LM scatter plot
        if len(self.lm_points) > 0:
            # Clear old plot
            self.lm_plot.clearSeries()
            self.lm_plot.clearDragPoints()
            self.lm_plot.clearAnnotations()

            points = np.array([self.lm_maps[0].px2microns(point) for point in self.lm_points])
            self.lm_plot.addSeries(points[:, 0], points[:, 1], label=f"lm_scatter", theme=f"scatter_theme4")
            for p, point in enumerate(points):
                self.lm_plot.addDragPoint(point[0], point[1], label=f"Point {p + 1}", callback=self.dragPointUpdate, user_data=f"lm_{p}", color=gui.THEME_COLORS[4])
                self.lm_plot.addAnnotation(f"{p + 1}", point[0], point[1], color=gui.THEME_COLORS[4])

    def clearPoints(self):
        """Clears all registration points."""

        self.em_points = []
        self.em_plot.clearAnnotations()
        self.em_plot.clearSeries()
        self.em_plot.clearDragPoints()

        self.lm_points = []
        self.lm_plot.clearAnnotations()
        self.lm_plot.clearSeries()
        self.lm_plot.clearDragPoints()

        dpg.hide_item(self.lm_menu.all_elements["flm_delpts"])

    def clearAll(self, keep_em=False):
        """Clears maps and points."""

        self.clearPoints()
        self.lm_plot.clearAll()
        # Get list of LM maps to delete from EM plot and main plot
        delete_list = [plot["label"] for plot in self.em_plot.img if "lm_map" in plot["label"]] + [""]
        if not keep_em:
            self.em_plot.clearImg(delete_textures=False) # Don't delete textures as they will be deleted by main plot
            self.em_plot.clearAll()
        else:
            self.em_plot.clearImg(delete_list, delete_textures=False) # Don't delete textures as they will be deleted by main plot
        self.main_plot.clearImg(delete_list)

        self.rotated = 0
        self.flipped = [False, False]

        self.lm_maps = []
        self.makeChannelTable()
        dpg.hide_item(self.lm_menu.all_elements["flm_delmap"])
        dpg.hide_item(self.lm_menu.all_elements["flm_transform"])

    def calcTransMatrix(self):
        """Calculates affine transformation matrix that maps lm_points to em_points."""
        
        if len(self.em_points) < 3 or len(self.lm_points) < 3:
            log(f"ERROR: Select at least 3 registration points before applying the transform!")
            return np.identity(3), np.identity(3), np.identity(3), np.identity(3)

        # Truncate points to same length
        A = np.array(self.em_points[:min(len(self.em_points), len(self.lm_points))])
        B = np.array(self.lm_points[:min(len(self.em_points), len(self.lm_points))])

        N = A.shape[0]
        A_padded = np.hstack([A, np.ones((N, 1))])
        
        # Solve for transformation matrix
        T, _, _, _ = np.linalg.lstsq(A_padded, B, rcond=None)
        T = np.vstack([T.T, [0, 0, 1]]) # Add bottom row to make it a 3x3 matrix

        # Split transformation matrix in components
        tx, ty = T[0, 2], T[1, 2]
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

        scale_x = np.sqrt(T[0, 0]**2 + T[1, 0]**2)
        scale_y = np.sqrt(T[0, 1]**2 + T[1, 1]**2)
        scaling_matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ])

        rotation_matrix = np.array([
            [T[0, 0] / scale_x, T[0, 1] / scale_y, 0],
            [T[1, 0] / scale_x, T[1, 1] / scale_y, 0],
            [0, 0, 1]
        ])

        return T, translation_matrix, scaling_matrix, rotation_matrix
    
    def showTransform(self, sender=None, app_data=None, user_data=None, subset=[]):
        """Plots transformed LM map to EM plot."""

        # Clear previous
        if not subset:
            delete_list = [plot["label"] for plot in self.em_plot.img if "lm_map" in plot["label"]] + [""]
        else:
            delete_list = [f"lm_map_{sub}" for sub in subset]
        self.em_plot.clearImg(delete_list)
        self.main_plot.clearImg(delete_list)

        for m, lm_map in enumerate(self.lm_maps):
            # In case of subset selection only transform subset
            if subset and m not in subset:
                continue

            # Get color channel index
            color = ["R", "G", "B", "W"].index(lm_map.channel)

            # Get thresholds
            min_threshold = np.min(lm_map.img_bin) + dpg.get_value(f"flm_thrmin_{m}") * (np.max(lm_map.img_bin) - np.min(lm_map.img_bin))
            max_threshold = np.min(lm_map.img_bin) + dpg.get_value(f"flm_thrmax_{m}") * (np.max(lm_map.img_bin) - np.min(lm_map.img_bin))

            # Flatten image if not white
            if color < 3:
                img = exposure.rescale_intensity(lm_map.img_bin[:, :, color], in_range=(min_threshold, max_threshold), out_range=(0, 255))
            else:
                img = exposure.rescale_intensity(lm_map.img_bin[:, :, :3], in_range=(min_threshold, max_threshold), out_range=(0, 255))

            # Get transformation matrices
            total_matrix, trans_matrix, scale_matrix, rot_matrix = self.calcTransMatrix()
            if (np.all(total_matrix == np.identity(3))):
                return

            # Only rotate and translate and crop, leave scaling to plot to save memory (double output_shape to make sure nothing is cut off during rotation)
            if np.ndim(img) == 3:
                trans_lm_img = np.dstack([affine_transform(img[:, :, dim], trans_matrix @ rot_matrix, output_shape=2 * np.array(img.shape[:2]))[:int(round(scale_matrix[0, 0] * self.loaded_map.img.shape[0])), :int(round(scale_matrix[1, 1] * self.loaded_map.img.shape[1]))] for dim in range(3)])
            else:
                trans_lm_img = affine_transform(img, trans_matrix @ rot_matrix, output_shape=2 * np.array(img.shape[:2]))[:int(round(scale_matrix[0, 0] * self.loaded_map.img.shape[0])), :int(round(scale_matrix[1, 1] * self.loaded_map.img.shape[1]))]

            # Prepare color image
            new_img = np.zeros((trans_lm_img.shape[0], trans_lm_img.shape[1], 4))
            # Set channel and alpha to pixel values
            if color == 3:
                new_img[:, :, :3] = trans_lm_img[:, :, :3]
                new_img[:, :, 3] = np.average(trans_lm_img, axis=2)
            else:
                new_img[:, :, color] = trans_lm_img
                new_img[:, :, 3] = trans_lm_img

            # Add modified image
            self.loaded_map.img_mod = new_img
            texture = self.loaded_map.getTexture(mod=True)

            dims_plot = np.flip(np.array(self.loaded_map.img.shape[:2]) * self.loaded_map.pix_size / 10000)

            # Add to EM plot
            self.em_plot.addImg(texture, [[0, dims_plot[0]], [0, dims_plot[1]]], self.loaded_map.binning, label=f"lm_map_{m}")

            # Add to main plot
            self.main_plot.addImg(texture, [[0, dims_plot[0]], [0, dims_plot[1]]], self.loaded_map.binning, label=f"lm_map_{m}")

            if not dpg.get_value(f"flm_chkbox_{m}"):
                self.changeShow(None, False, m)

    def show(self):
        dpg.show_item(self.window)

    def hide(self):
        dpg.hide_item(self.window)
