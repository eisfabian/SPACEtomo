#!/usr/bin/env python
# ===================================================================
# ScriptName:   thmb
# Purpose:      User interface utility to generate window with thumbnails for map selection
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/07
# Revision:     v1.3
# Last Change:  2025/03/14: simplified and added threading for thumbnail creation
#               2024/08/07: separated from old SPACEtomo_tgt.py 
# ===================================================================

import os
os.environ["__GLVND_DISALLOW_PATCHING"] = "1"           # helps to minimize Segmentation fault crashes on Linux when deleting textures
import sys
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
import concurrent.futures

from SPACEtomo.modules.gui import gui
from SPACEtomo.modules.utils import log

class MapWindow:
    """Windows showing thumbnails of all lamella maps in map list."""

    def __init__(self, cur_dir, map_name, map_list, map_list_tgtnum, callback_selectMap, thumbnail_size=(100, 100), executor=None) -> None:
        self.cur_dir = Path(cur_dir)
        self.map_name = map_name
        self.map_list = map_list
        self.map_list_tgtnum = map_list_tgtnum
        self.callback_selectMap = callback_selectMap
        self.thumbnail_size = thumbnail_size

        if isinstance(executor, concurrent.futures.ThreadPoolExecutor):
            self.executor = executor
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor()
        self.futures = []

        self.map_window = dpg.add_window(label="Lamella maps", tag="map_window", no_collapse=True, popup=True, show=False)
        self.map_table = None
        self.map_labels = {}
        self.map_buttons = {}
        self.map_textures = {}
        self.generic_texture = None
        self.texture_registry = dpg.add_texture_registry()

        self.loadThumbnails(first=True)

    def makeMapTable(self, grid_cols=5):
        """Generates table with thumbnails."""

        if self.map_table is None:
            with dpg.group(tag="map_stats", parent=self.map_window):
                dpg.add_text(f"Choose a map to load!", color=gui.COLORS["heading"])
                dpg.add_text(f"Number of maps:    {len(self.map_list)}\nNumber of targets: {np.sum(self.map_list_tgtnum)}")

            with dpg.table(header_row=False, parent=self.map_window) as self.map_table:
                # Add table columns
                for j in range(grid_cols):
                    dpg.add_table_column(init_width_or_weight=self.thumbnail_size[0] + 10, width_fixed=True)

                # Add table rows
                for i in range(int(np.ceil(len(self.map_list) / grid_cols))):
                    with dpg.table_row():
                        for j in range(grid_cols):
                            m_id = i * grid_cols + j
                            # Check if maps are remaining
                            if m_id >= len(self.map_list): break
                            m_name = self.map_list[m_id]

                            try:
                                texture = self.map_textures[m_name]
                            except KeyError:
                                texture = self.generic_texture

                            with dpg.table_cell():
                                # Generate button
                                if m_name == self.map_name:
                                    self.map_buttons[m_name] = dpg.add_image_button(texture, label=m_name, callback=self.callback_selectMap, user_data=m_name, tint_color=gui.COLORS["heading"], enabled=False)
                                    self.map_labels[m_name] = dpg.add_text(m_name, color=gui.COLORS["heading"])
                                else:
                                    self.map_buttons[m_name] = dpg.add_image_button(texture, label=m_name, callback=self.callback_selectMap, user_data=m_name)
                                    self.map_labels[m_name] = dpg.add_text(m_name)
                                
                                # Check target number
                                if self.map_list_tgtnum[m_id] > 0:
                                    dpg.add_text(str(self.map_list_tgtnum[m_id]) + " targets")
                                else:
                                    dpg.add_text("No targets", color=gui.COLORS["error"])

                                # Check inspected status
                                if (self.cur_dir / (m_name + "_inspected.txt")).exists():
                                    dpg.add_text("Inspected", color=gui.COLORS["heading"])
                                else:
                                    # Show inspect button if not currently loaded
                                    if m_name == self.map_name:
                                        dpg.add_text("Not inspected")
                                    else:
                                        dpg.add_button(tag=f"map_btn_ins_{m_id}", label="Mark inspected", callback=self.markInspected, user_data=m_name)

    def deleteMapTable(self):
        """Deletes thumbnail table."""

        if self.map_table is not None:
            if dpg.does_item_exist(self.map_table):
                dpg.delete_item(self.map_table)
            self.map_table = None

        # Delete stats and button as well
        if dpg.does_item_exist("map_stats"):
            dpg.delete_item("map_stats")
        if dpg.does_item_exist("map_butthumb"):
            dpg.delete_item("map_butthumb")

    def loadThumbnails(self, first=False):
        """Creates thumbnails of lamella maps or load from file if they already exist."""

        if first: 
            self.makeGenericThumbnail()

        for m_name in self.map_list:
            if m_name not in self.map_textures.keys():
                self.futures.append(self.executor.submit(self.makeThumbnail, m_name))

    def makeThumbnail(self, m_name, map_img=None, override=False):
        """Makes thumbnail from map or loads it if file is available."""

        # Make folder 
        if not (self.cur_dir / "thumbnails").exists():
            (self.cur_dir / "thumbnails").mkdir(parents=True)
        
        # If thumbnail file already exists
        thumbnail_path = self.cur_dir / "thumbnails" / (m_name + ".png")
        if thumbnail_path.exists():
            map_img = Image.open(thumbnail_path)
        else:
            # Open full map if not opened yet
            if map_img is None:
                map_img = Image.open(self.cur_dir / (m_name + ".png"))
            # Save thumbnail
            map_img.thumbnail(self.thumbnail_size)
            map_img.save(thumbnail_path)

        log(f"Creating thumbnail for {m_name}...")
        self.map_textures[m_name] = self.makeTexture(map_img)

    def makeGenericThumbnail(self):
        """Makes a generic schematic lamella thumbnail."""

        size = np.flip(self.thumbnail_size)

        thumbnail = np.zeros(size)
        thumbnail[size[0] // 3: size[0] // 3 * 2, 0: size[1] // 5] = 1
        thumbnail[size[0] // 3: size[0] // 3 * 2, size[1] // 5 * 4: size[1]] = 1
        thumbnail[size[0] // 3: size[0] // 3 * 2, size[1] // 5: size[1] // 5 * 4] = 0.6
        thumbnail[size[0] // 3: size[0] // 3 * 2, size[1] // 5: size[1] // 7 * 2] = 0.3

        # Expansion joints
        thumbnail[size[0] // 5: size[0] // 4, size[1] // 7: size[1] // 7 * 6] = 1
        thumbnail[size[0] // 4 * 3: size[0] // 5 * 4, size[1] // 7: size[1] // 7 * 6] = 1

        self.generic_texture = self.makeTexture(thumbnail)

    def makeTexture(self, image):
        """Creates texture from thumbnail."""

        # Make sure image is of type float
        image = np.array(image)
        if np.max(image) > 1:
            image = image / 255

        # Pad image to size
        map_thumb = np.zeros(np.flip(self.thumbnail_size), dtype=float)
        map_thumb[(self.thumbnail_size[1] - image.shape[0]) // 2: (self.thumbnail_size[1] + image.shape[0]) // 2, (self.thumbnail_size[0] - image.shape[1]) // 2: (self.thumbnail_size[0] + image.shape[1]) // 2] = image

        # Add texture
        map_thumb = np.ravel(np.dstack([map_thumb, map_thumb, map_thumb, np.ones(map_thumb.shape)]))
        return dpg.add_static_texture(width=self.thumbnail_size[0], height=self.thumbnail_size[1], default_value=map_thumb, parent=self.texture_registry)
    
    def show(self):
        dpg.show_item(self.map_window)

    def hide(self):
        dpg.hide_item(self.map_window)

    def update(self, map_name, map_list, map_list_tgtnum):
        """Updates map data and reloads table."""

        self.map_name = map_name
        self.map_list = map_list
        self.map_list_tgtnum = map_list_tgtnum

        if self.futures:
            self.futures = [future for future in self.futures if not future.done()]

        self.loadThumbnails()
        self.deleteMapTable()
        self.makeMapTable()

    def updateMap(self, map_name):
        """Updates only map name for highlighting."""

        if map_name in self.map_list:
            if self.map_name in self.map_list:
                # Unhighlight old selected map
                dpg.configure_item(self.map_buttons[self.map_name], tint_color=(255, 255, 255, 255), enabled=True)
                dpg.configure_item(self.map_labels[self.map_name], color=(255, 255, 255, 255))

            # Highlight new selected map
            self.map_name = map_name
            dpg.configure_item(self.map_buttons[self.map_name], tint_color=gui.COLORS["heading"], enabled=False)
            dpg.configure_item(self.map_labels[self.map_name], color=gui.COLORS["heading"])

    def markInspected(self, sender, app_data, user_data):
        """Makes single map as inspected."""

        map_name = user_data
        (self.cur_dir / (map_name + "_inspected.txt")).touch()
        self.deleteMapTable()
        self.makeMapTable()