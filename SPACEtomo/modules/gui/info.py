#!/usr/bin/env python
# ===================================================================
# ScriptName:   gui_info
# Purpose:      User interface utility to generate and manage info boxes.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2025/04/10
# Revision:     v1.3
# Last Change:  2025/04/12: outsourced StatusLine, finalized InfoBoxManager 
#               2025/04/10: made class from showInfoBox function (copilot)
# ===================================================================

import dearpygui.dearpygui as dpg
import numpy as np
import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from SPACEtomo.modules.gui.gui import COLORS
from SPACEtomo.modules.utils import log

class InfoBoxManager:
    """Manages a stack of InfoBox instances to ensure only one is displayed at a time."""
    _stack = []
    blocking_windows = [] # List of blocking windows that delay showing InfoBoxes

    @classmethod
    def push(cls, infobox):
        """Pushes a new InfoBox onto the stack and shows it if it's the only one."""

        log(f"DEBUG: Pushing InfoBox to stack: {infobox.title}")
        cls._stack.append(infobox)
        if len(cls._stack) == 1:
            cls.show()

    @classmethod
    def unblock(cls):
        """Determines if InfoBox stack should be unblocked."""

        if not dpg.does_item_exist("show_infobox_handler"):
            if cls._stack and (not dpg.does_item_exist(cls._stack[0].infobox) or not dpg.is_item_shown(cls._stack[0].infobox)):
                for window in cls.blocking_windows:
                    if dpg.is_item_focused(window):
                        return

                # Setting up show InfoBox on mouse move
                with dpg.handler_registry(tag="show_infobox_handler"):
                    dpg.add_mouse_move_handler(callback=cls.show)

    @classmethod
    def show(cls):
        """Shows the current InfoBox in the stack, if any."""

        if dpg.does_item_exist("show_infobox_handler"):
            dpg.delete_item("show_infobox_handler")

        if cls._stack and (not dpg.does_item_exist(cls._stack[0].infobox) or not dpg.is_item_shown(cls._stack[0].infobox)):
            log(f"DEBUG: Showing InfoBox: {cls._stack[0].title}")
            cls._stack[0].show()
        elif cls._stack:
            log(f"DEBUG: NOT YET showing InfoBox: {cls._stack[0].title}")
        else:
            log(f"DEBUG: No InfoBox to show.")

    @classmethod
    def pop(cls):
        """Removes the current InfoBox from the stack and shows the next one, if any."""

        if cls._stack:
            if dpg.does_item_exist(cls._stack[0].infobox):
                dpg.delete_item(cls._stack[0].infobox)
                dpg.split_frame()
            log(f"DEBUG: Popping InfoBox: {cls._stack[0].title}")
            cls._stack.pop(0) # Remove the current InfoBox
            if cls._stack:
                log(f"DEBUG: Showing next InfoBox: {cls._stack[0].title}")
                cls._stack[0].show() # Show the next InfoBox in the stack

class InfoBox:
    """Class to manage the creation and display of info or confirmation pop-up boxes."""

    def __init__(self, title, message, callback=None, options=None, options_data=None, loading=False):
        self.title = title
        self.message = message
        self.callback = callback or self.default_callback
        self.options = options or []
        self.options_data = options_data or []
        self.loading = loading
        self.infobox = None
        self.loading_icon = None
        self.infobtns = None

    def default_callback(self, sender, app_data, user_data):
        """Default callback to delete the info box and manage the stack."""

        #dpg.delete_item(user_data[0])
        InfoBoxManager.pop()

    def show(self):
        """Displays the info box."""

        # Set color
        color = COLORS["error"] if self.title == "ERROR" else (255, 255, 255, 255)

        viewport_size = np.array((dpg.get_viewport_client_width(), dpg.get_viewport_client_height()))

        # Check if window already exists and rerender
        if dpg.does_item_exist(self.infobox):
            dpg.delete_item(self.infobox)

        # Create popup window
        with dpg.window(label=self.title, modal=True, no_close=True) as self.infobox:
            dpg.add_text(self.message, color=color)
            if self.options:
                dpg.add_text()
                with dpg.group(horizontal=True) as self.infobtns:
                    for o, option in enumerate(self.options):
                        user_data = self.options_data[o] if o < len(self.options_data) else o
                        dpg.add_button(label=option, user_data=[self.infobox, user_data], callback=self.callback)
            elif self.loading:
                self.loading_icon = dpg.add_loading_indicator(circle_count=6)
            else:
                with dpg.group(horizontal=True) as self.infobtns:
                    dpg.add_button(label="OK", user_data=[self.infobox, None], callback=self.callback)

        # Adjust position and size
        self._adjust_position(viewport_size)

    def _adjust_position(self, viewport_size):
        """Adjusts the position of the info box and its elements."""

        dpg.split_frame()
        window_size = np.array((dpg.get_item_width(self.infobox), dpg.get_item_height(self.infobox)))
        dpg.set_item_pos(self.infobox, pos=(viewport_size - window_size) / 2)
        #dpg.split_frame()

        # Center buttons
        if not self.loading:
            group_size = np.array(dpg.get_item_rect_size(self.infobtns))
            dpg.set_item_pos(self.infobtns, ((window_size[0] - group_size[0]) / 2, dpg.get_item_pos(self.infobtns)[1]))

        # Center loading icon
        if self.loading and self.loading_icon:
            icon_size = dpg.get_item_rect_size(self.loading_icon)
            dpg.set_item_pos(self.loading_icon, ((window_size[0] - icon_size[0]) / 2, dpg.get_item_pos(self.loading_icon)[1]))


class StatusLine:
    def __init__(self, item=None) -> None:
        self.status = ""
        if item is None:
            self.item = dpg.add_text(default_value=self.status)
        else:
            self.item = item
        dpg.hide_item(self.item)
        dpg.configure_item(self.item, color=COLORS["subtle"])
        self.processing_box = None

    def update(self, status="", color=None, box=False):
        self.status = status

        while InfoBoxManager._stack and InfoBoxManager._stack[0].loading:
            InfoBoxManager.pop()

        dpg.set_value(self.item, self.status)
        if color is not None:
            dpg.configure_item(self.item, color=color)
        else:
            dpg.configure_item(self.item, color=(255, 255, 255, 255))
        if self.status == "":
            dpg.hide_item(self.item)
        else:
            dpg.show_item(self.item)
            if box:
                self.processing_box = InfoBoxManager.push(InfoBox("PROCESSING", self.status, options=None, loading=True))


# TODO: Find better place for shared functions like this
def saveSnapshot(element, file_path):
    """Saves frame buffer to temp file, loads it and crops out element to save as image file."""

    # Save whole frame as temp file (callback to directly obtain frame as array crashes)
    log(f"DEBUG: Saving temp frame buffer image...")
    temp_path = file_path.parent / "temp.png"
    dpg.output_frame_buffer(str(temp_path))

    # Wait until temp file is written
    max_count = 4
    counter = 0
    while not temp_path.exists():
        log(f"DEBUG: Waiting for temp frame buffer image [{temp_path}]...")
        time.sleep(0.5)
        counter += 1
        if counter >= max_count:
            log(f"ERROR: Snapshot failed! Saving snapshots might not be supported yet on your OS.")
            InfoBoxManager.push(InfoBox("ERROR", "Snapshot failed! Saving snapshots might not be supported yet on your OS."))
            return

    # Load temp frame buffer image
    log(f"DEBUG: Loading temp frame buffer image...")
    frame = np.array(Image.open(temp_path))

    # Get bounds of element
    left, top = dpg.get_item_pos(element)
    width, height = dpg.get_item_rect_size(element)

    # Crop element
    log(f"DEBUG: Element cropping dimensions: [{top}: {top + height}, {left}: {left + width}]")
    cropped_frame = frame[top: top + height, left: left + width]

    # Save cropped frame as image
    Image.fromarray(cropped_frame).save(file_path)
    log(f"NOTE: Saved snapshot at {file_path}")
    InfoBoxManager.push(InfoBox("NOTE", f"Snapshot was saved to {file_path}"))

    # Delete temp frame buffer image
    temp_path.unlink()