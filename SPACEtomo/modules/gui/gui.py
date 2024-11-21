#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_functions_gui
# Purpose:      Functions necessary to run SPACEtomo and that can be run externally without SerialEM.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/04/26
# Revision:     v1.2
# Last Change:  2024/11/20: added a variety of icons, removed static tag from info box
#               2024/08/09: moved most classes to their own module
#               2024/08/08: added coordinate conversion in MMap class
#               2024/07/30: added combo type to Menu
#               2024/06/05: added Menu class
#               2024/06/04: added status box
#               2024/06/03: added contingency for mrc files containing multiple montages, added info box
#               2024/05/31: added status line, fixed tiles bounds
#               2024/05/30: added rescaling, fixed tile_bounds
#               2024/05/29: outsourced plot clearing
#               2024/05/23: added labels to plot boxes, added padding to edge tiles, added loading of mrc files, fixed tile bounds
#               2024/05/10: added map class
#               2024/04/26
# ===================================================================

import numpy as np
from skimage import transform, draw
import dearpygui.dearpygui as dpg

### GUI CONFIG ###

COLORS = {
    "heading":  (255, 200, 0, 255),
    "error":    (200, 0, 0, 255),
    "subtle":   (255, 255, 255, 64)
}
THEME_COLORS = (
    (87, 138, 191, 255),    # blue
    (229, 242, 255, 255),   # light blue
    (59, 92, 128, 255),     # dark blue
    (200, 0, 0, 255),       # red
    (238, 136, 68, 255)     # orange
)

def configureGlobalTheme():
    """Sets global theme settings for all GUIs."""
    
    # Global theme
    with dpg.theme(tag="global_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 5)
        with dpg.theme_component(dpg.mvImageButton):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 5)

    # Theme for large buttons
    with dpg.theme(tag="large_btn_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS["heading"])

    # Theme for small buttons
    with dpg.theme(tag="small_btn_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 1, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
        with dpg.theme_component(dpg.mvImageButton):
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 2, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)

### END CONFIG ###

def cancel_callback():
    pass

def makeLogo(radius=100, stroke=3, oversampling=2):
    radius = radius * oversampling
    stroke = stroke * oversampling

    img = np.zeros([2 * radius, 2 * radius, 4])
    i, j = draw.disk((radius, radius), radius=radius, shape=img.shape[:2])
    img[i, j] = [0, 0, 0, 1]

    i, j = draw.disk((radius, radius), radius=radius - stroke, shape=img.shape[:2])
    gradient = np.array([(x - radius) / radius * np.array([0.965, 0.659, 0.098, 1])  + (1 - (x - radius) / radius) * np.array([0.788, 0.184, 0.157, 1]) for x in i])
    img[i, j] = gradient

    i, j = draw.disk((radius, radius + stroke - 1), radius=radius - stroke * 2, shape=img.shape[:2])
    img[i, j] = [0, 0, 0, 1]

    i, j = draw.disk((radius, radius + stroke - 1), radius=radius - stroke * 3, shape=img.shape[:2])
    img[i, j, :] = 0 

    img = img[radius:2 * radius, :, :]

    img = transform.rescale(img, (1/oversampling, 1/oversampling, 1), anti_aliasing=True)

    logo = np.ravel(img[::-1])
    logo_dims = img.shape

    with dpg.texture_registry():
        dpg.add_static_texture(width=logo_dims[1], height=logo_dims[0], default_value=logo, tag="logo")

    return logo_dims

def makeIconRotation(ccw=False):
    """Makes clockwise arrow icon."""

    if ccw:
        tag = "icon_ccw"
    else:
        tag = "icon_cw"

    if dpg.does_item_exist(tag):
        return tag

    img = np.zeros((13, 13))
    dims = img.shape[:2]

    radius = 5
    i, j, val = draw.circle_perimeter_aa(dims[0] // 2, dims[1] // 2, radius)
    img[i, j] = val

    img[5:10, 9:12] = 0
    img[4, 8:12] = 1
    img[1:4, 11] = 1

    if ccw:
        img = np.flip(img, axis=1)

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconFlip(horizontal=False):
    """Makes double sided arrow icon."""

    if horizontal:
        tag = "icon_fliph"
    else:
        tag = "icon_flipv"

    if dpg.does_item_exist(tag):
        return tag

    img = np.zeros((13, 13))
    dims = img.shape[:2]

    i, j, val = draw.line_aa(1, dims[0] // 2, 3, dims[0] // 2 + 2)
    img[i, j] = val
    i, j, val = draw.line_aa(1, dims[0] // 2, 3, dims[0] // 2 - 2)
    img[i, j] = val

    i, j, val = draw.line_aa(dims[1] - 2, dims[0] // 2, dims[1] - 4, dims[0] // 2 + 2)
    img[i, j] = val
    i, j, val = draw.line_aa(dims[1] - 2, dims[0] // 2, dims[1] - 4, dims[0] // 2 - 2)
    img[i, j] = val

    i, j, val = draw.line_aa(1, dims[0] // 2, dims[1] - 2, dims[0] // 2)
    img[i, j] = val

    if horizontal:
        img = img.T

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconDelete():
    """Makes trashcan icon."""

    tag = "icon_delete"

    if dpg.does_item_exist(tag):
        return tag

    img = np.zeros((13, 13))
    dims = img.shape[:2]
    radius = 2

    # Corners
    i, j, val = draw.circle_perimeter_aa(9, 4, radius)
    img[i, j] = val
    i, j, val = draw.circle_perimeter_aa(9, 8, radius)
    img[i, j] = val
    # Clean center
    img[0:10, 3:10] = 0
    img[0:12, 5:8] = 0
    # Bottom
    i, j, val = draw.line_aa(11, 4, 11, 8)
    img[i, j] = val
    # Sides
    i, j, val = draw.line_aa(4, 2, 9, 2)
    img[i, j] = val
    i, j, val = draw.line_aa(5, 4, 9, 4)
    img[i, j] = val
    i, j, val = draw.line_aa(5, 6, 9, 6)
    img[i, j] = val
    i, j, val = draw.line_aa(5, 8, 9, 8)
    img[i, j] = val
    i, j, val = draw.line_aa(4, 10, 9, 10)
    img[i, j] = val
    # Top

    i, j, val = draw.line_aa(3, 3, 1, 5)
    img[i, j] = val
    i, j, val = draw.line_aa(1, 7, 3, 9)
    img[i, j] = val
    i, j, val = draw.line_aa(3, 1, 3, 11)
    img[i, j] = val
    img[1, 6] = 1

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconEdit():
    """Makes pencil icon."""

    tag = "icon_edit"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    # Pencil sides
    i, j, val = draw.line_aa(7, 3, 0, 10)
    img[i, j] = val
    i, j, val = draw.line_aa(9, 5, 2, 12)
    img[i, j] = val

    # Pencils top/bottom
    i, j, val = draw.line_aa(0, 10, 2, 12)
    img[i, j] = val
    i, j, val = draw.line_aa(7, 3, 9, 5)
    img[i, j] = val
    i, j, val = draw.line_aa(2, 8, 4, 10)
    img[i, j] = val

    # Pencil tip
    i, j, val = draw.line_aa(7, 3, 10, 2)
    img[i, j] += val
    i, j, val = draw.line_aa(9, 5, 10, 2)
    img[i, j] += val

    # Paper
    i, j, val = draw.line_aa(2, 0, 12, 0)
    img[i, j] = val
    i, j, val = draw.line_aa(2, 0, 2, 5)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 0, 12, 8)
    img[i, j] = val
    i, j, val = draw.line_aa(9, 8, 12, 8)
    img[i, j] = val

    img = np.clip(img, 0, 1)

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def window_size_change(logo_dims):
    # Update items anchored to side of window
    dpg.set_item_pos("logo_img", pos=(10, dpg.get_viewport_client_height() - 40 - logo_dims[0]))
    dpg.set_item_pos("logo_text", pos=(10 + logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() - 40 - logo_dims[0] / 2))
    dpg.set_item_pos("version_text", pos=(10 + logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() - 27 - logo_dims[0] / 2))

def askForSave():
    # TODO check for saving conditions
    dpg.stop_dearpygui()

def fileNav(tag, callback, dir=False, extensions=[]):
    with dpg.file_dialog(directory_selector=dir, show=False, callback=callback, tag=tag, cancel_callback=cancel_callback, width=700 ,height=400): 
        dpg.add_file_extension(".*") 
        for ext in extensions:
            dpg.add_file_extension(ext, color=COLORS["heading"])


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

        if dpg.does_item_exist(self.processing_box):
            dpg.delete_item(self.processing_box)
            dpg.split_frame()

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
                self.processing_box = showInfoBox("PROCESSING", self.status, options=None)

def showInfoBox(title, message, callback=None, options=[], options_data=[]):
    """Shows info or confirmation pop up box."""

    # Set color
    if title == "ERROR":
        color = COLORS["error"]
    else:
        color = (255, 255, 255, 255)

    viewport_size = np.array((dpg.get_viewport_client_width(), dpg.get_viewport_client_height()))

    # Create popup window
    with dpg.window(label=title, modal=True, no_close=True) as infobox:
        dpg.add_text(message, color=color)
        if options is not None:
            if callback is None:
                callback = lambda: dpg.delete_item(infobox)
            dpg.add_text()
            with dpg.group(horizontal=True) as infobtns:
                if len(options) > 0:
                    for o, option in enumerate(options):
                        if o < len(options_data):
                            user_data = options_data[o]
                        else:
                            user_data = o
                        dpg.add_button(label=option, user_data=[infobox, user_data], callback=callback)
                else:
                    dpg.add_button(label="OK", callback=callback)

    # Wait for next frame so size and position can be adjusted
    dpg.split_frame()

    window_size = np.array((dpg.get_item_width(infobox), dpg.get_item_height(infobox)))
    dpg.set_item_pos(infobox, pos=(viewport_size - window_size) / 2)    # pos needs to be a float from dpg>=2.0

    # Center buttons
    if options is not None:
        group_size=np.array(dpg.get_item_rect_size(infobtns))
        dpg.set_item_pos(infobtns, ((window_size[0] - group_size[0]) / 2, dpg.get_item_pos(infobtns)[1]))    # pos needs to be a float from dpg>=2.0

    return infobox
