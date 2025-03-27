#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_functions_gui
# Purpose:      Functions necessary to run SPACEtomo and that can be run externally without SerialEM.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/04/26
# Revision:     v1.3
# Last Change:  2025/03/14: added dense pattern icon
#               2025/03/07: added more icons, added toggled button theme
#               2024/12/20: added arrow icons for cursor
#               2024/11/20: added a variety of icons, removed static tag from info box
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

import time
import numpy as np
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage import transform, draw
import dearpygui.dearpygui as dpg

from SPACEtomo.modules.utils import log

### GUI CONFIG ###

COLORS = {
    "heading":  (255, 200, 0, 255), # yellow
    "error":    (200, 0, 0, 255),   # red
    "geo":      (238, 136, 68, 255),# orange
    "subtle":   (255, 255, 255, 64) # white, semi-transparent
}
THEME_COLORS = (
    (87, 138, 191, 255),    # blue
    (229, 242, 255, 255),   # light blue
    (59, 92, 128, 255),     # dark blue
    (138, 87, 191, 255),    # purple
    (200, 180, 255),        # light purple
    (92, 59, 128, 255),     # dark purple
    (87, 191, 138, 255),    # green
    (180, 255, 200, 255),   # light green
    (59, 128, 92, 255),     # dark green
)

ICON_SIZE = np.array((13, 13))

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

    # Theme for active buttons
    with dpg.theme(tag="active_btn_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, COLORS["heading"])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (np.array(COLORS["heading"]) * 0.9).astype(int).tolist())
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (np.array(COLORS["heading"]) * 0.8).astype(int).tolist())
        with dpg.theme_component(dpg.mvImageButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, COLORS["heading"])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (np.array(COLORS["heading"]) * 0.9).astype(int).tolist())
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (np.array(COLORS["heading"]) * 0.8).astype(int).tolist())

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

    logo_text = np.flip(makeIco(), axis=0) / 255
    start_x, start_y = (np.array(img.shape[:2]) - np.array(logo_text.shape)) // 2 + np.array([-16, 0])
    img[start_x: start_x + logo_text.shape[0], start_y: start_y + logo_text.shape[1], :] = np.dstack([logo_text] * 4)

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

def makeIconSnapshot():
    """Makes camera icon."""

    tag = "icon_snapshot"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    radius = 3
    i, j, val = draw.circle_perimeter_aa(dims[0] // 2 + 1, dims[1] // 2, radius)
    img[i, j] = val

    # Flash
    i, j, val = draw.line_aa(3, 3, 1, 5)
    img[i, j] = val
    i, j, val = draw.line_aa(1, 5, 1, 7)
    img[i, j] = val
    i, j, val = draw.line_aa(1, 7, 3, 9)
    img[i, j] = val

    # Box
    i, j, val = draw.line_aa(3, 0, 11, 0)
    img[i, j] = val
    i, j, val = draw.line_aa(3, 0, 3, 12)
    img[i, j] = val
    i, j, val = draw.line_aa(11, 0, 11, 12)
    img[i, j] = val
    i, j, val = draw.line_aa(3, 12, 11, 12)
    img[i, j] = val

    img = np.clip(img, 0, 1)

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconResetZoom():
    """Makes zoom reset icon."""

    tag = "icon_resetzoom"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    # Top left
    i, j, val = draw.line_aa(1, 1, 4, 4)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 0, 0, 3)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 0, 3, 0)
    img[i, j] = val

    # Top right
    i, j, val = draw.line_aa(4, 8, 1, 11)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 12, 3, 12)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 12, 0, 9)
    img[i, j] = val

    # Bottom right
    i, j, val = draw.line_aa(8, 8, 11, 11)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 12, 9, 12)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 12, 12, 9)
    img[i, j] = val

    # Bottom left
    i, j, val = draw.line_aa(8, 4, 11, 1)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 0, 9, 0)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 0, 12, 3)
    img[i, j] = val

    img = np.clip(img, 0, 1)

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconShift():
    """Makes shift icon."""

    tag = "icon_shift"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    # Top down
    i, j, val = draw.line_aa(0, 6, 2, 8)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 6, 2, 4)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 6, 10, 8)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 6, 10, 4)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 6, 12, 6)
    img[i, j] = val

    # Left Right
    i, j, val = draw.line_aa(6, 0, 8, 2)
    img[i, j] = val
    i, j, val = draw.line_aa(6, 0, 4, 2)
    img[i, j] = val
    i, j, val = draw.line_aa(6, 12, 8, 10)
    img[i, j] = val
    i, j, val = draw.line_aa(6, 12, 4, 10)
    img[i, j] = val
    i, j, val = draw.line_aa(6, 0, 6, 12)
    img[i, j] = val

    # Center
    img[5:8, 5:8] = 0

    img = np.clip(img, 0, 1)

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconShiftLR():
    """Makes shift icon."""

    tag = "icon_shift_lr"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    # Left Right
    i, j, val = draw.line_aa(6, 0, 8, 2)
    img[i, j] = val
    i, j, val = draw.line_aa(6, 0, 4, 2)
    img[i, j] = val
    i, j, val = draw.line_aa(6, 12, 8, 10)
    img[i, j] = val
    i, j, val = draw.line_aa(6, 12, 4, 10)
    img[i, j] = val
    i, j, val = draw.line_aa(6, 0, 6, 12)
    img[i, j] = val

    # Center
    img[5:8, 5:8] = 0

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconShiftUD():
    """Makes shift icon."""

    tag = "icon_shift_ud"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    # Top down
    i, j, val = draw.line_aa(0, 6, 2, 8)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 6, 2, 4)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 6, 10, 8)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 6, 10, 4)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 6, 12, 6)
    img[i, j] = val

    # Center
    img[5:8, 5:8] = 0

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconShiftULDR():
    """Makes shift icon."""

    tag = "icon_shift_uldr"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    # Top left
    i, j, val = draw.line_aa(1, 1, 4, 4)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 0, 0, 3)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 0, 3, 0)
    img[i, j] = val

    # Bottom right
    i, j, val = draw.line_aa(8, 8, 11, 11)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 12, 9, 12)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 12, 12, 9)
    img[i, j] = val

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconShiftURDL():
    """Makes shift icon."""

    tag = "icon_shift_urdl"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    # Top right
    i, j, val = draw.line_aa(4, 8, 1, 11)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 12, 3, 12)
    img[i, j] = val
    i, j, val = draw.line_aa(0, 12, 0, 9)
    img[i, j] = val

    # Bottom left
    i, j, val = draw.line_aa(8, 4, 11, 1)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 0, 9, 0)
    img[i, j] = val
    i, j, val = draw.line_aa(12, 0, 12, 3)
    img[i, j] = val

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconGrid():
    """Makes grid icon."""

    tag = "icon_grid"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    img[:, [2, 6, 10]] = 1
    img[[2, 6, 10], :] = 1

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconHoles():
    """Makes holes icon."""

    tag = "icon_holes"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    top_left = [[0, 0], [0, 5], [0, 10], [5, 0], [5, 5], [5, 10], [10, 0], [10, 5], [10, 10]]

    for x, y in top_left:
        img[np.array([1, 0, 2, 1]) + x, np.array([0, 1, 1, 2]) + y] = 1
        img[np.array([0, 2, 0, 2]) + x, np.array([0, 0, 2, 2]) + y] = 0.5

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconDense():
    """Makes dense pattern icon."""

    tag = "icon_dense"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    top_left = [[3, 1], [1, 5], [3, 9], [7, 1], [5, 5], [9, 5], [7, 9]]

    for x, y in top_left:
        img[np.array([1, 0, 2, 1]) + x, np.array([0, 1, 1, 2]) + y] = 1
        img[np.array([0, 2, 0, 2]) + x, np.array([0, 0, 2, 2]) + y] = 0.5

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconPolygon():
    """Makes polygon icon."""

    tag = "icon_polygon"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    i, j, val = draw.line_aa(0, 6, 5, 0)
    img[i, j] = val

    i, j, val = draw.line_aa(0, 6, 5, 12)
    img[i, j] = val

    i, j, val = draw.line_aa(12, 3, 5, 0)
    img[i, j] = val

    i, j, val = draw.line_aa(12, 9, 5, 12)
    img[i, j] = val

    i, j, val = draw.line_aa(12, 3, 12, 9)
    img[i, j] = val

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIconSave():
    """Makes save icon."""

    tag = "icon_save"

    if dpg.does_item_exist(tag):
        return tag
    img = np.zeros((13, 13))
    dims = img.shape[:2]

    # Left
    i, j, val = draw.line_aa(1, 1, 11, 1)
    img[i, j] = val

    # Right
    i, j, val = draw.line_aa(5, 11, 11, 11)
    img[i, j] = val

    # Bottom
    i, j, val = draw.line_aa(11, 1, 11, 11)
    img[i, j] = val

    # Top
    i, j, val = draw.line_aa(1, 1, 1, 8)
    img[i, j] = val

    # Corner
    i, j, val = draw.line_aa(4, 11, 1, 8)
    img[i, j] = val

    # Bottom rect
    i, j, val = draw.line_aa(7, 3, 11, 3)
    img[i, j] = val
    i, j, val = draw.line_aa(7, 9, 11, 9)
    img[i, j] = val
    i, j, val = draw.line_aa(7, 3, 7, 9)
    img[i, j] = val

    # Top rect
    i, j, val = draw.line_aa(2, 3, 4, 3)
    img[i, j] = val
    i, j, val = draw.line_aa(2, 8, 4, 8)
    img[i, j] = val
    i, j, val = draw.line_aa(2, 7, 4, 7)
    img[i, j] = val

    i, j, val = draw.line_aa(4, 4, 4, 8)
    img[i, j] = val

    tex_img = np.ravel(np.dstack([np.ones(dims), np.ones(dims), np.ones(dims), img]))

    with dpg.texture_registry():
        dpg.add_static_texture(width=dims[1], height=dims[0], default_value=tex_img, tag=tag)

    return tag

def makeIco():
    """Makes icon for ICO file."""

    logo_center = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 125, 170, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 66, 0, 0, 0, 0, 156, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 197, 156, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 141, 184, 233, 255, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 156, 197, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 141, 0, 0, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 125, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 233, 0, 0, 0, 0, 0, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 197, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 233, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 209, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 107, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0], [0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 170, 0, 0, 0, 0, 88, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 233, 255, 255, 255, 255, 255, 255, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 244, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0], [0, 0, 0, 0, 125, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 125, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 221, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 244, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 184, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0], [0, 0, 0, 0, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 0, 0, 0, 0, 0, 221, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0, 0, 0, 0, 233, 255, 255, 255, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 221, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 141, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 233, 0, 0, 0, 0, 0], [0, 0, 0, 88, 255, 255, 255, 255, 255, 221, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 244, 255, 255, 255, 255, 233, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 244, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0, 0, 233, 255, 255, 255, 255, 107, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 221, 255, 255, 255, 255, 255, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 107, 255, 255, 255, 255, 255, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 125, 170, 221, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 141, 255, 255, 255, 255, 184, 0, 184, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0, 66, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 38, 255, 255, 255, 255, 255, 244, 125, 107, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 233, 255, 255, 255, 255, 66, 0, 66, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 156, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 88, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 209, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 233, 170, 66, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 255, 38, 0, 0, 0, 141, 255, 255, 255, 255, 209, 0, 0, 0, 197, 255, 255, 255, 255, 184, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 88, 244, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 141, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 184, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 209, 255, 255, 255, 255, 255, 221, 0, 0, 0, 0, 233, 255, 255, 255, 255, 107, 0, 0, 0, 107, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 88, 244, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 141, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 125, 0, 0, 0, 141, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 184, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 66, 184, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 38, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 184, 0, 0, 0, 0, 233, 255, 255, 255, 255, 141, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 66, 0, 0, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 38, 107, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 233, 255, 255, 255, 255, 255, 156, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 170, 0, 0, 0, 0, 141, 255, 255, 255, 255, 233, 0, 0, 0, 0, 0, 0, 0, 221, 255, 255, 255, 255, 184, 0, 0, 0, 107, 255, 255, 255, 255, 255, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 184, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 141, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 233, 255, 255, 255, 255, 197, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 233, 184, 66, 0, 0, 0, 0, 0, 233, 255, 255, 255, 255, 141, 0, 66, 107, 107, 107, 107, 107, 170, 255, 255, 255, 255, 255, 66, 0, 0, 38, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 209, 0, 0, 0, 255, 255, 255, 255, 255, 156, 107, 107, 107, 107, 107, 107, 107, 107, 107, 66, 0, 0, 0, 0, 0, 0, 0, 0, 125, 255, 255, 255, 255, 233, 0, 0, 197, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 170, 0, 0, 0, 221, 255, 255, 255, 255, 244, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 244, 255, 255, 255, 255, 197, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 156, 0, 66, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 38, 0, 0, 141, 255, 255, 255, 255, 255, 221, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 197, 244, 255, 255, 255, 255, 255, 156, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 244, 38, 0, 170, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 221, 255, 255, 255, 255, 255, 255, 209, 170, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 107, 0, 0, 255, 255, 255, 255, 255, 184, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 0, 0, 0, 0], [0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 38, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 156, 0, 38, 244, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 38, 0, 0, 88, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 125, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 221, 0, 0, 0, 0], [0, 0, 0, 0, 170, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 125, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 244, 255, 255, 255, 255, 156, 0, 0, 0, 141, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0], [0, 0, 0, 0, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 233, 107, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 244, 38, 0, 0, 0, 125, 244, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 197, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 88, 0, 0, 0, 0], [0, 0, 0, 66, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 209, 141, 38, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 244, 255, 255, 255, 255, 156, 0, 0, 0, 0, 38, 156, 209, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 141, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 233, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 107, 107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 107, 107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 141, 0, 0, 0, 0, 0, 0, 0, 107, 170, 221, 255, 255, 255, 255, 255, 255, 255, 209, 156, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 170, 221, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 209, 209, 209, 209, 209, 88, 0, 0, 0, 0, 0, 0, 0, 0, 107, 170, 221, 255, 255, 255, 255, 255, 255, 255, 209, 156, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 88, 0, 0, 0, 0, 0, 107, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 88, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 107, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 233, 0, 0, 0, 0, 0, 184, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 184, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 184, 0, 0, 0, 0, 221, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 170, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 233, 255, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 221, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 125, 0, 0, 0, 170, 255, 255, 255, 255, 255, 255, 255, 221, 184, 156, 156, 156, 184, 221, 255, 255, 255, 255, 255, 255, 255, 125, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 233, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 255, 255, 107, 0, 0, 0, 170, 255, 255, 255, 255, 255, 255, 255, 221, 184, 156, 156, 156, 184, 221, 255, 255, 255, 255, 255, 255, 255, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 255, 184, 66, 0, 0, 0, 0, 0, 0, 0, 66, 209, 255, 255, 255, 255, 255, 244, 38, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 244, 255, 255, 255, 255, 255, 255, 255, 107, 0, 0, 107, 255, 255, 255, 255, 255, 255, 184, 66, 0, 0, 0, 0, 0, 0, 0, 66, 209, 255, 255, 255, 255, 255, 244, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 197, 255, 255, 255, 255, 255, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 184, 255, 255, 255, 255, 255, 156, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 244, 38, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 255, 255, 255, 107, 0, 0, 197, 255, 255, 255, 255, 255, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 184, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 255, 255, 255, 255, 255, 197, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 233, 255, 255, 255, 255, 221, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0, 0, 0, 0, 66, 244, 255, 255, 255, 255, 255, 255, 255, 255, 107, 0, 38, 255, 255, 255, 255, 255, 197, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 233, 255, 255, 255, 255, 221, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 255, 255, 255, 255, 255, 38, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 66, 0, 0, 0, 0, 0, 0, 184, 255, 255, 255, 255, 255, 255, 255, 255, 255, 107, 0, 107, 255, 255, 255, 255, 255, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 255, 255, 255, 255, 255, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 184, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 244, 255, 255, 255, 255, 255, 107, 0, 156, 255, 255, 255, 255, 255, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 255, 255, 255, 255, 255, 197, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0, 209, 255, 255, 255, 255, 156, 255, 255, 255, 255, 255, 107, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 255, 255, 255, 255, 255, 107, 233, 255, 255, 255, 255, 209, 0, 0, 0, 125, 255, 255, 255, 255, 233, 0, 255, 255, 255, 255, 255, 107, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 255, 255, 255, 255, 255, 107, 107, 255, 255, 255, 255, 255, 107, 0, 0, 233, 255, 255, 255, 255, 107, 0, 255, 255, 255, 255, 255, 107, 0, 156, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 66, 0, 0, 255, 255, 255, 255, 255, 107, 0, 197, 255, 255, 255, 255, 209, 0, 156, 255, 255, 255, 255, 209, 0, 0, 255, 255, 255, 255, 255, 107, 0, 125, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 255, 255, 255, 255, 255, 184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 244, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 66, 244, 255, 255, 255, 255, 156, 244, 255, 255, 255, 255, 88, 0, 0, 255, 255, 255, 255, 255, 107, 0, 66, 255, 255, 255, 255, 255, 184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 221, 255, 255, 255, 255, 255, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 184, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 156, 255, 255, 255, 255, 255, 255, 255, 255, 255, 184, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 221, 255, 255, 255, 255, 255, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 244, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 255, 66, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 38, 233, 255, 255, 255, 255, 255, 255, 255, 244, 38, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 141, 255, 255, 255, 255, 255, 244, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 255, 255, 255, 255, 255, 255, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 255, 255, 255, 255, 255, 255, 221, 156, 125, 107, 107, 107, 141, 170, 233, 255, 255, 255, 255, 255, 255, 170, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 125, 255, 255, 255, 255, 255, 255, 255, 156, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 209, 255, 255, 255, 255, 255, 255, 221, 156, 125, 107, 107, 107, 141, 170, 233, 255, 255, 255, 255, 255, 255, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 221, 38, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 209, 255, 255, 255, 255, 255, 233, 38, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 38, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 221, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 221, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 197, 38, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 88, 255, 255, 255, 255, 255, 141, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 66, 221, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 197, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 170, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 156, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 170, 255, 255, 255, 221, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 38, 170, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 170, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 141, 38, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 38, 107, 107, 107, 66, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 107, 0, 0, 0, 0, 0, 0, 0, 66, 170, 233, 255, 255, 255, 255, 255, 255, 255, 255, 255, 209, 141, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 107, 156, 156, 156, 156, 125, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 107, 156, 156, 156, 156, 125, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    logo = np.zeros((128, 128), dtype=np.uint8)
    start_x, start_y = (np.array(logo.shape) - np.array(logo_center.shape)) // 2
    logo[start_x: start_x + logo_center.shape[0], start_y: start_y + logo_center.shape[1]] = logo_center

    if not Path(__file__).parent.joinpath("logo.ico").exists():
        # Default standard icon sizes
        ico_sizes = [
            (16, 16),   # System tray, taskbar
            (32, 32),   # Default desktop icon
            (48, 48),   # Larger desktop icon
            (64, 64),   # High DPI
            (128, 128), # Modern high-res
        ]
        img = Image.fromarray(logo)
        img.save(Path(__file__).parent.joinpath("logo.ico"), sizes=ico_sizes)

    return logo_center

def window_size_change(logo_dims):
    # Update items anchored to side of window
    dpg.set_item_pos("logo_img", pos=(10, dpg.get_viewport_client_height() - 40 - logo_dims[0]))
    #dpg.set_item_pos("logo_text", pos=(10 + logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() - 40 - logo_dims[0] / 2))
    dpg.set_item_pos("version_text", pos=(10 + logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() + 5 - logo_dims[0] / 2))

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
                self.processing_box = showInfoBox("PROCESSING", self.status, options=None, loading=True)

def showInfoBox(title, message, callback=None, options=[], options_data=[], loading=False):
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
        elif loading:
            loading_icon = dpg.add_loading_indicator(circle_count=6)

    # Wait for next frame so size and position can be adjusted
    dpg.split_frame()

    window_size = np.array((dpg.get_item_width(infobox), dpg.get_item_height(infobox)))
    dpg.split_frame() # seems to help reducing misplaced windows
    dpg.set_item_pos(infobox, pos=(viewport_size - window_size) / 2)    # pos needs to be a float from dpg>=2.0

    # Center buttons
    if options is not None:
        group_size = np.array(dpg.get_item_rect_size(infobtns))
        dpg.set_item_pos(infobtns, ((window_size[0] - group_size[0]) / 2, dpg.get_item_pos(infobtns)[1]))    # pos needs to be a float from dpg>=2.0

    # Center loading icon
    if loading:
        icon_size = dpg.get_item_rect_size(loading_icon)
        dpg.set_item_pos(loading_icon, ((window_size[0] - icon_size[0]) / 2, dpg.get_item_pos(loading_icon)[1]))    # pos needs to be a float from dpg>=2.0

    return infobox

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
            showInfoBox("ERROR", "Snapshot failed! Saving snapshots might not be supported yet on your OS.")
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
    showInfoBox("NOTE", f"Snapshot was saved to {file_path}")

    # Delete temp frame buffer image
    temp_path.unlink()