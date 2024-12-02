#!/usr/bin/env python
# ===================================================================
# ScriptName:   gui_plt
# Purpose:      User interface utility to generate plot
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/08
# Revision:     v1.2
# Last Change:  2024/11/18: moved all texture deletions after plot deletions
#               2024/11/12: added annotations
#               2024/11/08: added border dragging of boxes
#               2024/08/08: separated from gui.py 
# ===================================================================

import time
import numpy as np
import dearpygui.dearpygui as dpg

from SPACEtomo.modules.gui.gui import COLORS

class Plot:
    """Class to hold and organize all plotted objects."""

    def __init__(self, tag) -> None:
        self.plot = None
        self.x_axis = None
        self.y_axis = None

        self.tag = tag

        self.img = []
        self.boxes = []
        self.overlays = []
        self.series = []
        self.drag_points = []
        self.annotations = []

        self.bounds = np.zeros((2, 2))  # [[xmin, xmax], [ymin, ymax]]

    def sortList(self, list_name, sortable):
        """Sorts list of plots according to label."""

        unsorted_list = getattr(self, list_name)
        if isinstance(unsorted_list, list) and len(unsorted_list) > 1:
            if hasattr(unsorted_list[0], sortable):
                sortable_list = [getattr(item, sortable) for item in unsorted_list]
                sorted_list = [item for _, item in sorted(zip(sortable_list, unsorted_list))]
                setattr(self, list_name, sorted_list)

    def makePlot(self, x_axis_label="x", y_axis_label="y", **kwargs):
        """Creates base plot object."""

        self.plot = dpg.add_plot(tag=self.tag, **kwargs)

        self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label=x_axis_label, parent=self.plot)#, tag="x_axis")
        self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=y_axis_label, parent=self.plot)#, tag="y_axis")

    def resetZoom(self):
        """Auto zooms to plot bounds."""

        # Fit axes
        dpg.fit_axis_data(self.x_axis)
        dpg.fit_axis_data(self.y_axis)

    def updateLabel(self, label=""):
        """Updates the plot heading."""

        dpg.set_item_label(self.plot, label)

    def addImg(self, texture, bounds, binning=1, label=""):
        """Plots an image."""

        bounds = np.array(bounds)
        self.img.append({"label": label, "tex": texture, "plot": dpg.add_image_series(texture, bounds_min=(bounds[0, 0], bounds[1, 0]), bounds_max=(bounds[0, 1], bounds[1, 1]), parent=self.x_axis), "binning": binning})
        self.updateBounds(bounds)

    def addSeries(self, x_vals, y_vals, label="", theme=""):
        """Plots a scatter series."""

        self.series.append({"label": label, "plot": dpg.add_scatter_series(list(x_vals), list(y_vals), parent=self.x_axis)})
        if theme and dpg.does_item_exist(theme):
            dpg.bind_item_theme(self.series[-1]["plot"], theme)

    def addAnnotation(self, text, x, y, offset=(10, -10), label="", color=COLORS["heading"]):
        """Plots annotation label."""

        self.annotations.append({"label": label, "annotation": dpg.add_plot_annotation(label=text, default_value=(x, y), offset=offset, color=color, clamped=False, parent=self.plot)})

    def addDragPoint(self, x, y, label, callback=None, user_data=None, color=(255, 255, 255, 255)):
        """Creates a drag point."""

        self.drag_points.append({"label": label, "plot": dpg.add_drag_point(default_value=(x, y), user_data=user_data, color=color, label=label, callback=callback, parent=self.plot)})

    def addOverlay(self, texture, bounds, label):
        """Plots an image (meant for small overlays)."""

        self.overlays.append({"label": label, "tex": texture, "plot": dpg.add_image_series(texture, bounds_min=(bounds[0][0], bounds[1][0]), bounds_max=(bounds[0][1], bounds[1][1]), parent=self.x_axis)})

    def addCustomPlot(self, list, label, plot, theme=None):
        """Adds custom series to any list of self."""

        list.append({"label": label, "plot": plot})
        if theme:
            dpg.bind_item_theme(plot, theme)

    def updateBounds(self, bounds):
        """Updates bounds of plot."""

        self.bounds[0, 0] = min(self.bounds[0, 0], bounds[0, 0])        # xmin
        self.bounds[0, 1] = max(self.bounds[0, 1], bounds[0, 1])        # xmax
        self.bounds[1, 0] = min(self.bounds[1, 0], bounds[1, 0])        # ymin
        self.bounds[1, 1] = max(self.bounds[1, 1], bounds[1, 1])        # ymax

    def withinBounds(self, coords):
        """Checks if coordinates are within the plot bounds."""

        x, y = coords
        if self.bounds[0, 0] <= x <= self.bounds[0, 1] and self.bounds[1, 0] <= y <= self.bounds[1, 1]:
            return True
        else:
            return False
        
    def updateImg(self, id, texture=None, bounds=None):
        """Updates texture of plotted image series."""

        if texture is not None:
            # Old texture
            old_tex = self.img[id]["tex"]

            # Update texture
            self.img[id]["tex"] = texture
            dpg.configure_item(self.img[id]["plot"], texture_tag=texture)

            # Delete old texture
            if dpg.does_item_exist(old_tex): 
                dpg.delete_item(old_tex)
                dpg.split_frame(delay=10)                       # helps to reduce Segmentation fault crashes

        # Update bounds
        if bounds is not None:
            dpg.configure_item(self.img[id]["plot"], bounds_min=(bounds[0][0], bounds[1][0]), bounds_max=(bounds[0][1], bounds[1][1]))

    def updateOverlay(self, id, texture=None, bounds=None):
        """Updates texture of plotted overlay image series."""

        if texture is not None:
            # Old texture
            old_tex = self.overlays[id]["tex"]

            # Update texture
            self.overlays[id]["tex"] = texture
            dpg.configure_item(self.overlays[id]["plot"], texture_tag=texture)

            # Delete old texture
            if dpg.does_item_exist(old_tex): 
                dpg.delete_item(old_tex)
                dpg.split_frame(delay=10)                       # helps to reduce Segmentation fault crashes

        # Update bounds
        if bounds is not None:
            dpg.configure_item(self.overlays[id]["plot"], bounds_min=(bounds[0][0], bounds[1][0]), bounds_max=(bounds[0][1], bounds[1][1]))

    """Methods to find specific plot elements whose label contains a keyword."""
    def getImgByKeyword(self, keyword):
        return [item["label"] for item in self.img if keyword in item["label"]]
    
    def getSeriesByKeyword(self, keyword):
        return [item["label"] for item in self.series if keyword in item["label"]]
    
    def getDragPointsByKeyword(self, keyword):
        return [item["label"] for item in self.drag_points if keyword in item["label"]]

    def getOverlaysByKeyword(self, keyword):
        return [item["label"] for item in self.overlays if keyword in item["label"]]    

    def clearImg(self, labels=[], delete_textures=True):
        """Delete all or some plotted images and their textures."""

        remaining_list = []
        for img in self.img:
            if labels and img["label"] not in labels:       # Option for removing only certain images
                remaining_list.append(img)
                continue
            if dpg.does_item_exist(img["plot"]): 
                dpg.delete_item(img["plot"])
            if delete_textures and dpg.does_item_exist(img["tex"]): 
                dpg.delete_item(img["tex"])
                dpg.split_frame(delay=10)                   # helps to reduce Segmentation fault crashes

        self.img = remaining_list

    def clearSeries(self, labels=[]):
        """Delete all or some plotted scatter series."""

        remaining_list = []
        for series in self.series:
            if labels and series["label"] not in labels:    # Option for removing only certain series
                remaining_list.append(series)
                continue
            if dpg.does_item_exist(series["plot"]): 
                dpg.delete_item(series["plot"])
        self.series = remaining_list

    def clearBoxes(self):
        """Delete all plotted boxes."""

        for box in self.boxes:
            box.remove()
            if dpg.does_item_exist(box):
                dpg.delete_item(box)
        self.boxes = []

    def clearOverlays(self, labels=[], delete_textures=True):
        """Delete all or some plotted overlays and their textures."""

        remaining_list = []
        for overlay in self.overlays:
            if labels and overlay["label"] not in labels:   # Option for removing only certain series
                remaining_list.append(overlay)
                continue
            if dpg.does_item_exist(overlay["plot"]): 
                dpg.delete_item(overlay["plot"])
            if delete_textures and dpg.does_item_exist(overlay["tex"]):
                dpg.delete_item(overlay["tex"])
                dpg.split_frame(delay=10)                   # helps to reduce Segmentation fault crashes

        self.overlays = remaining_list

    def clearDragPoints(self, labels=[]):
        """Delete all or some drag points."""

        remaining_list = []
        for point in self.drag_points:
            if labels and point["label"] not in labels:     # Option for removing only certain series
                remaining_list.append(point)
                continue
            if dpg.does_item_exist(point["plot"]):
                dpg.delete_item(point["plot"])
        self.drag_points = remaining_list

    def clearAnnotations(self, labels=[]):
        """Delete all or some annotations."""

        remaining_list = []
        for annotation in self.annotations:
            if labels and annotation["label"] not in labels:     # Option for removing only certain series
                remaining_list.append(annotation)
                continue
            if dpg.does_item_exist(annotation["annotation"]):
                dpg.delete_item(annotation["annotation"])
        self.annotations = remaining_list

    def clearAll(self):
        """Reset the plot."""

        self.clearImg()
        self.clearSeries()
        self.clearBoxes()
        self.clearOverlays()
        self.clearDragPoints()
        self.clearAnnotations()
        self.bounds = np.zeros((2, 2))


class PlotBox:
    """Class for bounding boxes on plot."""
    
    def __init__(self, coords, parent, color=COLORS["error"], thickness=1) -> None:
        """Initializes box with bounds at same coords defined by 2 points."""

        self.p1 = coords
        self.p2 = coords

        self.parent = parent
        self.color = color
        self.thickness = 1
        self.rect = None
        self.annotation = None

        # Keep track of update mode
        self.update_mode = None
        self.update_start = None

    @property
    def center(self):
        return (self.p1 + self.p2) / 2
    
    @property
    def top(self):
        return max([self.p1[1], self.p2[1]])
    
    @property
    def bottom(self):
        return min([self.p1[1], self.p2[1]])
    
    @property
    def height(self):
        return self.top - self.bottom
    
    @property
    def left(self):
        return min([self.p1[0], self.p2[0]])
    
    @property
    def right(self):
        return max([self.p1[0], self.p2[0]])
    
    @property
    def width(self):
        return self.right - self.left
    
    def startUpdate(self, coords):
        """Determines which update mode is appropriate depending on where coords are."""

        self.update_start = coords
        margin = 0.1 # 10 %

        # Top edge
        if self.top - margin * self.height <= coords[1] <= self.top + margin * self.height: 
            # Left
            if self.left - margin * self.width <= coords[0] <= self.left + margin * self.width:
                self.update_mode = ["top", "left"]
            # Right
            elif self.right - margin * self.width <= coords[0] <= self.right + margin * self.width:
                self.update_mode = ["top", "right"]
            # Edge
            else:
                self.update_mode = ["top"]
        # Bottom edge
        elif self.bottom - margin * self.height <= coords[1] <= self.bottom + margin * self.height: 
            # Left
            if self.left - margin * self.width <= coords[0] <= self.left + margin * self.width:
                self.update_mode = ["bottom", "left"]
            # Right
            elif self.right - margin * self.width <= coords[0] <= self.right + margin * self.width:
                self.update_mode = ["bottom", "right"]
            # Edge
            else:
                self.update_mode = ["bottom"]
        # Left edge
        elif self.left - margin * self.width <= coords[0] <= self.left + margin * self.width:
            self.update_mode = ["left"]
        # Right edge
        elif self.right - margin * self.width <= coords[0] <= self.right + margin * self.width:
            self.update_mode = ["right"]
        # Edge
        else:
            self.update_mode = ["top", "bottom", "left", "right"]

    def endUpdate(self):
        """Finished update mode."""

        self.update_mode = None
        self.update_start = None

        # Make sure p1 is top left and p2 is bottom right
        top_left = np.array([self.left, self.top])
        bottom_right = np.array([self.right, self.bottom])
        self.p1, self.p2 = top_left, bottom_right

    def update(self, coords):
        """Updates dependent on mode."""

        shift = coords - self.update_start

        if "top" in self.update_mode:
            self.p1[1] = self.p1[1] + shift[1]
        if "bottom" in self.update_mode:
            self.p2[1] = self.p2[1] + shift[1]
        if "left" in self.update_mode:
            self.p1[0] = self.p1[0] + shift[0]
        if "right" in self.update_mode:
            self.p2[0] = self.p2[0] + shift[0]

        dpg.configure_item(self.rect, pmin=self.p1, pmax=self.p2)

        if dpg.does_item_exist(self.annotation):
            dpg.set_value(self.annotation, (self.right, self.top))

        self.update_start = coords

    def updateP2(self, coords):
        """Updates second point to define bounds."""

        self.p2 = coords
        if self.rect is None:
            self.draw()
        else:
            dpg.configure_item(self.rect, pmax=self.p2)

        if dpg.does_item_exist(self.annotation):
            dpg.set_value(self.annotation, (self.right, self.top))

    def updateP1(self, coords):
        """Updates first point to redefine bounds."""

        self.p1 = coords
        self.p2 = coords
        dpg.configure_item(self.rect, pmin=self.p1, pmax=self.p2)

        if dpg.does_item_exist(self.annotation):
            dpg.set_value(self.annotation, (self.right, self.top))

    def updateThickness(self, thickness):
        """Updates border thickness of box plot."""

        self.thickness = thickness
        if dpg.does_item_exist(self.rect):
            dpg.configure_item(self.rect, thickness=self.thickness)

    def updateColor(self, color):
        """Updates border color of box plot."""

        self.color = color
        if dpg.does_item_exist(self.rect):
            dpg.configure_item(self.rect, color=self.color)
        if dpg.does_item_exist(self.annotation):
            dpg.configure_item(self.annotation, color=self.color)

    def within(self, coords, margin=0):
        """Checks if coords are within the box (or within margin of the box)."""

        if min(self.p1[0], self.p2[0]) - margin <= coords[0] < max(self.p1[0], self.p2[0]) + margin and min(self.p1[1], self.p2[1]) - margin <= coords[1] < max(self.p1[1], self.p2[1]) + margin:
            return True
        else:
            return False
        
    def drawLabel(self, label, offset=(10, -10)):
        """Draws an annotation on the plot."""

        if dpg.does_item_exist(self.annotation):
            dpg.delete_item(self.annotation)
        self.annotation = dpg.add_plot_annotation(label=label, default_value=(self.right, self.top), offset=offset, color=self.color, clamped=False, parent=self.parent)

    def draw(self):
        """Draws the box on the plot."""

        self.rect = dpg.draw_rectangle(pmin=self.p1, pmax=self.p2, color=self.color, thickness=self.thickness, parent=self.parent)

    def remove(self):
        """Deletes the box and annotation from the plot."""
        
        if dpg.does_item_exist(self.annotation):
            dpg.delete_item(self.annotation)
        dpg.delete_item(self.rect)
