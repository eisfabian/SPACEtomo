#!/usr/bin/env python
# ===================================================================
# ScriptName:   gui_menu
# Purpose:      User interface utility to generate menus
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/08
# Revision:     v1.2
# Last Change:  2024/10/02: added slider support
#               2024/08/08: separated from gui.py 
# ===================================================================

import dearpygui.dearpygui as dpg

class Menu:
    """Class to structure menu tables and control locking and unlocking of rows."""

    def __init__(self) -> None:
        self.makeTable()
        self.rows = {}                  # visible rows
        self.rows_locked = {}           # conditionally hidden rows
        self.rows_adv = {}              # advanced rows
        self.rows_adv_locked = {}       # advanced conditionally hidden rows
        self.elements = {}              # elements within rows
        self.elements_adv = {}          # advanced elements
        self.separators = {}            # separators between rows
        self.last_group = None          # group to which new elements will be added

        self.show_advanced = False      # toggle advance elements/rows

    def makeTable(self):
        """Creates base table."""

        with dpg.table(header_row=False, width=-1, borders_innerH=True) as self.table:
            dpg.add_table_column()

    def newRow(self, tag, horizontal=False, separator=False, locked=True, advanced=False):
        """Creates new row and necessary containers for elements."""

        with dpg.table_row(parent=self.table):
            with dpg.table_cell():
                self.last_group = dpg.add_group(horizontal=horizontal, show=not locked and not advanced)
                if separator:
                    self.separators[tag] = dpg.add_separator(show=not locked and not advanced)
        if advanced and locked:
            self.rows_adv_locked[tag] = self.last_group
        elif advanced:
            self.rows_adv[tag] = self.last_group
        elif locked:
            self.rows_locked[tag] = self.last_group
        else:
            self.rows[tag] = self.last_group

    def unlockRows(self, tags=[]):
        """Makes list of rows visible."""

        for tag in tags:
            # Normal rows
            if tag in self.rows_locked.keys():
                self.rows[tag] = self.rows_locked[tag]
                self.rows_locked.pop(tag)
                dpg.show_item(self.rows[tag])
                # Check for accompanying row separator
                if tag in self.separators:
                    dpg.show_item(self.separators[tag])
            # Advanced rows
            if tag in self.rows_adv_locked.keys():
                self.rows_adv[tag] = self.rows_adv_locked[tag]
                self.rows_adv_locked.pop(tag)
                # Only show in advanced mode
                if self.show_advanced:
                    dpg.show_item(self.rows_adv[tag])
                    # Check for accompanying row separator
                    if tag in self.separators:
                        dpg.show_item(self.separators[tag])

    def lockRows(self, tags=[]):
        """Hides list of rows until unlocked."""

        for tag in tags:
            # Normal rows
            if tag in self.rows.keys():
                self.rows_locked[tag] = self.rows[tag]
                self.rows.pop(tag)
                dpg.hide_item(self.rows_locked[tag])
                # Check for accompanying row separator
                if tag in self.separators:
                    dpg.hide_item(self.separators[tag])
            # Advanced rows
            if tag in self.rows_adv.keys():
                self.rows_adv_locked[tag] = self.rows_adv[tag]
                self.rows_adv.pop(tag)
                dpg.hide_item(self.rows_adv_locked[tag])
                # Check for accompanying row separator
                if tag in self.separators:
                    dpg.hide_item(self.separators[tag])

    def addText(self, tag, value="", color=(255, 255, 255, 255), advanced=False):
        """Adds a text element to the current row."""

        element = dpg.add_text(default_value=value, color=color, show=not advanced, parent=self.last_group)
        # Add to element list
        if advanced:
            self.elements_adv[tag] = element
        else:
            self.elements[tag] = element

    def addButton(self, tag, label="", callback=None, user_data=None, theme=None, show=True, advanced=False):
        """Adds a button to the current row."""

        # Add button to group
        element = dpg.add_button(label=label, callback=callback, user_data=user_data, show=show and not advanced, parent=self.last_group)
        # Add to element list
        if advanced:
            self.elements_adv[tag] = element
        else:
            self.elements[tag] = element

        if theme and dpg.does_item_exist(theme):
            dpg.bind_item_theme(element, theme)

    def addInput(self, tag, label="", value=None, width=50, callback=None, advanced=False):
        """Adds an input field to the current row and uses type of value."""

        # Add input to group
        if isinstance(value, float):
            element = dpg.add_input_float(default_value=value, step=0, width=width, label=label, format="%.3f", callback=callback, show=not advanced, parent=self.last_group)
        elif isinstance(value, int):
            element = dpg.add_input_int(default_value=value, step=0, width=width, label=label, callback=callback, show=not advanced, parent=self.last_group)
        elif isinstance(value, str):
            element = dpg.add_input_text(default_value=value, width=width, label=label, callback=callback, show=not advanced, parent=self.last_group)
        else:
            raise ValueError("No input element available for this type!")
        # Add to element list
        if advanced:
            self.elements_adv[tag] = element
        else:
            self.elements[tag] = element

    def addCheckbox(self, tag, label="", value=False, callback=None, advanced=False):
        """Adds checkbox to current row."""

        # Add checkbox to group
        element = dpg.add_checkbox(label=label, default_value=value, callback=callback, show=not advanced, parent=self.last_group)
        # Add to element list
        if advanced:
            self.elements_adv[tag] = element
        else:
            self.elements[tag] = element

    def addCombo(self, tag, label="", combo_list=[], value="", callback=None, user_data=None, width=0, advanced=False):
        """Adds dropdown menu to current row."""

        # Check if list and value are valid
        if not combo_list:
            combo_list = [""]
        if not value or value not in combo_list:
            value = combo_list[0]
        # Add combo list to group
        element = dpg.add_combo(combo_list, default_value=value, label=label, callback=callback, user_data=user_data, width=width, show=not advanced, parent=self.last_group)
        # Add to element list
        if advanced:
            self.elements_adv[tag] = element
        else:
            self.elements[tag] = element

    def addSlider(self, tag, label="", value=0, value_range=[0, 1], width=50, callback=None, advanced=False):
        """Adds a slider to current row."""

        # Add slider to group
        if isinstance(value, float):
            element = dpg.add_slider_float(default_value=value, min_value=value_range[0], max_value=value_range[1], width=width, label=label, callback=callback, show=not advanced, parent=self.last_group)
        elif isinstance(value, int):
            element = dpg.add_slider_int(default_value=value, min_value=value_range[0], max_value=value_range[1], width=width, label=label, callback=callback, show=not advanced, parent=self.last_group)
        else:
            raise ValueError("No slider available for this type!")
        # Add to element list
        if advanced:
            self.elements_adv[tag] = element
        else:
            self.elements[tag] = element

    @property
    def all_rows(self):
        """Provides list of all rows."""

        return self.rows | self.rows_locked | self.rows_adv | self.rows_adv_locked

    @property
    def all_elements(self):
        """Provides list of all elements."""

        return self.elements | self.elements_adv

    def show(self):
        dpg.show_item(self.table)

    def hide(self):
        dpg.hide_item(self.table)

    def toggleAdvanced(self):
        """Shows or hides rows and elements with the advanced flag."""

        self.show_advanced = not self.show_advanced

        if self.show_advanced:
            for tag, row in self.rows_adv.items():
                dpg.show_item(row)
                if tag in self.separators:
                    dpg.show_item(self.separators[tag])
            for element in self.elements_adv.values():
                dpg.show_item(element)
        else:
            for tag, row in self.rows_adv.items():
                dpg.hide_item(row)
                if tag in self.separators:
                    dpg.hide_item(self.separators[tag])
            for element in self.elements_adv.values():
                dpg.hide_item(element)