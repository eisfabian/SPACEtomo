#!/usr/bin/env python
# ===================================================================
# ScriptName:   gui_set
# Purpose:      User interface for SPACEtomo settings
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/05/31
# Revision:     v1.4
# Last Change:  2026/01/31: finished GUI rework
#               2025/06/01: added external SerialEM connection
#               2025/05/31: Copy most of lam_sel
# ===================================================================

import os
os.environ["__GLVND_DISALLOW_PATCHING"] = "1"           # helps to minimize Segmentation fault crashes on Linux when deleting textures

import json
import sys
import SPACEtomo.config as config

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

from SPACEtomo import __version__
from SPACEtomo.modules.gui import gui
from SPACEtomo.modules.gui.menu import Menu
from SPACEtomo.modules.gui.info import InfoBoxManager, InfoBox, StatusLine
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log
from SPACEtomo.modules.ext import MMModel

import faulthandler
faulthandler.enable()

DEFAULT_SETTINGS = {
    "automation_level": 4,
    "grid_list": [],
    "lamella": True,
    "exclude_lamella_classes": ["gone"] if "gone" in config.WG_model_categories else [],
    "WG_distance_threshold": 5,
    "WG_image_state": "",
    "IM_image_state": "",
    "MM_image_state": [],
    "WG_wait_for_inspection": True,
    "manual_selection": True,
    "MM_wait_for_inspection": True,
    "target_list": ["mitos"],
    "avoid_list": ['black', 'white', 'ice', 'crack', 'dynabeads'],
    "target_score_threshold": 0.01,
    "penalty_weight": 0.3,
    "sparse_targets": True,
    "target_edge": False,
    "extra_tracking": False,
    "max_tilt": 60,
    "beam_margin": 5,
    "IS_limit": 15,
    "script_numbers": [1, 2, 3],
    "session_dir": "",
    "external_map_dir": ""
}

class SettingsGUI:

    def loadLastSettings(self):
        """Loads default settings from default file and updates the GUI elements accordingly."""

        default_settings_file = Path(__file__).parent.parent.parent / "config/default_settings.ini"
        if default_settings_file.exists():
            settings = utils.loadSettings(default_settings_file)
            self.updateSettings(settings)

            log(f"DEBUG: Loaded default settings from {default_settings_file}")

        self.loaded_defaults = True

    def selectSettingsFile(self, sender, app_data, user_data):
        """Callback for settings file selection, loads settings from file."""

        # Check if a file was selected
        if app_data and "selections" in app_data:
            settings_file = Path(list(app_data["selections"].values())[0])
            if not settings_file.is_file():
                log(f"ERROR: Selected path {settings_file} is not a file!")
                InfoBoxManager.push(InfoBox("ERROR", f"Selected path {settings_file} is not a file!"))
                return
            if settings_file.suffix != ".ini":
                log(f"ERROR: Selected file {settings_file} is not a .ini file!")
                InfoBoxManager.push(InfoBox("ERROR", f"Selected file {settings_file} is not a .ini file!"))
                return
        else:
            log("ERROR: No file selected!")
            return

        # Load settings from file
        log(f"NOTE: Loading settings from {settings_file}")
        settings = utils.loadSettings(settings_file)

        self.updateSettings(settings)

    def changeLevel(self, sender, app_data, user_data):
        """Callback for automation level combo box, changes shown settings."""

        # Get current value
        level = app_data #int(app_data.split(" ")[-1])

        if level >= 2:
            self.menu3.showElements(["set_is_im", "set_is_mm0"])
        else:
            self.menu3.hideElements(["set_is_im", "set_is_mm0"])
            dpg.configure_item(self.menu2.all_elements["set_inspect_grid"], default_value=False)

        if level >= 3:
            self.menu3.showElements(["set_inspect_tgt", "set_select_tgt"])
            if not dpg.get_value(self.menu3.all_elements["set_select_tgt"]):
                self.menu_target.show()
                dpg.hide_item("menu_target_manual")
        else:
            self.menu3.hideElements(["set_inspect_tgt", "set_select_tgt"])
            self.menu_target.hide()
            dpg.show_item("menu_target_manual")

        if level >= 5:
            self.menu1.unlockRows(["row_scripts"])
        else:
            self.menu1.lockRows(["row_scripts"])


    def isLamella(self, sender, app_data, user_data):
        """Callback for lamella checkbox, shows or hides lamella settings."""

        if app_data:
            self.menu_lamella.show()
        else:
            self.menu_lamella.hide()

    def manualSelection(self, sender, app_data, user_data):
        """Callback for manual selection checkbox, shows or hides target settings."""

        if app_data:
            self.menu_target.hide()
            dpg.show_item("menu_target_manual")
            dpg.configure_item(self.menu3.all_elements["set_inspect_tgt"], default_value=True, enabled=False)
        else:
            self.menu_target.show()
            dpg.hide_item("menu_target_manual")
            dpg.configure_item(self.menu3.all_elements["set_inspect_tgt"], enabled=True)

    def showMMCombo(self, sender, app_data, user_data):
        """Callback to show another MM imaging state combo box."""

        # Get current number of MM imaging states
        mm_count = sum([1 for i in range(1, 4) if dpg.is_item_shown(self.menu3.all_elements[f"set_is_mm{i}"])])

        # Show next combo box if available
        if mm_count < 3:
            self.menu3.showElements([f"set_is_mm{mm_count + 1}"])
            if mm_count == 2:
                # Hide the add button if the last combo box is shown
                self.menu3.hideElements(["set_is_mm_add"])
        else:
            log("ERROR: Maximum number of MM imaging states reached!")

    def selectExtDir(self, sender, app_data, user_data):
        """Callback for external directory selection, updates input field with selected directory."""

        # Get selected directory
        if app_data and "file_path_name" in app_data:
            if Path(app_data["file_path_name"]).is_dir():
                dpg.set_value(self.menu1_advanced.all_elements["set_external_dir"], app_data["file_path_name"])
            else:
                log(f"ERROR: Selected path {app_data['file_path_name']} is not a directory!")
                InfoBoxManager.push(InfoBox("ERROR", f"Selected path {app_data['file_path_name']} is not a directory!"))
                return
        else:
            log("ERROR: No directory selected!")
            return

        log(f"NOTE: External processing directory set to {app_data['file_path_name']}")

    def clearExtDir(self, sender, app_data, user_data):
        """Callback to clear the external directory input field."""

        dpg.set_value(self.menu1_advanced.all_elements["set_external_dir"], "")

    def selectSessionDir(self, sender, app_data, user_data):
        """Callback for session directory selection, updates input field with selected directory."""

        # Get selected directory
        if app_data and "file_path_name" in app_data:
            if Path(app_data["file_path_name"]).is_dir():
                dpg.set_value(self.menu1.all_elements["session_dir_path"], app_data["file_path_name"])
            else:
                log(f"ERROR: Selected path {app_data['file_path_name']} is not a directory!")
                InfoBoxManager.push(InfoBox("ERROR", f"Selected path {app_data['file_path_name']} is not a directory!"))
                return
        else:
            log("ERROR: No directory selected!")
            return

        log(f"NOTE: Session directory set to {app_data['file_path_name']}")

    def clearSessionDir(self, sender, app_data, user_data):
        """Callback to clear the session directory input field."""

        dpg.set_value(self.menu1.all_elements["session_dir_path"], "")

    def updateSettings(self, settings):
        """Loads settings from a dictionary and updates the GUI elements accordingly."""

        # Update session directory
        if "session_dir" in settings:
            dpg.set_value(self.menu1.all_elements["session_dir_path"], settings['session_dir'])

        # Update automation level
        if "automation_level" in settings:
            dpg.set_value(self.menu1.all_elements["set_level"], settings['automation_level'])

        # Update grid list checkboxes
        if "grid_list" in settings:
            for i in range(12):
                dpg.set_value(self.menu1.all_elements[f"set_grid_{i + 1}"], i + 1 in settings["grid_list"])

        # Update lamella checkbox and settings
        if "lamella" in settings:
            dpg.set_value(self.menu1.all_elements["set_lamella"], settings["lamella"])
            if settings["lamella"]:
                self.menu_lamella.show()
                if "WG_distance_threshold" in settings:
                    dpg.set_value(self.menu_lamella.all_elements["set_lam_distance"], settings["WG_distance_threshold"])
                if "exclude_lamella_classes" in settings:
                    for cls in config.WG_model_categories:
                        dpg.set_value(self.menu_lamella.all_elements[f"set_exclude_{cls}"], cls in settings["exclude_lamella_classes"])
            else:
                self.menu_lamella.hide()

        # Update imaging states
        if self.imaging_states:
            for imaging_state in self.imaging_states:
                if "WG_image_state" in settings and settings["WG_image_state"] == imaging_state[0] or settings["WG_image_state"] == imaging_state[1]:
                    wg_image_state = imaging_state
                if "IM_image_state" in settings and settings["IM_image_state"] == imaging_state[0] or settings["IM_image_state"] == imaging_state[1]:
                    im_image_state = imaging_state
                if "MM_image_state" in settings and isinstance(settings["MM_image_state"], list) and len(settings["MM_image_state"]) > 0:
                    mm_image_states = []
                    for state in settings["MM_image_state"]:
                        if state == imaging_state[0] or state == imaging_state[1]:
                            mm_image_states.append(imaging_state)
            dpg.set_value(self.menu2.all_elements["set_is_wg"], f"{int(wg_image_state[0])}: {wg_image_state[1]} (Mag index: {wg_image_state[4]})")
            dpg.set_value(self.menu3.all_elements["set_is_im"], f"{int(im_image_state[0])}: {im_image_state[1]} (Mag index: {im_image_state[4]})")
            
            dpg.set_value(self.menu3.all_elements["set_is_mm0"], f"{int(mm_image_states[0][0])}: {mm_image_states[0][1]} (Mag index: {mm_image_states[0][4]})")
            dpg.set_value(self.menu3.all_elements["set_is_mm1"], f"{int(mm_image_states[1][0])}: {mm_image_states[1][1]} (Mag index: {mm_image_states[1][4]})" if len(mm_image_states) > 1 else "")
            dpg.set_value(self.menu3.all_elements["set_is_mm2"], f"{int(mm_image_states[2][0])}: {mm_image_states[2][1]} (Mag index: {mm_image_states[2][4]})" if len(mm_image_states) > 2 else "")
            dpg.set_value(self.menu3.all_elements["set_is_mm3"], f"{int(mm_image_states[3][0])}: {mm_image_states[3][1]} (Mag index: {mm_image_states[3][4]})" if len(mm_image_states) > 3 else "")
        else:
            # If SerialEM is not available, just set the imaging states to empty strings
            if "WG_image_state" in settings:
                dpg.set_value(self.menu2.all_elements["set_is_wg"], settings["WG_image_state"])
            if "IM_image_state" in settings:
                dpg.set_value(self.menu3.all_elements["set_is_im"], settings["IM_image_state"])
            if "MM_image_state" in settings:
                if isinstance(settings["MM_image_state"], list):
                    dpg.set_value(self.menu3.all_elements["set_is_mm0"], ", ".join([str(state) for state in settings["MM_image_state"]]))
                else:
                    dpg.set_value(self.menu3.all_elements["set_is_mm0"], settings["MM_image_state"])

        # Update wait for inspection checkboxes
        if "WG_wait_for_inspection" in settings:
            dpg.set_value(self.menu2.all_elements["set_inspect_grid"], settings["WG_wait_for_inspection"])
        if "manual_selection" in settings:
            dpg.set_value(self.menu3.all_elements["set_select_tgt"], settings["manual_selection"])
            if settings["manual_selection"]:
                self.menu_target.hide()
                dpg.configure_item(self.menu3.all_elements["set_inspect_tgt"], default_value=True, enabled=False)
            else:
                self.menu_target.show()
                dpg.configure_item(self.menu3.all_elements["set_inspect_tgt"], enabled=True)
        if "MM_wait_for_inspection" in settings:
            dpg.set_value(self.menu3.all_elements["set_inspect_tgt"], settings["MM_wait_for_inspection"])

        # Update target list and avoid list
        if "target_list" in settings:
            dpg.set_value(self.menu_target.all_elements["target_list"], ", ".join(settings["target_list"]))
        if "avoid_list" in settings:
            dpg.set_value(self.menu_target.all_elements["avoid_list"], ", ".join(settings["avoid_list"]))
        # Update target settings
        if "target_score_threshold" in settings:
            dpg.set_value(self.menu_target.all_elements["target_score_threshold"], float(settings["target_score_threshold"]))
        if "penalty_weight" in settings:
            dpg.set_value(self.menu_target.all_elements["penalty_weight"], float(settings["penalty_weight"]))
        if "sparse_targets" in settings:
            dpg.set_value(self.menu_target.all_elements["sparse_targets"], settings["sparse_targets"])
        if "target_edge" in settings:
            dpg.set_value(self.menu_target.all_elements["target_edge"], settings["target_edge"])
        if "extra_tracking" in settings:
            dpg.set_value(self.menu_target.all_elements["extra_tracking"], settings["extra_tracking"])
        if "max_tilt" in settings:
            dpg.set_value(self.menu_target.all_elements["max_tilt"], int(settings["max_tilt"]))
        if "beam_margin" in settings:
            dpg.set_value(self.menu_target.all_elements["beam_margin"], int(settings["beam_margin"]))
        if "IS_limit" in settings:
            dpg.set_value(self.menu_target.all_elements["IS_limit"], int(settings["IS_limit"]))
        # Update script numbers
        if "script_numbers" not in settings:
            for i in range(1, 4):
                dpg.set_value(self.menu1.all_elements[f"set_script_{i}"], str(settings["script_numbers"][i - 1]) if i - 1 < len(settings["script_numbers"]) else "")
        # Update external directory
        if "external_map_dir" in settings:
            dpg.set_value(self.menu1_advanced.all_elements["set_external_dir"], settings["external_map_dir"])


    def saveSettings(self):
        """Collects all settings and saves them to a file."""
        # Collect settings
        settings = {
            "automation_level": dpg.get_value(self.menu1.all_elements["set_level"]),
            "grid_list": [i + 1 for i in range(12) if dpg.get_value(self.menu1.all_elements[f"set_grid_{i + 1}"])],
            "lamella": dpg.get_value(self.menu1.all_elements["set_lamella"]),
            "exclude_lamella_classes": [cls for cls in config.WG_model_categories if dpg.get_value(self.menu_lamella.all_elements[f"set_exclude_{cls}"])],
            "WG_distance_threshold": dpg.get_value(self.menu_lamella.all_elements["set_lam_distance"]),
            "WG_image_state": dpg.get_value(self.menu2.all_elements["set_is_wg"]) if "Mag index:" not in dpg.get_value(self.menu2.all_elements["set_is_wg"]) else dpg.get_value(self.menu2.all_elements["set_is_wg"]).split(":")[0],
            "IM_image_state": dpg.get_value(self.menu3.all_elements["set_is_im"]) if "Mag index:" not in dpg.get_value(self.menu3.all_elements["set_is_im"]) else dpg.get_value(self.menu3.all_elements["set_is_im"]).split(":")[0],
            "MM_image_state": dpg.get_value(self.menu3.all_elements["set_is_mm0"]).split(",") if "Mag index:" not in dpg.get_value(self.menu3.all_elements["set_is_mm0"]) else [dpg.get_value(self.menu3.all_elements["set_is_mm0"]).split(":")[0]],
            "WG_wait_for_inspection": dpg.get_value(self.menu2.all_elements["set_inspect_grid"]),
            "manual_selection": dpg.get_value(self.menu3.all_elements["set_select_tgt"]),
            "MM_wait_for_inspection": dpg.get_value(self.menu3.all_elements["set_inspect_tgt"]),
            "target_list": [t.strip() for t in dpg.get_value(self.menu_target.all_elements["target_list"]).split(",") if t.strip()],
            "avoid_list": [t.strip() for t in dpg.get_value(self.menu_target.all_elements["avoid_list"]).split(",") if t.strip()],
            "target_score_threshold": float(dpg.get_value(self.menu_target.all_elements["target_score_threshold"])),
            "penalty_weight": float(dpg.get_value(self.menu_target.all_elements["penalty_weight"])),
            "sparse_targets": dpg.get_value(self.menu_target.all_elements["sparse_targets"]),
            "target_edge": dpg.get_value(self.menu_target.all_elements["target_edge"]),
            "extra_tracking": dpg.get_value(self.menu_target.all_elements["extra_tracking"]),
            "max_tilt": int(dpg.get_value(self.menu_target.all_elements["max_tilt"])),
            "beam_margin": float(dpg.get_value(self.menu_target.all_elements["beam_margin"])),
            "IS_limit": float(dpg.get_value(self.menu_target.all_elements["IS_limit"])),
            "script_numbers": [int(dpg.get_value(self.menu1.all_elements[f"set_script_{i}"])) for i in range(1, 4)],
            "session_dir": dpg.get_value(self.menu1.all_elements["session_dir_path"]),
            "external_map_dir": dpg.get_value(self.menu1_advanced.all_elements["set_external_dir"])
        }

        cur_dir = utils.getCurDir()
        utils.saveSettings(cur_dir / "SPACEtomo_settings.ini", settings)
        log(f"NOTE: Settings saved to {cur_dir / 'SPACEtomo_settings.ini'}")

        # Also save settings internally as defaults for next run
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        utils.saveSettings(config_dir / "default_settings.ini", settings)

    def startRun(self, sender, app_data, user_data):
        """Starts the SPACEtomo run with the current settings."""

        self.saveSettings()
        dpg.stop_dearpygui()  # Stop the GUI to run the main script

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
    
    def selectTab(self, sender, app_data, user_data):
        """Callback to select a tab from the custom tab bar."""

        # Use user data instead of sender if available
        if user_data:
            sender = user_data

        # Deselect all buttons
        for i in range(1, 5):
            dpg.bind_item_theme(f"tab_{i}", "tab_btn_theme")

        # Select clicked button
        dpg.bind_item_theme(sender, "tab_btn_sel_theme")

        dpg.show_item("tab_session") if sender == "tab_1" else dpg.hide_item("tab_session")
        dpg.show_item("tab_wgmap") if sender == "tab_2" else dpg.hide_item("tab_wgmap")
        dpg.show_item("tab_mmap") if sender == "tab_3" else dpg.hide_item("tab_mmap")
        dpg.show_item("tab_target") if sender == "tab_4" else dpg.hide_item("tab_target")

        self.current_tab = int(sender.split("_")[-1])

    def checkLockedTabs(self):
        """Checks which tabs should be locked or unlocked based on settings."""

        automation_level = int(dpg.get_value(self.menu1.all_elements["set_level"]))

        # Unlock tabs based on level
        if automation_level >= 1 \
        and dpg.get_value(self.menu1.all_elements["session_dir_path"]) \
        and any([dpg.get_value(self.menu1.all_elements[f"set_grid_{i + 1}"]) for i in range(12)]):
            dpg.configure_item("tab_2", enabled=True)
            self.menu_tab1_options.hideElements(["continue_disabled"])
            self.menu_tab1_options.showElements(["continue"])
        else:
            dpg.configure_item("tab_2", enabled=False)
            self.menu_tab1_options.hideElements(["continue"])
            self.menu_tab1_options.showElements(["continue_disabled"])
            
            dpg.configure_item("tab_3", enabled=False)
            dpg.configure_item("tab_4", enabled=False)
            return 1
        if automation_level >= 2 \
        and dpg.get_value(self.menu2.all_elements["set_is_wg"]):
            dpg.configure_item("tab_3", enabled=True)
            self.menu_tab2_options.hideElements(["continue_disabled"])
            self.menu_tab2_options.showElements(["continue"])

        else:
            dpg.configure_item("tab_3", enabled=False)
            self.menu_tab2_options.hideElements(["continue"])
            self.menu_tab2_options.showElements(["continue_disabled"])

            dpg.configure_item("tab_4", enabled=False)
            return 2
        if automation_level >= 4 \
        and not dpg.get_value(self.menu3.all_elements["set_select_tgt"]):
            dpg.configure_item("tab_4", enabled=True)
            self.menu_tab3_options.hideElements(["continue_disabled"])
            self.menu_tab3_options.showElements(["continue"])
        else:
            dpg.configure_item("tab_4", enabled=False)
            self.menu_tab3_options.hideElements(["continue"])
            self.menu_tab3_options.showElements(["continue_disabled"])
            return 3
        return 4

    def updateSummary(self):
        """Updates the summary text with current settings."""

        dpg.configure_item(self.menu_run.all_elements["summary_level_value"], default_value=dpg.get_value(self.menu1.all_elements["set_level"]))
        dpg.configure_item(self.menu_run.all_elements["summary_grids_value"], default_value=(", ".join([str(i + 1) for i in range(12) if dpg.get_value(self.menu1.all_elements[f"set_grid_{i + 1}"])]) or "-"))
        dpg.configure_item(self.menu_run.all_elements["summary_lamellae_value"], default_value="Yes" if dpg.get_value(self.menu1.all_elements["set_lamella"]) else "No")
        wg_state = dpg.get_value(self.menu2.all_elements["set_is_wg"]) or "-"
        im_state = dpg.get_value(self.menu3.all_elements["set_is_im"]) or "-"
        mm_states = dpg.get_value(self.menu3.all_elements["set_is_mm0"]).split(",") if "Mag index:" not in dpg.get_value(self.menu3.all_elements["set_is_mm0"]) else [dpg.get_value(self.menu3.all_elements[f"set_is_mm{i}"]).split(":")[0] for i in range(0, 4) if dpg.get_value(self.menu3.all_elements[f"set_is_mm{i}"])]
        #print(mm_states)
        dpg.configure_item(self.menu_run.all_elements["summary_states_value"], default_value=f"{wg_state} | {im_state} | {', '.join(mm_states)}")
        dpg.configure_item(self.menu_run.all_elements["summary_inspection_value"], default_value=f"{'Yes' if dpg.get_value(self.menu2.all_elements['set_inspect_grid']) else 'No'} | {'Yes' if dpg.get_value(self.menu3.all_elements['set_inspect_tgt']) else 'No'}")
        dpg.configure_item(self.menu_run.all_elements["summary_targeting_value"], default_value=f"{'Manual' if dpg.get_value(self.menu3.all_elements['set_select_tgt']) else 'Auto'}")

    def checkRunReady(self):
        """Checks if all required settings are set to enable the start run button."""

        automation_level = int(dpg.get_value(self.menu1.all_elements["set_level"]))

        # Check required settings based on level
        if automation_level >= 1:
            if not dpg.get_value(self.menu1.all_elements["session_dir_path"]):
                return False
            if not any([dpg.get_value(self.menu1.all_elements[f"set_grid_{i + 1}"]) for i in range(12)]):
                return False
        if automation_level >= 2:
            if not dpg.get_value(self.menu2.all_elements["set_is_wg"]):
                return False
        if automation_level >= 3:
            if not dpg.get_value(self.menu3.all_elements["set_is_im"]):
                return False
            if not dpg.get_value(self.menu3.all_elements["set_is_mm0"]):
                return False
        if automation_level >= 4:
            if not dpg.get_value(self.menu3.all_elements["set_select_tgt"]):
                if not dpg.get_value(self.menu_target.all_elements["target_list"]):
                    return False
        if automation_level >= 5:
            if not dpg.get_value(self.menu1.all_elements["set_script_1"]) or not dpg.get_value(self.menu1.all_elements["set_script_2"]) or not dpg.get_value(self.menu1.all_elements["set_script_3"]):
                return False
        return True

    def configureAvailableTabs(self):
        """Configures the available tabs and start run button based on whether all required settings are set."""

        max_tab = self.checkLockedTabs()

        if self.checkRunReady():
            if self.current_tab == max_tab:
                self.menu_run.showElements(["start_run_primary"])
                self.menu_run.hideElements(["start_run", "start_run_disabled"])
            else:
                self.menu_run.showElements(["start_run"])
                self.menu_run.hideElements(["start_run_primary", "start_run_disabled"])
        else:
            self.menu_run.showElements(["start_run_disabled"])
            self.menu_run.hideElements(["start_run", "start_run_primary"])
            

    def __init__(self, path="", run_mode=False) -> None:
        log("\n########################################\nRunning SPACEtomo Settings GUI\n########################################\n")

        self.session_dir = Path(path) if path else Path("")
        self.run_mode = run_mode

        # Keep list of popup blocking windows
        self.blocking_windows = []

        # Make logo
        self.logo_dims = gui.makeLogo()

        self.configureHandlers()
        self.configureThemes()

        # Initialise globals
        self.menu = None
        self.menu_lamella = None
        self.menu_target = None
        self.status = None
        self.imaging_states = None

        self.model = MMModel()

        # Load autoloader data from JSON if available
        self.autoloader = {}
        autoloader_file = self.session_dir / "autoloader.json"
        if autoloader_file.exists():
            with open(autoloader_file) as f:
                self.autoloader = {int(k): v for k, v in json.load(f).items()}

        # Load imaging states from JSON if available
        imaging_states_file = self.session_dir / "imaging_states.json"
        if imaging_states_file.exists():
            with open(imaging_states_file) as f:
                self.imaging_states = [tuple(s) for s in json.load(f)]

        self.current_tab = 1                        # Current selected tab

        # One-time calls
        self.loaded_defaults = False                # Needed to load defaults only once

    def configureHandlers(self):
        """Sets up dearpygui registries and handlers."""

        # Create file dialogues
        gui.fileNav("nav_session_dir", self.selectSessionDir, dir=True, default_path=str(self.session_dir))
        gui.fileNav("nav_settings_file", self.selectSettingsFile, extensions=[".ini"])
        gui.fileNav("nav_ext_dir", self.selectExtDir, dir=True)

        with dpg.item_handler_registry(tag="class_input_handler"):
            dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=self.openClassSelection)

        dpg.set_viewport_resize_callback(callback=lambda: gui.window_size_change(self.logo_dims, align="right"))

    @staticmethod
    def configureThemes():
        """Sets up dearpygui themes."""

        gui.configureGlobalTheme()

    def show(self):
        """Structures and launches main window of GUI."""

        # Setup window
        dpg.create_viewport(title="SPACEtomo Settings", disable_close=True, small_icon=str(Path(__file__).parent / "logo.ico"), large_icon=str(Path(__file__).parent / "logo.ico"))
        dpg.setup_dearpygui()



        # Create main window
        with dpg.window(label="GUI", tag="GUI", no_scrollbar=True, no_scroll_with_mouse=True):

            with dpg.viewport_menu_bar():
                with dpg.menu(label="Settings"):
                    dpg.add_menu_item(label="Load from File", callback=lambda: dpg.show_item("nav_settings_file"))
                    dpg.add_menu_item(label="Save as File", callback=self.saveSettings)
                    dpg.add_menu_item(label="Reset to Defaults", callback=lambda: self.updateSettings(DEFAULT_SETTINGS))
            dpg.add_spacer(height=10)

            """
            # Start run button
            self.menu_run = Menu(outline=False, indent=dpg.get_viewport_client_width() - 150)
            self.menu_run.newRow(tag="menu_tab1_options", locked=False)
            self.menu_run.addImageTextButton(tag="start_run", texture=gui.makeIconFromImg("rocket"), text="Start Run", callback=None, tooltip="Start the SPACEtomo run.")
            self.menu_run.addImageTextButton(primary=True, tag="start_run_primary", texture=gui.makeIconFromImg("rocket", "#252525"), text="Start Run", callback=None, tooltip="Start the SPACEtomo run.", show=False)
            self.menu_run.addImageTextButton(disabled=True, tag="start_run_disabled", texture=gui.makeIconFromImg("rocket", "#555555"), text="Start Run", callback=None, tooltip="Start the SPACEtomo run. (There are still missing settings!)", show=False)
            """


            # Custom tab bar
            with dpg.group(tag="tab_bar", horizontal=True):
                dpg.add_button(tag="tab_1", label="1. Session", width=150, height=40, callback=self.selectTab)
                dpg.add_button(tag="tab_2", label="2. Whole Grid Map", width=150, height=40, callback=self.selectTab, enabled=False)
                dpg.add_button(tag="tab_3", label="3. Medium Mag Map", width=150, height=40, callback=self.selectTab, enabled=False)
                dpg.add_button(tag="tab_4", label="4. Target Selection", width=150, height=40, callback=self.selectTab, enabled=False)

            dpg.bind_item_theme("tab_bar", "tab_bar_theme")
            dpg.bind_item_theme("tab_1", "tab_btn_sel_theme")

            with dpg.drawlist(width=dpg.get_viewport_client_width() - 300, height=1):
                dpg.draw_line((0, 0), (dpg.get_viewport_client_width() - 300, 0), color=gui.COLORS["heading"], thickness=2)


            
            #dpg.show_style_editor()
            
            with dpg.tab_bar(label="tabbar", tag="tabbar", show=False):

                with dpg.tab(label="Run settings", tag="tab_run"):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(init_width_or_weight=400, width_fixed=True)
                        dpg.add_table_column()

                        with dpg.table_row():
                            with dpg.table_cell(tag="tblleft"):
                                self.menu = Menu()
                                """
                                self.menu.newRow(tag="row_header1", horizontal=False, locked=False)
                                self.menu.addText(tag="set_header1", value="Settings", color=gui.COLORS["heading"])

                                # Load settings file
                                self.menu.newRow(tag="row_set", horizontal=True, locked=False)
                                self.menu.addButton(tag="set_session", label="Change session dir", callback=lambda: dpg.show_item("nav_session_dir"), tooltip="In the session directory the settings will be saved and a new folder for each grid will be created.")
                                self.menu.addButton(tag="set_load", label="Load settings", callback=lambda: dpg.show_item("nav_settings_file"), tooltip="Load old settings from file.")

                                # Automation level
                                self.menu.newRow(tag="row_header2", horizontal=False, locked=False)
                                self.menu.addText(tag="set_header2", value="\nAutomation Level", color=gui.COLORS["heading"])
                                self.menu.addCombo(tag="set_level", label="", value="Level 4", combo_list=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"], callback=self.changeLevel, width=80, tooltip="Automation level for the SPACEtomo run.\nLevel 1: Collect WG map (and find lamellae)\nLevel 2: Collect MM maps for each ROI\nLevel 3: Segment lamellae maps (if no manual selection)\nLevel 4: Setup targets manually or based on segmentation\nLevel 5: Start PACEtomo batch acquisition.")

                                # Grid list
                                self.menu.addText(tag="set_grid_header", value="Grids", color=gui.COLORS["heading"])
                                self.menu.newRow(tag="row_grid", horizontal=True, locked=False)
                                for i in range(12):
                                    if self.autoloader:
                                        if i + 1 in self.autoloader.keys():
                                            enabled = True
                                            tooltip = f"Grid {i + 1} in autoloader: {self.autoloader[i + 1]}"
                                        else:
                                            enabled = False
                                            tooltip = f"Grid {i + 1} in autoloader not available."
                                    else:
                                        enabled = True
                                        tooltip = f"Grid {i + 1} in autoloader (if available)."
                                    self.menu.addCheckbox(tag=f"set_grid_{i + 1}", label=f"{str(i + 1).ljust(2)}", value=False, callback=None, tooltip=tooltip, enabled=enabled)
                                    if i == 5:
                                        self.menu.newRow(tag="row_grid2", horizontal=True, locked=False)
                                """
                                # Is lamella
                                self.menu.newRow(tag="row_set_lam", horizontal=False, locked=False)
                                self.menu.addText(tag="set_lamella_header", value="Lamella", color=gui.COLORS["heading"])
                                self.menu.addCheckbox(tag="set_lamella", label="Grid contains lamellae", value=False, callback=self.isLamella, tooltip="Enables automated lamella detection.")
                                self.menu.addText(tag="set_spacer1", value="")


                                self.menu.newRow(tag="row_set3", horizontal=False, locked=False)
                                self.menu.addText(tag="set_imaging_header", value="Imaging states", color=gui.COLORS["heading"])
                                if self.imaging_states:
                                    log(f"DEBUG: Imaging states found:\n{self.imaging_states}")
                                    # Imaging states
                                    self.menu.addCombo(tag="set_is_wg", label="WG imaging state", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose < 0], callback=None, tooltip="Imaging state name or index for whole grid montage maps. (Set up in SerialEM before running!)")
                                    self.menu.addCombo(tag="set_is_im", label="IM imaging state", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose < 0], callback=None, tooltip="Imaging state name or index for intermediate mag used for recentering. (Set up in SerialEM before running!)")
                                    self.menu.addCombo(tag="set_is_mm0", label="MM imaging state(s)", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose >= 0], callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)")
                                    self.menu.addCombo(tag="set_is_mm1", label="MM imaging state(s)", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose >= 0], callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)", show=False)
                                    self.menu.addCombo(tag="set_is_mm2", label="MM imaging state(s)", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose >= 0], callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)", show=False)
                                    self.menu.addCombo(tag="set_is_mm3", label="MM imaging state(s)", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose >= 0], callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)", show=False)
                                    self.menu.addButton(tag="set_is_mm_add", label="+", callback=self.showMMCombo, tooltip="Add another MM imaging state for Low Dose Mode.", theme="small_btn_theme")

                                else:
                                    self.menu.addInput(tag="set_is_wg", label="LM imaging state for WG maps", value="WG", callback=None, tooltip="Imaging state name or index for whole grid montage maps. (Set up in SerialEM before running!)")
                                    self.menu.addInput(tag="set_is_im", label="IM imaging state", value="IM", callback=None, tooltip="Imaging state name or index for intermediate mag used for recentering. (Set up in SerialEM before running!)")
                                    self.menu.addInput(tag="set_is_mm0", label="LD imaging state for MM maps", value="MM", callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)")
                                self.menu.addText(tag="set_spacer2", value="")

                                self.menu.newRow(tag="row_set4", horizontal=False, locked=False)
                                self.menu.addText(tag="set_inspection_header", value="Inspection", color=gui.COLORS["heading"])
                                self.menu.addCheckbox(tag="set_inspect_grid", label="Inspect WG map", value=True, callback=None, tooltip="SPACEtomo waits for inspection of ROIs/lamellae before collecting MM maps.")
                                self.menu.addCheckbox(tag="set_inspect_tgt", label="Inspect targets", value=True, callback=None, enabled=False, tooltip="SPACEtomo waits for inspection of targets by operator before finalizing.")
                                self.menu.addCheckbox(tag="set_select_tgt", label="Select targets manually", value=True, callback=self.manualSelection, tooltip="SPACEtomo will allow manual target selection. If not selected, targets will be selected automatically based on segmentation of MM maps.")
                                self.menu.addText(tag="set_spacer3", value="")

                                self.menu.newRow(tag="row_scripts", horizontal=False, locked=True)
                                self.menu.addText(tag="set_scripts_header", value="Script indices", color=gui.COLORS["heading"])
                                self.menu.addCombo(tag="set_script_1", label="SPACEtomo_run.py", combo_list=[str(i) for i in range(1, 61)], value="10", callback=None, width=50, tooltip="Script number for running the SPACEtomo run script in SerialEM.")
                                self.menu.addCombo(tag="set_script_2", label="SPACEtomo_prepareTargets.py", combo_list=[str(i) for i in range(1, 61)], value="11", callback=None, width=50, tooltip="Script number for running the SPACEtomo prepare targets script in SerialEM.")
                                self.menu.addCombo(tag="set_script_3", label="PACEtomo.py", combo_list=[str(i) for i in range(1, 61)], value="5", callback=None, width=50, tooltip="Script number for running the PACEtomo script in SerialEM.")
                                self.menu.addText(tag="set_spacer4", value="")

                                self.menu.newRow(tag="row_ext1", horizontal=False, locked=False)
                                self.menu.addText(tag="set_external_header", value="External processing directory", color=gui.COLORS["heading"])
                                self.menu.newRow(tag="row_ext2", horizontal=True, locked=False)
                                self.menu.addInput(tag="set_external_dir", label="", value="", width=150, callback=None, tooltip="Directory where maps are saved and segmentations are expected (run \"SPACEmonitor\" on external machine to manage runs and queue).")
                                self.menu.addButton(tag="set_external_find", label="Browse", callback=lambda: dpg.show_item("nav_ext_dir"), tooltip="Browse for external processing directory.", theme="small_btn_theme")
                                self.menu.addButton(tag="set_external_clear", label="Clear", callback=self.clearExtDir, theme="small_btn_theme")

                                self.menu.newRow(tag="row_save", horizontal=False, locked=False)
                                self.menu.addButton(tag="set_save", label="Save settings", callback=self.saveSettings, tooltip="Save current settings to file.")
                                self.menu.addButton(tag="set_run", label="Run SPACEtomo", callback=self.startRun, tooltip="Start the SPACEtomo run with the current settings.", theme="large_btn_theme", show=self.run_mode)


                                self.status = StatusLine()

                            with dpg.table_cell(tag="tblright"):
                                pass

                                """self.menu_lamella = Menu()
                                self.menu_lamella.newRow(tag="row_header", separator=False, locked=False)
                                self.menu_lamella.addText(tag="set_header", value="Lamella settings", color=gui.COLORS["heading"])
                                self.menu_lamella.addText(tag="set_header2", value="Keep lamellae of class:")
                                self.menu_lamella.newRow(tag="row_exclude", horizontal=True, locked=False)
                                for cls in config.WG_model_categories:
                                    self.menu_lamella.addCheckbox(tag=f"set_exclude_{cls}", label=f"{cls}", value=False, callback=None, tooltip=f"Include {cls} lamellae for Medium Magnification maps.")

                                self.menu_lamella.newRow(tag="row_distance", horizontal=True, locked=False)
                                self.menu_lamella.addSlider(tag="set_lam_distance", label="Lamella distance threshold [microns]", value=5, value_range=[0, 50], width=75, tooltip="Minimum distance [microns] between lamellae to not be considered duplicate detection.")
                                
                                self.menu_lamella.hide()


                                self.menu_target = Menu()
                                self.menu_target.newRow(tag="settings1", separator=False, locked=False)
                                self.menu_target.addText(tag="target_heading", value="Target selection settings", color=gui.COLORS["heading"])
                                self.menu_target.addText(tag="target_list_label", value="Target classes:")
                                self.menu_target.addInput(tag="target_list", label="", value=",".join(['mitos']), width=-1, tooltip="List of target classes (comma separated). For exhaustive acquisition use \"lamella\".")
                                dpg.bind_item_handler_registry(self.menu_target.all_elements["target_list"], "class_input_handler")

                                self.menu_target.newRow(tag="settings2", separator=False, locked=False)
                                self.menu_target.addText(tag="avoid_list_label", value="Avoid classes:")
                                self.menu_target.addInput(tag="avoid_list", label="", value=",".join(['black', 'white', 'ice', 'crack', 'dynabeads']), width=-1, tooltip="List of classes to avoid (comma separated).")
                                dpg.bind_item_handler_registry(self.menu_target.all_elements["avoid_list"], "class_input_handler")

                                self.menu_target.newRow(tag="settings3", separator=False, locked=False)
                                self.menu_target.addText(tag="settings_heading", value="\nTargeting options:")
                                self.menu_target.addSlider(tag="target_score_threshold", label="Score threshold", value=0.01, value_range=[0, 1], width=75, tooltip="Score threshold [0-1] below targets will be excluded.")
                                self.menu_target.addSlider(tag="penalty_weight", label="Penalty weight", value=0.3, value_range=[0, 1], width=75, tooltip="Relative weight of avoided classes to target classes.")
                                self.menu_target.newRow(tag="settings4", separator=False, locked=False)
                                self.menu_target.addSlider(tag="max_tilt", label="Max. tilt angle", value=60, value_range=[0, 80], width=75, tooltip="Maximum tilt angle [degrees] to consider electron beam exposure.")
                                self.menu_target.addSlider(tag="beam_margin", label="[%] Beam margin", value=5, value_range=[0, 50], width=75, advanced=True, tooltip="Margin around target area to avoid exposure [% of beam diameter].")
                                self.menu_target.addSlider(tag="IS_limit", label="Image shift limit", value=15, value_range=[5, 20], width=75, tooltip="Image shift limit [µm] for PACEtomo acquisition. If targets are further apart, target area will be split.")
                                self.menu_target.newRow(tag="settings5", separator=False, locked=False)
                                self.menu_target.addCheckbox(tag="sparse_targets", label="Use sparse targets", value=True, tooltip="Target positions will be initialized only on target classes and refined independently (instead of grid based target target setup to minimize exposure overlap).")
                                self.menu_target.addCheckbox(tag="target_edge", label="Enable edge targeting", value=False, tooltip="Targets will be centered on edge of segmented target instead of maximising coverage.")
                                self.menu_target.addCheckbox(tag="extra_tracking", label="Add extra tracking target", value=False, tooltip="An extra target will be placed centrally for tracking.")

                                self.menu_target.hide()"""
                with dpg.tab(label="Hide", tag="tab_hide"):
                    pass


            with dpg.group(horizontal=True):
                    
                with dpg.child_window(tag="tab_session", width=dpg.get_viewport_client_width() - 300, autosize_x=False, autosize_y=True, border=False):

                    with dpg.table(header_row=False):
                        dpg.add_table_column(init_width_or_weight=(dpg.get_viewport_client_width() - 300) // 2, width_fixed=True)
                        dpg.add_table_column(init_width_or_weight=(dpg.get_viewport_client_width() - 300) // 2, width_fixed=True)
                        #dpg.add_table_column()

                        with dpg.table_row():
                            with dpg.table_cell(tag="session_tblleft"):
                                self.menu1 = Menu(outline=False)

                                #self.menu1.newRow(tag="row_header1", horizontal=False, locked=False)
                                #self.menu1.addText(tag="set_header1", value="Settings", color=gui.COLORS["heading"])

                                #self.menu1.newRow(tag="row_new", horizontal=True, locked=False)
                                #self.menu1.addImageTextButton(tag="set_new", texture=gui.makeIconFromImg("rocket"), text="New Session", callback=None, tooltip="Start a new session for SPACEtomo.")
                                #self.menu1.addImageTextButton(tag="set_load", texture=gui.makeIconFromImg("settings"), text="Load Settings", callback=None, tooltip="Load old settings from file.")
                                
                                self.menu1.newRow(tag="row_header2", horizontal=False, locked=False)                            
                                self.menu1.addImageText(tag="set_header2", texture=gui.makeIconFromImg("folder"), text="Session directory", color=gui.COLORS["heading"], tooltip="In the session directory the settings will be saved and a new folder for each grid will be created.")
                                #self.menu1.addText(tag="set_header2", value="\nSession directory", color=gui.COLORS["heading"], tooltip="In the session directory the settings will be saved and a new folder for each grid will be created.")

                                self.menu1.newRow(tag="row_session", horizontal=True, locked=False)
                                self.menu1.addInput(tag="session_dir_path", label="", value="", width=300, callback=None, tooltip="In the session directory the settings will be saved and a new folder for each grid will be created.")
                                self.menu1.addButton(tag="session_dir_browse", label="Browse", callback=lambda: dpg.show_item("nav_session_dir"), tooltip="Browse for session directory.", theme="small_btn_theme")
                                self.menu1.addButton(tag="session_dir_clear", label="Clear", callback=self.clearSessionDir, theme="small_btn_theme")



                                # Load settings file
                                #self.menu1.newRow(tag="row_set", horizontal=True, locked=False)
                                #self.menu1.addImageTextButton(tag="set_session2", texture=gui.makeIconFromImg("check-circle", "#009900"), text="Session Directory", disabled=True, show=False, tooltip="In the session directory the settings will be saved and a new folder for each grid will be created.")
                                #self.menu1.addText(tag="session_dir", value="No session directory selected.", color=gui.COLORS["subtle"])
                                #self.menu1.addButton(tag="session_dir_path", label="No session directory selected.", callback=lambda: dpg.show_item("nav_session_dir"), theme="invisible_btn_theme", height=40)
                                #self.menu1.addImageTextButton(tag="set_session", texture=gui.makeIconFromImg("folder"), text="Browse", callback=lambda: dpg.show_item("nav_session_dir"), )
                                #self.menu1.addButton(tag="session_dir_clear", label="Clear", callback=self.clearSessionDir, show=False, height=40)


                                #self.menu1.addButton(tag="set_session", label="Change session dir", callback=lambda: dpg.show_item("nav_session_dir"), tooltip="In the session directory the settings will be saved and a new folder for each grid will be created.")
                                #self.menu1.addButton(tag="set_load", label="Load settings", callback=lambda: dpg.show_item("nav_settings_file"), tooltip="Load old settings from file.")

                                # Automation level
                                self.menu1.newRow(tag="row_header2", horizontal=False, locked=False)
                                self.menu1.addImageText(tag="set_header2", texture=gui.makeIconFromImg("settings"), text="Automation Level", color=gui.COLORS["heading"], tooltip="Automation level for the SPACEtomo run.\nLevel 1: Collect WG map (and find lamellae)\nLevel 2: Collect MM maps for each ROI\nLevel 3: Segment lamellae maps (if no manual selection)\nLevel 4: Setup targets manually or based on segmentation\nLevel 5: Start PACEtomo batch acquisition.")
                                #self.menu1.addText(tag="set_header2", value="\nAutomation Level\n\n", color=gui.COLORS["heading"])
                                #self.menu1.addCombo(tag="set_level", label="", value="Level 4", combo_list=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"], callback=self.changeLevel, width=80, tooltip="Automation level for the SPACEtomo run.\nLevel 1: Collect WG map (and find lamellae)\nLevel 2: Collect MM maps for each ROI\nLevel 3: Segment lamellae maps (if no manual selection)\nLevel 4: Setup targets manually or based on segmentation\nLevel 5: Start PACEtomo batch acquisition.")
                                self.menu1.addSlider(tag="set_level", label="Automation level", value=4, value_range=[1, 5], width=200, callback=self.changeLevel, tooltip="Automation level for the SPACEtomo run.\nLevel 1: Collect WG map (and find lamellae)\nLevel 2: Collect MM maps for each ROI\nLevel 3: Segment lamellae maps (if no manual selection)\nLevel 4: Setup targets manually or based on segmentation\nLevel 5: Start PACEtomo batch acquisition.")

                                # Grid list
                                #self.menu1.addText(tag="set_grid_header", value="\nGrids\n\n", color=gui.COLORS["heading"])
                                self.menu1.addImageText(tag="set_header3", texture=gui.makeIconFromImg("layers"), text="Grids", color=gui.COLORS["heading"], tooltip="Select which grids to image.")

                                self.menu1.newRow(tag="row_grid", horizontal=True, locked=False)
                                for i in range(12):
                                    if self.autoloader:
                                        if i + 1 in self.autoloader.keys():
                                            enabled = True
                                            tooltip = f"Grid {i + 1} in autoloader: {self.autoloader[i + 1]}"
                                        else:
                                            enabled = False
                                            tooltip = f"Grid {i + 1} in autoloader not available."
                                    else:
                                        enabled = True
                                        tooltip = f"Grid {i + 1} in autoloader (if available)."
                                    self.menu1.addCheckbox(tag=f"set_grid_{i + 1}", label=f"{str(i + 1).ljust(2)}", value=False, callback=None, tooltip=tooltip, enabled=enabled)
                                    if i == 5:
                                        self.menu1.newRow(tag="row_grid2", horizontal=True, locked=False)

                                self.menu1.newRow(tag="row_header4", horizontal=False, locked=False)
                                #self.menu1.addText(tag="set_header3", value="\nGrid type\n\n", color=gui.COLORS["heading"])
                                self.menu1.addImageText(tag="set_header4", texture=gui.makeIconFromImg("grid"), text="Grid type", color=gui.COLORS["heading"], tooltip="Select if the grids contain FIB-milled lamellae or not.")
                                self.menu1.addCheckbox(tag="set_lamella", label="Grid contains FIB-milled lamellae", value=True, callback=self.isLamella, tooltip="Enables automated lamella detection.")


                                self.menu1.newRow(tag="row_scripts", horizontal=False, locked=True)
                                #self.menu1.addText(tag="set_scripts_header", value="Script indices", color=gui.COLORS["heading"])
                                self.menu1.addImageText(tag="set_scripts_header", texture=gui.makeIconFromImg("file-code"), text="Script indices for full automation", color=gui.COLORS["heading"], tooltip="For full automation the PACEtomo script is needed as well as the SPACEtomo scripts for additional setup of targets as new maps become available.")
                                self.menu1.addCombo(tag="set_script_1", label="SPACEtomo_run.py", combo_list=[str(i) for i in range(1, 61)], value="10", callback=None, width=50, tooltip="Script number for running the SPACEtomo run script in SerialEM.")
                                self.menu1.addCombo(tag="set_script_2", label="SPACEtomo_prepareTargets.py", combo_list=[str(i) for i in range(1, 61)], value="11", callback=None, width=50, tooltip="Script number for running the SPACEtomo prepare targets script in SerialEM.")
                                self.menu1.addCombo(tag="set_script_3", label="PACEtomo.py", combo_list=[str(i) for i in range(1, 61)], value="5", callback=None, width=50, tooltip="Script number for running the PACEtomo script in SerialEM.")


                            with dpg.table_cell(tag="session_tblright"):
                                with dpg.collapsing_header(label="Advanced settings", default_open=False):
                                    self.menu1_advanced = Menu(outline=False)
                                    self.menu1_advanced.newRow(tag="row_header_adv", horizontal=False, locked=False)
                                    self.menu1_advanced.addImageText(tag="set_external_header", texture=gui.makeIconFromImg("server"), text="External processing directory", color=gui.COLORS["heading"])
                                    #self.menu1_advanced.addText(tag="set_external_header", value="External processing directory", color=gui.COLORS["heading"])
                                    self.menu1_advanced.newRow(tag="row_ext2", horizontal=True, locked=False)
                                    self.menu1_advanced.addInput(tag="set_external_dir", label="", value="", width=300, callback=None, tooltip="Directory where maps are saved and segmentations are expected (run \"SPACEmonitor\" on external machine to manage runs and queue).")
                                    self.menu1_advanced.addButton(tag="set_external_find", label="Browse", callback=lambda: dpg.show_item("nav_ext_dir"), tooltip="Browse for external processing directory.", theme="small_btn_theme")
                                    self.menu1_advanced.addButton(tag="set_external_clear", label="Clear", callback=self.clearExtDir, theme="small_btn_theme")

                        with dpg.table_row():
                            with dpg.table_cell(tag="tab1_tblbottom1"):
                                self.menu_tab1_options = Menu(outline=False)
                                self.menu_tab1_options.newRow(tag="row_header_options", locked=False)
                                self.menu_tab1_options.addImageTextButton(primary=True, tag="continue", texture=gui.makeIconFromImg("arrow-circle-right", "#252525"), text="Continue", callback=self.selectTab, user_data="tab_2", tooltip="Continue to next tab", show=False)
                                self.menu_tab1_options.addImageTextButton(disabled=True, tag="continue_disabled", texture=gui.makeIconFromImg("arrow-circle-right", "#555555"), text="Continue", callback=None, tooltip="Continue to next tab")

                with dpg.child_window(tag="tab_wgmap", width=dpg.get_viewport_client_width() - 300, autosize_x=False, autosize_y=True, show=False, border=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(init_width_or_weight=(dpg.get_viewport_client_width() - 300) // 2, width_fixed=True)
                        dpg.add_table_column(init_width_or_weight=(dpg.get_viewport_client_width() - 300) // 2, width_fixed=True)

                        with dpg.table_row():
                            with dpg.table_cell(tag="tab2_tblleft"):
                                self.menu2 = Menu(outline=False)

                                self.menu2.newRow(tag="row_header1", horizontal=False, locked=False)
                                self.menu2.addImageText(tag="set_header1", texture=gui.makeIconFromImg("aperture"), text="Imaging conditions", color=gui.COLORS["heading"])
                                #self.menu2.addText(tag="set_header1", value="Imaging conditions\n\n", color=gui.COLORS["heading"])
                                if self.imaging_states:
                                    log(f"DEBUG: Imaging states found:\n{self.imaging_states}")
                                    # Imaging states
                                    self.menu2.addCombo(tag="set_is_wg", label="LM imaging state for WG maps", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose < 0], callback=None, tooltip="Imaging state name or index for whole grid montage maps. (Set up in SerialEM before running!)")
                                else:
                                    self.menu2.addInput(tag="set_is_wg", label="LM imaging state for WG maps", value="WG", callback=None, tooltip="Imaging state name or index for whole grid montage maps. (Set up in SerialEM before running!)")

                                self.menu2.addImageText(tag="set_inspection_header", texture=gui.makeIconFromImg("user-check"), text="Inspection", color=gui.COLORS["heading"])
                                #self.menu2.addText(tag="set_inspection_header", value="\nInspection\n\n", color=gui.COLORS["heading"])
                                self.menu2.addCheckbox(tag="set_inspect_grid", label="Inspect WG map to manually add/edit ROIs", value=True, callback=None, tooltip="SPACEtomo waits for inspection of ROIs/lamellae before collecting MM maps.")
                            
                            with dpg.table_cell(tag="tab2_tblright"):

                                self.menu_lamella = Menu(outline=False)
                                self.menu_lamella.newRow(tag="row_header", separator=False, locked=False)
                                self.menu_lamella.addImageText(tag="set_header", texture=gui.makeIconFromImg("lamella"), text="Lamella settings", color=gui.COLORS["heading"])
                                #self.menu_lamella.addText(tag="set_header", value="Lamella settings\n\n", color=gui.COLORS["heading"])
                                self.menu_lamella.addText(tag="set_header2", value="Keep lamellae of class:")
                                self.menu_lamella.newRow(tag="row_exclude", horizontal=True, locked=False)
                                for cls in config.WG_model_categories:
                                    self.menu_lamella.addCheckbox(tag=f"set_exclude_{cls}", label=f"{cls}", value=False if cls == "gone" else True, callback=None, tooltip=f"Include {cls} lamellae for Medium Magnification maps.")

                                self.menu_lamella.newRow(tag="row_distance", horizontal=True, locked=False)
                                self.menu_lamella.addSlider(tag="set_lam_distance", label="Lamella distance threshold [microns]", value=5, value_range=[0, 50], width=75, tooltip="Minimum distance [microns] between lamellae to not be considered duplicate detection.")
                                
                                if dpg.get_value(self.menu1.all_elements["set_lamella"]):
                                    self.menu_lamella.show()
                                else:
                                    self.menu_lamella.hide()

                                dpg.add_text(tag="todo1", default_value="\nComing soon: Model selection as advanced collapsable", color=gui.COLORS["subtle"])

                        with dpg.table_row():
                            with dpg.table_cell(tag="tab2_tblbottom1"):
                                self.menu_tab2_options = Menu(outline=False)
                                self.menu_tab2_options.newRow(tag="row_header_options", horizontal=True, locked=False)
                                self.menu_tab2_options.addImageTextButton(primary=True, tag="continue", texture=gui.makeIconFromImg("arrow-circle-right", "#252525"), text="Continue", callback=self.selectTab, user_data="tab_3", tooltip="Continue to next tab", show=False)
                                self.menu_tab2_options.addImageTextButton(disabled=True, tag="continue_disabled", texture=gui.makeIconFromImg("arrow-circle-right", "#555555"), text="Continue", callback=None, tooltip="Continue to next tab")

                with dpg.child_window(tag="tab_mmap", width=dpg.get_viewport_client_width() - 300, autosize_x=False, autosize_y=True, show=False, border=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(init_width_or_weight=(dpg.get_viewport_client_width() - 300) // 2, width_fixed=True)
                        dpg.add_table_column(init_width_or_weight=(dpg.get_viewport_client_width() - 300) // 2, width_fixed=True)

                        with dpg.table_row():
                            with dpg.table_cell(tag="tab3_tblleft"):
                                self.menu3 = Menu(outline=False)      

                                self.menu3.newRow(tag="row_set3", horizontal=False, locked=False)
                                self.menu3.addImageText(tag="set_header1", texture=gui.makeIconFromImg("aperture"), text="Imaging conditions", color=gui.COLORS["heading"])
                                #self.menu3.addText(tag="set_imaging_header", value="Imaging states\n\n", color=gui.COLORS["heading"])
                                if self.imaging_states:
                                    log(f"DEBUG: Imaging states found:\n{self.imaging_states}")
                                    # Imaging states
                                    self.menu3.addCombo(tag="set_is_im", label="Intermediate Mag [IM] imaging state for realignment", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose < 0], callback=None, tooltip="Imaging state name or index for intermediate mag used for recentering. (Set up in SerialEM before running!)")
                                    self.menu3.addCombo(tag="set_is_mm0", label="MM imaging state(s)", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose >= 0], callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)")
                                    self.menu3.addCombo(tag="set_is_mm1", label="MM imaging state(s)", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose >= 0], callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)", show=False)
                                    self.menu3.addCombo(tag="set_is_mm2", label="MM imaging state(s)", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose >= 0], callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)", show=False)
                                    self.menu3.addCombo(tag="set_is_mm3", label="MM imaging state(s)", combo_list=[f"{int(index)}: {name} (Mag index: {int(mag_index)}, Pixel size: {pixel_size})" for index, name, low_dose, camera, mag_index, pixel_size in self.imaging_states if low_dose >= 0], callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)", show=False)
                                    self.menu3.addButton(tag="set_is_mm_add", label="+", callback=self.showMMCombo, tooltip="Add another MM imaging state for Low Dose Mode.", theme="small_btn_theme")

                                else:
                                    self.menu3.addInput(tag="set_is_im", label="Intermediate Mag [IM] imaging state for realignment", value="IM", callback=None, tooltip="Imaging state name or index for intermediate mag used for recentering. (Set up in SerialEM before running!)")
                                    self.menu3.addInput(tag="set_is_mm0", label="Low Dose imaging state for MM maps", value="MM", callback=None, tooltip="Imaging state name or index for Low Dose Mode, this can be several imaging states to specify Record, View, ... (Set up in SerialEM before running!)")

                                self.menu3.newRow(tag="row_set4", horizontal=False, locked=False)
                                self.menu3.addImageText(tag="set_inspection_header", texture=gui.makeIconFromImg("user-check"), text="Inspection", color=gui.COLORS["heading"])
                                #self.menu3.addText(tag="set_inspection_header", value="\nInspection\n\n", color=gui.COLORS["heading"])
                                self.menu3.addCheckbox(tag="set_inspect_tgt", label="Inspect MM maps", value=True, callback=None, enabled=False, tooltip="SPACEtomo waits for inspection of targets by operator before finalizing.")
                                self.menu3.addCheckbox(tag="set_select_tgt", label="Select targets manually", value=True, callback=self.manualSelection, tooltip="SPACEtomo will allow manual target selection. If not selected, targets will be selected automatically based on segmentation of MM maps.")
                                self.menu3.addText(tag="set_spacer3", value="")

                            with dpg.table_cell(tag="tab3_tblright"):
                                dpg.add_text(tag="todo2", default_value="\nComing soon: Model selection/segmentation as advanced collapsable", color=gui.COLORS["subtle"])

                        with dpg.table_row():
                            with dpg.table_cell(tag="tab3_tblbottom1"):
                                self.menu_tab3_options = Menu(outline=False)
                                self.menu_tab3_options.newRow(tag="row_header_options", horizontal=True, locked=False)
                                self.menu_tab3_options.addImageTextButton(primary=True, tag="continue", texture=gui.makeIconFromImg("arrow-circle-right", "#252525"), text="Continue", callback=self.selectTab, user_data="tab_4", tooltip="Continue to next tab", show=False)
                                self.menu_tab3_options.addImageTextButton(disabled=True, tag="continue_disabled", texture=gui.makeIconFromImg("arrow-circle-right", "#555555"), text="Continue", callback=None, tooltip="Continue to next tab")


                with dpg.child_window(tag="tab_target", width=dpg.get_viewport_client_width() - 300, autosize_x=False, autosize_y=True, show=False, border=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(init_width_or_weight=(dpg.get_viewport_client_width() - 300) // 2, width_fixed=True)
                        dpg.add_table_column(init_width_or_weight=(dpg.get_viewport_client_width() - 300) // 2, width_fixed=True)

                        with dpg.table_row():
                            with dpg.table_cell(tag="tab4_tblleft"):

                                dpg.add_text(tag="menu_target_manual", default_value="Manual target selection is enabled.\nPlease select targets in the SPACEtomo window after MM map acqusition.", color=gui.COLORS["subtle"], show=False)

                                self.menu_target = Menu()
                                self.menu_target.newRow(tag="settings1", separator=False, locked=False)
                                self.menu_target.addText(tag="target_heading", value="Target selection settings", color=gui.COLORS["heading"])
                                self.menu_target.addText(tag="target_list_label", value="Target classes:")
                                self.menu_target.addInput(tag="target_list", label="", value=",".join(['mitos']), width=-1, tooltip="List of target classes (comma separated). For exhaustive acquisition use \"lamella\".")
                                dpg.bind_item_handler_registry(self.menu_target.all_elements["target_list"], "class_input_handler")

                                self.menu_target.newRow(tag="settings2", separator=False, locked=False)
                                self.menu_target.addText(tag="avoid_list_label", value="Avoid classes:")
                                self.menu_target.addInput(tag="avoid_list", label="", value=",".join(['black', 'white', 'ice', 'crack', 'dynabeads']), width=-1, tooltip="List of classes to avoid (comma separated).")
                                dpg.bind_item_handler_registry(self.menu_target.all_elements["avoid_list"], "class_input_handler")

                                self.menu_target.newRow(tag="settings3", separator=False, locked=False)
                                self.menu_target.addText(tag="settings_heading", value="\nTargeting options:")
                                self.menu_target.addSlider(tag="target_score_threshold", label="Score threshold", value=0.01, value_range=[0, 1], width=75, tooltip="Score threshold [0-1] below targets will be excluded.")
                                self.menu_target.addSlider(tag="penalty_weight", label="Penalty weight", value=0.3, value_range=[0, 1], width=75, tooltip="Relative weight of avoided classes to target classes.")
                                self.menu_target.newRow(tag="settings4", separator=False, locked=False)
                                self.menu_target.addSlider(tag="max_tilt", label="Max. tilt angle", value=60, value_range=[0, 80], width=75, tooltip="Maximum tilt angle [degrees] to consider electron beam exposure.")
                                self.menu_target.addSlider(tag="beam_margin", label="[%] Beam margin", value=5, value_range=[0, 50], width=75, advanced=True, tooltip="Margin around target area to avoid exposure [% of beam diameter].")
                                self.menu_target.addSlider(tag="IS_limit", label="Image shift limit", value=15, value_range=[5, 20], width=75, tooltip="Image shift limit [µm] for PACEtomo acquisition. If targets are further apart, target area will be split.")
                                self.menu_target.newRow(tag="settings5", separator=False, locked=False)
                                self.menu_target.addCheckbox(tag="sparse_targets", label="Use sparse targets", value=True, tooltip="Target positions will be initialized only on target classes and refined independently (instead of grid based target target setup to minimize exposure overlap).")
                                self.menu_target.addCheckbox(tag="target_edge", label="Enable edge targeting", value=False, tooltip="Targets will be centered on edge of segmented target instead of maximising coverage.")
                                self.menu_target.addCheckbox(tag="extra_tracking", label="Add extra tracking target", value=False, tooltip="An extra target will be placed centrally for tracking.")

                                self.menu_target.hide()

                with dpg.child_window(tag="tab_finalize", autosize_x=True, autosize_y=True, show=True, border=True):
                    # Start run button
                    self.menu_run = Menu(outline=False)
                    self.menu_run.newRow(tag="menu_tab1_options", locked=False)
                    self.menu_run.addImageText(tag="header1", texture=gui.makeIconFromImg("check-circle"), text="Summary", color=gui.COLORS["heading"], tooltip="Double-check the most important settings and start the SPACEtomo run.")

                    self.menu_run.newRow(tag="summary_level", horizontal=True, separator=False, locked=False)
                    self.menu_run.addText(tag="summary_level", value="Automation:     ", color=gui.COLORS["subtle"])
                    self.menu_run.addText(tag="summary_level_value", value=str(dpg.get_value(self.menu1.all_elements["set_level"])))

                    self.menu_run.newRow(tag="summary_grids", horizontal=True, separator=False, locked=False)
                    self.menu_run.addText(tag="summary_grids", value="Grid(s):        ", color=gui.COLORS["subtle"])
                    self.menu_run.addText(tag="summary_grids_value", value=", ".join([str(i + 1) for i in range(12) if dpg.get_value(self.menu1.all_elements[f"set_grid_{i + 1}"])]))

                    self.menu_run.newRow(tag="summary_lamellae", horizontal=True, separator=False, locked=False)
                    self.menu_run.addText(tag="summary_lamellae", value="Lamellae:       ", color=gui.COLORS["subtle"])
                    self.menu_run.addText(tag="summary_lamellae_value", value="Yes" if dpg.get_value(self.menu1.all_elements["set_lamella"]) else "No")

                    self.menu_run.newRow(tag="summary_states", horizontal=True, separator=False, locked=False)
                    self.menu_run.addText(tag="summary_states", value="Imaging states: ", color=gui.COLORS["subtle"])
                    wg_state = dpg.get_value(self.menu2.all_elements["set_is_wg"])
                    im_state = dpg.get_value(self.menu3.all_elements["set_is_im"])
                    mm_states = [dpg.get_value(self.menu3.all_elements[f"set_is_mm{i}"]) for i in range(4) if dpg.does_item_exist(self.menu3.all_elements.get(f"set_is_mm{i}"))]
                    self.menu_run.addText(tag="summary_states_value", value=f"{wg_state} | {im_state} | {', '.join(mm_states)}")

                    self.menu_run.newRow(tag="summary_inspection", horizontal=True, separator=False, locked=False)
                    self.menu_run.addText(tag="summary_inspection", value="Inspection:     ", color=gui.COLORS["subtle"])
                    self.menu_run.addText(tag="summary_inspection_value", value=f"{'Yes' if dpg.get_value(self.menu2.all_elements['set_inspect_grid']) else 'No'} | {'Yes' if dpg.get_value(self.menu3.all_elements['set_inspect_tgt']) else 'No'}")
                    
                    self.menu_run.newRow(tag="summary_targeting", horizontal=True, separator=True, locked=False)
                    self.menu_run.addText(tag="summary_targeting", value="Targeting:      ", color=gui.COLORS["subtle"])
                    self.menu_run.addText(tag="summary_targeting_value", value=f"{'Manual' if dpg.get_value(self.menu3.all_elements['set_select_tgt']) else 'Auto'}")

                    self.menu_run.newRow(tag="summary_button", separator=False, locked=False)
                    self.menu_run.addImageTextButton(tag="start_run", texture=gui.makeIconFromImg("rocket"), text="Start Run", callback=self.startRun, tooltip="Start the SPACEtomo run with the current settings.", show=False)
                    self.menu_run.addImageTextButton(primary=True, tag="start_run_primary", texture=gui.makeIconFromImg("rocket", "#252525"), text="Start Run", callback=self.startRun, tooltip="Start the SPACEtomo run with the current settings.", show=False)
                    self.menu_run.addImageTextButton(disabled=True, tag="start_run_disabled", texture=gui.makeIconFromImg("rocket", "#555555"), text="Start Run", callback=None, tooltip="Start the SPACEtomo run. (There are still missing settings!)", show=True)


                """
                with dpg.tab(label="Models", tag="tab_model"):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(init_width_or_weight=400, width_fixed=True)
                        dpg.add_table_column()
                        with dpg.table_row():
                            with dpg.table_cell(tag="tblleft_model"):
                                self.menu_model = Menu()
                                self.menu_model.newRow(tag="row_header", separator=False, locked=False)
                                self.menu_model.addText(tag="set_header", value="Model settings", color=gui.COLORS["heading"])


                with dpg.tab(label="Config", tag="tab_config"):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(init_width_or_weight=400, width_fixed=True)
                        dpg.add_table_column()
                        with dpg.table_row():
                            with dpg.table_cell(tag="tblleft_config"):
                                self.menu_config = Menu()
                                self.menu_config.newRow(tag="row_header", separator=False, locked=False)
                                self.menu_config.addText(tag="set_header", value="Configuration", color=gui.COLORS["heading"])
                """


            # Show logo
            dpg.add_image("logo", pos=(dpg.get_viewport_client_width() - 10 - self.logo_dims[1], dpg.get_viewport_client_height() - 40 - self.logo_dims[0]), tag="logo_img")
            dpg.add_text(default_value="v" + __version__, pos=(dpg.get_viewport_client_width() - 10  - self.logo_dims[1] / 2 - (30), dpg.get_viewport_client_height() + 5 - self.logo_dims[0] / 2), tag="version_text")



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

        dpg.set_exit_callback(dpg.stop_dearpygui)

        dpg.set_primary_window("GUI", True)
        dpg.configure_item("GUI", no_scrollbar=True, horizontal_scrollbar=False)
        dpg.show_viewport()

        # Render loop
        next_update = time.time() + 0.1
        while dpg.is_dearpygui_running():

            # Load defaults once
            if not self.loaded_defaults:
                self.loadLastSettings()

            # Recheck folder for segmentation every minute
            now = time.time()
            if now > next_update:
                next_update = now + 0.1

                self.updateSummary()
                self.configureAvailableTabs()

                # Check if info box needs to be shown
                InfoBoxManager.unblock()

            gui.flush_deferred()

            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    # Run GUI
    dpg.create_context()
    settings_gui = SettingsGUI()
    settings_gui.show()
    dpg.destroy_context()