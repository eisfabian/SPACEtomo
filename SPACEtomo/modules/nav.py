#!/usr/bin/env python
# ===================================================================
# ScriptName:   nav
# Purpose:      Interface for SerialEM navigator.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/13
# Revision:     v1.2
# Last Change:  2024/09/04: added newPolygon function
#               2024/08/23: allowed creation of minimal map without template
#               2024/08/19: added creation of new map and point items
#               2024/08/16: added change item functions, added writeToFile, newMap, replaceItem functions 
#               2024/08/14: added search function 
# ===================================================================

# Check SerialEM availability
try:
    import serialem as sem
    # Check if PySEMSocket is connected
    try:
        sem.Delay(0, "s")
        SERIALEM = True
    except sem.SEMmoduleError:
        SERIALEM = False
except ModuleNotFoundError:
    SERIALEM = False

from pathlib import Path
import numpy as np
import mrcfile

from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.utils import log
from SPACEtomo.modules.utils import serialem_check, dummy_skip, dummy_replace

class NavItem:
    """Nav item."""

    def __init__(self, index, label) -> None:
        self.nav_index = int(index)      # SerialEM navigator item index (starting from 1, SPACEtomo Navigator IDs start from 0)
        self.label = str(label)
        self.entries = {}

        # Instantiate attributes
        self.stage = np.zeros(3)
        self.item_type = 0
        self.note = ""
        self.map_file = None

    def readBlock(self, open_file):
        """Reads block of open navigator file until next empty line."""

        for line in open_file:
            col = line.strip().split(" ")
            if col[0] == "":
                return
            self.entries[col[0]] = [val for val in col[2:]]

    def getBlock(self):
        """Converts attributes back to navigator file entry."""

        # Check existence of map and downgrade to polygon if map file cannot be found to prevent error when loading nav
        if self.item_type == 2 and self.map_file:
            if not self.map_file.exists():
                self.item_type = 1

        # Update changed attributes
        self.revertEntries()

        # Create text block
        text = f"[Item = {self.label}]\n"
        for key, attr in self.entries.items():
            text += f"{key} = {' '.join(attr)}\n"
        text += "\n"

        return text

    def convertEntries(self):
        """Exposes relevant entries as attributes. [StageXYZ, Type, Note, MapFile]"""

        self.stage = np.array(self.entries["StageXYZ"], dtype=np.float32)               # Stage position
        self.item_type = int(self.entries["Type"][0])                                   # 0-2 for point, polygon, map
        if "Note" in self.entries.keys():
            self.note = " ".join(self.entries["Note"])                                  # Note string
        else:
            self.note = ""
        if "MapFile" in self.entries.keys():
            self.map_file = Path(" ".join(self.entries["MapFile"]))                     # Parse map file
        else:
            self.map_file = None

    def revertEntries(self):
        """Updates entries according to changed attributes. [StageXYZ, Type, Note, MapFile]"""

        self.entries["StageXYZ"] = [str(coord) for coord in self.stage]                 # Stage position          
        self.entries["Type"] = [str(self.item_type)]                                    # 0-2 for point, polygon, map
        if self.note:
            self.entries["Note"] = self.note.split(" ")                                 # Note string
        if self.map_file:
            self.entries["MapFile"] = [str(self.map_file)]                              # Parse map file

    def changeColor(self, color_id):
        """Changes color of nav item (0 = red; 1 = green; 2 = blue; 3 = yellow; 4 = magenta; 5 = no realign)."""

        self.entries["Color"] = [str(color_id)]

        if SERIALEM:
            sem.ChangeItemColor(self.nav_index, color_id)

    def changeLabel(self, label):
        """Changes label of nav item."""

        self.label = label

        if SERIALEM:
            sem.ChangeItemLabel(self.nav_index, label)
            
    def changeNote(self, note):
        """Changes note of nav item."""

        self.note = note

        if SERIALEM:
            sem.ChangeItemNote(self.nav_index, note)

    def changeDraw(self, draw):
        """Changes if nav item is drawn."""

        self.entries["Drawn"] = [str(int(draw))]

        if SERIALEM:
            sem.ChangeItemDraw(self.nav_index, int(draw))

    def changeAcquire(self, acquire):
        """Changes acquire flag for item."""

        self.entries["Acquire"] = [str(acquire)]

        if SERIALEM:
            sem.SetItemAcquire(self.nav_index)

    def changeStage(self, coords, relative=False):
        """Changes stage coordinates."""

        # Add z coord if not given
        if len(coords) < 3:
            if relative:
                coords = np.append(coords, 0)
            else:
                coords = np.append(coords, self.stage[2])

        # Get relative pts coords
        pts_x = [float(x) - self.stage[0] for x in self.entries["PtsX"]]
        pts_y = [float(y) - self.stage[1] for y in self.entries["PtsY"]]

        if relative:
            self.stage += coords
        else:
            self.stage = coords

        self.entries["PtsX"] = [str(x + self.stage[0]) for x in pts_x]
        self.entries["PtsY"] = [str(y + self.stage[1]) for y in pts_y]

        self.revertEntries()

    def changeZ(self, z, relative=False):
        """Updates z coordinates of nav item."""

        if relative:
            self.stage[2] += z
        else:
            self.stage[2] = z
        self.revertEntries()

    def addEntry(self, key, value):
        """Adds any entry to nav item."""

        if isinstance(value, list):
            self.entries[key] = [str(item) for item in value]
        else:
            self.entries[key] = [str(value)]

    def getDistance(self, coords):
        """Calculates distance in x and y from given coords."""

        coords = np.array(coords)
        #log(f"DEBUG: Distance: {self.stage[:2]}, {coords}, {np.linalg.norm(self.stage[:2] - coords)}")
        return np.linalg.norm(self.stage[:2] - coords)
    
    def getVector(self, coords):
        """Calculates vector to coords."""

        coords = np.array(coords)
        return coords - self.stage[:2]
    
    def createMap(self, template, stage, file, map_id, note=""):
        """Creates new map item from image file and template."""

        with mrcfile.open(file) as mrc:
            min_max_scale = [str(np.min(mrc.data)), str(np.max(mrc.data))]
            dims = np.flip(mrc.data.shape)

        # Generate default template if no template
        if not template:
            template = self.getMinimalTemplate()

        # Get stage Z from template if missing
        if len(stage) < 3:
            stage = np.array([stage[0], stage[1], template.stage[2]])

        # Calculate point coords using dimensions and inverse of MapScaleMat matrix
        points = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]) * dims
        px2stage = np.linalg.inv(np.array(template.entries["MapScaleMat"], dtype=np.float32).reshape((2, 2)))
        points = np.array([px2stage @ point for point in points]) + stage[:2]
        pts_x, pts_y = points[:, 0], points[:, 1]

        # Go through all required entries
        self.entries["Color"] = ["2"]                                           # Default map color is blue
        self.entries["StageXYZ"] = [str(x) for x in stage]                      # XYZ stage coords
        self.entries["NumPts"] = ["5"]                                          # 5 points to define rectangle
        self.entries["Regis"] = template.entries["Regis"]                       # Registration
        self.entries["Type"] = ["2"]                                            # Type map is 2
        self.entries["Note"] = note.split(" ") if isinstance(note, str) else note
        self.entries["SamePosId"] = [str(map_id)]                               # ID of map at same coords
        self.entries["MapFile"] = [str(file)]                                   # Image file of map
        self.entries["MapID"] = [str(map_id)]                                   # Map has to be unique
        self.entries["MapMontage"] = ["0"]                                      # Map is not a montage
        self.entries["MapSection"] = ["0"]                                      # Map is only section in file
        self.entries["MapBinning"] = template.entries["MapBinning"]             # Map binning
        self.entries["MapMagInd"] = template.entries["MapMagInd"]               # Map mag index
        self.entries["MapCamera"] = template.entries["MapCamera"]               # Camera number
        self.entries["MapScaleMat"] = template.entries["MapScaleMat"]           # Stage to pixel scale matrix for drawing (might be off when taken from montage template)
        self.entries["MapWidthHeight"] = [str(dim) for dim in dims]             # Size of initial map image

        # Optional entries
        self.entries["MapMinMaxScale"] = min_max_scale                          # Min and max scale values of image when map was defined
        self.entries["MapFramesXY"] = ["0", "0"]                                # Number of montage frames
        self.entries["ImageType"] = ["0"]                                       # 0 for mrc file

        other_entries = ["MontBinning", "MapExposure", "MapSpotSize", "MapIntensity", "MapSlitIn", "MapSlitWidth", "DefocusOffset", "K2ReadMode", "NetViewShiftXY", "ViewBeamShiftXY", "ViewBeamTiltXY", "MapProbeMode", "MapLDConSet", "MapTiltAngle"]
        for entry in other_entries:
            if entry in template.entries.keys():
                self.entries[entry] = template.entries[entry]

        self.entries["PtsX"] = [str(pt) for pt in pts_x]
        self.entries["PtsY"] = [str(pt) for pt in pts_y]

        self.convertEntries()

    def createPoint(self, coords, map_id, group_id=None, note=""):
        """Creates new point item."""

        self.entries["Color"] = ["0"]
        if len(coords) < 3:
            raise ValueError("Point item needs 3D coords but only 2D were given!")
        self.entries["StageXYZ"] = [str(x) for x in coords]
        self.entries["NumPts"] = ["1"]
        self.entries["Regis"] = ["1"]
        self.entries["Type"] = ["0"]
        if group_id is not None:
            self.entries["GroupID"] = [str(group_id)]
        self.entries["MapID"] = [str(map_id)] 
        self.entries["PtsX"] = [str(coords[0])]
        self.entries["PtsY"] = [str(coords[1])]

        self.entries["Note"] = note.split(" ") if isinstance(note, str) else note

        self.convertEntries()

    def createPolygon(self, pts_x, pts_y, z, map_id, note=""):
        """Creates new point item."""

        center = np.array([np.mean(pts_x), np.mean(pts_y), z])

        self.entries["Color"] = ["1"]
        self.entries["StageXYZ"] = [str(x) for x in center]
        self.entries["NumPts"] = [str(len(pts_x))]
        self.entries["Regis"] = ["1"]
        self.entries["Type"] = ["1"]
        self.entries["MapID"] = [str(map_id)] 
        self.entries["PtsX"] = [str(x) for x in pts_x]
        self.entries["PtsY"] = [str(y) for y in pts_y]

        self.entries["Note"] = note.split(" ") if isinstance(note, str) else note

        self.convertEntries()

    """
    def defaults(self):
        self.color = 0              # Color         (0 = red; 1 = green; 2 = blue; 3 = yellow; 4 = magenta; 5 = no realign)
        self.stage = [0, 0, 0]      # StageXYZ      Stage position
        self.num_pts = 0            # NumPts        Number of points
        self.draw = 1               # Draw          Flag to draw
        self.item_type = 0          # Type          0-2 for point, polygon, map
        self.note = ""              # Note          Note string
        self.group_id = 0           # GroupID       ID of group that item belongs to
        self.imported = 0           # Imported      Indicator of an imported map or point drawn on one
        self.drawn_id = 0           # DrawnID       ID of map point/polygon was drawn on
        self.same_pos_id = 0        # SamePosId     Items with matching IDs were taken at same raw stage position
        self.acquire = 0            # Acquire       Flag for acquiring
        self.map_file = ""          # MapFile       Full or relative path of map file
        self.map_id = 0             # MapID         Unique ID (all items get one, not just maps)
        self.map_montage = 0        # MapMontage    Flag that map is a montage
        self.map_section = 0        # MapSection    Section number in file
        self.map_binning = 1        # MapBinning    Binning at which map was taken, or of initial overview map image for montage
        self.map_mag_id = 0         # MapMagInd     Magnification index of map, or of non-map image a point or polygon was drawn on
        self.grid_map_transform = 

        self.attr_entries = ["Color", "StageXYZ", "NumPts", "Draw", "Type", "Note", ]
        self.other_entries = {}     # Catch all for entries that don't have attribute
    """
    @staticmethod
    def getMinimalTemplate():
        template = NavItem(index=0, label="template")
        template.stage = np.zeros(3)

        template.entries["Regis"] = ["1"]                                       # Registration
        template.entries["MapBinning"] = ["1"]                                  # Map binning
        template.entries["MapMagInd"] = ["0"]                                   # Map mag index
        template.entries["MapCamera"] = ["1"]                                   # Camera number
        template.entries["MapScaleMat"] = ["1.0", "0.0", "0.0", "1.0"]          # Stage to pixel scale matrix for drawing (might be off when taken from montage template)

        return template

class Navigator:

    map_id_counter = 11

    def __init__(self, file=None, is_open=False) -> None:
        self.file = file
        self.header = ""
        self.items = []

        if self.file:
            self.file = Path(file)
            if file.exists():
                self.loadOld()
                self.readFromFile()
            else:
                self.openNew()
        elif is_open:
            self.getOpen()

    def readFromFile(self):
        """Reads lines from navigator text file and delegates to NavItem to read item blocks."""

        with open(self.file) as f:
            for line in f:
                col = line.strip().split(" ")
                if col[0] == "": continue
                if col[0] == "[Item":
                    label = col[2].strip("]")
                    self.items.append(NavItem(index=len(self.items) + 1, label=label))
                    self.items[-1].readBlock(f)
                    self.items[-1].convertEntries()
                else:
                    self.header += line

    def writeToFile(self, file=None):
        """Write new navigator file."""

        if file:
            file = Path(file)
        else:
            file = self.file
        if not file:
            raise ValueError("File for saving navigator could not be identified!")

        # Make sure header contains at least adoc version
        if not self.header:
            self.header = "AdocVersion = 2.00\n"

        text = self.header + "\n"

        for item in self.items:
            text += item.getBlock()

        with open(file, "w") as f:
            f.write(text)

    @dummy_skip
    @serialem_check
    def loadOld(self):
        """Load nav file."""

        nav_status = sem.ReportIfNavOpen()
        # If a nav file is open
        if nav_status == 2:
            sem.SaveNavigator()

            # Check if current nav file is from loaded grid
            nav_file = Path(sem.ReportNavFile())
            if nav_file == self.file:
                return

        # If nav is not saved
        elif nav_status == 1:
            sem.SaveNavigator("temp.nav")
            log("WARNING: Open navigator was saved as temp.nav and closed!")

        # If nav is not open
        else:
            sem.OpenNavigator()

        # Load nav file
        sem.ReadNavFile(str(self.file))

    @dummy_skip
    @serialem_check
    def openNew(self):
        """Open new nav file."""

        nav_status = sem.ReportIfNavOpen()
        # If a nav file is open
        if nav_status == 2:
            sem.SaveNavigator()
            sem.CloseNavigator()

        # If nav is not saved
        elif nav_status == 1:
            sem.SaveNavigator("temp.nav")
            sem.CloseNavigator()
            log("WARNING: Open navigator was saved as temp.nav and closed!")

        # Open new nav
        sem.OpenNavigator(str(self.file))

    @dummy_skip
    @serialem_check
    def getOpen(self):
        """Gets file from open nav in SerialEM."""

        self.file = Path(sem.ReportNavFile())
        self.pull()

    def pull(self):
        """Reloads nav file pulling changes from SerialEM."""

        # If SerialEM is open, save navigator
        if SERIALEM:
            sem.SaveNavigator(str(self.file))
        
        # Reset attributes and reload
        self.header = ""
        self.items = []
        self.readFromFile()

    def push(self):
        """Reloads nav file pushing changes in instance to SerialEM."""

        # If SerialEM is open, save navigator
        if SERIALEM:
            sem.SaveNavigator(str(self.file))
            sem.CloseNavigator()
        
        # Reset attributes and reload
        self.writeToFile()

        if SERIALEM:
            sem.ReadNavFile(str(self.file))

    @dummy_skip
    @serialem_check
    def shiftItems(self, shift):
        """Shift all navigator items by shift in microns."""
        
        log(f"Shifting items by {shift}!")
        sem.ShiftItemsByMicrons(*shift)

        self.pull()

    @dummy_replace("newMapFromImg")
    @serialem_check
    def newMap(self, buffer=None, img_file="", label="", note="", **kwargs):                                               # Only accept keyword args to make compatible with newMapFromImg
        """Makes new navigator map from buffer."""
        
        if not isinstance(buffer, Buffer):
            raise ValueError("Map has to be a buffer object!")
        
        img_file = Path(img_file)

        # Load buffer
        buffer.show()

        # Load or create file
        if img_file.exists():
            if not sem.IsImageFileOpen(str(img_file)):
                sem.OpenOldFile(str(img_file))
        else:
            sem.OpenNewFile(str(img_file))
            sem.S()

        # Make map
        nav_id = int(sem.NewMap()) - 1

        if label:
            sem.ChangeItemLabel(nav_id + 1, label)
        if note:
            sem.ChangeItemNote(nav_id + 1, note)

        # Close file
        sem.CloseFile()

        # Add nav_id to buffer
        buffer.nav_id = nav_id

        # Reload nav
        self.pull()

        return nav_id
    
    def newMapFromImg(self, img_file="", template_id=None, coords=(0, 0, 0), label="", note="", **kwargs):      # Only accept keyword args to make compatible with newMap
        """Makes new map from image file without SerialEM."""

        if template_id is not None:     # If template is None, a minimal template with default values will be used
            template = self.items[template_id]
        else:
            template = None

        nav_item = NavItem(len(self), label=label)
        nav_item.createMap(template, coords, img_file, self.map_id_counter, note)
        self.map_id_counter += 1

        self.items.append(nav_item)

        # Reload nav
        self.push()

        return len(self) - 1
    
    def newPoint(self, coords, label="", note="", color_id=0, group_id=None, update=True):
        """Adds new point to navigator."""

        if not label:
            label = str(len(self))

        nav_item = NavItem(len(self), label=label)
        nav_item.createPoint(coords, self.map_id_counter, group_id=group_id, note=note)
        self.map_id_counter += 1
        nav_item.entries["Color"] = [str(color_id)]

        self.items.append(nav_item)

        # Reload nav
        if update:
            self.push()

        return len(self) - 1    # nav_id
    
    def newPointGroup(self, points, label_prefix, color_id=0, stage_z=0):
        """Adds group of points to navigator."""

        group_id = self.map_id_counter
        self.map_id_counter += 1

        for p, point in enumerate(points):
            label = label_prefix + str(p + 1).zfill(len(str(len(points))))
            coords = np.array([point[0], point[1], stage_z])
            nav_id = self.newPoint(coords, label, label, color_id, group_id, update=False)

        # Reload nav
        self.push()

    def newPolygon(self, pts_x, pts_y, z, label="", note="", color=3, update=True):
        """Adds polygon to navigator."""

        if not label:
            label = str(len(self))

        nav_item = NavItem(len(self), label=label)
        nav_item.createPolygon(pts_x, pts_y, z, map_id=self.map_id_counter, note=note)
        self.map_id_counter += 1

        nav_item.entries["Color"] = [str(color)]

        self.items.append(nav_item)

        # Reload nav
        if update:
            self.push()

        return len(self) - 1    # nav_id

    def replaceItem(self, old_id, new_id):
        """Replaces an old item with a new item."""

        self.items[old_id] = self.items[new_id]
        self.items.pop(new_id)

        # Reload nav
        self.push()

    def getIDfromNote(self, note, warn=True):
        """Finds index if item by note."""

        candidates = [id for id, item in enumerate(self.items) if item.note == note]

        if not candidates:
            if warn:
                log(f"WARNING: No navigator item has the note [{note}]!")
            return None
            #raise LookupError(f"No navigator item found with note [{note}]!")

        if len(candidates) > 1:
            if warn:
                log(f"WARNING: More than one navigator items has the note [{note}]!")

        return candidates[0]
    
    def searchByEntry(self, entry, query, partial=False, subset=[]):
        """Search navigator items.

        Args:
            entry: Name of navigator item entry.
            query: Search query for specified entry.
            partial: Return items where query in entry instead of query == entry.
            subset: Only include if item ID is in this list.
        """

        results = []
        for id, item in enumerate(self.items):
            # Exclude items not in subset
            if subset and id not in subset: continue

            # Deal with special item entries
            if entry == "label":
                value = item.label
            elif entry == "stage":
                value = item.stage
            else:
                if entry in item.entries.keys():
                    # Concatenate entry back to string
                    value = " ".join(item.entries[entry])
                else:
                    continue
            
            # Check for match
            if not partial:
                if value == query:
                    results.append(id)
            else:
                if query in value:
                    results.append(id)

        return results
    
    def searchByCoords(self, coords, margin=0, subset=None):
        """Search navigator stage coords in X and Y.

        Args:
            coords: Coords at which to search.
            margin: Accepted distance from given coords.
            subset: Only include if item ID is in this list.
        """

        coords = np.array(coords, dtype=np.float32)

        results = []
        for id, item in enumerate(self.items):
            # Exclude items not in subset
            if subset is not None and id not in subset: continue

            # Calculate distance
            dist = item.getDistance(coords)

            # Check for match
            if dist <= margin:
                results.append(id)

        return results

    def __len__(self):
        return len(self.items)
    
    def __iter__(self):
        return iter(self.items)








if __name__ == "__main__":
    test_nav = Path("/Users/eifabian/Documents/temp/GW282_g4.nav")

    nav = Navigator(test_nav)

    print(len(nav))
    print(nav.header)

    print(nav.getIDfromNote("good (86%)"))

    print(nav.getIDfromNote("20240528_GW282_g4_l10_tgts.txt"))

    #print(nav.getIDfromNote("fail"))

    result = nav.searchByEntry("label", "PL", partial=True)
    print(result)

    coords = (318.903, -699.347)
    result = nav.searchByCoords(coords, 5, result)
    print(result)

    for item in result:
        print(nav.items[item].label, nav.items[item].stage)

    nav.writeToFile(test_nav.parent / (test_nav.stem + "_rewritten.nav"))

    print(nav.items[5].entries["Color"])