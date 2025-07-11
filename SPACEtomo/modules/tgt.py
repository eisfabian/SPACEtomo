#!/usr/bin/env python
# ===================================================================
# ScriptName:   tgt
# Purpose:      Class to hold and process target areas
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/07
# Revision:     v1.3
# Last Change:  2025/04/08: added tracking centering upon splitting areas
#               2025/01/30: added rectangular search threshold, small fixes
#               2024/09/02: fixed area name not considered when preparing targets
#               2024/08/19: added virtual map creation, nav preparation, target export
#               2024/08/16: added PACEArea child of TargetArea
#               2024/08/12: implemented local optimization
#               2024/08/09: added method to change target area for point
#               2024/08/07: separated from old SPACEtomo_tgt.py
# ===================================================================

import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2

from SPACEtomo.modules.nav import Navigator
from SPACEtomo.modules.buf import Buffer
from SPACEtomo.modules.ext import calcScore_cluster, calcScore_point
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log


class Targets:
    """Class to hold all target areas of lamella with functionality to redistribute points among target areas and facilitate access on the points level."""

    def __init__(self, map_dir, map_name, tgt_params, map_dims, map_pix_size=None) -> None:
        self.map_dir = Path(map_dir)
        self.map_name = map_name            # needed for file names
        self.map_dims = map_dims            # needed to check bounds
        self.map_pix_size = map_pix_size    # nm/px, needed for meta_data
        self.tgt_params = tgt_params        # needed for rec_dims and target score calculation

        # Get camera dims
        self.rec_dims = np.array(tgt_params.weight.shape)

        self.areas = []
        self.suggestions = []
        self.settings = None

        # Set class attributes for areas
        TargetArea.tgt_params = self.tgt_params
        TargetArea.rec_dims = self.rec_dims



    def addTarget(self, coords, new_area=False, ld_area="R"):
        """Checks conditions and adds new target to closest target area."""

        # Check if coords are out of bounds
        if not self.checkBounds(coords):
            log("WARNING: Can't add target. Too close to the edge!")
            return False
        
        # Create target areas to allow manual picking without auto run
        if not self.areas:
            self.areas.append(TargetArea())

        # Check if coords are too close to existing point
        closest_point_id, in_range = self.getClosestPoint(coords, np.min(self.rec_dims))
        if in_range and ld_area == "R": # allow overlapping targets for non Record targets
            log("WARNING: Target is too close to an existing target! It will not be added.")
            return False
        
        if new_area:
            # Create new area only if last area has points
            if self.areas[-1]:
                self.areas.append(TargetArea())

                # Add all geo points to new target area
                for geo_point in self.areas[0].geo_points:
                    self.areas[-1].addPoint(geo_point, geo=True)
            closest_area = -1

        else:
            # Figure out which target area tracking targets is closest
            closest_area = self.getClosestArea(coords)

        # Add point
        self.areas[closest_area].addPoint(coords, ld_area=ld_area)
        log("NOTE: Added new target!")
        return True

    def addGeoPoint(self, coords):
        """Similar to addTargets for geo points with different checks."""

        # Check if coords are out of bounds
        if not self.checkBounds(coords):
            log("WARNING: Can't add point. Too close to the edge!")
            return False
        
        # No geo points without any targets selected
        if not self.areas:
            log("WARNING: Can't add geo point without any selected targets!")
            return False

        # Check if coords are too close to existing point
        closest_point_id, in_range = self.getClosestPoint(coords, np.min(self.rec_dims))
        if in_range:
            log("WARNING: Point is too close to an existing target! It will not be added.")
            return False
        
        # Add geo point to all target areas
        for area in self.areas:
            area.addPoint(coords, geo=True)
        log("NOTE: Added geo point!")
        return True
    
    def movePointToArea(self, id, new_area_id):
        """Moves point from old target area to new target area."""

        old_area_id = id[0]
        old_point_id = id[1]
    
        # Get point data
        coords = self.areas[old_area_id].points[old_point_id]
        ld_area = self.areas[old_area_id].ld_areas[old_point_id]
        score = self.areas[old_area_id].scores[old_point_id]

        # Add to new area
        self.areas[new_area_id].addPoint(coords, ld_area=ld_area, score=score)

        # Remove from old area (and area if no points remaining)
        self.areas[old_area_id].removePoint(old_point_id)
        if len(self.areas[old_area_id]) == 0:
            self.areas.pop(old_area_id)
    
    def getAllPoints(self, include_geo=False):
        """Creates list of all points and list of original ids."""
        
        points = np.empty([0, 2])
        ids = []
        for a, area in enumerate(self.areas):
            points = np.vstack([points, area.points])
            ids += [[a, p] for p in range(len(area.points))]        # keep track of area and point ids for each point
        return points, ids

    def getClosestPoint(self, coords, threshold):
        """Finds closest point and checks if distance within threshold."""

        points, point_ids = self.getAllPoints()
        # Check for point within range
        if len(points) > 0:
            closest_id = np.argmin(np.linalg.norm(points - coords, axis=1))

            # Check if distance is within threshold
            if isinstance(threshold, np.number):
                valid = np.linalg.norm(points[closest_id] - coords) < threshold
            elif isinstance(threshold, np.ndarray):
                valid = np.all(np.abs(points[closest_id] - coords) < threshold)
            else:
                log(f"WARNING: Invalid threshold type for closest point {type(threshold)}! Need float or array!")
                return None, False
            return point_ids[closest_id], valid
        else:
            return None, False

    def getClosestArea(self, coords):
        """Finds area with closest tracking point."""

        if len(self.areas) > 1:
            # Make list of center points and exclude empty areas
            track_points = [target_area.center for target_area in self.areas if target_area.center is not None]
            # Get ID of closest center point
            closest_area = np.argmin(np.linalg.norm(track_points - coords, axis=1))
        else:
            closest_area = 0
        return closest_area
    
    def getClosestGeoPoint(self, coords, threshold):
        """Finds closest geo point and checks if distance within threshold."""

        if self.areas:
            points = self.areas[0].geo_points
            if len(points) > 0:
                closest_id = np.argmin(np.linalg.norm(points - coords, axis=1))

                # Check if distance is within threshold
                if isinstance(threshold, np.number):
                    valid = np.linalg.norm(points[closest_id] - coords) < threshold
                elif isinstance(threshold, np.ndarray):
                    valid = np.all(np.abs(points[closest_id] - coords) < threshold)
                else:
                    log(f"WARNING: Invalid threshold type for closest point {type(threshold)}! Need float or array!")
                    return None, False
                return closest_id, valid
        return None, False
    
    def getClosestSuggestion(self, coords, threshold):
        """Finds closest target suggestion and checks if distance within threshold."""

        if len(self.suggestions) > 0:
            closest_id = np.argmin(np.linalg.norm(self.suggestions - coords, axis=1))

            # Check if distance is within threshold
            if isinstance(threshold, np.number):
                valid = np.linalg.norm(self.suggestions[closest_id] - coords) < threshold
            elif isinstance(threshold, np.ndarray):
                valid = np.all(np.abs(self.suggestions[closest_id] - coords) < threshold)
            else:
                log(f"WARNING: Invalid threshold type for closest point {type(threshold)}! Need float or array!")
                return None, False
            return closest_id, valid
        return None, False
    
    def checkBounds(self, coords):
        """Checks if coords are out of bounds of map."""

        if not self.rec_dims[0] / 2 <= coords[0] < self.map_dims[0] - self.rec_dims[0] / 2 or not self.rec_dims[1] / 2 <= coords[1] < self.map_dims[1] - self.rec_dims[1] / 2:
            return False
        return True

    def resetGeo(self):
        for area in self.areas:
            area.geo_points = np.empty([0, 2])

    def removeGeoPoint(self, point_id):
        """Removes single geo_point from all areas. (Assumes geo_points are kept synced for all target areas.)"""

        for area in self.areas:
            area.removeGeoPoint(point_id)

    def mergeAreas(self):
        """Merge all target areas."""

        # Check if more than area exist
        if len(self.areas) > 1:

            # Create new area
            merged_area = TargetArea()

            # Extract points from all previous areas
            for area in self.areas:
                for point, ld_area, score in zip(area.points, area.ld_areas, area.scores):
                    merged_area.addPoint(point, ld_area=ld_area, score=score)

                for point in area.geo_points:
                    # Check if geo point was already added to merged_area
                    if np.any([np.array_equal(point, geo_point) for geo_point in merged_area.geo_points]):
                        continue
                    merged_area.addPoint(point, geo=True)

            # Keep only merged area
            self.areas = [merged_area]

    def splitArea(self, area_num=2):
        """Use kmeans clustering to split target area."""

        # Merge previous areas
        self.mergeAreas()

        # Cluster all points
        centroids, labels = kmeans2(self.areas[0].points, area_num, minit="++")
        counts = np.bincount(labels)
        log(f"NOTE: Split targets into {len(counts)} areas with {', '.join(counts.astype(str))} targets, respectively.")

        # Create new areas and add points according to their label
        new_areas = []
        for point_id, area_id in enumerate(labels):
            while area_id >= len(new_areas):
                new_areas.append(TargetArea())
            new_areas[area_id].addPoint(self.areas[0].points[point_id], ld_area=self.areas[0].ld_areas[point_id], score=self.areas[0].scores[point_id])

        # Center tracking target
        for area_id, area in enumerate(new_areas):
            if len(area.points) > 0:
                area.centerTrack(centroids[area_id])

        # Add geo point to all target areas
        for geo_point in self.areas[0].geo_points:
            for area in new_areas:
                area.addPoint(geo_point, geo=True)

        # Overwrite target areas
        self.areas = new_areas

    def splitAreaManual(self):
        """Split targets by distance to existing tracking points."""

        remove_list = []
        for area_id, area in enumerate(self.areas):
            for point_id, (point, ld_area, score) in enumerate(zip(area.points, area.ld_areas, area.scores)):
                # Skip tracking target
                if point_id == 0: continue

                # Find closest tracking point
                closest_area_id = self.getClosestArea(point)

                # If closest tracking point is not current area
                if closest_area_id != area_id:
                    # Move point from current to closest area
                    self.areas[closest_area_id].addPoint(point, ld_area=ld_area, score=score)
                    remove_list.append((area_id, point_id))

        # Remove points that were moved (starting from highest ids)
        for area_id, point_id in sorted(remove_list, key=lambda x: x[1], reverse=True):
            self.areas[area_id].removePoint(point_id)

    def loadAreas(self):
        """Loads json data for all point files."""

        # Find all point files
        point_files = sorted(self.map_dir.glob(self.map_name + "_points*.json"))

        if len(point_files) > 0:
            # Instantiate target area for each file
            self.areas = []
            for file in point_files:
                area = TargetArea(file=file)
                # Only load non-empty point files
                if len(area) > 0:
                    self.areas.append(area)
            log("NOTE: Loaded target coordinates from file.")
        else:
            log("NOTE: No target coordinates found.")
            self.areas = []

        # Check if settings were loaded from file
        if TargetArea.settings:
            self.settings = TargetArea.settings

    def exportTargets(self, settings=None):
        """Exports all target areas to json files."""

        # Check for existing point files and delete them
        point_files = sorted(self.map_dir.glob(self.map_name + "_points*.json"))
        if len(point_files) > 0:
            log("NOTE: Previous targets were deleted.")
            for file in point_files:
                # Delete file on path
                file.unlink()

        meta_data = {"pix_size": self.map_pix_size, "img_size": self.map_dims, "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

        if self.areas:
            for a, area in enumerate(self.areas):
                if not area: continue   # Don't save empty areas
                file = self.map_dir / (self.map_name + f"_points{a}.json")
                area.exportToJson(file, settings, meta_data)
        else:
            # Write empty points file to ensure empty targets file is written and map is considered processed
            with open(self.map_dir / (self.map_name + "_points.json"), "w") as f:
                json.dump({"points": [], "scores": [], "geo_points": [], "meta_data": meta_data}, f)    

        # Update settings
        if settings:
            self.settings = settings

    def __bool__(self):
        """Return False if no points exist."""

        return bool(self.getAllPoints()[1])     # Have to check list of ids since points is a numpy array
    
    def __len__(self):
        return len(self.getAllPoints()[0])
    
    def __str__(self):
        text = "Target areas:\n"
        for a, area in enumerate(self.areas):
            text += f"Area {a}:\t{len(area)} points\n"
        return text

class TargetArea:
    """Class to hold all data for one target area."""

    # Class attributes
    tgt_params = None
    rec_dims = None

    settings = None

    def __init__(self, file=None) -> None:
        self.points = np.empty([0, 2])
        self.scores = np.empty(0)
        self.ld_areas = np.empty(0, dtype=str)  # Low Dose areas, e.g. "R" for Record
        self.geo_points = np.empty([0, 2])

        self.meta_data = {}

        # Dict to hold class dependent scores
        self.class_scores = {}

        if file:
            self.loadFromJson(file)

        if len(self.points) > 0:
            self.center = self.points[0]
        else:
            self.center = None

    def loadFromJson(self, file):
        """Reads targets from json file."""

        # Load json data
        with open(file, "r") as f:
            area = json.load(f, object_hook=utils.revertTaggedString)

        if np.any(area["points"]):
            self.points = area["points"]
        if "ld_areas" in area and np.any(area["ld_areas"]):
            self.ld_areas = area["ld_areas"]
        if np.any(area["scores"]):
            self.scores = area["scores"]
        if np.any(area["geo_points"]):
            self.geo_points = area["geo_points"]
        if "settings" in area.keys():
            TargetArea.settings = area["settings"]
        if "meta_data" in area.keys():
            self.meta_data = area["meta_data"]

    def exportToJson(self, file, settings=None, meta_data=None):
        """Writes targets to json file."""

        # Convert to dictionary
        output_dict = {"points": self.points, "ld_areas": self.ld_areas, "scores": self.scores, "geo_points": self.geo_points}

        # Set measureGeo to True if geo points exist
        if len(self.geo_points) > 0:
            if settings:
                settings["measureGeo"] = True
            else:
                settings = {"measureGeo": True}

        if settings:
            output_dict["settings"] = settings
            # Update class attribute
            TargetArea.settings = settings
        output_dict["meta_data"] = meta_data if meta_data else self.meta_data

        # Write to file
        with open(file, "w") as f:
            json.dump(output_dict, f, indent=4, default=utils.convertToTaggedString)

    def addPoint(self, coords, geo=False, ld_area="R", score=100):
        """Adds new point at coords."""

        if not geo:
            self.points = np.vstack([self.points, coords])
            self.ld_areas = np.append(self.ld_areas, [ld_area])
            self.scores = np.append(self.scores, [score])
            if self.center is None:
                self.center = coords
        else:
            self.geo_points = np.vstack([self.geo_points, coords])

        log(f"Added point {coords} with score {score} using Low Dose area {ld_area}.")

    def updatePoint(self, id, coords):
        """Updates point by id."""

        self.points[id] = coords
        if id == 0:
            self.center = coords

    def removePoint(self, id):
        """Removes point by id."""

        self.points = np.delete(self.points, id, axis=0)
        self.ld_areas = np.delete(self.ld_areas, id, axis=0)
        self.scores = np.delete(self.scores, id, axis=0)

        # Update center
        if len(self.points) > 0:
            self.center = self.points[0]

    def removeGeoPoint(self, point_id):
        """Removes geo point by id."""

        self.geo_points = np.delete(self.geo_points, point_id, axis=0)

    def makeTrack(self, id):
        self.points[0: id + 1] = np.roll(self.points[0: id + 1], shift=1, axis=0)
        self.ld_areas[0: id + 1] = np.roll(self.ld_areas[0: id + 1], shift=1, axis=0)
        self.scores[0: id + 1] = np.roll(self.scores[0: id + 1], shift=1, axis=0)

        # Update center
        self.center = self.points[0]

    def centerTrack(self, centroid):
        """Chooses tracking target closest to centroid."""

        if len(self.points) > 0:
            # Get closest point
            closest_id = np.argmin(np.linalg.norm(self.points - centroid, axis=1))
            # Move tracking target to closest point
            self.makeTrack(closest_id)

    def getPointInfo(self, id):
        """Gets information about point by id."""

        dist = np.linalg.norm(self.points[id] - self.center)
        return self.scores[id], dist
    
    def getClassScores(self, cat, mask):
        """Calculates scores for single classes."""

        self.class_scores[cat] = np.empty(0)
        for point in self.points:
            score = calcScore_point(point, mask, np.zeros(mask.shape), 0, self.tgt_params.weight, self.tgt_params.edge_weights)
            self.class_scores[cat] = np.append(self.class_scores[cat], [score])

    def getGeoInRange(self, is_limit_px):
        """Returns list of geo points within image shift limit [pixels]."""

        distances = np.linalg.norm(self.geo_points - self.center, axis=1)
        return [geo_point for g, geo_point in enumerate(self.geo_points) if distances[g] < is_limit_px]
    
    def optimizeLocally(self, point_id, mask, penalty_mask=None):
        """Use mask to optimize target position locally."""

        # Check if class attributes were set
        if not self.tgt_params or self.rec_dims is None:
            log("ERROR: Target params have not been set for target area!")
            return
        
        # Check if penalty mask was given
        if not penalty_mask:
            penalty_mask = np.zeros(mask.shape)
        
        # Get initial coords and score
        point = self.points[point_id, :]
        score = calcScore_point(point, mask, penalty_mask, self.tgt_params.penalty, self.tgt_params.weight, self.tgt_params.edge_weights)
        log(f"Initial score: {round(score, 3)}")

        # Call optimization
        start_offset = np.zeros(2)
        offset_raw = minimize(calcScore_cluster, start_offset, args=(point[np.newaxis, :], mask, penalty_mask, self.tgt_params.penalty, self.tgt_params.weight, self.tgt_params.edge_weights), method="nelder-mead", bounds=((-0.1, 0.1), (-0.1, 0.1)))
        offset_pixel = offset_raw.x * max(self.rec_dims) * 10
        point += offset_pixel

        # Calculate new score
        log(f"Target was moved by {round(np.linalg.norm(offset_pixel))} pixels.")
        score = calcScore_point(point, mask, penalty_mask, self.tgt_params.penalty, self.tgt_params.weight, self.tgt_params.edge_weights)
        log(f"New score: {round(score, 3)}")

        # Update point
        self.points[point_id] = point
        self.scores[point_id] = score
    
    def __bool__(self):
        return bool(len(self.points))

    def __len__(self):
        return len(self.points)


class PACEArea(TargetArea):
    """Target area with additional functions to create PACEtomo tgts file."""

    model = None
    imaging_params = None

    def __init__(self, file, model, imaging_params, buffer: Buffer) -> None:
        super().__init__(file)

        PACEArea.model = model
        PACEArea.imaging_params = imaging_params

        self.pix_size = self.meta_data["pix_size"] if "pix_size" in self.meta_data else self.model.pix_size
        self.map_buffer = buffer
        self.nav_id = buffer.nav_id

        self.stageCoords()
        self.specimenCoords()

    def getClassScores(self, cat, mask):
        raise NotImplementedError("This method is not implemented for PACEArea!")
    
    def optimizeLocally(self, point_id, mask, penalty_mask=None):
        raise NotImplementedError("This method is not implemented for PACEArea!")

    def stageCoords(self):
        """Makes list of points in stage coords."""

        self.scaleCoordsBuffer()
        self.points_stage = np.array([self.map_buffer.px2stage(np.flip(point)) for point in self.points])
        if self.center is not None:
            self.center_stage = self.points_stage[0]
        else:
            self.center_stage = None
        self.geo_points_stage = np.array([self.map_buffer.px2stage(np.flip(point)) for point in self.geo_points])

    def specimenCoords(self):
        """Makes list of points in relative specimen coords."""

        if self.center_stage is not None:
            self.points_ss = np.array([self.imaging_params.s2ss_matrix @ (point - self.center_stage) for point in self.points_stage])
            self.geo_points_ss = np.array([self.imaging_params.s2ss_matrix @ (point - self.center_stage) for point in self.geo_points_stage])
        else:
            log("WARNING: Can't convert to relative specimen coordinates because tracking target is not defined!")

    def scaleCoordsBuffer(self):
        self.points *= self.pix_size / self.map_buffer.pix_size
        self.geo_points *= self.pix_size / self.map_buffer.pix_size
        self.pix_size = self.map_buffer.pix_size

    def scaleCoordsModel(self):
        self.points *= self.pix_size / self.model.pix_size
        self.geo_points *= self.pix_size / self.model.pix_size
        self.pix_size = self.model.pix_size

    def getSSDistances(self):
        return np.linalg.norm(self.points_ss, axis=1)
    
    def checkISLimit(self, remove=False):
        """Checks if all points are within image shift limits."""

        for p, (point, dist) in enumerate(zip(self.points_stage, self.getSSDistances())):
            if dist > self.imaging_params.IS_limit:
                log(f"WARNING: Point {p + 1} {point} requires image shifts ({round(dist, 1)} microns) beyond the image shift limit ({self.imaging_params.IS_limit})!")
                if remove:
                    log(f"ERROR: Removing of points is not implemented yet!")
                    pass # TODO remove point from all lists
    
    def makeTrack(self, id, nav: Navigator):
        super().makeTrack(id)
        log("ERROR: Making track is not fully implemented yet.")
        #TODO deal with nav changes

    def prepareForAcquisition(self, map_file, nav: Navigator, grid_name=""):
        """Prepares virtual maps and navigator for acquisition."""

        # Check if map_file contains area
        if grid_name:
            suffix = map_file.stem.split(grid_name)[-1]
            if "_A" in suffix:
                area_name = map_file.stem
                map_name = grid_name + suffix.split("_A")[0]
            else:
                map_name = area_name = map_file.stem
        else:
            if len(map_file.stem.split("_A")) > 1:
                area_name = map_file.stem
                map_name = map_file.stem.rsplit("_A", 1)[0]
            else:
                map_name = area_name = map_file.stem

        # Get lamella montage nav_id for template
        #template_id = nav.getIDfromNote(map_name + ".mrc") # <= could not deal with Notes like: Sec 0 - map_name.mrc
        #template_id = nav.searchByEntry("Note", f"{map_name}.mrc", partial=True)
        template_id = self.nav_id
        if not template_id:
            # Try finding map using note
            log(f"DEBUG: Could not find template map {map_name}.mrc in navigator. Trying to find by note...")
            template_id = nav.searchByEntry("Note", f"{map_name}.mrc", partial=True)
            if not template_id:
                log(f"ERROR: Could not find template map {map_name}.mrc in navigator!")
                return
            elif len(template_id) > 1:
                log(f"WARNING: Found multiple entries for template map {map_name}.mrc in navigator!")
            template_id = template_id[0]

        # Convert to buffer pix size
        self.scaleCoordsBuffer()

        for p, point in enumerate(self.points):
            # Make virtual map from montage in buffer
            virt_map_file = map_file.parent / (area_name + "_tgt_" + str(p + 1).zfill(3) + "_view.mrc")
            virt_map = np.flip(self.map_buffer.getCropImage(point, np.flip(self.imaging_params.cam_dims)), axis=0)       # crop image and flip y-axis
            log(f"DEBUG: Virtual map dimenstions: {virt_map.shape}")
            utils.writeMrc(virt_map_file, virt_map, self.map_buffer.pix_size)

            # Add map to navigator
            nav_id = nav.newMapFromImg(img_file=virt_map_file, template_id=template_id, coords=self.points_stage[p], label=str(p + 1).zfill(3), note=virt_map_file.name)

            # Tracking target id for nav adjustments
            if p == 0:
                track_id = nav_id

        # Push once per area to save nav reloads
        nav.push()
        nav.items[track_id].changeAcquire(1)
        nav.items[track_id].changeNote(area_name + "_tgts.txt")

        nav.newPointGroup(self.geo_points_stage, "geo", color_id=5, stage_z=nav.items[template_id].stage[2])    

    def getTargetsInfo(self, area_name):
        """Returns target area data in dicts."""

        targets = []
        for p, point in enumerate(self.points_stage):
            targets.append({
                "tsfile": area_name + "_ts_" + str(p + 1).zfill(3) + ".mrc", 
                "viewfile": area_name + "_tgt_" + str(p + 1).zfill(3) + "_view.mrc", 
                "SSX": self.points_ss[p][0], 
                "SSY": self.points_ss[p][1], 
                "stageX": point[0], 
                "stageY": point[1], 
                "LDArea": self.ld_areas[p],
                "SPACEscore": self.scores[p], 
                "skip": "False"})

        geo_points = []
        for g, geo_point in enumerate(self.geo_points_stage):
            geo_points.append({
                "SSX": self.geo_points_ss[g][0], 
                "SSY": self.geo_points_ss[g][1], 
                "stageX": geo_point[0], 
                "stageY": geo_point[1], 
            })

        return targets, geo_points, self.settings