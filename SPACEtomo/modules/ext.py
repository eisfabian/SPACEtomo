#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_functions_ext
# Purpose:      Functions necessary to run SPACEtomo and that can be run externally without SerialEM.
#               More information at http://github.com/eisfabian/PACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/10/04
# Revision:     v1.2dev
# Last Change:  2024/08/21: removed SPACE_DIR, fixed map staying in run list when error is detected
#               2024/07/31: allowed for empty seg to be 1x1 image
#               2024/07/25: added overlap when tiling in findLamellae function
#               2024/06/04: fixed findLamellae crash when img dims equal side lengths
#               2024/04/12: fixed weight_matrix dimension issue
#               2024/04/09: fixes after Rado Krios test, added PID output to SPACE_runs.json
#               2024/03/27: fixes after monitor test
#               2024/03/25: fixes after Krios 3 test, added check for truncated map in error file
#               2024/03/20: changed ultralytics import to be conditional to allow import of space_ext on K2 computer
#               2024/03/18: fixed extra tracking
#               2024/03/11: bug fixes and YOLO device control, added array conversion for json, got target selection working, adjusted plotTargets
#               2024/03/07: added TgtParams class to condense parameters for runTargetSelection function
#               2024/03/06: implemented clustering when targets spread over more area than image shift limits (still needs consideration in export)
#               2024/02/14: wrote general lamella detection function generally usable, fixed findGeoPoints crash when using lamella targeting
#               2024/02/13: split collectWGMap from lamella detection, split functions in to external functions file not requiring SerialEM
#               2023/12/12: fixed proximity removal, added version check
#               2023/12/11: added removal of lamellae due to close proximity
#               2023/11/28: fixes after Krios test
#               2023/11/15: added center padding for IM lamella detection and consider multiple hits, removed lamellae close to grid edge, removed geo points beyond IS limit, fixes and improvements to findOffset
#               2023/11/06: used IM lamella detection as confirmation
#               2023/11/04: fixes after Krios test
#               2023/10/31: fixes after Krios test
#               2023/10/27: added lamella option to select all classes, added ice check for tracking point, adjusted plotting option
#               2023/10/25: added nav file management
#               2023/10/23: added extra tracking
#               2023/10/18: added virtual map for targeting
#               2023/10/16: fixes after Krios test, removed polygon (listToSEMarray not found in external script)
#               2023/10/12: fixes after Krios test, added findOffset for more lamellae if lamella not found
#               2023/10/04: outsourcing of functions from main SPACEtomo script
# ===================================================================

import os
import sys
import glob
import json
import time
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images
from scipy.ndimage import rotate
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2

import SPACEtomo.config as config
from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log


##### MM functions #####

class MMModel:
    def __init__(self, alt_model_dir=None):
        self.script = Path(__file__).parent.parent / config.MM_model_script

        # Allow for loading of specific model
        if alt_model_dir is None or not Path(alt_model_dir).exists():
            self.dir = Path(config.MM_model_folder)
        else:
            self.dir = Path(alt_model_dir)

        if not self.dir.exists() or not (self.dir / "dataset.json").exists():
            raise FileNotFoundError("Lamella segmentation model does not exist! Please import a model using the SPACEmodel command!")

        # Get all classes from model folder
        with open(self.dir / "dataset.json", "r") as f:
            dataset_json = json.load(f)
        self.categories = dataset_json["labels"]

        # Check json for pix size and fall back to config if not present
        if "pixel_size" in dataset_json.keys():
            self.pix_size = dataset_json["pixel_size"]
        else:
            self.pix_size = config.MM_model_pix_size

    def setDimensions(self, mic_params):
        self.beam_diameter = mic_params.rec_beam_diameter * 1000 / self.pix_size
        self.cam_dims = (mic_params.cam_dims[[1, 0]] * mic_params.rec_pix_size / self.pix_size).astype(int)    # record camera dimensions in pixels on view montage

    # Make empty segmentation to skip nnUnet and go to manual target selection
    @staticmethod
    def saveEmptySeg(map_dir, map_name):
        #map_image = Image.open(os.path.join(map_dir, map_name + ".png"))
        #seg = np.zeros(map_image.size, dtype=np.uint8)
        seg = np.zeros((1, 1), dtype=np.uint8)
        seg_image = Image.fromarray(seg)
        seg_image.save(os.path.join(map_dir, map_name + "_seg.png"))

# Class for lamella to enable target selection
class Lamella:
    #def __init__(self, map_name, map_dir, target_list, penalty_list, model, score_weights, score_weights_edge, grid_vecs, mic_params, max_tilt, plot=False):
    def __init__(self, map_name, map_dir, model, mic_params, tgt_params, plot=False, alt_seg_path=None):
        self.map_name = map_name
        self.map_dir = map_dir

        # Make subfolder for intermediary files
        self.out_dir = os.path.join(self.map_dir, self.map_name, "")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.model = model
        self.mic_params = mic_params
        self.tgt_params = tgt_params

        self.plot = plot

        self.map = np.array(Image.open(os.path.join(map_dir, map_name + ".png")))

        # Load specific segmentation if specified or _seg suffix if not
        if alt_seg_path is not None and os.path.exists(alt_seg_path):
            self.segmentation = np.array(Image.open(alt_seg_path))
        else:
            self.segmentation = np.array(Image.open(os.path.join(map_dir, map_name + "_seg.png")))
        log("Generating target mask...")
        self.target_mask = self.makeMask(self.tgt_params.target_list)
        log("Generating background mask...")
        self.geo_mask = self.makeMask([name for name in self.model.categories if name not in self.tgt_params.target_list and name not in self.tgt_params.penalty_list])
        log("Generating penalty mask...")
        self.penalty_mask = self.makeMask(self.tgt_params.penalty_list)

        self.points = []
        self.point_scores = []
        self.clusters = []
        self.geo_points = []
        self.targets = []

    # Create mask from segmentation and selected classes
    def makeMask(self, class_names):
        mask = np.zeros(self.segmentation.shape)

        # Check if no class selected (sometimes happens for geo points when choosing whole lamella)
        if len(class_names) == 0:
            log("WARNING: Creating empty mask! This could cause an error later.")
            return mask

        for name in class_names:
            if name not in self.model.categories:
                log(f"ERROR: Unknown target class: {name}")
                log("Possible classes are: ")
                log(str(self.model.categories.keys()))
                break
            mask[self.segmentation == self.model.categories[name]] = 1

        # Save mask as binary image
        mask_img = Image.fromarray(mask, mode="1")
        mask_img.save(os.path.join(self.out_dir, self.map_name + "_mask_" + class_names[0] + ".png"))

        return mask

    # Define hexagonal grid of points depending on beam diameter
    def definePoints_grid(self, penalty_weight, threshold=0.0, alternative_mask=None):
        x1, x2 = self.tgt_params.vecs
        rec_dims = np.array(self.tgt_params.weight.shape)
        if alternative_mask is None:
            mask = self.target_mask
        else:
            mask = alternative_mask
        points = []
        point_scores = []
        max_rows = int(mask.shape[0] / abs(x1[0]) * 1.5)
        max_cols = int(mask.shape[1] / abs(x2[1]) * 1.5)
        for i in range(max_rows):
            for j in range(1, max_cols):
                point = i * x1 + j * x2 - max_rows // 2 * x1 - max_cols // 2 * x2 + np.array(mask.shape) // 2
            
                score = calcScore_point(point, mask, self.penalty_mask, penalty_weight, self.tgt_params.weight)
                if score > threshold:
                    points.append(point)
                    point_scores.append(score)
        return np.array(points), point_scores

    # Define sparse points depending on segmentation
    def definePoints_sparse(self, penalty_weight, threshold=0.0, alternative_mask=None):
        rec_dims = np.array(self.tgt_params.weight.shape)
        if alternative_mask is None:
            mask = self.target_mask
        else:
            mask = alternative_mask
        points = []
        point_scores = []
        i = rec_dims[0] / 2
        while i < mask.shape[0]:
            j = rec_dims[1] / 2
            while j < mask.shape[1]:
                point = np.array([i, j])
                score = calcScore_point(point, mask, self.penalty_mask, penalty_weight, self.tgt_params.weight)
                if score > threshold:
                    overlap = False
                    for pt in points:
                        dist = np.linalg.norm(pt - point)
                        if dist < self.model.beam_diameter:
                            overlap = True
                            break
                    if not overlap:
                        points.append(point)
                        point_scores.append(score)
                j += rec_dims[1]
            i += rec_dims[0]
        return np.array(points), point_scores
    
    # Add additional points surrounding existing points
    def addAdjacentPoints(self, penalty_weight, threshold=0.0):
        x1, x2 = self.tgt_params.vecs
        new_points = []
        new_point_scores = []
        for p, point in enumerate(self.points):
            new_points.append(point)
            new_point_scores.append(self.point_scores[p])
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == j == 0:
                        continue
                    new_point = point + i * x1 + j * x2
                    for other_point in new_points:
                        dif_real = other_point - new_point
                        dif_squeezed = (self.mic_params.view_rotM.T @ dif_real) * np.array([1, np.cos(np.radians(self.tgt_params.max_tilt))])   # scale distance according to maximum tilt angle
                        if np.linalg.norm(dif_squeezed) < self.model.beam_diameter:
                            break  
                    score = calcScore_point(new_point, self.target_mask, self.penalty_mask, penalty_weight, self.tgt_params.weight)
                    if score > threshold:
                        new_points.append(new_point)
                        new_point_scores.append(score)
        log(f"Added {len(new_points) - len(self.points)} adjacent points!")
        log(f"{len(new_points)} points remaining.")
        self.points = np.array(new_points)
        self.point_scores = new_point_scores

    # Find clusters of points based on beam diameter
    def findClusters(self, points):
        if len(points) < 2:
            return [[0]]
        squeezed_points = np.array([(self.mic_params.view_rotM.T @ point) * np.array([1, np.cos(np.radians(self.tgt_params.max_tilt))]) for point in points])
        distM = distance_matrix(squeezed_points, squeezed_points)
        clusters = []
        for p1 in range(len(points)):
            clusters.append([p[0] for p in np.argwhere(distM[p1] <= 1.01 * self.model.beam_diameter)])
        
        clusters = sorted(clusters)
        
        for i in range(3):
            if len(clusters) > 1:
                final_clusters = [clusters[0]]
                for cluster in clusters[1:]:
                    cluster_set = set(cluster)
                    overlap = False
                    for fc in range(len(final_clusters)):
                        fcluster_set = set(final_clusters[fc])
                        if len(cluster_set & fcluster_set) != 0:
                            final_clusters[fc] = list(cluster_set | fcluster_set)
                            overlap = True
                            break
                    if not overlap:
                        final_clusters.append(cluster)  
                clusters = final_clusters
            
        log(f"Clusters found: {len(clusters)}")
        return clusters    

    # Clean points based on score threshold
    def cleanPoints_score(self, penalty_weight, threshold=0.0, clusters=None):
        cleaned_points = []
        cleaned_point_scores = []
        removed_point_ids = []
        for p, point in enumerate(self.points):
            score = calcScore_point(point, self.target_mask, self.penalty_mask, penalty_weight, self.tgt_params.weight)
            if score > threshold:
                cleaned_points.append(point)
                cleaned_point_scores.append(score)
            else:
                removed_point_ids.append(p)
        if clusters is not None:
            removed_point_ids = sorted(removed_point_ids, reverse=True)
            for p in removed_point_ids:
                cleaned_clusters = []
                for cluster in clusters:
                    cleaned_clusters.append([])
                    for i in cluster:
                        if i == p: continue
                        if i > p: cleaned_clusters[-1].append(i - 1)
                        else: cleaned_clusters[-1].append(i)
                clusters = [cluster for cluster in cleaned_clusters if cluster != []]
                
        log(f"Cleaned {len(removed_point_ids)} points by score threshold!")
        log(f"{len(cleaned_points)} points remaining.")
        self.points = np.array(cleaned_points)
        self.point_scores = cleaned_point_scores
        if clusters is not None:
            self.clusters = clusters

    # Clean points based on beam overlap (needs more flexible input and output to clean against other groups of points, e.g. geo points)
    def cleanPoints_dist(self, points, max_tilt=0, tolerance=0.1, point_scores=None, clusters=None):
        if point_scores is not None:    # sort points according to score to remove low score points first
            point_zip = sorted(zip(list(range(len(points))), points, point_scores), key=lambda x: x[2])
            points = [p for _, p, _ in point_zip]
            original_point_ids = [pid for pid, _, _ in point_zip]
        else:
            original_point_ids = list(range(len(points)))
        removed_point_ids = []
        moved_point_ids = []
        for p1 in range(len(points)):
            too_close = False
            for p2 in range(len(points)):
                if p1 == p2 or original_point_ids[p2] in removed_point_ids: continue
                dif_real = points[p2] - points[p1]
                dif_squeezed = (self.mic_params.view_rotM.T @ dif_real) * np.array([1, np.cos(np.radians(max_tilt))])   # scale distance according to maximum tilt angle
                if np.linalg.norm(dif_squeezed) < self.model.beam_diameter * (1 - 2 * tolerance * np.linalg.norm(dif_squeezed) / np.linalg.norm(dif_real)) or (np.linalg.norm(dif_squeezed) < 0.99 * self.model.beam_diameter and (p1 in moved_point_ids or p2 in moved_point_ids)):
                    too_close = True
                    break
                elif np.linalg.norm(dif_squeezed) < self.model.beam_diameter:
                    overlap = (self.model.beam_diameter - np.linalg.norm(dif_squeezed))
                    move = self.mic_params.view_rotM @ (overlap * dif_squeezed / np.linalg.norm(dif_squeezed) / np.array([1, np.cos(np.radians(max_tilt))]) / 2)
                    points[p1] = points[p1] - move
                    points[p2] = points[p2] + move
                    moved_point_ids.extend([p1, p2])
            if too_close:
                removed_point_ids.append(original_point_ids[p1])
        if point_scores is not None:
            point_zip = sorted(zip(original_point_ids, points), key=lambda x: x[0])
            points = [p for _, p in point_zip]
        cleaned_points = [point for p, point in enumerate(points) if p not in removed_point_ids]
        
        if clusters is not None:
            removed_point_ids = sorted(removed_point_ids, reverse=True)
            for p in removed_point_ids:
                cleaned_clusters = []
                for cluster in clusters:
                    cleaned_clusters.append([])
                    for i in cluster:
                        if i == p: continue
                        if i > p: cleaned_clusters[-1].append(i - 1)
                        else: cleaned_clusters[-1].append(i)
                clusters = [cluster for cluster in cleaned_clusters if cluster != []]
                
        log(f"Cleaned {len(removed_point_ids)} points by distance threshold!")
        log(f"{len(cleaned_points)} points remaining.")
        return np.array(cleaned_points), clusters
    
    # Remove single point from from beam overlap
    def distancePoint(self, point, points):
        moved = False
        for p in range(len(points)):
            dif_real = points[p] - point
            dif_squeezed = (self.mic_params.view_rotM.T @ dif_real) * np.array([1, np.cos(np.radians(self.tgt_params.max_tilt))])   # scale distance according to maximum tilt angle
            if np.linalg.norm(dif_squeezed) < self.model.beam_diameter:
                overlap = (self.model.beam_diameter - np.linalg.norm(dif_squeezed))
                move = self.mic_params.view_rotM @ (overlap * dif_squeezed / np.linalg.norm(dif_squeezed) / np.array([1, np.cos(np.radians(self.tgt_params.max_tilt))]))
                point = point - move         
                moved = True
        return point, moved
    
    # Create plot of targets on montage
    def plotTargets(self, offsets=[], clusters=[], tracking_id=None, overlay=None, save="temp.png"):
        rec_dims = np.array(self.tgt_params.weight.shape)
        if len(self.points) > 0:   
            if tracking_id is None:
                # Check if target list already exists and use first positions of each target area as tracking target
                if len(self.targets) > 0:
                    closest_middle = [points["points"][0] for points in self.targets]     # had to change all closes_middle to a list re accommodate more than one tracking target
                else:
                    middle_point = [np.max(self.points[:, 0]) / 2 + np.min(self.points[:, 0]) / 2, np.max(self.points[:, 1]) / 2 + np.min(self.points[:, 1]) / 2]
                    closest_middle = [self.points[np.argmin(np.linalg.norm(self.points - middle_point, axis=1))]]
            else:
                closest_middle = [self.points[tracking_id]]
        else:
            closest_middle = [np.array(self.map.shape) // 2]
        fig, ax = plt.subplots(figsize=(16, 12))
        
        if overlay is None:
            plt.imshow(self.map, cmap="gray")
        else:
            alpha = 1 / 4
            overlay_map = np.clip(np.dstack([self.map / 255 * (1 - alpha) + overlay * alpha, self.map / 255 * (1 - alpha), self.map / 255 * (1 - alpha)]), 0, 1)
            plt.imshow(overlay_map)
        plt.axline(closest_middle[0][[1, 0]], slope=-1 / np.tan(np.radians(self.mic_params.view_ta_rotation)), linestyle="--", color="#ffffff")
        
        if len(offsets) > 0 and len(clusters) > 1:
            for c, cluster in enumerate(clusters):
                data = np.array(self.points[cluster, :])
                plt.scatter(data[:, 1] + rec_dims[1] / 2, data[:, 0] + rec_dims[0] / 2, marker="*", s=100)
                offset_cluster_pixel = offsets[c] * max(rec_dims) * 10
                plt.arrow(self.points[cluster, :][0][1] - offset_cluster_pixel[1], self.points[cluster, :][0][0] - offset_cluster_pixel[0], offset_cluster_pixel[1], offset_cluster_pixel[0], length_includes_head=True, width=10, color="#ff7d00")
        
        for p, point in enumerate(self.points):
            ellipse = Ellipse((point[1], point[0]), self.model.beam_diameter / np.cos(np.radians(self.tgt_params.max_tilt)), self.model.beam_diameter, angle=self.mic_params.view_ta_rotation, fill=False, linewidth=1, color="#ffd700")
            ax.add_artist(ellipse)
            if np.any([np.array_equal(point, center) for center in closest_middle]):
                color = "#c92b27"
            else:
                color = "#578abf"
            rect = Rectangle([point[1] - rec_dims[1] / 2, point[0] - rec_dims[0] / 2], rec_dims[1], rec_dims[0], fill=False, linewidth=2, color=color)
            ax.add_artist(rect)
            plt.text(point[1] - 10, point[0] - 10, str(p + 1), color=color)

        if len(offsets) == 1:
            offset_pixel = offsets[0] * max(rec_dims) * 10
            plt.arrow(closest_middle[0][1] - offset_pixel[1], closest_middle[0][0] - offset_pixel[0], offset_pixel[1], offset_pixel[0], length_includes_head=True, width=10, color="#ff7d00")
        plt.axis("equal")
        plt.savefig(save, bbox_inches="tight", dpi=300)
        plt.clf()

    # Find targets based on mask
    def findPoints(self):
        # Precheck if mask is empty
        if not np.any(self.target_mask):
            return

        rec_dims = np.array(self.tgt_params.weight.shape)
        # Setup initial points
        if self.tgt_params.sparse:
            self.points, self.point_scores = self.definePoints_sparse(penalty_weight=0)
        else:
            self.points, self.point_scores = self.definePoints_grid(penalty_weight=0)

        log(f"Initial points found: {len(self.points)}")

        if self.plot: self.plotTargets(save=os.path.join(self.out_dir, self.map_name + "_points_initial.png"))

        if len(self.points) == 0:
            return False

        # Global translation optimization
        if not self.tgt_params.sparse:
            start_offset = np.zeros(2)
            offset = minimize(calcScore_cluster, start_offset, args=(self.points, self.target_mask, self.penalty_mask, 0.0, self.tgt_params.weight, self.tgt_params.edge_weights), method="nelder-mead", bounds=((-0.1, 0.1), (-0.1, 0.1)))
            offset_pixel = offset.x * max(rec_dims) * 10
            self.points += offset_pixel
            
            if self.plot: self.plotTargets(offsets=[offset.x], save=os.path.join(self.out_dir, self.map_name + "_points_globalSGD.png"))
            self.cleanPoints_score(penalty_weight=0, threshold=self.tgt_params.threshold)

        # Find clusters
        self.clusters = self.findClusters(self.points)

        # Iterative optimization
        if self.tgt_params.sparse: #len(self.score_weights_edge) > 0:
            adjacent_points = False
        else:
            adjacent_points = True
            
        for it in range(self.tgt_params.max_it):
            log("____________________")
            log(f"Iteration: {it + 1}")
            
            # Translation optimization for each cluster
            cluster_offsets = []
            total_offset = 0
            for cluster in self.clusters:
                start_offset = np.zeros(2)
                offset_cluster = minimize(calcScore_cluster, start_offset, args=(self.points[cluster, :], self.target_mask, self.penalty_mask, self.tgt_params.penalty, self.tgt_params.weight, self.tgt_params.edge_weights), method="nelder-mead", bounds=((-0.1, 0.1), (-0.1, 0.1)))
                offset_cluster_pixel = offset_cluster.x * max(rec_dims) * 10
                self.points[cluster, :] += offset_cluster_pixel
                cluster_offsets.append(offset_cluster.x)
                total_offset += np.linalg.norm(offset_cluster_pixel)

            if self.plot: self.plotTargets(offsets=cluster_offsets, clusters=self.clusters, save=os.path.join(self.out_dir, self.map_name + "_points_it" + str(it + 1).zfill(2) + ".png"))
            
            # Clean points
            prev_number = len(self.points)
            self.cleanPoints_score(self.tgt_params.penalty, self.tgt_params.threshold, self.clusters)
            self.points, self.clusters = self.cleanPoints_dist(self.points, self.tgt_params.max_tilt, tolerance=0.1, point_scores=self.point_scores, clusters=self.clusters)
            new_number = len(self.points)
            
            # Find new clusters
            if new_number != prev_number:
                if new_number == 0:
                    break
                self.clusters = self.findClusters(self.points)
            
            # Check for convergence
            log(f"Total offset: {total_offset}")
            if total_offset < 1 and new_number == prev_number:
                if not adjacent_points and it < self.tgt_params.max_it - 1:
                    log("Adding additional adjacent points...")
                    self.addAdjacentPoints(self.tgt_params.penalty)
                    self.clusters = self.findClusters(self.points)
                    adjacent_points = True
                if len(self.points) == prev_number:
                    log("##### Converged! #####\n")
                    break
        
        if len(self.points) > 0:
            # Calculate final SPACE scores
            for p, point in enumerate(self.points):
                self.point_scores[p] = calcScore_point(point, self.target_mask, self.penalty_mask, self.tgt_params.penalty, self.tgt_params.weight, self.tgt_params.edge_weights)

            # Check maximum distance
            coords_max = self.points.max(axis=0)
            coords_min = self.points.min(axis=0)
            max_delta = (coords_max - coords_min) * self.model.pix_size / 1000

            # Check distance to center point
            middle_point = [np.max(self.points[:, 0]) / 2 + np.min(self.points[:, 0]) / 2, np.max(self.points[:, 1]) / 2 + np.min(self.points[:, 1]) / 2]
            #print("middle", middle_point)
            closest_middle = self.points[np.argmin(np.linalg.norm(self.points - middle_point, axis=1))]
            #print("closest", closest_middle)
            furthest_middle = self.points[np.argmax(np.linalg.norm(self.points - closest_middle, axis=1))]
            #print("furthest", furthest_middle)
            middle_delta = np.abs(furthest_middle - closest_middle) * self.model.pix_size / 1000
            #print(middle_delta, max_delta)

            # If distance above threshold cluster targets in 2 clusters
            if max_delta[0] > 1.9 * self.mic_params.IS_limit or max_delta[1] > 1.9 * self.mic_params.IS_limit or middle_delta[0] > 0.95 * self.mic_params.IS_limit or middle_delta[1] > 0.95 * self.mic_params.IS_limit:
                log("Distance between points above threshold. Splitting targets in 2 acquisition areas...")
                centroids, label = kmeans2(self.points, 2, minit="++")
                counts = np.bincount(label)

                # Run up to 10 times in case one cluster remains empty
                for i in range(10):
                    if len(counts) > 1: break
                    centroids, label = kmeans2(self.points, 2, minit="++")
                    counts = np.bincount(label)    

                log(f"Split: {counts}")  
                self.clusters = [np.where(label == c)[0] for c in range(len(centroids))]

            else:
                self.clusters = [[i for i in range(len(self.points))]]

            log("Choosing tracking target...")

            # Consider tilted FOV to avoid ice coming in and compromising tracking
            expanded_dims = (self.mic_params.view_rotM @ ((self.mic_params.view_rotM.T @ np.array(self.tgt_params.weight.shape)) / np.array([1, np.cos(np.radians(self.tgt_params.max_tilt))]))).astype(int)

            self.targets = []       # list of array of targets for export as file
            # Choose tracking target for each cluster (in most cases it will be one)
            for cluster in self.clusters:
                points = self.points[cluster, :]

                # Find middle point, choose tracking target closest to it and move it to start of list of points
                middle_point = [np.max(points[:, 0]) / 2 + np.min(points[:, 0]) / 2, np.max(points[:, 1]) / 2 + np.min(points[:, 1]) / 2]

                if self.tgt_params.extra_track: # or len(self.points) == 1:
                    for it in range(3):
                        # Use middle point as initial tracking point and refine based on expanded dims and penalty mask only 
                        start_offset = (np.random.random(2) - 0.5) / 10   # add a random offset in case middle point coincides with only point
                        offset = minimize(calcScore_cluster, start_offset, args=([middle_point], np.zeros(self.target_mask.shape), self.penalty_mask, 1, np.ones(expanded_dims)), method="nelder-mead", bounds=((-0.1, 0.1), (-0.1, 0.1)))
                        offset_pixel = offset.x * max(rec_dims) * 10
                        middle_point += offset_pixel
                        # In case of overlap remove tracking point from other point
                        middle_point, moved = self.distancePoint(middle_point, points)
                        # No need for additional iterations if point was not moved after minimize
                        if not moved:
                            break
                    points = np.vstack([middle_point, points])

                    # Get the respective point scores
                    point_scores = np.array(self.point_scores)[cluster]
                    point_scores = np.insert(point_scores, 0, 0)
                    self.targets.append({"points": points, "scores": point_scores, "geo_points": []})
                    log("Successfully added additional tracking target.")
                else:                
                    closest_middle_ids = np.argsort(np.linalg.norm(points - middle_point, axis=1))
                    # Check 3 closest points to middle for ice
                    ice_scores = []
                    for min_id in range(0, min(3, len(closest_middle_ids))):
                        # Calculate score using expanded dims and only penalty mask
                        ice_scores.append(calcScore_point(points[closest_middle_ids[min_id]], np.zeros(self.target_mask.shape), self.penalty_mask, 1, np.ones(expanded_dims)))
                    log(f"Candidate ice scores: {ice_scores}")
                    # Use point with least ice for tracking
                    min_ice = np.argmax(ice_scores)

                    closest_middle = points[closest_middle_ids[min_ice]]
                    points = np.vstack([closest_middle, np.delete(points, closest_middle_ids[min_ice], 0)])

                    # Get the respective point scores
                    point_scores = np.array(self.point_scores)[cluster]
                    point_scores = np.concatenate([[point_scores[closest_middle_ids[min_ice]]], np.delete(point_scores, closest_middle_ids[min_ice], 0)])

                    self.targets.append({"points": points, "scores": point_scores, "geo_points": []})
                    log("Successfully chose tracking target.")


    # Find points to measure geometry
    def findGeoPoints(self):
        # Precheck is geo mask is empty
        if not np.any(self.geo_mask):
            return

        log("Finding geo points...")
        # Start from grid of points with high threshold
        geo_points, _ = self.definePoints_grid(penalty_weight=0.5, threshold=0.8, alternative_mask=self.geo_mask)

        if len(geo_points) > 0:
            # Remove points overlapping with target points at zero tilt
            all_points, _ = self.cleanPoints_dist(np.concatenate([geo_points, self.points]), max_tilt=0, tolerance=0.0)
            geo_points = all_points[:-len(self.points), :]

            for t, target_area in enumerate(self.targets):
                # Remove points beyond IS limits from tracking point
                area_geo_points = np.array([point for point in geo_points if np.linalg.norm(point - target_area["points"][0]) * self.model.pix_size < self.mic_params.IS_limit * 1000])
                if len(area_geo_points) < len(geo_points):
                    log(f"NOTE: Removed {len(geo_points) - len(area_geo_points)} geo points due to image shift limit.")

                # Only use 1 point per cluster if >=5 clusters
                if len(area_geo_points) > 5:
                    geo_clusters = self.findClusters(area_geo_points)
                    if len(geo_clusters) >= 5:
                        area_geo_points = area_geo_points[[cluster[0] for cluster in geo_clusters]]

                if len(area_geo_points) > 10:
                    # Choose 10 random geo points (TODO: update to use extreme points)
                    area_geo_points = np.random.default_rng().choice(area_geo_points, 10, replace=False)

                log(f"{len(area_geo_points)} geo points were selected.")

                if len(area_geo_points) < 3:
                    log("WARNING: Not enough geo points found to measure the sample geometry. The user defined pretilt and rotation values will be used.")
                #self.geo_points = geo_points
                self.targets[t]["geo_points"] = area_geo_points
        else:
            log("WARNING: Not enough geo points found to measure the sample geometry. The user defined pretilt and rotation values will be used.")

    # Export points
    def exportPoints(self):
        if len(self.targets) > 0:
            for t, target_area in enumerate(self.targets):
                #target_area["geo_points"] = self.geo_points
                #lamella = {"points": self.points, "point_scores": self.point_scores, "geo_points": self.geo_points}
                with open(os.path.join(self.map_dir, self.map_name + "_points" + str(t) + ".json"), "w+") as f:
                    json.dump(target_area, f, indent=4, default=utils.convertArray)
        else:
            # Write empty points file to ensure empty targets file is written and map is considered processed
            with open(os.path.join(self.map_dir, self.map_name + "_points.json"), "w+") as f:
                json.dump({"points": [], "scores": [], "geo_points": []}, f)            


# Function to instantiate lamella and run target selection from any script
def runTargetSelection(map_dir, map_name, tgt_params, mic_params, MM_model, save_plot=False, save_final_plot=False, alt_seg_path=None):
    map_name = os.path.basename(map_name)
    log(f"Setting up targets for {map_name}...")

    # Instantiate lamella and find points
    lamella = Lamella(map_name, map_dir, MM_model, mic_params, tgt_params, save_plot, alt_seg_path)
    lamella.findPoints()

    if len(lamella.points) == 0:
        log("WARNING: No targets found in automatic target selection!")
        # Write empty points file
        lamella.exportPoints()
        # Early return
        return
    
    log(f"Final targets: {len(lamella.points)}")
    if save_final_plot:
        log("Saving overview image...")
        lamella.plotTargets(overlay=lamella.target_mask, clusters=lamella.clusters, offsets=np.zeros([len(lamella.clusters), 2]), save=os.path.join(lamella.out_dir, map_name + "_" + tgt_params.target_list[0] +"_targets.png"))
        log(f"Saved at {os.path.join(lamella.out_dir, map_name + '_targets_' + tgt_params.target_list[0] + '.png')}")

    # Find geo points for sample geometry measurement
    lamella.findGeoPoints()

    # Export target points
    lamella.exportPoints()


##### UTILITY FUNCTIONS #####   

class MicParams_ext:
    def __init__(self, map_dir):
        with open(os.path.join(map_dir, "mic_params.json"), "r") as f:
            params = json.load(f, object_hook=utils.revertArray)
        for key, value in params.items():
            vars(self)[key] = value

    def export(self, map_dir):
        with open(os.path.join(map_dir, "mic_params.json"), "w+") as f:
             json.dump(vars(self), f, indent=4, default=utils.convertArray)
        log(f"NOTE: Saved microscope parameters at {os.path.join(map_dir, 'mic_params.json')}")

class TgtParams:
    def __init__(self, target_list=[], penalty_list=[], MM_model=None, sparse_targets=False, edge_targets=False, penalty_weight=0, score_threshold=0, max_tilt=0, mic_params=[], extra_tracking=False, max_iterations=10, file_dir=None):
        # Check if file can be imported
        if file_dir is None:
            self.target_list = target_list
            self.penalty_list = penalty_list

            # Replace keyword classes
            self.parseLists(MM_model)

            # Check if all lists classes are known to the model
            if not self.checkLists(MM_model):
                sys.exit()

            self.sparse = sparse_targets
            self.edge = edge_targets

            self.weight, self.edge_weights = makeScoreWeights(MM_model, self.edge)

            self.penalty = penalty_weight
            self.threshold = score_threshold

            self.max_tilt = max_tilt

            self.vecs = findGridVecs(MM_model, self.max_tilt, mic_params)

            self.extra_track = extra_tracking
            self.max_it = max_iterations
        else:
            self.loadFromFile(file_dir, MM_model)

    def parseLists(self, MM_model):
        # Replace lamella keyword
        if self.target_list[0] == "lamella":
            self.target_list = [name for name in MM_model.categories if name not in self.penalty_list]

        # Make sure penalty list includes black and white
        for guaranteed_penalty in ["black", "white", "dynabeads"]:
            if guaranteed_penalty in MM_model.categories and guaranteed_penalty not in self.penalty_list:
                log(f"WARNING: Added {guaranteed_penalty} class to avoid list!")
                self.penalty_list.append(guaranteed_penalty)

    def checkLists(self, MM_model):
        for cat in self.target_list + self.penalty_list:
            if cat not in MM_model.categories.keys() and cat not in ["lamella"]:
                log(f"ERROR: {cat} not found in model!")
                return False
        return True

    def loadFromFile(self, file_dir, MM_model):
        with open(os.path.join(file_dir, "tgt_params.json"), "r") as f:
            params = json.load(f, object_hook=utils.revertArray)
        for key, value in params.items():
            vars(self)[key] = value

        if self.weight is None:
            if MM_model is not None:
                self.weight, self.edge_weights = makeScoreWeights(MM_model, self.edge)
            else:
                log("ERROR: Need loaded MM model to regenerate weights!")
                return

    def export(self, map_dir):
        # Remove weights to make file smaller
        weight_matrix = self.weight
        weight_matrix_edge = self.edge_weights
        self.weight = None
        self.edge_weights = None

        # Write file
        with open(os.path.join(map_dir, "tgt_params.json"), "w+") as f:
             json.dump(vars(self), f, indent=4, default=utils.convertArray)

        # Restore weights
        self.weight = weight_matrix
        self.edge_weights = weight_matrix_edge

        log(f"NOTE: Saved target selection parameters at {os.path.join(map_dir, 'tgt_params.json')}")


# Find vectors for hexagonal grid
def findGridVecs(model, max_tilt, mic_params):
    rotM_pattern = np.array([[np.cos(np.radians(60)), -np.sin(np.radians(60))], [np.sin(np.radians(60)), np.cos(np.radians(60))]])
    x1 = np.array([model.beam_diameter, 0])
    x2 = rotM_pattern @ x1
    x2[1] /= np.cos(np.radians(max_tilt))

    x1 = mic_params.view_rotM @ x1
    x2 = mic_params.view_rotM @ x2
    if abs(mic_params.view_ta_rotation) > 45:
        xt = x1[:]
        x1 = x2[:]
        x2 = xt

    return np.vstack([x1, x2])

# Create score weights
def makeScoreWeights(model, edge=False, edge_angle_sampling=45):
    # Setup weight mask
    weight_mask = np.zeros(model.cam_dims)
    x, y = np.meshgrid(np.linspace(-model.cam_dims[1] // 2, model.cam_dims[1] // 2, model.cam_dims[1]), np.linspace(-model.cam_dims[0] // 2, model.cam_dims[0] // 2, model.cam_dims[0]))
    d = np.sqrt(x * x + y * y)
    sigma, mu = max(model.cam_dims) // 2, 0
    weight_mask = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    weight_mask /= np.sum(weight_mask) / np.prod(model.cam_dims)

    # TODO add larger weight mask for penalty to account for ice chunks coming into field of view (needs max_tilt and boolean user setting?)
    # weight_mask_stretched = resize(weight_mask, [model.cam_dims[0] * np.cos(np.radians(max_tilt)), model.cam_dims[1]]) 

    # Setup edge masks
    edge_weight_masks = []
    if edge:
        x, y = np.meshgrid(np.linspace(-max(model.cam_dims), max(model.cam_dims), 2 * max(model.cam_dims)), np.linspace(-max(model.cam_dims), max(model.cam_dims), 2 * max(model.cam_dims)))
        d = np.sqrt(x * x + y * y)
        sigma, mu = max(model.cam_dims) // 2, 0
        square_weight = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        square_weight /= np.sum(weight_mask) / np.prod(model.cam_dims) / 2
        
        edge_weight = np.array(square_weight)
        edge_weight[0:square_weight.shape[0] // 2, :] =  -square_weight[0:square_weight.shape[0] // 2, :]
        
        dif_shape = np.array(square_weight.shape) - np.array(weight_mask.shape)
        
        for i in range(0, 360, edge_angle_sampling):
            edge_weight_masks.append(rotate(edge_weight, angle=i, reshape=False)[dif_shape[0] // 2: dif_shape[0] // 2 + weight_mask.shape[0], dif_shape[1] // 2: dif_shape[1] // 2 + weight_mask.shape[1]])

    return weight_mask, edge_weight_masks

# Calculate score of a point depending on camera dimensions
def calcScore_point(point, mask, penalty_mask, penalty_weight, weight_mask, edge_weight_masks=[]):
    rec_dims = np.array(weight_mask.shape)
    if not rec_dims[0] / 2 <= point[0] < mask.shape[0] - rec_dims[0] / 2 or not rec_dims[1] / 2 <= point[1] < mask.shape[1] - rec_dims[1] / 2:
        return 0
    x_min = int(round(point[0] - rec_dims[0] / 2))
    y_min = int(round(point[1] - rec_dims[1] / 2))
    if len(edge_weight_masks) > 0:
        edge_scores = []
        for edge_mask in edge_weight_masks:
            score = np.sum(mask[x_min: x_min + edge_mask.shape[0], y_min: y_min + edge_mask.shape[1]] * edge_mask) / np.prod(rec_dims)
            score -= np.sum(penalty_mask[x_min: x_min + weight_mask.shape[0], y_min: y_min + weight_mask.shape[1]] * weight_mask) / np.prod(rec_dims) * penalty_weight
            edge_scores.append(score)
        score = np.max(edge_scores) 
    else:
        score = np.sum(mask[x_min: x_min + weight_mask.shape[0], y_min: y_min + weight_mask.shape[1]] * weight_mask) / np.prod(rec_dims)
        score -= np.sum(penalty_mask[x_min: x_min + weight_mask.shape[0], y_min: y_min + weight_mask.shape[1]] * weight_mask) / np.prod(rec_dims) * penalty_weight
    return score

# Calculate the total score of a cluster of points
def calcScore_cluster(offset, points, mask, penalty_mask, penalty_weight, weight_mask, edge_weight_masks=[]):
    rec_dims = np.array(weight_mask.shape)
    pixel_offset = offset * max(rec_dims) * 10
    total_score = 0
    for pt in points:
        point = pt + pixel_offset
        score = calcScore_point(point, mask, penalty_mask, penalty_weight, weight_mask, edge_weight_masks)
        total_score -= score
    return total_score


##### MONITOR FUNCTIONS #####

# Make list of map files in inventory
def monitorFiles(map_dir):
    file_list = sorted(glob.glob(os.path.join(map_dir, "*.png")))
    wg_list = []
    mm_list = []
    seg_list = []
    for file in file_list:
        if os.path.splitext(file)[0].split("_")[-1] == "wg":                # maps for lamella detection
            wg_list.append(os.path.splitext(os.path.basename(file))[0])
        elif os.path.splitext(file)[0].split("_")[-1] == "seg":             # segmentations
            seg_list.append(os.path.basename(file).rsplit("_seg", 1)[0])
        elif os.path.splitext(file)[0].split("_")[-1] == "segtemp":         # segmentation temp file
            continue
        else:
            mm_list.append(os.path.splitext(os.path.basename(file))[0])                       # all other maps
    # Check for unprocessed segmentation maps
    unprocessed_mm_list = []
    unprocessed_seg_list = []
    for map_name in mm_list:
        if map_name not in seg_list:
            unprocessed_mm_list.append(map_name)
        else:
            point_files = sorted(glob.glob(os.path.join(map_dir, map_name + "_points*.json")))
            if len(point_files) == 0:
                unprocessed_seg_list.append(map_name)

    # Check for unprocessed lamella detection maps
    unprocessed_wg_list = [wg_map for wg_map in wg_list if not os.path.exists(os.path.join(map_dir, wg_map + "_boxes.json"))]

    return unprocessed_mm_list, unprocessed_seg_list, unprocessed_wg_list 

def checkErr(map_dir, map_name):
    error_file = os.path.join(map_dir, map_name + "_SPACE.err")
    if os.path.exists(error_file):
        with open(error_file, "r", encoding="utf-8") as f:
            content = f.read()
            if "not enough memory" in content or "truncated" in content:
                log(f"ERROR: Out of memory or truncated segmentation error! Trying again...")
                return True
            elif "No such file or directory" in content:
                log(f"ERROR: File not found! Please ensure the config contains the proper segmentation script!")
                return True
            else:
                return False
    else:
        return False

# Control queue for external inference
def updateQueue(map_dir, WG_model=None, MM_model=None, mic_params=None, tgt_params=None, gpu_list=[0], save_plot=False):
    # Check run file
    if os.path.exists(os.path.join(map_dir, "SPACE_runs.json")):
        with open(os.path.join(map_dir, "SPACE_runs.json"), "r") as f:
            runs = json.load(f)
    else:
        runs = {"runs": [], "queue": []}

    # Get unprocessed maps
    mm_list, seg_list, wg_list  = monitorFiles(map_dir)

    # Remove finished runs
    for map_name in seg_list:
        run_list = [run["map"] for run in runs["runs"]]
        if map_name in run_list:
            runs["runs"].pop(run_list.index(map_name))

    # Determine free GPU
    free_gpus = []
    for gpu in gpu_list:
        if gpu not in [run["gpu"] for run in runs["runs"]]:
            free_gpus.append(gpu)

    # Run lamella detection
    if WG_model is not None:
        for wg_map in wg_list:
            # Check if GPU is available
            if len(free_gpus) > 0:
                device = "cuda:" + str(free_gpus[0])
            else:
                device = "cpu"
            # Check if file size is still changing
            file_size = os.path.getsize(os.path.join(map_dir, wg_map + ".png"))
            time.sleep(1)
            if file_size != os.path.getsize(os.path.join(map_dir, wg_map + ".png")):
                continue
            # Find lamellae
            log(f"Detecting lamellae on {os.path.basename(wg_map)}.png...")
            bboxes = WG_model.findLamellae(map_dir, os.path.basename(wg_map), suffix=".png", device=device)
            log(f"Detected {len(bboxes)} lamellae.\n")

    # Run target selection
    if tgt_params is not None:
        for seg_map in seg_list:
            runTargetSelection(map_dir, seg_map, tgt_params, mic_params, MM_model, save_plot)

    # Add unprocessed maps to queue if not already running
    add_to_queue = []   # add to queue after submitting new runs to give file time to finish copying
    for map_name in mm_list:
        if map_name in runs["queue"]:
            continue
        if map_name in [run["map"] for run in runs["runs"]]:
            if not checkErr(map_dir, map_name):
                continue
            else:
                # In case of detected error, remove map from run list and add to queue
                runs["runs"].pop(utils.findIndex(runs["runs"], "map", map_name))
                add_to_queue.append(map_name)
        else:
            add_to_queue.append(map_name)

    # Submit new run
    while len(free_gpus) > 0 and len(runs["queue"]) > 0:
        map_name = runs["queue"].pop(0)
        run_gpu = free_gpus.pop(0)

        out_file = open(os.path.join(map_dir, map_name + "_SPACE.err"), "w+")
        process = subprocess.Popen([sys.executable, Path(__file__).parent.parent / config.MM_model_script, os.path.join(map_dir, map_name + ".png")], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(run_gpu)), stdout=out_file, stderr=subprocess.STDOUT, text=True)
        log(f"\nStarting inference for {os.path.basename(map_name)}.png on GPU {run_gpu}...")
        runs["runs"].append({"map": map_name, "gpu": run_gpu, "PID": process.pid})

    # Add to queue after submitting new runs to give file time to finish copying
    for map_name in add_to_queue:
        runs["queue"].append(map_name)

    # Output
    if len(runs["runs"]) > 0:
        log("\n# Processing:")
        for run in runs["runs"]:
            log(f"{os.path.basename(run['map'])} (gpu: {run['gpu']})")

    if len(runs["queue"]) > 0:
        log("# Queue:")
        for map_name in runs["queue"]:
            log(os.path.basename(map_name))

    # Write run file
    with open(os.path.join(map_dir, "SPACE_runs.json"), "w+") as f:
        json.dump(runs, f, indent=4)