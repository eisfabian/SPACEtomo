#!/usr/bin/env python
# ===================================================================
# ScriptName:   mod_wg
# Purpose:      Classes for lamella detection.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/08/14
# Revision:     v1.2
# Last Change:  2024/08/21: minor fixes after external test run
#               2024/08/16: added sorting function, added meta data
# ===================================================================

import time
import json
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images

from SPACEtomo.modules import utils
from SPACEtomo.modules.utils import log
import SPACEtomo.config as config

class WGModel:
    def __init__(self, external=False):
        if external:
            self.model = None
        else:
            from ultralytics import YOLO, settings
            model_file = Path(config.WG_model_file)
            if not model_file.exists():
                raise FileNotFoundError("Lamella detection model file does not exist! Please import a model using the SPACEmodel command!")
            self.model = YOLO(Path(config.WG_model_file))
            settings.update({'sync': False})    # no tracking by google analytics

        # Get params from config file
        self.pix_size = config.WG_model_pix_size
        self.sidelen = config.WG_model_sidelen
        self.categories = config.WG_model_categories
        self.cat_colors = np.array(config.WG_model_gui_colors)
        for c in range(len(self.cat_colors)):
            if not isinstance(self.cat_colors[c], str) and any([val > 1 for val in self.cat_colors[c]]): 
                self.cat_colors[c] = self.cat_colors[c] / 255        # convert colors to float if given as RGB
        self.cat_nav_colors = config.WG_model_nav_colors

    def findLamellae(self, map_dir, map_name, suffix="_wg.png", overlap_microns=10, threshold=0, save_boxes=True, device="cpu"):
        """Runs YOLO lamella detection on image file and compiles boxes."""

        # Load image from file
        map_dir = Path(map_dir)
        image = np.array(Image.open(map_dir / (map_name + suffix)))

        # Calculate overlap of tiles in pixels
        overlap = overlap_microns * 1000 // self.pix_size

        # Split up image if larger than model input
        if image.shape[0] > self.sidelen or image.shape[1] > self.sidelen:
            num_cols = -int(image.shape[0] // -(self.sidelen - overlap))      # Using -(a // -b) instead of a // b for ceil int division
            num_rows = -int(image.shape[1] // -(self.sidelen - overlap))
        else:       # avoid additional columns consisting just of overlap
            num_cols = num_rows = 1

        # Pad if smaller than model input
        if image.shape[0] < self.sidelen or image.shape[1] < self.sidelen:
            img_pad = np.zeros((self.sidelen, self.sidelen))
            img_pad[(self.sidelen - image.shape[0]) // 2: (self.sidelen + image.shape[0]) // 2, (self.sidelen - image.shape[1]) // 2: (self.sidelen + image.shape[1]) // 2] = image
            padding = np.array([self.sidelen - image.shape[0], self.sidelen - image.shape[1]])
        else:
            img_pad = image
            padding = np.zeros(2)

        bboxes = []
        for c in range(num_cols):
            for r in range(num_rows):
                x = int(c * (self.sidelen - overlap))
                y = int(r * (self.sidelen - overlap))

                crop = img_pad[x: x + self.sidelen, y: y + self.sidelen]
                crop = np.dstack([crop, crop, crop])

                results = self.model.predict(crop, device=device)                                         # YOLO inference
                
                if len(results[0].boxes) > 0:
                    bbox = np.array(results[0].boxes.xyxy.to("cpu"))

                    # Subtract tile coords and/or padding from coords and make sure coords go not out of image bounds
                    bbox[:, 0] = np.clip(bbox[:, 0] + y - padding[1] // 2, 0, image.shape[1])
                    bbox[:, 2] = np.clip(bbox[:, 2] + y - padding[1] // 2, 0, image.shape[1])
                    bbox[:, 1] = np.clip(bbox[:, 1] + x - padding[0] // 2, 0, image.shape[0])
                    bbox[:, 3] = np.clip(bbox[:, 3] + x - padding[0] // 2, 0, image.shape[0])
                    
                    cat = np.reshape(results[0].boxes.cls.to("cpu"), (bbox.shape[0], 1))
                    conf = np.reshape(results[0].boxes.conf.to("cpu"), (bbox.shape[0], 1))

                    bbox = np.hstack([bbox, cat, conf])

                    if len(bboxes) > 0:
                        bboxes = np.concatenate([bboxes, bbox])
                    else:
                        bboxes = bbox
                    
        bboxes = Boxes(bboxes, label_prefix="PL", pix_size=config.WG_model_pix_size, img_size=image.shape)
        log(f"Lamellae found (initial): \t{len(bboxes)}")

        if len(bboxes) > 0:
            # Merge bboxes found on different crop windows
            bboxes.mergeOverlappingBoxes()
            #bboxes = self.mergeOverlappingBoxes(bboxes, overlap)
            log(f"Lamellae found (merged): \t{len(bboxes)}")

        # Clean lamellae based on user defined confidence threshold
        #bboxes = np.array([bbox for bbox in bboxes if bbox[5] >= threshold])
        bboxes.removeLowConfidence(threshold)
        log(f"Lamellae found (final): \t{len(bboxes)}")

        # Save bboxes to output file
        if save_boxes:
            box_file = map_dir / (Path(map_name + suffix).stem + "_boxes.json")
            bboxes.saveFile(box_file)

        return bboxes

class Boxes:
    """Collection of bounding boxes"""

    def __init__(self, bboxes, label_prefix="", exclude_cats=[], pix_size=None, img_size=None) -> None:
        """Initializes from box list or from box file."""

        self.pix_size = float(pix_size) if pix_size is not None else None       # nm
        self.img_size = np.array(img_size) if img_size is not None else None

        if isinstance(bboxes, Path):
            bboxes = self.getFromFile(bboxes)

        self.boxes = []
        self.excluded_boxes = []
        for b, box in enumerate(bboxes):
            # Ignore boxes of exclude_cats
            if config.WG_model_categories[int(box[4])] in exclude_cats:
                self.excluded_boxes.append(BBox(box[:4], *box[5:], label=label_prefix + str(b + 1)))
            else:
                self.boxes.append(BBox(box[:4], *box[4:], label=label_prefix + str(b + 1)))

        if len(self.boxes) - len(bboxes) > 0:
            log(f"NOTE: {len(self.boxes) - len(bboxes)} lamellae were excluded because they were {', '.join(exclude_cats)}!")
    
    def mergeOverlappingBoxes(self, margin=0.1):
        """Merges overlapping bboxes."""

        # Sort boxes by x values
        self.sortBy("x")

        merged_boxes = []
        for box in self.boxes:
            if merged_boxes:
                # Get the previous merged box
                prev_box = merged_boxes[-1]

                #size = [box[2] - box[0], box[3] - box[1]]

                # Check if the current box overlaps with the previous box within margin
                if box.x_min <= prev_box.x_max + margin * box.size[0] and box.y_min <= prev_box.y_max + margin * box.size[1] and box.y_max >= prev_box.y_min - margin * box.size[1]:
                    # Update the previous box to encompass both boxes
                    prev_box.x_max = max(box.x_max, prev_box.x_max)
                    prev_box.y_max = max(box.y_max, prev_box.y_max)
                    prev_box.y_min = min(box.y_min, prev_box.y_min)
                    # Change class if probability is higher for new box OR if old box was "undesirable" class
                    if prev_box.prob < box.prob or config.WG_model_categories[prev_box.cat] == "broken" or config.WG_model_categories[prev_box.cat] == "gone":
                        prev_box.cat = box.cat
                        prev_box.prob = box.prob
                else:
                    merged_boxes.append(box)
            else:
                merged_boxes.append(box)
        
        self.boxes = merged_boxes

    def removeLowConfidence(self, threshold):
        """Removes boxes below confidence threshold."""

        new_boxes = []
        for box in self.boxes:
            if box.prob > threshold:
                new_boxes.append(box)
        self.boxes = new_boxes

    def saveFile(self, file):
        """Saves json file with bounding boxes."""

        bboxes = []
        for box in self.boxes:
            bboxes.append(box.xyxycc.tolist())
        
        meta_data = {"pix_size": self.pix_size, "img_size": self.img_size, "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

        data = {"boxes": bboxes, "meta_data": meta_data}
        with open(file, "w") as f:
            json.dump(data, f, indent=4, default=utils.convertArray)

    def getFromFile(self, file):
        """Reads json box file."""

        with open(file, "r") as f:
            data = json.load(f, object_hook=utils.revertArray)
            if isinstance(data, dict):
                bboxes = data["boxes"]
                meta_data = data["meta_data"]
            else:
                bboxes = data           # keep compatibility with boxes.json before meta_data

        if "pix_size" in meta_data.keys():
            self.pix_size = meta_data["pix_size"]
        if "img_size" in meta_data.keys():
            self.img_size = meta_data["img_size"]

        return bboxes
    
    def sortBy(self, key, reverse=False):
        """Sorts boxes by coords, distance or attribute."""

        if len(self) <= 1: return

        if key == "x":
            if not reverse:
                self.boxes = sorted(self.boxes, key=lambda box: box.x_min)
            else:
                self.boxes = sorted(self.boxes, key=lambda box: box.x_max, reverse=True)
        elif key == "y":
            if not reverse:
                self.boxes = sorted(self.boxes, key=lambda box: box.y_min)
            else:
                self.boxes = sorted(self.boxes, key=lambda box: box.y_max, reverse=True)
        elif key == "center_dist":
            if self.img_size is not None:
                self.boxes = sorted(self.boxes, key=lambda box: np.linalg.norm(box.center - self.img_size // 2), reverse=reverse)
            else:
                log("ERROR: Can't sort by distance from center, because image size is not known!")
        elif hasattr(self.boxes[0], key):
            self.boxes = sorted(self.boxes, key=lambda box: getattr(box, key), reverse=reverse)
        else:
            raise NotImplementedError("Sorting key not available!")

    def __bool__(self):
        return bool(self.boxes)

    def __len__(self):
        return len(self.boxes)
    

class BBox:
    """Class for bounding box."""

    def __init__(self, bbox, cat, prob=1, label="") -> None:
        self.x_min = bbox[0]
        self.x_max = bbox[2]
        self.y_min = bbox[1]
        self.y_max = bbox[3]
        self.cat = int(cat)
        self.prob = prob

        self.label = label

    @property
    def center(self):
        return np.array([(self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2])
    
    @property
    def size(self):
        return np.array([self.x_max - self.x_min, self.y_max - self.y_min])
    
    @property
    def yolo(self):
        """Box in format CXYWH (category, center x, center y, width, height)"""
        return np.array([self.cat, *self.center, *self.size])
    
    @property
    def xyxycc(self):
        """Box in format XYXYCC (min x, min y, max x, max y, category, confidence)"""
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max, self.cat, self.prob])
    
    def scale(self, factor):
        size = self.size * factor
        self.x_min, self.y_min = self.center - size / 2
        self.x_max, self.y_max = self.center + size / 2
       
    def __mul__(self, mult):
        bbox = self.xyxycc[:4] * mult
        return BBox(bbox, self.cat, self.prob, self.label)
    
    __rmul__ = __mul__
        