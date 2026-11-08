#!/usr/bin/env python
# ===================================================================
# ScriptName:   specimen
# Purpose:      Synthetic lamella grid used by the dummy microscope as ground truth for images at all mags.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein / Claude
# Created:      2026/11/07
# Revision:     v1.4
# Last Change:  2026/11/08: faster shot noise (Gaussian approximation of Poisson)
#               2026/11/07: created
# ===================================================================
#
# Coordinates: the world is the nominal stage frame of the whole grid map (µm). An image of pixel size p (µm)
# centered on world position c shows column -> +x and row -> -y:
#     x = c_x + (col - W / 2) * p        y = c_y - (row - H / 2) * p
# so the stage-to-pixel matrix of such an image is diag(1, -1) / p (used as MapScaleMat for dummy maps).
#
# Each imaging state shows the world with an offset (like uncorrected image shift offsets between mags):
# an image taken at stage s in state k is centered on world position s + offsets[k].

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy import ndimage

# Segmentation classes and relative intensities of the lamella segmentation model
CATS = {"background": 0, "white": 1, "black": 2, "crack": 3, "coating": 4, "cell": 5, "cellwall": 6, "nucleus": 7,
        "vacuole": 8, "mitos": 9, "lipiddroplets": 10, "vesicles": 11, "multivesicles": 12, "membranes": 13,
        "dynabeads": 14, "ice": 15, "cryst": 16}
INTENSITY = {"background": 0.5, "white": 0.95, "black": 0.1, "crack": 0.9, "coating": 0.25, "cell": 0.4,
             "cellwall": 0.35, "nucleus": 0.37, "vacuole": 0.32, "mitos": 0.3, "ice": 0.12, "bar": 0.03, "square": 0.25}

# Lamella classes of the lamella detection model
LAMELLA_CLASSES = ("good", "thick", "wedge", "broken", "contaminated", "gone")


def stageToPixelMatrix(pix_size_nm):
    """Stage (µm) to pixel matrix of a dummy image (see module header)."""
    return np.array([[1.0, 0.0], [0.0, -1.0]]) * 1000 / pix_size_nm


class Lamella:
    """Rectangular lamella slab between two milled trenches; long axis along y if vertical."""

    def __init__(self, center, size, vertical, category, seed):
        self.center = np.array(center, dtype=float)     # world µm
        self.size = np.array(size, dtype=float)         # width, length µm
        self.vertical = bool(vertical)
        self.category = str(category)
        self.seed = int(seed)
        self.features = self._makeFeatures()

    def _toWorld(self, u, v):
        """Lamella-local (across u, along v) µm to world µm."""
        return (self.center + (np.array([u, v]) if self.vertical else np.array([v, u])))

    def rect(self, across, along, offset_along=0.0):
        """World rectangle (x_min, x_max, y_min, y_max) of a region centered on the lamella axis."""
        corners = np.array([self._toWorld(su * across / 2, offset_along + sv * along / 2) for su in (-1, 1) for sv in (-1, 1)])
        return corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max()

    @property
    def slab(self):
        return self.rect(self.size[0], self.size[1])

    @property
    def trenches(self):
        return [self.rect(self.size[0], self.size[1] / 2, sign * 0.75 * self.size[1]) for sign in (-1, 1)]

    def _makeFeatures(self):
        """Deterministic interior objects as (class, u, v, width, height) in lamella-local µm (relative to center)."""
        rng = np.random.default_rng(self.seed)
        half = self.size / 2
        features = []

        # Cells with organelles (cell walls are drawn as outline)
        cell_size = np.array([5.0, 7.0])
        cells = []
        for coords in rng.uniform(-half + cell_size / 2, half - cell_size / 2, size=(30, 2)):
            if all(np.linalg.norm(coords - cell) > cell_size.min() for cell in cells):
                cells.append(coords)
        for cell in cells:
            features.append(("cell", *cell, *cell_size))
            for _ in range(3):
                features.append(("mitos", *(cell + rng.uniform(-0.3, 0.3, 2) * cell_size), 1.0, 2.0))
            features.append(("nucleus", *(cell + rng.uniform(-0.2, 0.2, 2) * cell_size), *(np.array([3.0, 3.0]) * rng.uniform(0.3, 1))))
            features.append(("vacuole", *(cell + rng.uniform(-0.2, 0.2, 2) * cell_size), *(np.array([3.0, 3.0]) * rng.uniform(0.5, 1))))

        # Ice blobs on the lamella surface
        for coords in rng.uniform(-half + 1.5, half - 1.5, size=(10, 2)):
            size = rng.uniform(0, 3)
            features.append(("ice", *coords, size, size))

        # Class specific defects
        if self.category == "contaminated":
            for coords in rng.uniform(-half + 2, half - 2, size=(6, 2)):
                size = rng.uniform(2, 4)
                features.append(("contamination", *coords, size, size))
        return features

    def toDict(self):
        return {"center": self.center.tolist(), "size": self.size.tolist(), "vertical": self.vertical, "category": self.category}


class Specimen:
    """Seeded synthetic grid: grid bars, squares with varying ice and lamellae in random squares."""

    def __init__(self, seed, center=(0, 0), region=1000, pitch=100, n_lamellae=None, categories=("good",), vertical=True,
                 lamella_size=(15, 25), offsets=None):
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.center = np.array(center, dtype=float)
        self.region = region
        self.pitch = pitch
        self.bar_width = pitch / 3
        self.origin = self.center - region / 2

        # Thickness variation: smooth separable field over the whole grid
        self.wave_length = rng.uniform(150, 400, 2)
        self.wave_phase = rng.uniform(0, 2 * np.pi, 2)

        # Lamellae in distinct grid squares away from the region edge
        n_squares = int(region // pitch)
        squares = [(i, j) for i in range(1, n_squares - 1) for j in range(1, n_squares - 1)]
        if n_lamellae is None:
            n_lamellae = int(rng.integers(1, 5))
        self.lamellae = []
        for idx in rng.permutation(len(squares))[:n_lamellae]:
            square_center = self.origin + (np.array(squares[idx]) + 0.5) * pitch + self.bar_width / 2
            size = np.array(lamella_size) * rng.uniform(0.9, 1.1, 2)
            self.lamellae.append(Lamella(square_center, size, vertical, rng.choice(categories), rng.integers(2**31)))

        # Image offsets per imaging state (like uncorrected image shift offsets between mags)
        if offsets is None:
            offsets = {"WG": [0.0, 0.0], "IM": rng.uniform(-4, 4, 2).tolist(), "MM": rng.uniform(-1.5, 1.5, 2).tolist()}
        self.offsets = {key: np.array(val, dtype=float) for key, val in offsets.items()}

    def _squareFactor(self, i, j):
        """Ice thickness factor of grid square (i, j); about 5% of squares are broken (empty)."""
        rng = np.random.default_rng([self.seed, int(i) + 10000, int(j) + 10000])
        return 0.0 if rng.random() < 0.05 else rng.uniform(0.85, 1.15)

    def _axes(self, center, shape, pix_size):
        """World coordinates of pixel columns (x) and rows (y)."""
        p = pix_size / 1000
        xs = center[0] + (np.arange(shape[1]) - shape[1] / 2 + 0.5) * p
        ys = center[1] - (np.arange(shape[0]) - shape[0] / 2 + 0.5) * p
        return xs, ys

    @staticmethod
    def _toPx(center, shape, pix_size, x, y):
        """World µm to (col, row) pixel coordinates."""
        p = pix_size / 1000
        return (x - center[0]) / p + shape[1] / 2, -(y - center[1]) / p + shape[0] / 2

    def render(self, center, shape, pix_size, detail=False, seg=False):
        """Renders relative intensities (0-1) of the view centered on world position center.

        Args:
            center: World position (µm) of the image center.
            shape: Image shape (rows, cols).
            pix_size: Pixel size (nm).
            detail: Draw lamella interior (cells, organelles, ice blobs).
            seg: Also return a segmentation (uint8, CATS) of the view.
        """

        shape = tuple(int(s) for s in shape)
        xs, ys = self._axes(center, shape, pix_size)

        # Grid squares with per-square ice thickness and smooth thickness variation
        gx, gy = (xs - self.origin[0]) / self.pitch, (ys - self.origin[1]) / self.pitch
        ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
        ux, inv_x = np.unique(ix, return_inverse=True)
        uy, inv_y = np.unique(iy, return_inverse=True)
        factors = np.array([[self._squareFactor(i, j) for i in ux] for j in uy], dtype=np.float32)
        img = INTENSITY["square"] * factors[np.ix_(inv_y, inv_x)]
        broken = factors[np.ix_(inv_y, inv_x)] == 0
        img[broken] = INTENSITY["white"]
        wave_x = 1 + 0.1 * np.sin(2 * np.pi * xs / self.wave_length[0] + self.wave_phase[0])
        wave_y = 1 + 0.1 * np.sin(2 * np.pi * ys / self.wave_length[1] + self.wave_phase[1])
        img *= np.outer(wave_y, wave_x).astype(np.float32)

        # Grid bars
        bar_cols = (gx - ix) * self.pitch < self.bar_width
        bar_rows = (gy - iy) * self.pitch < self.bar_width
        img[bar_rows, :] = INTENSITY["bar"]
        img[:, bar_cols] = INTENSITY["bar"]

        seg_map = np.full(shape, CATS["black"], dtype=np.uint8) if seg else None

        # Lamellae intersecting the view
        half = np.array([shape[1], shape[0]]) * pix_size / 2000
        for lamella in self.lamellae:
            reach = lamella.size.max() * 1.5
            if np.any(np.abs(lamella.center - center) > half + reach):
                continue
            self._drawLamella(lamella, img, seg_map, center, shape, pix_size, detail)

        return (img, seg_map) if seg else img

    def _drawLamella(self, lamella, img, seg_map, center, shape, pix_size, detail):
        def pxRect(rect):
            c0, r1 = self._toPx(center, shape, pix_size, rect[0], rect[2])
            c1, r0 = self._toPx(center, shape, pix_size, rect[1], rect[3])
            clip = lambda val, limit: min(max(0, int(round(val))), limit)
            return slice(clip(r0, shape[0]), clip(r1, shape[0])), slice(clip(c0, shape[1]), clip(c1, shape[1]))

        # Milled trenches are empty (bright)
        for trench in lamella.trenches:
            rows, cols = pxRect(trench)
            img[rows, cols] = INTENSITY["white"]
            if seg_map is not None:
                seg_map[rows, cols] = CATS["white"]

        rows, cols = pxRect(lamella.slab)
        if lamella.category == "gone":
            img[rows, cols] = INTENSITY["white"]
            if seg_map is not None:
                seg_map[rows, cols] = CATS["white"]
            return

        # Slab intensity by lamella class
        slab = np.full(img[rows, cols].shape, INTENSITY["background"], dtype=np.float32)
        if slab.size == 0:
            return
        if lamella.category == "thick":
            slab *= 0.7
        elif lamella.category == "wedge":
            ramp = np.linspace(1.2, 0.6, slab.shape[0] if lamella.vertical else slab.shape[1], dtype=np.float32)
            slab *= ramp[:, None] if lamella.vertical else ramp[None, :]
        img[rows, cols] = slab
        if seg_map is not None:
            seg_map[rows, cols] = CATS["background"]

        # Interior features and contamination drawn in pixel space
        features = [feature for feature in lamella.features if feature[0] == "contamination" or detail]
        if features:
            self._drawFeatures(lamella, features, img, seg_map, center, shape, pix_size)

        if lamella.category == "broken":
            # Crack across the lamella and a missing corner
            crack_rows, crack_cols = pxRect(lamella.rect(lamella.size[0], 1.0, 0.2 * lamella.size[1]))
            corner_rows, corner_cols = pxRect(lamella.rect(lamella.size[0] / 2, lamella.size[1] / 4, -0.375 * lamella.size[1]))
            img[crack_rows, crack_cols] = INTENSITY["crack"]
            img[corner_rows, corner_cols] = INTENSITY["white"]
            if seg_map is not None:
                seg_map[crack_rows, crack_cols] = CATS["crack"]
                seg_map[corner_rows, corner_cols] = CATS["white"]

        # Protective coating at the front edge of the lamella (4 µm)
        coat_rows, coat_cols = pxRect(lamella.rect(lamella.size[0], 4.0, -lamella.size[1] / 2 + 2.0))
        img[coat_rows, coat_cols] = INTENSITY["coating"]
        if seg_map is not None:
            seg_map[coat_rows, coat_cols] = CATS["coating"]

    def _drawFeatures(self, lamella, features, img, seg_map, center, shape, pix_size):
        img_canvas = Image.fromarray(img)
        img_draw = ImageDraw.Draw(img_canvas)
        seg_canvas = Image.fromarray(seg_map) if seg_map is not None else None
        seg_draw = ImageDraw.Draw(seg_canvas) if seg_canvas is not None else None

        def ellipse(kind, u, v, w, h, outline=None):
            corners = [lamella._toWorld(u + su * w / 2, v + sv * h / 2) for su in (-1, 1) for sv in (-1, 1)]
            pts = np.array([self._toPx(center, shape, pix_size, *corner) for corner in corners])
            box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            width = max(1, int(0.5 / (pix_size / 1000))) if outline else 0
            intensity = 0.12 if kind == "contamination" else INTENSITY[kind]
            img_draw.ellipse(box, fill=intensity, outline=INTENSITY[outline] if outline else None, width=width)
            if seg_draw is not None:
                seg_draw.ellipse(box, fill=CATS["ice" if kind == "contamination" else kind], outline=CATS[outline] if outline else None, width=width)

        for kind, u, v, w, h in features:
            ellipse(kind, u, v, w, h, outline="cellwall" if kind == "cell" else None)

        img[:] = np.array(img_canvas)
        if seg_map is not None:
            seg_map[:] = np.array(seg_canvas)

    def toDict(self):
        return {"seed": self.seed, "center": self.center.tolist(), "region": self.region, "pitch": self.pitch,
                "lamellae": [lamella.toDict() for lamella in self.lamellae],
                "offsets": {key: val.tolist() for key, val in self.offsets.items()}}

    def save(self, file_path, extra=None):
        """Writes the ground truth (and optional extra entries like acquired maps) as json."""
        data = self.toDict()
        if extra:
            data.update(extra)
        Path(file_path).write_text(json.dumps(data, indent=2))


def toCounts(img, mean_counts=2000, blur_px=0.8):
    """Converts relative intensities into int16 camera counts with blur and shot noise.

    Shot noise is Poisson; at these counts it is drawn as Gaussian with variance = counts (much faster for camera frames).
    The generator is seeded from numpy's global random state, so seeded runs stay reproducible.
    """
    if blur_px:
        img = ndimage.gaussian_filter(img, blur_px)
    counts = np.clip(img, 0, None).astype(np.float32) * 2 * mean_counts
    rng = np.random.default_rng(np.random.randint(2**31))
    counts += np.sqrt(counts) * rng.standard_normal(counts.shape, dtype=np.float32)
    return np.clip(np.round(counts), 0, np.iinfo(np.int16).max).astype(np.int16)
