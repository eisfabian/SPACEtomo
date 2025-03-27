#!/usr/bin/env python
# ===================================================================
# Purpose:      Handles sample geometry calculations.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2024/10/08
# Last Change:  2025/02/12: added geometry from beam tilt image pair
#               2025/01/21: added handling of Position objects
#               2024/10/15: added all conversions
# ===================================================================

import numpy as np
from scipy import optimize

from SPACEtomo.modules.utils import log, alignCC
from SPACEtomo.modules.scope import Microscope
from SPACEtomo.modules.buf import Buffer

def paraboloid(x, a, b, c, d, e):
    return a * x[0] + b * x[1] + c * (x[0]**2) + d * (x[1]**2) + e * x[0] * x[1]

class Geometry:
    """Handles sample geometry."""

    microscope = None # set to None before use

    def __init__(self, microscope=None) -> None:

        if microscope:
            self.__class__.microscope = microscope

        # Instantiate coefficients
        self.a = self.b = self.c = self.d = self.e = 0

    def fromPretilt(self, pretilt, rotation):
        """Creates geometry function from pretilt and rotation values."""

        # Assuming pretilt being a rotation around the x-axis and rotation being a rotation around the z-axis,
        # the normal vector will be N = ( SIN(pretilt) * SIN(rotation), -SIN(pretilt) * COS(rotation), COS(pretilt) ) and z = (Nx / Nz) * x + (Ny / Nz) * y

        self.a = np.sin(np.radians(pretilt)) * np.sin(np.radians(rotation)) / np.cos(np.radians(pretilt))
        self.b = -np.sin(np.radians(pretilt)) * np.cos(np.radians(rotation)) / np.cos(np.radians(pretilt))

    def fromGeoPoints(self, geo_points):
        """Creates geometry function from defocus measurements at geo points."""

        if len(geo_points) < 3:
            log(f"WARNING: Sample geometry could not be measured, because there are not enough geo points!")
            return

        # If z values are missing, measure by beam tilt routine
        if (hasattr(geo_points[0], "relative_z") and geo_points[0].relative_z is None) or (isinstance(geo_points, np.ndarray) and geo_points.shape[0] >= 3 and geo_points.shape[1] == 2):
            new_geo_points = self.measureDefocus(geo_points)
            # Call self to undergo check again
            self.fromGeoPoints(new_geo_points)
            return

        # If geo_points are Position objects, convert to numpy array
        if hasattr(geo_points[0], "specimen_shift"):
            geo_points = np.array([np.append(geo_point.specimen_shift, geo_point.relative_z) for geo_point in geo_points])

        # Remove 0 defocus values
        num = geo_points.shape[0]
        log(f"NOTE: Obtained {num} geo points.")
        geo_points = geo_points[geo_points[:, 2] != 0]
        if num - geo_points.shape[0] > 0:
            log(f"WARNING: Removed {num - geo_points.shape[0]} geo points with 0 defocus values.")

        # If geo points contain z values, transpose
        if geo_points.shape[0] >= 3 and geo_points.shape[1] == 3:
            # Subtract first point from all points
            geo_points -= geo_points[0]
            geo_xyz = geo_points.T
        else:
            log(f"WARNING: Sample geometry plane could not be measured, because there are not enough geo points!")
            return        
        
        ##########
        # Source: https://math.stackexchange.com/q/99317
        # subtract out the centroid and take the SVD, extract the left singular vectors, the corresponding left singular vector is the normal vector of the best-fitting plane
        svd = np.linalg.svd(geo_xyz - np.mean(geo_xyz, axis=1, keepdims=True))
        left = svd[0]
        norm = left[:, -1]
        ##########        
        log(f"DEBUG: Normal vector: {norm}")

        self.a = norm[0] / norm[2]
        self.b = norm[1] / norm[2]
        log(f"DEBUG: Set geometry parameters:\n    a = {self.a}\n    b = {self.b}")

        log(f"=> Pretilt: {round(self.pretilt, 2)} degrees | Rotation: {round(self.rotation, 2)} degrees")

        """
        # Plot measure geo fit
        import matplotlib.pyplot as plt
        ax = plt.subplot(111, projection='3d')
        ax.scatter(geo_xyz[0], geo_xyz[1], geo_xyz[2], s=50, label="Geo points", color="#ff0000")
        for i in range(len(geo_xyz[0])):
            ax.text(geo_xyz[0][i], geo_xyz[1][i], geo_xyz[2][i], ' %s' % (i + 1), size=20, zorder=1, color="#ff0000")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                        np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)

        center = np.mean(geo_xyz, axis=1)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r,c] = (-norm[0] * (X[r,c] - center[0]) - norm[1] * (Y[r,c] - center[1])) / norm[2] + center[2]
        ax.plot_wireframe(X,Y,Z, label="Fitted plane", color="#000000")

        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect("equal")
        plt.show()
        """

    def fromCtfFit(self, geo_points):
        """Creates geometry function from defocus estimations by CTF fit."""

        # If geo points contain z values, transpose
        if geo_points.shape[0] >= 5 and geo_points.shape[1] == 3:
            geo_xyz = geo_points.T
        else:
            log(f"WARNING: Sample geometry paraboloid could not be fitted, because there are not enought geo points or defocus was not estimated yet!")
            return       

        parameters, cov = optimize.curve_fit(paraboloid, [geo_xyz[0], geo_xyz[1]], geo_xyz[2])

        ss = 0
        for i in range(0, len(geo_xyz[2])):
            ss += (geo_xyz[2][i] - geo_xyz[2][0] - paraboloid([geo_xyz[0][i], geo_xyz[1][i]], *parameters))**2
        rmse = np.sqrt(ss / len(geo_xyz[2]))
        log(f"RMSE: {round(rmse, 3)}")

        self.a, self.b, self.c, self.d, self.e = parameters
        log(f"Set geometry parameters:\n    " + "\n    ".join(parameters.astype(str)))

    def fromBeamTiltPair(self, buf1: Buffer, buf2: Buffer, beam_tilt_mrad: float, patches=(4, 4)):
        """Measures geometry from single beam tilt [mrad] pair by measuring displacements of image patches and calculating defocus gradient."""

        # Check if buffers are of same size
        if not np.array_equal(buf1.shape, buf2.shape):
            log(f"ERROR: Buffers are not of same size.")
            return

        # Divide images into patches
        y, x = buf1.shape.astype(int)
        patch_x, patch_y = x // patches[0], y // patches[1]
        
        geo_points = []
        for i in range(patches[0]):
            for j in range(patches[1]):
                patch1 = buf1.img[i * patch_x:(i + 1) * patch_x, j * patch_y:(j + 1) * patch_y]
                patch2 = buf2.img[i * patch_x:(i + 1) * patch_x, j * patch_y:(j + 1) * patch_y]

                # Find displacement
                shift, confidence = alignCC(patch1, patch2)
                if displacement := np.linalg.norm(shift) <= 2:
                    log(f"WARNING: Patch image alignment failed with a confidence of {confidence}. Maybe try again at higher defocus.")
                    continue
                displacement_nm = np.linalg.norm(shift) * buf1.pix_size

                # Calculate defocus
                defocus_nm = -displacement_nm / (2 * beam_tilt_mrad / 1000) # Approximation
                log(f"DEBUG: Displacement: {round(displacement_nm, 1)} nm, Calculated defocus: {round(defocus_nm)} nm. (Mean: {round(np.mean(patch1))})")

                # Save in geo points
                x = j * patch_y * buf1.pix_size / 1000
                y = -i * patch_x * buf1.pix_size / 1000
                geo_points.append([x, y, defocus_nm / 1000])

        # Create geometry from geo points
        self.fromGeoPoints(np.array(geo_points))

    def measureDefocus(self, geo_points):
        """Measures defocus at geo_points."""

        defoci = []
        for point in geo_points:

            if hasattr(point, "specimen_shift"):
                point_coords = point.specimen_shift
            else:
                point_coords = point

            # Shift to point
            self.microscope.setSpecimenShift(point_coords, relative=True)
            # Measure defocus
            defocus, drift = self.microscope.autofocus(measure=True)
            log(f"DEBUG: Measured an autofocus of {defocus} and a drift of {drift} at {point_coords}.")
            # Check if values are reasonable
            if drift > 0 and abs(defocus) > 0:
                defoci.append(defocus)
            else:
                log("WARNING: Measured defocus or drift is 0, which indicates that the defocus measurement failed. This geo point will not be considered.")
                defoci.append(np.nan)

            # Save in Position object
            if hasattr(point, "relative_z"):
                point.relative_z = defoci[-1]

            # Reset shift
            self.microscope.setSpecimenShift(-point_coords, relative=True)

        # Append defoci to geo points
        if not hasattr(geo_points[0], "specimen_shift"): # Check if geo_points are Position objects
            geo_points = np.column_stack([geo_points, defoci])

            # Remove failed geo points
            geo_points = geo_points[~np.isnan(geo_points).any(axis=1)]
        else:
            geo_points = [point for point in geo_points if not np.isnan(point.relative_z)]

        return geo_points

    def rotateAroundX(self, tilt_angle: float):
        """Rotates geometry plane around x-axis."""
        
        cos_tilt = np.cos(np.radians(tilt_angle))
        sin_tilt = np.sin(np.radians(tilt_angle))
        
        # Rotate normal vector
        new_normal = np.dot(np.array([[1, 0, 0], [0, cos_tilt, -sin_tilt], [0, sin_tilt, cos_tilt]]), self.normal)
        # Set new coefficients
        self.a, self.b = new_normal[:2] / new_normal[2]

    def z(self, x):
        return paraboloid(x, self.a, self.b, self.c, self.d, self.e)
    
    @property
    def normal(self):
        """Calculates normal vector from linear coefficients."""

        if all([self.a == 0, self.b == 0]):
            log(f"WARNING: Normal vector of sample plane cannot be calculated, because a and b are 0.")
            return np.array([0, 0, 1])

        norm = np.array([self.a, self.b, 1])
        norm /= np.linalg.norm(norm)

        if any([self.c != 0, self.d != 0, self.e != 0]):
            log(f"WARNING: Normal vector of paraboloid cannot represent geometry fully.")

        return norm
    
    @property
    def pretilt(self):
        """Calculates pretilt relative to x-y-plane from linear coefficients."""

        return -np.sign(self.normal[1]) * np.degrees(np.arccos(self.normal[2]))
    
    @property
    def rotation(self):
        """Calculates rotation around z-axis from linear coefficients."""

        if self.normal[1] == 0:
            return 0.0

        return -np.degrees(np.arctan(self.normal[0] / self.normal[1]))