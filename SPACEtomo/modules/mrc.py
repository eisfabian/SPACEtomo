#!/usr/bin/env python
# ===================================================================
# ScriptName:   mrc
# Purpose:      Class for MRC file handling
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2025/04/13
# Revision:     v1.3
# Last Change:  2025/04/22: finished first draft
#               2025/04/16: implemented stitching
#               2025/04/15: implemented extended header
#               2025/04/13: started implementation
# ===================================================================

import mrcfile
import struct
import time
import json
import numpy as np
from pathlib import Path
from skimage import exposure
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images

from SPACEtomo import config
from SPACEtomo.modules.utils import log, castString, convertToTaggedString

class MRC:
    """Class for handling MRC files."""
    
    def __init__(self, file_path, new=False):
        """Initialize MRC object."""
    
        self.file_path = Path(file_path)
        self.data = None
        self.header = None
        self.header_labels = []
        self.extended_header = None
        self.extended_header_bitflags = None
        self.mdoc = {"header": {}, "items": [], "mont": []}

        self.map_type = None
        self.pix_size = None
        self.binning = 1

        if not new:
            self.readMrc()
            self.readExtendedHeader()
            self.readMdoc()

    def readMrc(self):
        """Read MRC file."""

        log(f"DEBUG: Reading mrc file...")
        with mrcfile.open(self.file_path, mode='r', permissive=True) as mrc:
            self.data = mrc.data
            self.pix_size = mrc.voxel_size.x / 10
            self.header = mrc.header
            self.header_labels = mrc.get_labels()

            if mrc.is_single_image():
                self.map_type = "2D"
            elif mrc.is_image_stack():
                self.map_type = "2D_stack"
            elif mrc.is_volume():
                self.map_type = "3D"
            elif mrc.is_volume_stack():
                self.map_type = "3D_stack"

        log(f"DEBUG: Image min/max/mean/dtype: {np.min(self.data)}, {np.max(self.data)}, {np.mean(self.data)}, {self.data.dtype}")

    def readExtendedHeader(self):
        """Read extended header."""

        with open(self.file_path, "rb") as mrc_file:
            mrc = mrc_file.read(4096)
            
        # MRC header format SERI (SERI is not covered by mrcfile)
        format = ""
        for char in struct.iter_unpack("s", mrc[104:108]):
            format += char[0].decode("utf-8")
        log(f"DEBUG: MRC format: {format}")

        if format == "SERI":
            # Get number of sections
            section_number = struct.unpack("i", mrc[8: 12])[0]
            log(f"DEBUG: Number of sections: {section_number}")

            # Get bytes per section
            bytes_per_section = struct.unpack("h", mrc[128: 130])[0]
            log(f"DEBUG: Bytes per section: {bytes_per_section}")

            # Bitflags
            self.extended_header_bitflags = struct.unpack("h", mrc[130: 132])[0]
            log(f"DEBUG: Bitflags: {self.extended_header_bitflags}")
            """
            https://bio3d.colorado.edu/imod/doc/mrc_format.txt
            1 = Tilt angle in degrees * 100  (2 bytes)
            2 = X, Y, Z piece coordinates for montage (6 bytes)
            4 = X, Y stage position in microns * 25   (4 bytes)
            8 = Magnification / 100 (2 bytes)
            16 = Intensity * 25000  (2 bytes)
            32 = Exposure dose in e-/A2, a float in 4 bytes
            128, 512: Reserved for 4-byte items
            64, 256, 1024: Reserved for 2-byte items
            If the number of bytes implied by these flags does
            not add up to the value in nint, then nint and nreal
            are interpreted as ints and reals per section
            """

            section_data = []
            for i in range(1024, 1024 + bytes_per_section * section_number, bytes_per_section): # extended header starts at byte 1024
                section_data.append(self.parseSectionData(mrc[i:i + bytes_per_section], self.extended_header_bitflags))

            self.extended_header = section_data
        
        elif "FEI" in format:
            with mrcfile.open(self.file_path, "rb") as mrc_file:
                self.extended_header = mrc_file.extended_header # numpy, indexed_extended_header can give part of extended header that was indexed

        else:
            log(f"WARNING: MRC extended header format not recognized: {format}")
            self.extended_header = None

    @staticmethod
    def parseSectionData(section_data, bitflags):
        """Parse a single section of extended header data based on bitflags."""

        parsed_data = {}
        offset = 0

        # Check each bitflag and extract corresponding data
        if bitflags & 1:  # Tilt angle in degrees * 100 (2 bytes)
            parsed_data["TiltAngle"] = struct.unpack('h', section_data[offset:offset + 2])[0] / 100
            offset += 2

        if bitflags & 2:  # X, Y, Z piece coordinates for montage (6 bytes)
            parsed_data["PieceCoordinates"] = struct.unpack('hhh', section_data[offset:offset + 6])
            offset += 6

        if bitflags & 4:  # X, Y stage position in microns * 25 (4 bytes)
            parsed_data["StagePosition"] = tuple(coord / 25 for coord in struct.unpack('hh', section_data[offset:offset + 4]))
            offset += 4

        if bitflags & 8:  # Magnification / 100 (2 bytes)
            parsed_data["Magnification"] = struct.unpack('h', section_data[offset:offset + 2])[0] * 100
            offset += 2

        if bitflags & 16:  # Intensity * 25000 (2 bytes)
            parsed_data["Intensity"] = struct.unpack('h', section_data[offset:offset + 2])[0] / 25000
            offset += 2

        if bitflags & 32:  # Exposure dose in e-/A² (4 bytes, float)
            parsed_data["ExposureDose"] = struct.unpack('f', section_data[offset:offset + 4])[0]
            offset += 4

        # Reserved fields (optional, if needed)
        if bitflags & 128 or bitflags & 512:  # Reserved for 4-byte items
            offset += 4
        if bitflags & 64 or bitflags & 256 or bitflags & 1024:  # Reserved for 2-byte items
            offset += 2

        return parsed_data

    def writeExtendedHeader(self, file_path=None):
        """Write extended header."""

        file_path = Path(file_path) if file_path else self.file_path

        # If extended header comes from SERI format, it is a list
        if isinstance(self.extended_header, (list, tuple)):
            with open(file_path, "r+b") as mrc_file:
                # Write the extended header type (e.g., "SERI") to bytes 104–108
                mrc_file.seek(104)
                mrc_file.write(b"SERI")

                # Write the number of sections to bytes 8–12
                section_number = len(self.extended_header)
                mrc_file.seek(8)
                mrc_file.write(struct.pack("i", section_number))

                # Write the bytes per section to bytes 128–130
                bytes_per_section = len(self.serializeSectionData(self.extended_header[0], self.extended_header_bitflags))
                mrc_file.seek(128)
                mrc_file.write(struct.pack("h", bytes_per_section))

                # Write the bitflags to bytes 130–132
                mrc_file.seek(130)
                mrc_file.write(struct.pack("h", self.extended_header_bitflags))

                # Write the section data
                mrc_file.seek(1024)
                mrc_file.write(b"".join([self.serializeSectionData(section, self.extended_header_bitflags) for section in self.extended_header]))

        # If extended header comes from FEI format, use mrcfile built-in function
        else:
            with mrcfile.open(file_path, "r+") as mrc_file:
                mrc_file.set_extended_header(self.extended_header)

    @staticmethod
    def serializeSectionData(section_data, bitflags):
        """Convert parsed section data back to bytes based on bitflags."""
        
        section_bytes = b""

        # Serialize each field based on the bitflags
        if bitflags & 1:  # Tilt angle in degrees * 100 (2 bytes)
            tilt_angle = int(section_data.get("TiltAngle", 0) * 100)
            section_bytes += struct.pack('h', tilt_angle)

        if bitflags & 2:  # X, Y, Z piece coordinates for montage (6 bytes)
            piece_coordinates = section_data.get("PieceCoordinates", (0, 0, 0))
            section_bytes += struct.pack('hhh', *piece_coordinates)

        if bitflags & 4:  # X, Y stage position in microns * 25 (4 bytes)
            stage_position = tuple(int(coord * 25) for coord in section_data.get("StagePosition", (0, 0)))
            section_bytes += struct.pack('hh', *stage_position)

        if bitflags & 8:  # Magnification / 100 (2 bytes)
            magnification = int(section_data.get("Magnification", 0) / 100)
            section_bytes += struct.pack('h', magnification)

        if bitflags & 16:  # Intensity * 25000 (2 bytes)
            intensity = int(section_data.get("Intensity", 0) * 25000)
            section_bytes += struct.pack('h', intensity)

        if bitflags & 32:  # Exposure dose in e-/A² (4 bytes, float)
            exposure_dose = section_data.get("ExposureDose", 0.0)
            section_bytes += struct.pack('f', exposure_dose)

        # Reserved fields (optional, if needed)
        if bitflags & 128 or bitflags & 512:  # Reserved for 4-byte items
            section_bytes += b'\x00\x00\x00\x00'
        if bitflags & 64 or bitflags & 256 or bitflags & 1024:  # Reserved for 2-byte items
            section_bytes += b'\x00\x00'

        return section_bytes

    def readMdoc(self):
        """Read MDOC file."""

        mdoc_file = self.file_path.with_suffix(".mrc.mdoc")
        if mdoc_file.exists():
            with open(mdoc_file, "r") as f:

                # Loop over line iterator
                for line in f:
                    col = line.strip().split()

                    # Skip empty lines
                    if not col or not col[0]: continue

                    # Instantiate new item on ZValue
                    if col[0] == "[ZValue" or col[0] == "[MontSection":
                        new_section = {}

                        # Item loop over line iterator
                        for entry in f:
                            item_col = entry.strip().split()

                            # Exit item on empty line
                            if not item_col or not item_col[0]: break

                            # Cast values
                            new_section[item_col[0]] = [castString(val) for val in item_col[2:]]

                            # Check if value is a list and only one value
                            if isinstance(new_section[item_col[0]], list) and len(new_section[item_col[0]]) == 1:
                                new_section[item_col[0]] = new_section[item_col[0]][0]

                        if col[0] == "[ZValue":
                            self.mdoc["items"].append(new_section)
                        elif col[0] == "[MontSection":
                            self.mdoc["mont"].append(new_section)

                    else:
                        if line.startswith("[T"):
                            # Append header line if not within item
                            if "T" not in self.mdoc["header"].keys():
                                self.mdoc["header"]["T"] = [line.strip()]
                            else:
                                self.mdoc["header"]["T"].append(line.strip())

                        else:
                            # Cast values
                            self.mdoc["header"][col[0]] = [castString(val) for val in col[2:]]

                            # Check if value is a list and only one value
                            if isinstance(self.mdoc["header"][col[0]], list) and len(self.mdoc["header"][col[0]]) == 1:
                                self.mdoc["header"][col[0]] = self.mdoc["header"][col[0]][0]

            log(f"DEBUG: Found {len(self.mdoc['items'])} items in MDOC file.")

            # Update binning
            if self.mdoc["items"] and "Binning" in self.mdoc["items"][0]:
                self.binning = self.mdoc["items"][0]["Binning"]
                log(f"DEBUG: Updated binning from MDOC: {self.binning}")
        else:
            log(f"WARNING: MDOC file not found: {mdoc_file}")

    def writeNewMrc(self, file_path=None):
        """Write MRC file."""

        log(f"DEBUG: Writing new MRC file...")

        file_path = Path(file_path) if file_path else self.file_path

        # Mrc cannot handle float64
        if self.data.dtype == np.float64:
            self.data = self.data.astype(np.float32)

        with mrcfile.new(file_path, overwrite=True) as mrc:
            mrc.set_data(self.data)
            mrc.voxel_size = self.pix_size * 10

            if self.data.ndim == 3:
                if self.map_type == "3D":
                    mrc.set_volume()
                else:
                    mrc.set_image_stack()  

            mrc.update_header_from_data()
            mrc.update_header_stats()
            # Need to reset mz because set_image_stack sets mz to 1 but that messes up voxel_size.z and 3dmod opens images as white 
            mrc.header.mz = mrc.header.nz

            # Add labels
            for label in self.header_labels:
                # Skip mrcfile labels that are already in the header
                if "mrcfile" in label:
                    continue
                # Stop if max number of labels is reached
                if mrc.header.nlabl >=10:
                    log(f"WARNING: MRC header labels are full. Cannot add all labels!")
                    break
                mrc.add_label(label)

        log(f"DEBUG: Image min/max/mean/dtype: {np.min(self.data)}, {np.max(self.data)}, {np.mean(self.data)}, {self.data.dtype}")

        if self.extended_header:
            self.writeExtendedHeader(file_path=file_path)

        if self.mdoc["items"]:
            self.writeMdoc(file_path=file_path)

    def updateMrc(self):
        """Update MRC file."""

        # Mrc cannot handle float64
        if self.data.dtype == np.float64:
            self.data = self.data.astype(np.float32)

        with mrcfile.open(self.file_path, mode='r+') as mrc:
            mrc.set_data(self.data)
            mrc.voxel_size = self.pix_size * 10

            mrc.update_header_from_data()
            mrc.update_header_stats()
            # Need to reset mz because set_image_stack sets mz to 1 but that messes up voxel_size.z and 3dmod opens images as white 
            #mrc.header.mz = mrc.header.nz

            # Add labels
            for l, label in enumerate(self.header_labels):
                # Skip mrcfile labels and labels that are already in the header
                if "mrcfile" in label or l < mrc.header.nlabl:
                    continue
                # Stop if max number of labels is reached
                if mrc.header.nlabl >=10:
                    log(f"WARNING: MRC header labels are full. Cannot add all labels!")
                    break
                mrc.add_label(label)

        log(f"DEBUG: Image min/max/mean/dtype: {np.min(self.data)}, {np.max(self.data)}, {np.mean(self.data)}, {self.data.dtype}")

        if self.extended_header:
            self.writeExtendedHeader()

        if self.mdoc["items"]:
            self.writeMdoc()

    def writeMdoc(self, file_path=None):
        """Write MDOC file."""

        file_path = Path(file_path) if file_path else self.file_path
        mdoc_file_path = file_path.with_suffix('.mrc.mdoc')

        def format_item(item, index, section_header="ZValue"):
            """Format a single MDOC item."""

            if section_header:
                lines = [f"[{section_header} = {index}]"]
            else:
                lines = []
            for key, attr in item.items():
                if key == "T":
                    lines.append("\n" + "\n\n".join(attr))
                else:
                    value = "  ".join(map(str, attr)) if key == "DateTime" else " ".join(map(str, attr if isinstance(attr, (list, tuple)) else [attr])) # Date and time is separated by double space in SerialEM
                    lines.append(f"{key} = {value}")
            return "\n".join(lines)

        # Build sections
        header_content = format_item(self.mdoc["header"], 0, "")
        items_content = "\n\n".join(format_item(item, i) for i, item in enumerate(self.mdoc["items"]))
        mont_content = "\n\n".join(format_item(item, i, "MontSection") for i, item in enumerate(self.mdoc["mont"]))

        # Write to file
        with open(mdoc_file_path, "w") as f:
            f.write(f"{header_content}\n\n{items_content}\n\n{mont_content}")

    def getIntensityCutoffs(self, quantile=0.01):
        """Get statistics of MRC file."""

        # Figure out scaling based on quantiles
        if config.DEBUG:
            vals = np.round(np.array((np.min(self.data), np.max(self.data), np.mean(self.data)), dtype=np.float32), 2)
            log(f"DEBUG: MRC statistics:")
            log(f"# Min, max, mean: {vals}")
            log(f"# Quantile: {round(quantile, 5)}")
            log(f"# Cutoffs: {np.quantile(self.data, quantile)}, {np.round(np.quantile(self.data, 1 - quantile), 2)}")

        return np.quantile(self.data, quantile), np.quantile(self.data, 1 - quantile)

    def scaleIntensity(self, quantile=0.01):
        """Rescale intensity of MRC file."""

        # Rescale intensity
        min_val, max_val = self.getIntensityCutoffs(quantile=quantile)
        log(f"DEBUG: Rescaling intensity to: {min_val}, {max_val}")

        self.data = exposure.rescale_intensity(self.data, in_range=(min_val, max_val), out_range=(0, 1))

        # Add header label
        self.header_labels.append(f"SPACEtomo: Rescaled intensity to [{min_val}, {max_val}]")
 
    def bin(self, factor: int):
        """Bin MRC file by factor."""

        img = self.data

        # TODO: implement for volumes and volume stacks
        if "2D" in self.map_type:

            # Save dtype
            dtype = img.dtype

            # Add third dimension if img is not stack
            if img.ndim == 2:
                img = np.expand_dims(img, 0)

            # Only bin in 2D and keep stack size
            factors = (1, factor, factor)

            # Calculate the new dimensions after cropping to allow even binning
            new_shape = tuple((dim // factor) * factor for dim, factor in zip(img.shape, factors))

            # Center crop the array to the new dimensions
            slices = tuple(slice((dim - new_dim) // 2, (dim + new_dim) // 2) for dim, new_dim in zip(img.shape, new_shape))
            cropped_img = img[slices]

            # Determine the new shape for reshaping
            reshaped_shape = np.array([(dim // factor, factor) for dim, factor in zip(cropped_img.shape, factors)]).reshape(-1)
            
            # Reshape the array
            reshaped_img = cropped_img.reshape(reshaped_shape)

            # Calculate the mean along the new axes
            for i in range(-1, -cropped_img.ndim-1, -1):
                reshaped_img = reshaped_img.mean(axis=i)

            # Remove added dimension
            if reshaped_img.shape[0] == 1:
                reshaped_img = reshaped_img[0, :, :]

            self.data = reshaped_img.astype(dtype)
            self.pix_size *= factor
            self.binning *= factor

            # Add label
            self.header_labels.append(f"SPACEtomo: Binned by {factor}")

            # Adjust mdoc header pixel size and binning
            if "PixelSpacing" in self.mdoc["header"].keys():
                self.mdoc["header"]["PixelSpacing"] *= factor
            if "Binning" in self.mdoc["header"].keys():
                self.mdoc["header"]["Binning"] *= factor

            # Adjust tile pixel size, binning and coordinates if existing
            if self.mdoc["items"]:
                for item in self.mdoc["items"]:
                    if "PixelSpacing" in item:
                        item["PixelSpacing"] *= factor
                    if "Binning" in item:
                        item["Binning"] *= factor

                    if "PieceCoordinates" in item:
                        item["PieceCoordinates"] = tuple(coord // factor for coord in item["PieceCoordinates"])
                    if "AlignedPieceCoords" in item:
                        item["AlignedPieceCoords"] = tuple(coord // factor for coord in item["AlignedPieceCoords"])
                    if "AlignedPieceCoordsVS" in item:
                        item["AlignedPieceCoordsVS"] = tuple(coord // factor for coord in item["AlignedPieceCoordsVS"])
            if self.extended_header:
                for section in self.extended_header:
                    if "PieceCoordinates" in section:
                        section["PieceCoordinates"] = tuple(coord // factor for coord in section["PieceCoordinates"])

        else:
            log(f"WARNING: Binning not implemented for {self.map_type}.")


    def sortStack(self, key=None):
        """Sort stack by key."""

        if "stack" in self.map_type and self.mdoc["items"]:

            # Make list of item values
            values = [item[key] for item in self.mdoc["items"]]
            log(f"DEBUG: Values before sorting: {values}")

            # Get extended header data
            if not self.extended_header:
                self.readExtendedHeader()
            log(f"DEBUG: MRC header sections: {len(self.extended_header)}")

            # Sort by values
            zipped_sections = sorted(zip(values, self.mdoc["items"], self.data, self.extended_header), key=lambda x: x[0])
            values, self.mdoc["items"], data, self.extended_header = zip(*zipped_sections)
            self.data = np.array(data)

            # Add label
            self.header_labels.append(f"SPACEtomo: Sorted by {key}")

            log(f"DEBUG: Values after sorting: {values}")
        
        else:
            log(f"WARNING: Sorting not implemented for {self.map_type}.")

    def stitch(self):
        """Stitch stack."""

        if "2D_stack" in self.map_type:# and self.mdoc["items"]:

            # Get tile coordinates
            if self.mdoc["items"] and "PieceCoordinates" in self.mdoc["items"][0]:
                # Check first for very sloppy aligned coords
                if "AlignedPieceCoordsVS" in self.mdoc["items"][0]:
                    tile_coords = [item["AlignedPieceCoordsVS"] for item in self.mdoc["items"]]
                # Then for aligned coords
                elif "AlignedPieceCoords" in self.mdoc["items"][0]:
                    tile_coords = [item["AlignedPieceCoords"] for item in self.mdoc["items"]]
                # Then for raw coords
                else:
                    tile_coords = [item["PieceCoordinates"] for item in self.mdoc["items"]]
                tile_coords = np.array(tile_coords)
                log(f"DEBUG: Tile coordinates [MDOC]: {tile_coords}")
            elif self.extended_header and "PieceCoordinates" in self.extended_header[0]:
                # Finally, get coords from extended header
                tile_coords = [section["PieceCoordinates"] for section in self.extended_header]
                tile_coords = np.array(tile_coords)
                log(f"DEBUG: Tile coordinates [EXT HEADER]: {tile_coords}")
            else:
                log(f"ERROR: No tile coordinates found. Cannot stitch map.")
                return

            # Calculate the dimensions of the stitched image
            min_x = np.min(tile_coords[:, 1])
            min_y = np.min(tile_coords[:, 0])
            # Adjust coordinates to start from (0, 0)
            tile_coords[:, 1] -= min_x
            tile_coords[:, 0] -= min_y
            # Calculate the maximum dimensions
            max_x = np.max(tile_coords[:, 1]) + self.data.shape[1]
            max_y = np.max(tile_coords[:, 0]) + self.data.shape[2]
            stitched_image = np.zeros((max_x, max_y), dtype=self.data.dtype)

            # Assemble image
            for t, tile in enumerate(self.data):
                # Stop when end of tile_coords is reached (in case there are more montages in file)
                if t >= len(tile_coords): 
                    break
                stitched_image[tile_coords[t][1]: tile_coords[t][1] + self.data.shape[1], tile_coords[t][0]: tile_coords[t][0] + self.data.shape[2]] = tile

            return stitched_image

        else:
            log(f"WARNING: Stitching not implemented for {self.map_type}.")

    def splitStack(self):
        """Split stack into images."""

        stack = []
        if "stack" in self.map_type:
            for i in range(self.data.shape[0]):
                new_mrc = MRC(self.file_path.parent / f"{self.file_path.stem}_{i}.mrc", new=True)
                new_mrc.data  = self.data[i]
                if self.extended_header:
                    new_mrc.extended_header = [self.extended_header[i]]
                    new_mrc.extended_header_bitflags = self.extended_header_bitflags
                if self.mdoc["items"]:
                    new_mrc.mdoc["header"] = self.mdoc["header"]
                    new_mrc.mdoc["items"] = [self.mdoc["items"][i]]
                new_mrc.pix_size = self.pix_size
                new_mrc.binning = self.binning

                # Update header labels
                new_mrc.header_labels = self.header_labels.copy()
                new_mrc.header_labels.append(f"SPACEtomo: Split from {self.file_path.name} (section {i})")

                stack.append(new_mrc)

                # Write to file
                new_mrc.writeNewMrc()
                new_mrc.writeMdoc()
        else:
            log(f"WARNING: Splitting not implemented for {self.map_type}.")

        return stack
    
    def saveImg(self, file_type="png", section=None):
        """Saves mrc map as image file."""

        if file_type.lower() not in ["png", "tif", "tiff"]:
            log(f"ERROR: Saving as {file_type} is not implements. Use PNG or TIF!")
            return

        if self.map_type != "2D" and section:
            log(f"DEBUG: Saving section {section} as image.")
            img = self.data[section]
        elif self.map_type == "2D":
            img = self.data
            section = 0
        else:
            log(f"ERROR: Cannot save full stack/volume as image.")
            return

        # Scale to uint8
        if np.min(img) < 0 or not 200 < np.max(img) <= 255:
            log(f"DEBUG: Rescaling image intensity to out_range=(0, 255)") 
            img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

        # Save image
        log(f"DEBUG: Saving image to file...")    
        Image.fromarray(img).save(self.file_path.with_suffix(f".{file_type}"))
        log(f"DEBUG: Finished saving {self.file_path.with_suffix(f'.{file_type}')}")

        # Save meta data to json
        log(f"DEBUG: Saving meta data to {self.file_path.with_suffix('.json')}")
        meta_data = {"datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
                     "img_size": img.shape,
                     "pix_size": self.pix_size,
                     "original_map": self.file_path
                     }

        # Get additional data from mdoc or extended header
        if self.mdoc["items"] and "StagePosition" in self.mdoc["items"][section]:
            meta_data["stage_coords"] = self.mdoc["items"][section]["StagePosition"] + [self.mdoc["items"][section]["StageZ"]]
        elif self.extended_header and "StagePosition" in self.extended_header[section]:
            meta_data["stage_coords"] = self.extended_header[section]["StagePosition"] + [0.]

        if self.mdoc["items"] and "TiltAngle" in self.mdoc["items"][section]:
            meta_data["tilt_angle"] = self.mdoc["items"][section]["TiltAngle"]
        elif self.extended_header and "TiltAngle" in self.extended_header[section]:
            meta_data["tilt_angle"] = self.extended_header[section]["TiltAngle"]
            # Can be extended if necessary (also extend in buf.py accordingly)
            
        with open(self.file_path.with_suffix('.json'), "w") as f:
            json.dump(meta_data, f, indent=4, default=convertToTaggedString)


# TODO:
# [x] Load MRC file
# [x] Load MDOC file
# [x] Write MRC file
# [x] Write MDOC file

# [x] Get statistics
# [x] Scale intensity

# [x] Resort MRC file and MDOC file
# [x] Read MRC extended header
# [x] Write MRC extended header
# [x] when writing extended header for a new file, also specify extended hader type!

# [x] Bin MRC file
# [x] Update header, tile coords, ... after binning
# [x] Update pixel size and binning in mdoc

# [x] Read tile coords from header or MDOC file
# [x] Read tilt angles from header or MDOC file

# [x] Create stitched image
# [x] Save as PNG
# [x] Save as TIFF
# [x] meta_data: datetime, pix_size, img_size, stage_coords, s2img_matrix, original_map, tilt_angle, grid_vectors

# [x] Split stack
# [x] Deal with mont section in mdoc

# [x] test on tilt series and volume: stitch, saveImg, ...
# [x] keep mrc header titles/labels

# [x] updateMrc method
# [x] intensities get messed up somewhere, caused by file mode changing to float32 from int16 during binning
"""
# [ ] Assemble stack [stretch]
# [ ] Calculate auto correlation and grid vectors [stretch]
"""

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Test the MRC class
    mrc = MRC("/Users/fabian/Downloads/Yeast3_L02.mrc")
    #mrc = MRC("/Users/fabian/Downloads/tomo2c.mrc")

    mrc.bin(4)

    #mrc.scaleIntensity(quantile=0.01)
    map_img = mrc.stitch()
    if map_img is not None:
        print(map_img.shape, np.min(map_img), np.max(map_img), np.mean(map_img))
        plt.imshow(map_img)
        plt.show()

    #mrc.splitStack()

    mrc.writeNewMrc("test_out.mrc")
    test = MRC("test_out.mrc")
