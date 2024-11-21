#!/usr/bin/env python
# ===================================================================
# Purpose:      Functions for utilities needed by other packages and scripts.
# Author:       Fabian Eisenstein
# Created:      2024/07/18
# Last Change:  2024/09/24: updated log to handle debug and line breaks
#               2024/09/04: converted settings to configparser
#               2024/09/02: added dummy_replace
#               2024/08/19: moved and updated writeTargets, moved writeMrc, moved saveSettings
# ===================================================================

import os
import sys
import time
import json
import struct
import numpy as np
import mrcfile
import hashlib
import subprocess
import concurrent.futures
from pathlib import Path
from functools import wraps
from configparser import ConfigParser
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images

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

from SPACEtomo import __version__
from SPACEtomo import config

# Check if torch and GPU are available
try:
    from torch.cuda import is_available
    if is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except:
    DEVICE = "cpu"

# Write to SerialEM log or print
def log(text, color=0, style=0):
    if text.startswith("DEBUG:") and not config.DEBUG:
        return
    if SERIALEM:
        if text.startswith("NOTE:"):
            color = 4
        elif text.startswith("WARNING:"):
            color = 5
        elif text.startswith("ERROR:"):
            color = 2
            style = 1 
        elif text.startswith("DEBUG:"):
            color = 1

        sem.SetNextLogOutputStyle(style, color)
        sem.EchoBreakLines(text)
    else:
        print(text)

def getCurDir():
    """Gets current directory from SerialEM or from main script."""

    if SERIALEM:
        return Path(sem.ReportDirectory())
    else:
        return Path.cwd()

# Find index in list of dicts for specific key, value pair
def findIndex(lst, key, val):
    if len(lst) > 0:
        if isinstance(lst[0], dict):
            for i, dic in enumerate(lst):
                if dic[key] == val:
                    return i
    return None

# Read generic column based text file
def loadColFile(file, force_type=str):
    data = []
    if os.path.exists(file):
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            cols = line.strip().split()
            if len(cols) > 0:
                data.append([])
                for col in cols:
                    data[-1].append(force_type(col))
    return data

# Convert numpy arrays to list for json (https://stackoverflow.com/a/65354261)
def convertArray(array):
    if hasattr(array, "tolist"):
        return {"$array": array.tolist()}       # convert to list in dict with tag
    raise TypeError(array)

def revertArray(array):
    if len(array) == 1:                         # check only dicts with one key
        key, value = next(iter(array.items()))
        if key == "$array":                     # if the key is the tag, convert
            return np.array(value)
    return array

def toNumpy(img):
    """Loads image with slightly faster (~3%) conversion from PIL.Image to numpy.array (https://uploadcare.com/blog/fast-import-of-pillow-images-to-numpy-opencv-arrays/)"""
    
    retries = 3
    for i in range(retries):
        try:
            img.load()
        except OSError as err:
            if i + 1 == retries:
                log(f"WARNING: Loading image failed. [{i + 1}/{retries}]")
                raise err
            else:
                log(f"WARNING: Loading image failed. Trying again... [{i + 1}/{retries}]")
    # unpack data
    e = Image._getencoder(img.mode, 'raw', img.mode)
    e.setimage(img.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(img)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast('B', (data.data.nbytes,)) # type: ignore

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data


def guessMontageDims(tile_num):
    # Guess based on factors as equal as possible
    i = int(tile_num ** 0.5 + 0.5)
    while tile_num % i != 0:
        i -= 1
    mont_shape = np.array([i, tile_num / i], dtype=int)
    return mont_shape

def serialem_check(func):
    """Decorator to check if SerialEM is running."""

    @wraps(func)
    def inner(*args, **kwargs):
        if "serialem" not in sys.modules:
            raise ModuleNotFoundError("SerialEM is not running!")
        return func(*args, **kwargs)
        
    return inner

def dummy_skip(func):
    """Decorator to skip function in DUMMY mode."""

    @wraps(func)    # wraps is needed to access proper function __name___
    def inner(*args, **kwargs):
        if not config.DUMMY:
            return func(*args, **kwargs)
        else:
            return log(f"WARNING: {func.__module__}.{func.__name__} is skipped in DUMMY mode.")
        
    return inner

def dummy_replace(replacement_func_name):
    """Decorator with argument to take replacement function for function uses in DUMMY mode."""

    def dummy_replace_decorator(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            log(f"DEBUG: {self}, {args}, {kwargs}")

            if not config.DUMMY:
                return func(self, *args, **kwargs)
            else:
                replacement_func = getattr(self, replacement_func_name)
                log(f"WARNING: {func.__module__}.{func.__name__} is replaced with {replacement_func.__module__}.{replacement_func.__name__} in DUMMY mode.")
                return replacement_func(*args, **kwargs)
        return inner
    return dummy_replace_decorator

def saveSettings(file, vars, start="automation_level", end="max_tilt"):
    """Saves SPACEtomo script settings."""

    config = ConfigParser(allow_no_value=True)
    config.optionxform = str
    config["INFO"] = {
        "SPACEtomo version": __version__,
        "Datetime": time.strftime('%d.%m.%Y %H:%M:%S', time.localtime())
    }
    config["SETTINGS"] = {}
    save = False
    for var in vars:                                    # globals() is ordered by creation, start and end points might have to be adjusted if script changes
        if var == start:                                # first var to save
            save = True
        if save:
            config["SETTINGS"][var] = str(vars[var])
        if var == end:                                  # last var to save
            break
    with open(file, "w") as f:
        config.write(f)

def loadSettings(file):
    if file.exists():
        config = ConfigParser()
        config.read(file)
        settings = {}
        for setting, value in config["SETTINGS"].items():
            # Recapitalize name
            if setting.startswith("wg") or setting.startswith("mm") or setting.startswith("im"):
                setting = setting[:2].upper() + setting[2:]
            # Cast to proper type
            settings[setting] = castString(value)
        return settings
    else:
        log("ERROR: Cannot start SPACEtomo because no SPACE settings file was found!")
        sys.exit()

def castString(string):
    """Casts string into int, float, Path, list(recursive), bool or string."""

    if not string:
        return string
    string = string.strip().strip("'\"")
    try:
        var = int(string)
    except ValueError:
        try:
             var = float(string)
        except ValueError:
            if Path(string).exists():
                var = Path(string)
            else:
                if string.startswith("[") and string.endswith("]"):
                    var = [castString(val) for val in string.strip("[]").split(",")]
                    if len(var) == 1 and var[0] == "":
                        var = []
                else:
                    if string.lower() == "true" or string.lower() == "yes" or string.lower() == "y" or string.lower() == "on":
                        var = True
                    elif string.lower() == "false" or string.lower() == "no" or string.lower() == "n" or string.lower() == "off":
                        var = False
                    elif string.lower() == "none":
                        var = None
                    else:
                        var = string
    return var

def waitForFile(file, message, function_call=None, msg_interval=60):
    """Stalls run until file was created."""

    start_time = time.time()
    next_update = start_time + msg_interval

    # Initial function call
    if function_call:
        function_call()

    # Use glob if path contains wildcard *
    if "*" in str(file):
        check = lambda: list(file.parent.glob(file.name))
    else:
        check = lambda: file.exists()

    while not check():
        now = time.time()
        if now > next_update:
            log(message + f" [{round((now - start_time) / 60)} min]")
            if function_call:
                function_call()
            next_update = now + msg_interval
        time.sleep(1)

def monitorExternal(external_dir):
    from SPACEtomo.modules import ext
    unprocessed_mm_list, unprocessed_seg_list, unprocessed_wg_list = ext.monitorFiles(external_dir)
    unprocessed_all = unprocessed_mm_list + unprocessed_seg_list + unprocessed_wg_list
    if unprocessed_all:
        log(f"# Current processing queue:\n" + "\n# ".join(unprocessed_all), color=1)

def writeMrc(file, image, pix_size):
    """Saves image as mrc file."""

    # Mrc cannot handle float64
    if image.dtype == np.float64:
        image = image.astype(np.float32)

    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(image)
        mrc.voxel_size = (pix_size * 10, pix_size * 10, pix_size * 10)
        mrc.update_header_from_data()
        mrc.update_header_stats()

# Write PACEtomo target file
def writeTargets(targets_file, targets, geo_points=[], saved_run=False, resume={"sec": 0, "pos": 0}, settings=None):
    """Writes tgts file in PACEtomo format."""

    output = ""
    if settings:
        for key, val in settings.items():
            if val != "":
                output += f"_set {key} = {val}\n"
        output += "\n"
    if resume["sec"] > 0 or resume["pos"] > 0:
        output += f"_spos = {resume['sec']},{resume['pos']}\n\n"
    for t, target in enumerate(targets):
        output += f"_tgt = {str(t + 1).zfill(3)}\n"
        for key in target.keys():
            output += f"{key} = {target[key]}\n"
        if saved_run:
            output += "_pbr\n"
            for key in saved_run[t][0].keys():
                output += f"{key} = {saved_run[t][0][key]}\n"
            output += "_nbr\n"
            for key in saved_run[t][1].keys():
                output += f"{key} = {saved_run[t][1][key]}\n"       
        output += "\n"
    for g, geo in enumerate(geo_points):
        output += f"_geo = {g + 1}\n"
        for key in geo.keys():
            output += f"{key} = {geo[key]}\n"
        output += "\n"
    with open(targets_file, "w") as f:
        f.write(output)


def loadDatasetJson(file_path):
    """Loads dataset.json from nnU-Net model."""

    file_path = Path(file_path)
    if file_path.exists():
        with open(file_path, "r") as f:
            dataset_json = json.load(f)

        # Get Categories
        cats = dataset_json["labels"]

        # Get number of images
        img_num = dataset_json["numTraining"]

        # Get pixel size
        if "pixel_size" in dataset_json.keys():
            pix_size = dataset_json["pixel_size"]
        else:
            pix_size = None

        return cats, pix_size, img_num
    else:
        return None, None, None
    

def cyclicRange(start, end, increment):
    """Generates cyclic values."""
    
    while True:
        #for i in range(start, end, increment):
        for i in np.arange(min(start, end), max(start, end), increment):
            yield i

def parseMdoc(mdoc_file):
    """Reads mdoc file into list of item dicts containing all entry values in lists."""
    
    header = ""
    items = []
    
    with open(mdoc_file) as f:
        # Loop over line iterator
        for line in f:
            col = line.strip().split(" ")
            # Skip empty lines
            if not col[0]: 
                continue
            # Instantiate new item on ZValue
            if col[0] == "[ZValue":
                items.append({"ZValue": col[2].strip("]")})

                # Item loop over line iterator
                for entry in f:
                    col = entry.strip().split(" ")
                    # Exit item on empty line
                    if not col[0]:
                        break
                    items[-1][col[0]] = [val for val in col[2:]]
            else:
                header += line

    return header, items
    
def synonymKeys(dictionary: dict, from_list: list, to_list: list):
    """Takes a dictionary and replaces keys in from_list with keys in to_list."""

    for from_key, to_key in zip(from_list, to_list):
        if from_key in dictionary.keys():
            dictionary[to_key] = dictionary[from_key]
            del dictionary[from_key]
    return dictionary

def hashFile(file_path):
    """Gets hash of file contents to monitor changes."""

    if file_path.exists():
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    else:
        # Empty
        return hashlib.md5(b'\x00').hexdigest()
        
def crossCorrCoeff(img1, img2):
    """Calculates maximum cross correlation coefficient."""

    # Ensure images have same size by center cropping
    if img1.shape != img2.shape:
        img1 = img1[(img1.shape[0] - min(img1.shape[0], img2.shape[0])) // 2: (img1.shape[0] + min(img1.shape[0], img2.shape[0])) // 2, (img1.shape[1] - min(img1.shape[1], img2.shape[1])) // 2, (img1.shape[1] + min(img1.shape[1], img2.shape[1])) // 2]
        img2 = img2[(img2.shape[0] - min(img1.shape[0], img2.shape[0])) // 2: (img2.shape[0] + min(img1.shape[0], img2.shape[0])) // 2, (img2.shape[1] - min(img1.shape[1], img2.shape[1])) // 2, (img2.shape[1] + min(img1.shape[1], img2.shape[1])) // 2]

    image_product = np.fft.fft2(img1) * np.fft.fft2(img2).conj()
    cc = np.fft.fftshift(np.fft.ifft2(image_product))
    return np.max(cc.real)

def getMrcExtendedHeader(file_path):
    """Parses extended header of mrc file and returns data for each section. (At time of implementation the mrcfile 1.5.3 package does not support SERI extended headers.)"""

    with open(file_path, "rb") as mrc_file:
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
        bitflags = struct.unpack("h", mrc[130: 132])[0]
        log(f"DEBUG: Bitflags: {bitflags}")
        # Figure out contained data
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
        data_byte_lengths = {"tilt_angle": [2], "piece_coords": [2, 2, 2], "stage": [2, 2], "mag": [2], "intensity": [2], "exposure": [4], "reserved1": [2], "reserved2": [4], "reserved3": [2], "reserved4": [4], "reserved5": [2]}
        data_conversion_factors = {"tilt_angle": 1 / 100, "piece_coords": 1, "stage": 1 / 25, "mag": 100, "intensity": 1 / 25000, "exposure": 1, "reserved1": 1, "reserved2": 1, "reserved3": 1, "reserved4": 1, "reserved5": 1}

        data_bits = reversed([1 if digit == "1" else 0 for digit in bin(bitflags)[2:]])
        data_entries = [entry for entry, bit in zip(data_byte_lengths.keys(), data_bits) if bit]
        log(f"DEBUG: Found data entries: {data_entries}")

        section_data = []
        section_data_raw = []
        # Loop over sections in extended header
        for i in range(1024, 1024 + bytes_per_section * section_number, bytes_per_section): # extended header starts at byte 1024
            section = {}
            # Get bytes for section
            section_raw = mrc[i:i + bytes_per_section]
            # Set pointer to start of section
            start = i
            # Extract data for each entry
            for entry in data_entries:
                # Exposure is formatted as float, rest is short
                byte_type = "h" if entry != "exposure" else "f"
                # Loop over byte lengths and append list if more than one value per entry
                if len(data_byte_lengths[entry]) > 1:
                    section[entry] = []
                    for length in data_byte_lengths[entry]:
                        # Convert bytes to value
                        section[entry].append(struct.unpack(byte_type, mrc[start:start + length])[0] * data_conversion_factors[entry])
                        # Move pointer by read length
                        start += length
                else:
                    # Convert bytes to value
                    section[entry] = struct.unpack(byte_type, mrc[start:start + data_byte_lengths[entry][0]])[0] * data_conversion_factors[entry]
                    # Move pointer by read length
                    start += data_byte_lengths[entry][0]
                    
            section_data.append(section)
            section_data_raw.append(section_raw)

        return section_data, section_data_raw
    
    # TODO: handle other header formats (FEI1 and FEI2 are already implemented in mrcfile)
    return None

def timeoutCall(function, timeout=5, retries=3):
    """Runs function as thread with timeout and retry limitations."""

    for attempt in range(retries):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(function)
            try:
                # Wait up to timeout for the function to complete
                result = future.result(timeout=timeout)
                log(f"DEBUG: Attempt {attempt + 1} was successful!")
                return result
            except concurrent.futures.TimeoutError:
                log(f"WARNING: Call to {function.__name__} timed out! Retrying...")
                # Cancel the future
                future.cancel()
                time.sleep(1)
    raise concurrent.futures.TimeoutError(f"Call to {function.__name__} timed out {retries} times!")

def guiProcess(gui_type, file_path="", auto_close=False):
    """Starts new process to call GUI."""

    file_path = str(file_path)

    args = [gui_type, file_path]
    if auto_close:
        args.append("--auto_close")

    log(f"DEBUG: GUI args: {args}")

    DETACHED_PROCESS = 0x00000008 # From here: https://stackoverflow.com/questions/89228/calling-an-external-command-in-python#2251026
    try:
        subprocess.Popen([sys.executable, Path(__file__).parent.parent / "GUI.py", *args], creationflags=DETACHED_PROCESS)
    except ValueError:      # Creationflags only supported on Windows
        subprocess.Popen([sys.executable, Path(__file__).parent.parent / "GUI.py", *args])