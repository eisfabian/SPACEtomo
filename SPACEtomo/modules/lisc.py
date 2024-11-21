# LisC filter with lamella alignment
# without plotting

import os
import time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None           # removes limit for large images
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import transform, filters, registration, exposure
from scipy.ndimage import gaussian_filter

PLOT = True

file_name = "/Users/eifabian/Documents/temp/Yeast8_L01_bin4.png"
pix_size = 18.9 * 4 #[nm/px]
tilt_axis = 0 #deg

highpass_threshold = 10000 #[nm]
exposure_crop = (0.7, 1.3)

# Import png image
image = np.array(Image.open(file_name))

# Only consider single channel
if len(image.shape) > 2:
    image = image[:, :, 0]

start = time.time()

print("Image")
plt.figure(figsize=(10,10))
plt.imshow(image, cmap="grey")
plt.axis("off")
plt.show()

ori_dims = image.shape


# FIND ROTATION

# Generate reference
ref_size = (400, 400)
ref = np.zeros(ref_size)
ref[int(ref_size[0] * 0.2): int(ref_size[0] * 0.8), :] = 1

if PLOT:
    print("Reference")
    plt.figure(figsize=(10,10))
    plt.imshow(ref)
    plt.axis("off")
    plt.show()

# Image resize to fit ref
image_sq = image[(image.shape[0] - min(image.shape)) // 2: (image.shape[0] + min(image.shape)) // 2, (image.shape[1] - min(image.shape)) // 2: (image.shape[1] + min(image.shape)) // 2]
image_sq = transform.resize(image_sq, ref_size, anti_aliasing=True)

# Apply soft edge mask
image_filt = image_sq * filters.window("hann", image_sq.shape)
ref_filt = ref * filters.window("hann", ref.shape)

if PLOT:
    print("Image resized to reference with soft edge")
    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(image_filt)
    ax[0].axis("off")
    ax[1].imshow(ref_filt)
    plt.show()

# Fourier transform
fft_img = abs(np.fft.fftshift(np.fft.fft2(image_filt)))
fft_ref = abs(np.fft.fftshift(np.fft.fft2(ref_filt)))

if PLOT:
    print("Fourier transform")
    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(np.log(abs(fft_img)))
    ax[0].axis("off")
    ax[1].imshow(np.log(abs(fft_ref)))
    ax[1].axis("off")
    plt.show()

# High-pass filter
yy = np.linspace(-np.pi / 2, np.pi / 2, fft_img.shape[0])[:, None]
xx = np.linspace(-np.pi / 2, np.pi / 2, fft_img.shape[1])[None, :]
rads = np.sqrt(xx ** 2 + yy ** 2)
filt = 1 - np.cos(rads) ** 2
filt[np.abs(rads) > np.pi / 2] = 1

fft_img = fft_img * filt
fft_ref = fft_ref * filt

# Log-polar conversion
lp_img = transform.warp_polar(fft_img, scaling="log", output_shape=(360,500))
lp_ref = transform.warp_polar(fft_ref, scaling="log", output_shape=(360,500))

if PLOT:
    print("Log-polar converted FFT")
    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(np.log(abs(lp_img)))
    ax[0].axis("off")
    ax[1].imshow(np.log(abs(lp_ref)))
    ax[1].axis("off")
    plt.show()

# Phase correlation (skimage implementation)
shifts, *_ = registration.phase_cross_correlation(lp_img, lp_ref)
#print(shifts)

# Calculate scale
base = np.exp(np.log(lp_img.shape[1] / 2) / lp_img.shape[1])
scale = base ** shifts[1]
print("Scale:", scale)

# Calc rotation
rotation = shifts[0]# - 180
print("Rotation:", rotation)

# Apply rotation
image_rot = transform.rotate(image, rotation, resize=True)

if PLOT:
    plt.figure(figsize=(10,10))
    plt.imshow(image_rot)
    plt.axis("off")
    plt.show()

print("Time to find rotation: ", time.time() - start, " s")

### START FILTERING ###

filter_threshold = highpass_threshold / pix_size
sigma_masks = 6.5 / pix_size

(w, h) = image_rot.shape
half_w, half_h = int(w/2), int(h/2)

# Fourier transform
fft = np.fft.fftshift(np.fft.fft2(image_rot))

if PLOT:
    plt.figure(figsize=(10,10))
    plt.imshow(np.log(abs(fft)))
    plt.axis("off")
    plt.show()

# high pass filter
filter_radius = pix_size * w / highpass_threshold # fourier pixels
print("Filter radius (fourier pixels):", filter_radius, filter_threshold)

mask = np.ones(fft.shape)
mask[:, range(half_h - int(filter_radius), half_h + int(filter_radius))] = 0
mask = gaussian_filter(mask, filter_radius / 4)

fft *= mask

if PLOT:
    plt.figure(figsize=(10,10))
    plt.imshow(np.log(abs(fft)))
    plt.axis("off")
    plt.show()

#highpassed = fftpack.ifft2(fftpack.ifftshift(fft)).real
highpassed = np.fft.ifft2(np.fft.ifftshift(fft)).real

if PLOT:
    plt.figure(figsize=(10,10))
    plt.imshow(highpassed, cmap="grey")
    plt.axis("off")
    plt.show()

highpassed_crop = highpassed[int(exposure_crop[0] * half_w):int(exposure_crop[1] * half_w), int(exposure_crop[0] * half_h):int(exposure_crop[1] * half_h)]
#in_range = (np.mean(highpassed_crop) - np.std(highpassed_crop), np.mean(highpassed_crop) + np.std(highpassed_crop))
#print(in_range)
in_range = (np.min(highpassed_crop), np.max(highpassed_crop))
print("Highpass exposure in-range:", in_range)
highpassed = exposure.rescale_intensity(highpassed, in_range=in_range, out_range=(0, 1))

#plt.figure(figsize=(10,10))
#plt.hist(highpassed)
#plt.show()


# Vacuum mask

lowpassed = gaussian_filter(image_rot, sigma_masks)

if PLOT:
    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(lowpassed, cmap="grey")
    rect = Rectangle((exposure_crop[0] * half_h, exposure_crop[0] * half_w), (exposure_crop[1] - exposure_crop[0]) * half_h, (exposure_crop[1] - exposure_crop[0]) * half_w, fill=False, linewidth=1, edgecolor="r")
    ax.add_patch(rect)
    plt.axis('off')
    plt.show()


# Crop to get histogram of lamella area
lowpassed_crop = lowpassed[int(exposure_crop[0] * half_w):int(exposure_crop[1] * half_w), int(exposure_crop[0] * half_h):int(exposure_crop[1] * half_h)]

# Create masks for vacuum and contamination
threshold = np.mean(lowpassed) + 1.5 * np.std(lowpassed)
print("Vac threshold:", threshold)
mask_vacuum = np.ones(image_rot.shape)
mask_vacuum[lowpassed > threshold] = 0


threshold = np.mean(lowpassed_crop) - 1.5 * np.std(lowpassed_crop)
print("Contamination threshold:", threshold)
mask_dark = np.ones(image_rot.shape)
mask_dark[lowpassed < threshold] = 0

# Combine masks
mask_both = (mask_vacuum * mask_dark).astype(bool)

if PLOT:
    plt.figure(figsize=(10,10))
    plt.imshow(mask_both, cmap="grey")
    plt.axis('off')
    plt.show()

# Create final image
final_rot = mask_both * highpassed + np.invert(mask_both) * lowpassed

# Rotate back to original
final = transform.rotate(final_rot, -rotation, resize=True)
final = final[(final.shape[0] - ori_dims[0]) // 2: (final.shape[0] + ori_dims[0]) // 2, (final.shape[1] - ori_dims[1]) // 2: (final.shape[1] + ori_dims[1]) // 2]

plt.figure(figsize=(10,10))
plt.imshow(final, cmap="grey")
plt.axis('off')
plt.show()

print("Total time elapsed:", (time.time() - start), "seconds")

image_save = Image.fromarray(np.uint8(final * 255))
image_save.save(os.path.splitext(file_name)[0] + "_lisc.png")