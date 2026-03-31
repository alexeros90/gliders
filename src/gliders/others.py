import numpy as np
from photutils.background import Background2D
from photutils.segmentation import detect_threshold, detect_sources, SegmentationImage
from .utils import convolve_with_psf, normalize_data, replace_mask_with_noise, grow_mask, sigma_clipping, adjust_image_resolution_to_scale, smooth_blend


def make_background_robust(image, sigma=5):
    bkg_lvl, bkg_std = sigma_clipping(image)
    outlier_mask = image < bkg_lvl - sigma * bkg_std
    image[outlier_mask] = np.random.normal(loc=bkg_lvl, scale=bkg_std, size=np.sum(outlier_mask))
    return image

def segmentation_map_smoothing(seg_map):
    smoothed_segmap_mask = convolve_with_psf(seg_map.make_source_mask().astype(int), kernel_std=2)
    smoothed_segmap_mask = np.round(smoothed_segmap_mask, 0)
    smoothed_segmap = SegmentationImage(smoothed_segmap_mask.astype(int))
    return smoothed_segmap

def clear_background(image, source_mask=None, pix_scale=1, return_mask=True):
    if source_mask is None:
        print("Invalid source mask, returning non modified image.")
        return image
    influence_radius = round(np.sqrt(np.sum(source_mask)) / 4)
    img_bkg_lvl, img_bkg_lvl_std = sigma_clipping(image)
    noise_sample = np.random.normal(loc=img_bkg_lvl, scale=img_bkg_lvl_std, size=np.sum(source_mask))
    noise_image = np.random.choice(noise_sample, size=image.shape)
    noise_image = adjust_image_resolution_to_scale(noise_image, pix_scale)
    image = smooth_blend(noise_image, image, source_mask, blend_width=influence_radius)
    cleaning_mask = ~grow_mask(source_mask, 2)
    if return_mask:
        return image, cleaning_mask
    else:
        return image

def mask_secondary_detections(image, convolved_image=None, safe_mask=None, det_sigma=5, pix_scale=1, return_mask=True):
    grow_prop = int(round(np.sum(safe_mask)*0.0005))
    try:
        safe_mask = grow_mask(safe_mask,grow_prop)
        if convolved_image is None:
            convolved_image = image
        if safe_mask is None:
            safe_mask = np.full_like(image, False)
        threshold = detect_threshold(convolved_image, nsigma=det_sigma)
        sec_detections_mask = (convolved_image > threshold) & ~(safe_mask)
        sec_detections_mask = grow_mask(sec_detections_mask, 3) & ~(safe_mask)
        image = replace_mask_with_noise(image, sec_detections_mask, noise_factor=3, use_median=False,
                                        scale=pix_scale, smoothing_factor=0)
        if return_mask:
            return image, sec_detections_mask
        else:
            return image
    except Exception as e:
        print("Could not mask secondary objects:")
        print(e)
        return image, np.full_like(image,False, dtype = bool)

def background_subtraction(image, show_plots=False, normalize=False, detection_threshold=2,
                        box_divisor=14, bkg_min_pixels=5, psf_kernel_std=2.0):
    image_convolved = convolve_with_psf(image, psf_kernel_std)
    bkg_threshold = detect_threshold(image_convolved, nsigma=detection_threshold)
    bkg_segmap = detect_sources(image_convolved, bkg_threshold, bkg_min_pixels, connectivity=8)
    if bkg_segmap is None:
        bkg_mask = np.full(image.shape, False)  # TEMP
    else:
        bkg_mask = bkg_segmap.make_source_mask(size=6)
    valid_background = False
    box_size = (int(np.floor(image.shape[0] / box_divisor)), int(np.floor(image.shape[1] / box_divisor)))
    percentile = 10
    while not valid_background and box_divisor > 1:
        try:
            Background = Background2D(image, box_size=box_size, mask=bkg_mask, filter_size=3,
                                      exclude_percentile=percentile)
            valid_background = True
        except:
            percentile += 5

    if box_divisor <= 1:
        raise Exception("An error occurred while performing back-subtraction.")
    background_subtracted_image = image - Background.background
    if normalize:
        background_subtracted_image = normalize_data(background_subtracted_image)
    if show_plots:
        import matplotlib.pyplot as plt
        masked_img = image * (np.invert(bkg_mask) + 1 - 1) + np.max(image) / 1000
        fig_bck, ax_bck = plt.subplots(ncols=3, nrows=2, figsize=(8, 6), layout="tight")
        ax_bck[0, 0].imshow(image - np.min(image), cmap='gray', origin='lower', norm="log")
        ax_bck[0, 0].set_title("Image")
        ax_bck[0, 1].imshow(bkg_mask, origin='lower')
        ax_bck[0, 1].set_title("Detected sources")
        ax_bck[1, 0].imshow(masked_img, cmap='gray', origin='lower', norm="log")
        ax_bck[1, 0].set_title("Masked image")
        ax_bck[1, 1].imshow(Background.background, origin='lower')
        ax_bck[1, 1].set_title("Background")
        ax_bck[1, 2].imshow(Background.background_rms, origin='lower')
        ax_bck[1, 2].set_title("Background RMS")
        ax_bck[0, 2].imshow(background_subtracted_image - np.min(background_subtracted_image), origin='lower',
                            cmap='gray', norm="log")
        ax_bck[0, 2].set_title("Subtracted image")
        plt.show()

    return background_subtracted_image
