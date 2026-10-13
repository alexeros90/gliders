import numpy as np
from photutils.background import Background2D
from photutils.segmentation import detect_threshold, detect_sources, SegmentationImage, SourceCatalog
from .utils import (convolve_with_psf, normalize_data, replace_mask_with_noise, grow_mask, sigma_clipping,
                    adjust_image_resolution_to_scale, smooth_blend, get_main_source_label)
import warnings

def remove_low_outliers(image, sigma=5):
    bkg_lvl, bkg_std = sigma_clipping(image)
    outlier_mask = image < bkg_lvl - sigma * bkg_std
    image[outlier_mask] = np.random.normal(loc=bkg_lvl, scale=bkg_std, size=np.sum(outlier_mask))
    return image

def segmentation_map_smoothing(seg_map, kernel_std = 2):
    seg_map_out = np.zeros_like(seg_map)
    segmap_data = seg_map.data
    for segment in seg_map.segments:
        mask = (segmap_data == segment.label).astype(int)
        smoothed_mask = np.round(convolve_with_psf(mask, kernel_std=kernel_std), 0).astype(int)
        seg_map_out += smoothed_mask * segment.label
    seg_map_out = SegmentationImage(seg_map_out)
    return seg_map_out

def clear_background(image, source_mask=None, pix_scale=1):
    if source_mask is None:
        print("Invalid source mask, returning non modified image.")
        return image, source_mask
    influence_radius = round(np.sqrt(np.sum(source_mask)) / 4)
    img_bkg_lvl, img_bkg_lvl_std = sigma_clipping(image)
    noise_image = np.random.normal(loc=img_bkg_lvl, scale=img_bkg_lvl_std, size=image.shape)
    noise_image = adjust_image_resolution_to_scale(noise_image, pix_scale)
    image = smooth_blend(noise_image, image, source_mask, blend_width=influence_radius)
    blend_w = np.round(np.sqrt(source_mask.sum()) / 4, 0)
    cleaning_mask = ~grow_mask(source_mask, blend_w // 2)
    return image, cleaning_mask

def mask_secondary_detections(image, seg_map=None, source_pos=None, pix_scale=1, verbose=False):
    if seg_map is None:
        if verbose:
            warnings.warn("Invalid segmentation map. Returning non modified image.",
                          UserWarning, stacklevel=2)
        return image, np.full(image.shape, False)

    if source_pos is None:
        if verbose:
            warnings.warn("No source position provided. Assuming source is located at the image center.",
                          UserWarning, stacklevel=2)
        ny, nx = image.shape
        source_pos = np.array((int(nx / 2), int(ny / 2)))

    source_cat = SourceCatalog(image, seg_map)
    sources_table = source_cat.to_table()
    main_segment_label = get_main_source_label(seg_map, source_pos=source_pos, table=sources_table)

    main_mask = seg_map.data == main_segment_label
    main_slack_layers = round(np.sqrt(main_mask.sum()) * 0.05) # 5% of the radius
    main_mask = grow_mask(main_mask, main_slack_layers)

    secondaries_mask = np.full(seg_map.shape, False)
    for segment in seg_map.segments:
        if segment.label == main_segment_label:
            continue
        segment_mask = seg_map.data == segment.label
        slack_layers = round(np.sqrt(segment_mask.sum()) * 0.05) # 5% of the radius
        segment_mask = grow_mask(segment_mask, slack_layers)
        secondaries_mask |= segment_mask

    secondaries_mask = secondaries_mask & ~main_mask

    image = replace_mask_with_noise(image, secondaries_mask, sigma_factor=3, use_median=False,
                                        scale=pix_scale, smoothing_factor=0)
    return image, secondaries_mask

# TODO: De aqui para abajo, y documentacion a las de arriba

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
