import numpy as np
from photutils.segmentation import detect_threshold, detect_sources, SegmentationImage
from photutils.background import Background2D, SExtractorBackground, MADStdBackgroundRMS
from scipy.signal import convolve2d
from astropy.convolution import Gaussian2DKernel
from .utils import (convolve_with_gaussian_kernel, replace_mask_with_noise, grow_mask, sigma_clipping,
                    adjust_image_resolution_to_scale, smooth_blend, get_main_source_label)
import warnings

def segmentation_map_smoothing(seg_map, kernel_std = 2):
    """Smooth the individual segments of a segmentation map.

        Each segment is smoothed using a point-spread-function (PSF)
        convolution.

        Parameters
        ----------
        seg_map : photutils.segmentation.SegmentationImage
            Input segmentation map.
        kernel_std : float, optional
            Standard deviation of the kernel used by
            `convolve_with_psf`, in pixels. Controls the spatial extent
            of the smoothing. Larger values produce more extensive
            smoothing. Default is 2.

        Returns
        -------
        photutils.segmentation.SegmentationImage
            New segmentation map containing the smoothed segment masks
            with their original labels.

        Notes
        -----
        Each segment is smoothed independently. Rounding the convolved
        mask to integer values determines which pixels remain assigned
        to that segment. The output is initialized to zero, so pixels
        not covered by any smoothed segment remain unassigned.

        If smoothed segments overlap, their labeled masks are added
        together. Consequently, overlapping labels can produce values
        that do not correspond to an original segment label.
        """
    seg_map_out = np.zeros_like(seg_map)
    segmap_data = seg_map.data
    for segment in seg_map.segments:
        mask = (segmap_data == segment.label).astype(int)
        smoothed_mask = np.round(convolve_with_gaussian_kernel(mask, kernel_std=kernel_std), 0).astype(int)
        seg_map_out += smoothed_mask * segment.label
    seg_map_out = SegmentationImage(seg_map_out)
    return seg_map_out

def clear_background(image, source_mask=None, pix_scale=1, verbose=False):
    """Replace the ell except the source regions with synthetic background noise.

        Parameters
        ----------
        image : numpy.ndarray
            Two-dimensional input image to be processed.
        source_mask : numpy.ndarray of bool, optional
            Boolean mask identifying the source region to keep in place.
            True pixels indicate the source region. If None, the input
            image is returned unchanged, together with None.
        pix_scale : float, optional
            Pixel-scale factor passed to `adjust_image_resolution_to_scale`.
            Controls the resolution adjustment applied to the synthetic
            background image. Default is 1.
        verbose : bool, optional
            If True, activates warning and messages.
            Default is False.
        Returns
        -------
        image_out : numpy.ndarray
            Image in which all but the source region has been replaced by
            synthetic background noise.
        cleaning_mask : numpy.ndarray of bool or None
            Boolean mask identifying pixels affected by the replacement.
            True pixels are retained by the mask, while False
            pixels correspond to the region excluded for subsequent
            cleaning or analysis. Returns None if `source_mask` is None.

        Notes
        -----
        The background level and noise standard deviation are estimated
        using `sigma_clipping`. Synthetic noise is generated using a
        Gaussian distribution with these estimated parameters.

        The synthetic image is blended into the source region using
        `smooth_blend`, with an influence radius derived from the area
        of the source mask. A separate, expanded mask is then created
        using `grow_mask` to exclude the source and its surrounding
        pixels from subsequent analysis.

        The synthetic background is randomly generated, so repeated
        calls can produce different results unless the random-number
        generator used by `np.random.normal` is seeded.

        """

    if source_mask is None:
        if verbose:
            warnings.warn("Invalid source mask, returning non modified image.", UserWarning, stacklevel=2)
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
    """Replace secondary detections with synthetic noise.

        All detected segments other than the main segment are replaced with
        synthetic noise.

        Parameters
        ----------
        image : numpy.ndarray
            Two-dimensional input image to be processed.
        seg_map : photutils.segmentation.SegmentationImage, optional
            Segmentation map identifying the detected sources. Each
            segment has a unique positive label. If None, the input image
            is returned unchanged along with an all-False mask.
        source_pos : tuple of float, optional
            Approximate position of the main source in image coordinates,
            given as (x, y) in pixels. If None, the geometric center of
            the image is used.
        pix_scale : float, optional
            Pixel-scale factor passed to `replace_mask_with_noise`,
            controlling the resolution adjustment of the replacement
            noise. Its precise interpretation depends on that helper
            function. Default is 1.
        verbose : bool, optional
            If True, activates warning and messages.
            Default is False.

        Returns
        -------
        image_out : numpy.ndarray
            Image in which secondary detections have been replaced
            with synthetic noise. The main source is preserved.
        secondaries_mask : numpy.ndarray of bool
            Boolean mask identifying the secondary-source pixels
            selected for replacement, excluding pixels protected by
            the expanded main-source mask. True pixels indicate
            the replacement region.

        Notes
        -----
        If `seg_map` is None, no image modification is performed and
        an all-False mask is returned. If `source_pos` is None, the
        image center is used as the main-source position.
        """
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

    main_segment_label = get_main_source_label(seg_map, source_pos=source_pos)

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

def flatten_background(image, box_size=(16, 16), filter_size=(5, 5), src_detection_thresh=5, src_min_pixels=5,
                       src_detection_kernel=None, show_plots=False):
    """Estimate and subtract the background from a 2D astronomical image.

    Parameters
    ----------
    image : array-like
        Two-dimensional science image.
    box_size : int or tuple of int
        Background mesh size in pixels.
    filter_size : tuple of int
        Median filter size applied to the background mesh.
    src_detection_thresh : float
        Detection threshold in sigma units on the convolved image.
    src_min_pixels : int
        Minimum connected pixels required to identify a source.
    src_detection_kernel : astropy.convolution.Kernel2D, optional
        Kernel used to convolve the image for source detection.
        Defaults to Gaussian2DKernel(2.0).
    show_plots : bool
        Display background-subtraction diagnostic plots.

    Returns
    -------
    flattened_image : ndarray
        Original image minus the estimated background.
    bkg_image : ndarray
        background map.
    bkg_rms_image : ndarray
        background RMS map
    """

    if np.isscalar(box_size):
        if not isinstance(box_size, (int, np.integer)) or box_size < 2:
            raise ValueError("box_size must be an integer >= 2.")
    else:
        if len(box_size) != 2 or any(not isinstance(v, (int, np.integer)) or v < 2 for v in box_size):
            raise ValueError("box_size must contain two integers >= 2.")

    if src_detection_thresh <= 0:
        raise ValueError("src_detection_thresh must be positive.")

    if src_min_pixels < 1:
        raise ValueError("src_min_pixels must be >= 1.")

    if src_detection_kernel is None:
        src_detection_kernel = Gaussian2DKernel(2.0)

    invalid_mask = ~np.isfinite(image)

    image_convolved = convolve2d(image, src_detection_kernel, mode='same', boundary='symm')
    det_threshold = detect_threshold(image_convolved, nsigma=src_detection_thresh, mask=invalid_mask)
    seg_map = detect_sources(image_convolved, det_threshold, src_min_pixels, mask=invalid_mask, connectivity=8)

    if seg_map is None:
        src_mask = np.zeros(image.shape, dtype=bool)
    else:
        src_mask = seg_map.make_source_mask(size=1)

    background_mask = src_mask | invalid_mask

    valid_background = False
    percentile = 5
    while not valid_background:
        try:
            bkg = Background2D(image, box_size=box_size, mask=background_mask, filter_size=filter_size,
                               exclude_percentile=percentile,
                               bkg_estimator=SExtractorBackground(), bkg_rms_estimator=MADStdBackgroundRMS())
            valid_background = True
        except:
            if percentile >= 100:
                raise Exception("An error occurred while performing back-subtraction.")
            else:
                percentile += 5

    bkg_image = bkg.background
    bkg_rms_image = bkg.background_rms
    flattened_image = image - bkg.background

    if show_plots:
        import matplotlib.pyplot as plt
        plot_structure = [['Raw image', 'Detected sources', 'Background-subtracted image'],
                          ['Background', 'Background RMS', '.']]
        fig, axs = plt.subplot_mosaic(plot_structure, figsize=(10, 6), layout="tight")
        image_set = [image, background_mask, flattened_image, bkg_image, bkg_rms_image]

        for i, ax_name in enumerate(
                ['Raw image', 'Detected sources', 'Background-subtracted image', 'Background', 'Background RMS']):
            ax = axs[ax_name]
            plt_image = image_set[i]
            if "image" in ax_name:
                plt_image = plt_image - plt_image.min()
                ax.imshow(plt_image, cmap='inferno', origin='lower', norm="log")
            else:
                ax.imshow(plt_image, cmap='inferno', origin='lower')
            ax.set_title(ax_name)
            ax.axis('off')
        plt.show()

    return flattened_image, bkg_image, bkg_rms_image
