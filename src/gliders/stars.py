import numpy as np
from photutils.detection import DAOStarFinder, IRAFStarFinder, StarFinder
from photutils.segmentation import detect_threshold
from astropy.nddata import Cutout2D
from astropy.table import vstack
from astropy.convolution import Gaussian2DKernel
from photutils.aperture import CircularAperture
from .star_filter import star_detection_filtering
from .utils import interpolate_image, add_noise_to_region, contour_interpolation, replace_mask_with_noise, grow_mask, convolve_with_gaussian_kernel, has_false_neighbor, sigma_clipping, _estimate_overlapping_portion
import warnings

def detect_and_mask_stars(image, main_source_mask, config = None, main_source_pos = None, phot_sources = None, image_psf = None, pix_scale = 1,
                          verbose = False, show_plots = False):
    """
    Detect stars in an image and mask or interpolate their contributions.

    This function identifies stellar sources using the provided configuration
    and source information, classifies their apertures according to their
    overlap with the main source, and applies an appropriate masking or
    interpolation strategy to reduce stellar contamination.

    Star apertures are divided into three categories:

    - Outer stars: apertures that do not overlap with the main source.
      Their masked pixels are replaced with noise.
    - Edge stars: apertures that partially overlap with the main source.
      Their contaminated regions are handled by a dedicated edge-star
      masking procedure.
    - Inner stars: apertures that overlap completely with the main source.
      Their contributions are interpolated within the main-source mask.

    A Boolean mask tracks the pixels replaced by the outer-star and
    edge-star masking procedures. Pixels modified by the inner-star
    interpolation procedure are not added to this mask.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional input image containing the main source and
        contaminating stars. The image is progressively modified by the
        masking and interpolation operations.

    main_source_mask : numpy.ndarray
        Boolean mask identifying the pixels belonging to the main source.
        Used to determine the region to preserve and to classify stellar
        apertures according to their overlap with the source.

    config : object or None, optional
        Configuration object controlling star-aperture detection, safe
        masking regions, and interpolation. If ``None``, the default
        configuration is loaded from ``cleaning_configs.configs['default']``.

        The configuration is expected to provide the attributes required
        by the detection and masking routines, including
        ``auto_safe_masking_region``, ``safe_star_masking_prop_radius``,
        ``safe_star_masking_radius``, and ``interpolation_extent``.

        If no configuration is provided and ``verbose=True``, a
        ``UserWarning`` is issued before the default configuration is used.
        Default is ``None``.

    main_source_pos : object or None, optional
        Position information for the main source, passed to
        ``find_all_star_apertures`` to assist with star-aperture detection
        and selection. The expected format depends on that function's
        implementation. Default is ``None``.

    phot_sources : object or None, optional
        Photometric source information passed to ``find_all_star_apertures``
        to assist in identifying stellar apertures. Default is ``None``.

    image_psf : numpy.ndarray or None, optional
        Reference point spread function (PSF) image passed to
        ``find_all_star_apertures`` for use during star detection.
        Default is ``None``.

    pix_scale : float, optional
        Pixel-scale factor passed to the noise-replacement and masking
        interpolation routines. It controls their behavior according to
        the conventions implemented by those routines. Default is 1.

    verbose : bool, optional
        If ``True``, enable verbose behavior in the aperture-detection
        routine and issue a warning when the default configuration is
        selected because ``config`` is ``None``. Default is ``False``.

    show_plots : bool, optional
        If ``True``, display a plot of the detected apertures after they
        have been classified into outer, edge, and inner groups, but before
        masking and interpolation are applied. Default is ``False``.

    Returns
    -------
    image : numpy.ndarray
        The processed image after applying the relevant noise replacement,
        masking, and interpolation operations.

    replaced_mask : numpy.ndarray
        Boolean array with the same shape as ``image`` indicating pixels
        replaced by the outer-star and edge-star masking procedures.
        Pixels modified exclusively by the inner-star interpolation
        procedure are not included in this mask.

    all_star_apertures_split : object or None
        Classified stellar-aperture collection returned by
        ``classify_apertures``, containing the ``'outer'``, ``'edge'``,
        and ``'inner'`` groups.

        If no stellar apertures are detected, returns ``None`` instead.

    Notes
    -----
    The safe masking radius is determined from the area of the main-source
    mask when ``config.auto_safe_masking_region`` is enabled. Specifically,
    it is calculated as::

        round(sqrt(N * safe_star_masking_prop_radius / pi))

    where ``N`` is the number of nonzero pixels in ``main_source_mask``.
    The resulting radius is constrained to be at least one pixel.

    If automatic safe-region sizing is disabled, the radius specified by
    ``config.safe_star_masking_radius`` is used instead.

    The function first calls ``find_all_star_apertures`` with photometric
    sources enabled, source-finder detection disabled, and nucleus clearing
    enabled. If no apertures are returned, the original image, an all-False
    Boolean mask, and ``None`` are returned immediately.

    For outer stars, the aperture radii are increased by half their original
    radii, rounded upward, before constructing and growing the combined
    masking region. The masked pixels are then replaced with noise using
    ``replace_mask_with_noise`` with ``sigma_factor=3`` and median-based
    replacement enabled.

    Edge stars are handled by ``_mask_edge_stars``. Inner stars are handled
    by ``interpolate_apertures`` using the configured interpolation extent,
    a region factor of 0.25, and an aperture factor of 2.

    The input image may be modified in place or replaced by a new array,
    depending on the behavior of the called masking and interpolation
    functions.

    Notes
    -----
    This function depends on ``find_all_star_apertures``,
    ``classify_apertures``, ``grow_mask``,
    ``_elliptical_aperture_masking``, ``replace_mask_with_noise``,
    ``_mask_edge_stars``, ``interpolate_apertures``, and, when plotting is
    enabled, ``_plot_apertures``.
    """

    if config is None:
        if verbose:
            warnings.warn("No configuration object was provided. Will use default configuration.",
                          UserWarning, stacklevel=2)
        from .cleaning_configs import configs
        config = configs['default']

    main_source_npixels = np.sum(main_source_mask)
    if config.auto_safe_masking_region:
        safe_star_masking_radius = round(np.sqrt(main_source_npixels * config.safe_star_masking_prop_radius / np.pi))
        if safe_star_masking_radius < 1: safe_star_masking_radius = 1
    else:
        safe_star_masking_radius = config.safe_star_masking_radius

    all_star_apertures = find_all_star_apertures(image, config, phot_sources, safe_star_masking_radius, main_source_pos,
                                                 use_phot_sources = True, use_finders = False, clear_nucleus = True,
                                                 image_psf = image_psf, verbose=verbose)

    replaced_mask = np.full_like(image, False, dtype = bool)
    if all_star_apertures is None:
        return image, replaced_mask, all_star_apertures

    # Separate stars that are overlapping, at the edge and non overlapping with the galaxy
    all_star_apertures_split = classify_apertures(all_star_apertures, main_source_mask)

    if show_plots:
        _plot_apertures(image, all_star_apertures_split)

    # Masking the stars that do not overlap with the galaxy
    if len(all_star_apertures_split["outer"]) > 0:
        for ap_key in all_star_apertures_split["outer"]:
            ap_key.r += np.ceil(ap_key.r/2)
        outer_stars_mask = grow_mask(_elliptical_aperture_masking(image, all_star_apertures_split["outer"]),1)
        image = replace_mask_with_noise(image, outer_stars_mask, sigma_factor=3, use_median=True, pix_scale=pix_scale)
        replaced_mask = replaced_mask | outer_stars_mask

    # Interpolating/masking over the stars that overlap partially with the galaxy
    if len(all_star_apertures_split["edge"]) > 0:
        image, edge_replaced_mask = _mask_edge_stars(image, main_source_mask, all_star_apertures_split["edge"], config, pix_scale = pix_scale)
        replaced_mask = replaced_mask | edge_replaced_mask

    # Interpolating over the stars that overlap completely with the galaxy
    if len(all_star_apertures_split["inner"]) > 0:
        image = interpolate_apertures(image, main_source_mask, all_star_apertures_split["inner"], config.interpolation_extent,
                                      interpolation_region_factor=0.25, is_linear_image=True, pix_scale=pix_scale,
                                      aperture_factor = 2)
    return image, replaced_mask, all_star_apertures_split

def find_all_star_apertures(image, config = None, phot_sources = None, safe_star_masking_radius = 1,
                            main_source_pos = None, use_phot_sources = True, use_finders = False, clear_nucleus = True,
                            image_psf = None, verbose = False):
    """
    Detect stars in the image using both photometric sources and star finders, and return their apertures.

    Parameters:
    -----------
    image : numpy.ndarray
        Two-dimensional input image to be processed.
    main_source_pos : tuple, optional
        The (x, y) position of the main source (e.g., galaxy nucleus) in the image.
    config : Config, optional
        Configuration object containing parameters for star detection and filtering. If not given will use default configuration.
    phot_sources : astropy Table, optional
        Table containing photometric sources with their properties (id, x_centroid, y_centroid).
    safe_star_masking_radius : float, optional
        Radius within which masking is not allowed around the main source.
    use_phot_sources : bool, optional
        Whether to use photometric sources for star detection.
    use_finders : bool, optional
        Whether to use star finders for star detection.
    clear_nucleus : bool, optional
        Whether to remove apertures that overlap with the main source position.
    image_psf : numpy.ndarray, optional
        Point spread function (PSF) of the image, used for filtering detected stars.
    verbose : bool, optional
        Whether to print warnings and additional information during processing.

    Returns:
    --------
    all_star_apertures : dict
        Dictionary containing detected star apertures, with keys as 's_<id>' and values as CircularAperture objects.
    """

    if not use_phot_sources and not use_finders:
        if verbose:
            warnings.warn("Both the finders and photometric sources are turned off, so no stars will be detected.",
                          UserWarning, stacklevel=2)
        return None

    if config is None:
        if verbose:
            warnings.warn("No configuration object was provided. Will use default configuration.",
                          UserWarning, stacklevel=2)
        from .cleaning_configs import configs
        config = configs['default']

    all_stars_detected_list = []

    if phot_sources is not None and use_phot_sources:
        # We use the detections from the input use_phot_sources table
        stars_from_phot_sources = phot_sources[phot_sources['type']=='PSF']
        all_stars_detected_list.append(stars_from_phot_sources)

    if use_finders:
        # Detecting stars in the image
        convolved_image = image
        for this_std in config.star_psf_stds:
            if this_std > 0:
                # Instead of convolving the original image with a psf with this_std we use the previous convoluted
                # image and convolve it with effective_std, which is equivalent but results in roughly half
                # the processing time.
                effective_std = np.sqrt(this_std*this_std-np.power(this_std-1,2))
                convolved_image = convolve_with_gaussian_kernel(convolved_image, effective_std)

            det_fwhm = -0.31 + 12.45 * this_std # Empirically derived relation to maximize detections

            stars_detected = detect_stars_in_image(convolved_image, config.star_detection_threshold, det_fwhm,
                                                   config.star_detector, config.star_roundness,
                                                   config.star_sharpness_low, config.star_min_radius, config.sigma_radius)

            if stars_detected is not None:
                # Controlling that centroids are inside the image
                stars_detected['x_centroid'][stars_detected['x_centroid'] > image.shape[1] - 1] = image.shape[1] - 1
                stars_detected['x_centroid'][stars_detected['x_centroid'] < 0] = 0
                stars_detected['y_centroid'][stars_detected['y_centroid'] > image.shape[0] - 1] = image.shape[0] - 1
                stars_detected['y_centroid'][stars_detected['y_centroid'] < 0] = 0
                stars_detected['type'] = 'PSF'
                all_stars_detected_list.append(stars_detected)

    if len(all_stars_detected_list)==0:
        return None
    all_stars_detected = vstack(all_stars_detected_list, metadata_conflicts = 'silent')
    all_stars_detected['id'] = np.arange(1, len(all_stars_detected) + 1)
    all_stars_detected = _remove_redundant_star_detections(all_stars_detected, config.star_min_radius)

    filter_psf_std = config.star_filter_psf_std
    if image_psf is not None:
        psf_for_stars = convolve_with_gaussian_kernel(image_psf, filter_psf_std) # Useful to filter stars
    else:
        psf_for_stars = None

    # Filtering out bad detections and obtaining radii of stars
    all_stars_detected = star_detection_filtering(image=convolve_with_gaussian_kernel(image, filter_psf_std), stars_detected=all_stars_detected, min_radius=config.star_min_radius,
                                                  max_radius=config.star_max_radius, slope_threshold = config.star_filter_slope_thresh,
                                                  delta_slope_threshold = config.star_filter_delta_slope_thresh,
                                                  profile_cos_distance_threshold = config.star_filter_prof_cos_dist_thresh,
                                                  symmetry_ratio_threshold = config.star_filter_symmetry_ratio_thresh,
                                                  gaussian_ratio_threshold = config.star_filter_gaussian_ratio_thresh,
                                                  image_psf = psf_for_stars, gaussian_fit_max_sigma = config.star_filter_gaussian_fit_max_sigma,
                                                  do_parallel = False)

    if all_stars_detected is None:
        return None

    all_star_apertures = _get_star_apertures(all_stars_detected)

    # Removing any stars that overlap with the galaxy nucleus
    if clear_nucleus and main_source_pos is not None:
        all_star_apertures = _remove_nucleus_apertures(all_star_apertures, main_source_pos, safe_star_masking_radius)

    return all_star_apertures

def classify_apertures(all_star_apertures, main_source_mask):
    """
    Classify star apertures into three categories: inner, edge, and outer
    based on their overlap with the main source mask.

    Parameters:
    -----------
    all_star_apertures : dict
        Dictionary containing detected star apertures, with keys as 's_<id>' and values as CircularAperture objects.
    main_source_mask : numpy.ndarray
        Boolean mask of the main source (e.g., galaxy) in the image.

    Returns:
    --------
    all_star_apertures_split : dict
        Dictionary containing classified star apertures with keys 'inner', 'edge', and 'outer'.
    """

    if all_star_apertures is None:
        return None
    all_star_apertures_split = {"inner":[],"edge":[],"outer":[]}
    for ap_key in all_star_apertures.keys():
        ap_mask = all_star_apertures[ap_key].to_mask(method='exact').to_image(main_source_mask.shape,dtype=bool)
        if np.any(main_source_mask & ap_mask):
            union_mask = main_source_mask | ap_mask
            ap_mask_grown = grow_mask(ap_mask, int(np.ceil(all_star_apertures[ap_key].r/2)))
            if has_false_neighbor(union_mask, ap_mask_grown, connect_type=4):
                all_star_apertures_split["edge"].append(all_star_apertures[ap_key])
            else:
                all_star_apertures_split["inner"].append(all_star_apertures[ap_key])
        else:
            all_star_apertures_split["outer"].append(all_star_apertures[ap_key])

    return all_star_apertures_split


def interpolate_apertures(image, main_source_mask, apertures, interpolation_extent=3,
                          interpolation_region_factor=0.5, is_linear_image=True, pix_scale = 1, aperture_factor = 1):
    """
    Interpolate over the regions defined by the given apertures in the image, replacing them with interpolated values.

    Parameters:
    -----------
    image : numpy.ndarray
        Two-dimensional input image to be processed.
    main_source_mask : numpy.ndarray
        Boolean mask of the main source (e.g., galaxy) in the image.
    apertures : list
        List of CircularAperture objects defining the regions to be interpolated.
    interpolation_extent : int, optional
        The extent of the interpolation region around each aperture.
    interpolation_region_factor : float, optional
        Factor to determine the size of the region to be interpolated based on the aperture radius.
    is_linear_image : bool, optional
        Whether the image is linear (True) or logarithmic (False).
    pix_scale : float, optional
        Pixel scale of the image, used for noise estimation.
    aperture_factor : float, optional
        Factor to grow the aperture mask for interpolation.

    Returns:
    --------
    image_cp : numpy.ndarray
        The image with the specified apertures interpolated over.
    """

    if interpolation_extent < 2:
        interpolation_extent = 2
    image_cp = image.copy()
    image_background, image_background_std = sigma_clipping(image, iteration_limit = 100, use_median = True)
    if not is_linear_image:
        q_img = image[image <= image_background+image_background_std/2]
        image_background = np.median(q_img)
        image_background_std = np.std(q_img, mean=image_background)

    # Masking overlapping stars and interpolating over the region left empty
    for this_aperture in apertures:
        ap_mask = this_aperture.to_mask(method='exact').to_image(main_source_mask.shape, dtype=bool)
        sub_img = Cutout2D(image_cp, this_aperture.positions,
                           this_aperture.r * interpolation_extent, copy=True)
        sub_img_mask = Cutout2D(ap_mask, this_aperture.positions,
                                this_aperture.r * interpolation_extent, copy=True)
        to_grow = int(round(this_aperture.r*interpolation_region_factor))

        sub_img_mask_grown = grow_mask(sub_img_mask.data, to_grow*aperture_factor)
        interpolated_img = interpolate_image(sub_img.data, sub_img_mask_grown,
                                             np.max((to_grow,1.)), method = "inpaint")
        interpolated_img = add_noise_to_region(interpolated_img, sub_img_mask_grown, noise_center= 0, noise_std=image_background_std, scale=pix_scale)
        image_cp[sub_img.ymin_original:sub_img.ymax_original + 1, sub_img.xmin_original:sub_img.xmax_original + 1] = interpolated_img

    return image_cp

def detect_stars_in_image(image, detection_thresh_sigma, star_fwhm, star_detector = None, star_roundness = 0.5,
                          star_sharpness_low = 0.2, star_min_radius = None, sigma_radius = 1.5):
    """
    Detect stars in the image using the specified star detection method and parameters.

    Parameters:
    -----------
    image : numpy.ndarray
        Two-dimensional input image to be processed.
    detection_thresh_sigma: float
        The number of standard deviations above the background to use as the detection threshold.
    star_fwhm: float
        The full width at half maximum of the stars to detect.
    star_detector: str
        The star detection method to use ('dao', 'iraf', or None for StarFinder).
    star_roundness: float
        The roundness parameter for the star detection algorithm.
    star_sharpness_low: float
        The lower bound for the sharpness parameter for the star detection algorithm.
    star_min_radius: float
        The minimum separation between detected stars.
    sigma_radius: float
        The radius in pixels for the Gaussian kernel used in the star detection algorithm.

    Returns:
    --------
    stars_detected: astropy Table
        Containing the detected stars with their properties (id, x_centroid, y_centroid).
    """


    detection_threshold = detect_threshold(image, nsigma=detection_thresh_sigma)[0][0]
    roundness_range = (-star_roundness,star_roundness)
    sharpness_range = (star_sharpness_low,1.0)
    if star_detector == "dao":
        finder = DAOStarFinder(threshold=detection_threshold, fwhm=star_fwhm, sigma_radius=sigma_radius,
                               roundness_range = roundness_range, sharpness_range=sharpness_range,
                               min_separation = star_min_radius)
    elif star_detector == "iraf":
        finder = IRAFStarFinder(threshold=detection_threshold, fwhm=star_fwhm, sigma_radius=sigma_radius,
                                roundness_range = roundness_range, sharpness_range=sharpness_range,
                                min_separation = star_min_radius)
    else:
        star_kernel = Gaussian2DKernel(2.0)
        finder = StarFinder(threshold=detection_threshold, kernel=star_kernel, min_separation=star_min_radius)

    stars_detected = finder(image)
    if stars_detected is not None:
        stars_detected = stars_detected['id', 'x_centroid', 'y_centroid']
    return stars_detected

def _plot_apertures(image, all_star_apertures):
    """
    Plot the detected star apertures on the image.
    """
    import matplotlib.pyplot as plt
    plt.imshow(image + np.abs(np.min(image)),norm = 'log', origin = "lower", cmap = "grey")
    plt.title('Detected stars')
    for ap_grp in all_star_apertures.items():
        if ap_grp[0] == "outer":
            color = "red"
        elif ap_grp[0] == "edge":
            color = "blue"
        else:
            color = "limegreen"
        ap_lbl = True
        for ap_obj in ap_grp[1]:
            if ap_lbl:
                label = ap_grp[0]
                ap_lbl = False
            else:
                label = None
            ap_obj.plot(color=color, lw=1, label = label)
    plt.legend()
    plt.show()

def _get_star_apertures(stars_detected, default_radius = 2):
    """
    Create circular apertures for the detected stars based on their centroids and radii.
    """

    if stars_detected is None:
        return stars_detected
    detected_star_apertures={}
    current_id = 1
    if 'r' not in stars_detected.colnames:
        use_default_radius = True
    else:
        use_default_radius = False
    for this_star in stars_detected:
        if use_default_radius:
            radius = default_radius
        else:
            radius = this_star['r']
        star_ap = CircularAperture((this_star['x_centroid'], this_star['y_centroid']), r=radius)
        detected_star_apertures['s_' + str(current_id)] = star_ap
        current_id += 1
    return detected_star_apertures

def _remove_redundant_star_detections(all_stars_detected, min_separation = 1.0):
    """
    Remove redundant star detections that are too close to each other based on the specified minimum separation

    Parameters:
    -----------
    all_stars_detected : astropy Table
        Table containing detected stars with their properties (id, x_centroid, y_centroid).
    min_separation : float
        Minimum separation distance between stars to consider them as distinct detections.

    Returns:
    --------
    astropy Table
        Filtered table containing only distinct star detections based on the minimum separation criteria.
    """

    ids_to_keep = []
    positions = np.array([[star['x_centroid'], star['y_centroid']] for star in all_stars_detected])
    ids = np.array([star['id'] for star in all_stars_detected])

    for i, pos_i in enumerate(positions):
        keep = True
        for kept_id in ids_to_keep:
            j = np.where(ids == kept_id)[0][0]
            pos_j = positions[j]
            dist = np.linalg.norm(pos_j - pos_i)
            if dist < min_separation:
                keep = False
                break
        if keep:
            ids_to_keep.append(ids[i])

    id_mask = np.isin(ids, ids_to_keep)
    return all_stars_detected[id_mask]

def _remove_nucleus_apertures(apertures, source_pos, safe_radius=1):
    """
    Remove apertures that are too close to the source position (likely the nucleus) based on a safe radius.
    """
    if apertures is None:
        return apertures
    to_remove = []
    filtered_apertures = apertures.copy()
    if not isinstance(apertures, type(None)):
        for this_apertue in apertures.items():
            d_x_src = this_apertue[1].positions[0] - source_pos[0]
            d_y_src = this_apertue[1].positions[1] - source_pos[1]
            safe_dist = safe_radius + this_apertue[1].r / 2
            if np.hypot(d_x_src, d_y_src) < safe_dist:
                to_remove.append(this_apertue[0])
        filtered_apertures = {k: v for k, v in apertures.items() if k not in to_remove}

    return filtered_apertures

def _mask_edge_stars(image,main_source_mask,all_star_apertures, config, pix_scale = 1):
    """
    Mask stars that are overlapping with the galaxy edge. Depending on
    the overlapping portion, either replace the star region with noise or
    perform contour interpolation.
    """
    edge_replaced_mask = np.full_like(main_source_mask, False, dtype = bool)
    background, background_std = sigma_clipping(image, 3)
    ed_main_lbl = 1
    lbl_count = 2
    label_map = np.zeros_like(image)
    label_map[main_source_mask] = ed_main_lbl
    for ed_ap in all_star_apertures:
        ed_mask = ed_ap.to_mask(method='exact').to_image(label_map.shape, dtype=bool)
        ed_mask = grow_mask(ed_mask, int(np.ceil(ed_ap.r/2)))
        label_map[ed_mask] = lbl_count
        overlapping_portion = _estimate_overlapping_portion(label_map, ed_main_lbl, lbl_count)
        if overlapping_portion < config.overlap_portion_threshold:
            image = replace_mask_with_noise(image, ed_mask, n_pixels=round(ed_ap.r / 2), sigma_factor=5,
                                            pix_scale=pix_scale,
                                            use_median=False, edge_smoothing_factor=0.4)
            edge_replaced_mask = edge_replaced_mask | ed_mask

        else:
            target_pix_number = np.sum(ed_mask)
            mask_radius = round(np.sqrt(target_pix_number) / 2)
            influence_radius = mask_radius * overlapping_portion * 2
            image = replace_mask_with_noise(image, ed_mask, sigma_factor=5, pix_scale=pix_scale)
            layer_noise_sample = np.random.normal(loc=background, scale=background_std,
                                                  size=np.sum(ed_mask))
            image = contour_interpolation(ed_mask, image, connectivity=8,
                                          noise_sample=layer_noise_sample, decay_threshold=0.1,
                                          influence_radius=influence_radius)
        lbl_count += 1

    return image, edge_replaced_mask

def star_contamination_check(main_source_mask, all_star_apertures, safe_pix_ratio=0.03, verbose=True):
    """
    Check if any of the detected stars are too large compared to the main source, which might indicate contamination.

    Parameters:
    -----------
    main_source_mask : numpy.ndarray
        Boolean mask of the main source.
    all_star_apertures : dict
        Dictionary containing star apertures classified into 'inner', 'edge', and 'outer'.
    safe_pix_ratio : float
        The maximum allowed ratio of star pixels to main source pixels before issuing a warning.
    verbose : bool
        If True, warnings will be printed when a large star is detected.

    Returns:
    --------
    bool
        True if a large star is detected that might result in bad masking, False otherwise.
    """

    if all_star_apertures is None:
        return False
    main_source_px = np.sum(main_source_mask)
    for star_group in ["inner", "edge"]:
        for this_ap in all_star_apertures[star_group]:
            ap_mask = this_ap.to_mask(method='exact').to_image(main_source_mask.shape, dtype=bool)
            size_ratio = np.sum(ap_mask) / main_source_px
            if size_ratio > safe_pix_ratio:
                if verbose:
                    warnings.warn("Detected a big star that might result in bad masking.", UserWarning,
                                  stacklevel=2)
                return True
    return False

def _elliptical_aperture_masking(image, all_apertures):
    """
    Create a boolean mask for the image based on the provided elliptical apertures.
    """
    output_mask = np.full_like(image, False, dtype=bool)
    for this_aperture in all_apertures:
        ap_mask = this_aperture.to_mask(method='exact')
        bool_mask = ap_mask.to_image(image.shape, dtype=bool)
        output_mask = output_mask | bool_mask
    return output_mask
