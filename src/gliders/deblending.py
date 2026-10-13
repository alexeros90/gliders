import numpy as np
from photutils.segmentation import SourceCatalog, deblend_sources, detect_threshold, SegmentationImage
from scipy.ndimage import generate_binary_structure, binary_dilation
import sep
import warnings
from .utils import (get_main_source_label, separate_segments, get_image_centroid, sigma_clipping, grow_mask,
                    replace_mask_with_noise, interpolate_image, add_noise_to_region, estimate_overlapping_portion,
                    contour_interpolation, smooth_region_edge, get_z_sources_in_image)


def deblend_and_mask_sources(image, image_filtered = None, seg_map = None, source_pos = None, config = None,
                             use_redshift = True, ref_redshift = None, max_delta_z = 0.005, z_sources = None, query_z_sources = True,
                             wcs = None, z_query_timeout = 60, z_query_synch = True, use_photo_type = True, phot_sources = None,
                             keep_undecided_segments = True, pix_scale=1, show_plots=False, verbose=False):
    """
    Deblend overlapping sources and remove contaminating objects from an image.

    This function identifies the main source in a segmentation map based on its
    provided position, performs source deblending, evaluates the resulting segments
    using redshift and/or photometric information, and removes segments that are
    determined not to belong to the target object. A boolean mask of all removed
    pixels is also returned.

    Parameters
    ----------
    image : ndarray
        Two-dimensional image containing the target source and surrounding
        objects.

    image_filtered : ndarray, optional
        Filtered or convolved version of ``image`` used for source deblending.
        If None, the input ``image`` is used.

    seg_map : photutils.segmentation.SegmentationImage, optional
        Segmentation map identifying detected sources in the image. If None,
        the input image is returned unchanged.

    source_pos : tuple of int, optional
        Pixel coordinates ``(x, y)`` of the target source. If None, the image
        center is assumed.

    config : object, optional
        Configuration object containing deblending and source-removal
        parameters. If None, the default cleaning configuration is used.

    use_redshift : bool, default=True
        Whether spectroscopic/photometric redshift information should be used
        to determine whether deblended segments belong to the main source.

    ref_redshift : float, optional
        Reference redshift of the target object. If provided, candidate sources
        with redshifts differing by more than ``max_delta_z`` may be removed.

    max_delta_z : float, default=0.005
        Maximum allowed redshift difference between a segment and the target
        source for them to be considered associated.

    z_sources : astropy.table.Table, optional
        Table containing redshift information for sources in the image. If
        ``query_z_sources=True``, this parameter is overwritten by the query
        results.

    query_z_sources : bool, default=True
        Whether to perform an external redshift query for sources detected in
        the deblended segmentation map.

    wcs : astropy.wcs.WCS, optional
        World Coordinate System used to convert source positions into sky
        coordinates for redshift catalog queries.

    z_query_timeout : int, default=60
        Timeout, in seconds, for redshift catalog queries.

    z_query_synch : bool, default=True
        If True, execute redshift queries synchronously. Otherwise, use the
        asynchronous query mode supported by the underlying query function.

    use_photo_type : bool, default=True
        Whether to use photometric object classifications for evaluating
        deblended segments. Artifacts (ATF) and stars (PSF) type objects
        will be deblended.

    phot_sources : astropy.table.Table, optional
        Table containing photometric classifications for image sources.

    keep_undecided_segments : bool, default=True
        If True, segments for which insufficient information is available to
        determine association with the main source are retained.

    pix_scale : float, default=1
        The scale of the artificial noise pixels added when removing contaminating
        sources. If the image is correctly sampled w.r.t. the resolution, this value
        should always be 1.

    show_plots : bool, default=False
        If True, display diagnostic plots generated during the analysis of
        deblended segments.

    verbose : bool, default=False
        If True, print status messages and issue warnings describing the
        processing steps and any fallback behavior.

    Returns
    -------
    image : ndarray
        Image after removal of contaminating deblended sources. If no valid
        deblending is possible, the original image is returned.

    deblending_mask : ndarray of bool
        Boolean mask marking pixels belonging to sources that were removed
        during the deblending process.

    deblended_flag : bool
        True if at least one contaminating source was successfully identified
        and removed. False otherwise.

    failed_z_query : bool
        True if a redshift query was attempted but failed. False otherwise.

    Notes
    -----
    The main source is identified as the segment associated with
    ``source_pos``. De-blending is performed using the method specified in
    ``config.deblend_tool`` and related configuration parameters.

    Candidate segments can be evaluated using:

    * Redshift consistency with the target source.
    * Photometric source classifications.
    * Geometric overlap criteria used during source removal.

    If no segmentation map is provided, or the deblending process produces only
    a single segment, the input image is returned unchanged.

    See Also
    --------
    get_deblended_map
        Generate a deblended segmentation map.

    analyze_deblended_map
        Classify deblended segments as associated or contaminating.

    remove_deblended_objects
        Remove contaminating sources from an image.
    """

    deblended_flag = False
    failed_z_query = False
    deblending_mask = np.full_like(image, False, dtype=bool)
    default_return = image, seg_map, deblending_mask, deblended_flag, failed_z_query
    if image_filtered is None:
        if verbose:
            warnings.warn("Invalid image_filtered. Will use raw image for deblending.",
                          UserWarning, stacklevel=2)
        image_filtered = image
    if seg_map is None:
        if verbose:
            warnings.warn("Invalid seg_map, returning unmodified image and default values.",
                          UserWarning, stacklevel = 2)
        return default_return
    if config is None:
        if verbose:
            warnings.warn("No configuration object was provided. Will use default configuration.",
                          UserWarning, stacklevel=2)
        from .cleaning_configs import configs
        config = configs['default']
    if source_pos is None:
        if verbose:
            warnings.warn("No main source position provided. Assuming main source is at image center.",
                      UserWarning, stacklevel=2)
        ny, nx = image.shape
        source_pos = (nx // 2, ny // 2)

    src_cat = SourceCatalog(image, seg_map)
    main_src_label = get_main_source_label(seg_map, source_pos=source_pos, table=src_cat.to_table())
    main_source_npixels = seg_map.get_area(main_src_label)

    # Determining the minimum number of pixels for de-blending
    npixels_deblend = max(int(main_source_npixels * config.npixels_deblend_factor), 1)

    # Getting the deblended candidates. We use the convolved image for better de-blending edges (especially if deblend_tool='sep').
    seg_map_deblend = get_deblended_map(image_filtered, seg_map, config.deblend_tool, npixels_deblend,
                                        config.deblend_contrast, config.deblend_levels, config.deblend_mode,
                                        config.connect_mode, config.galaxy_detection_threshold)
    if seg_map_deblend is None:
        if verbose:
            warnings.warn("No sources detected after de-blending, returning unmodified image and default values.",
                          UserWarning, stacklevel=2)
        return default_return

    elif len(seg_map_deblend.labels) > 1:
        # Segments may vary after deblending, so main source label has to be obtained again
        deb_cat = SourceCatalog(image, seg_map_deblend)
        main_label = get_main_source_label(seg_map_deblend, source_pos=source_pos, table=deb_cat.to_table())

        if use_redshift and query_z_sources:
            if wcs is None:
                if verbose:
                    warnings.warn("Cannot perform redshift query: The provided World Coordinate System (WCS) object is not valid",
                              UserWarning, stacklevel=2)
            else:
                if verbose:
                    print("Looking for nearby sources with available redshift...")
                # z_sources query overrides the current content of z_sources
                z_sources, failed_z_query = get_z_sources_in_image(seg_map_deblend, wcs, z_query_timeout, z_query_synch, z_query_attempts=2)

        seg_map_deblend = analyze_deblended_map(seg_map_deblend, main_label, use_redshift, ref_redshift,
                                                max_delta_z, z_sources, use_photo_type, phot_sources,
                                                keep_undecided_segments, show_plots, verbose)

        if len(seg_map_deblend.labels) > 1:
            image = remove_deblended_objects(image, seg_map_deblend, main_label,
                                             overlap_portion_threshold=config.overlap_portion_threshold,
                                             pix_scale=pix_scale)
            deblended_flag = True
            removal_segmap = seg_map_deblend.copy()
            removal_segmap.remove_label(main_label)
            deblending_mask = removal_segmap.make_source_mask()

    if verbose:
        print("Detected", len(seg_map_deblend.labels), "sources after de-blending.")

    return image, seg_map_deblend, deblending_mask, deblended_flag, failed_z_query


def get_deblended_map(image, seg_map, deblend_tool = 'sep', npixels = 50, contrast = 0.005, levels = 32,
                      deblend_mode = 'exponential', connect_mode = 8, detection_threshold = 5, verbose = False):
    """
    Deblend a segmentation map using either SEP or Photutils.

    Parameters
    ----------
    image : ndarray
        Input image used for source deblending.
    seg_map : photutils.segmentation.SegmentationImage
        Input segmentation map.
    deblend_tool : {'sep', 'phut'}, optional
        Deblending backend to use. Defaults to ``'sep'``.
    npixels : int, optional
        Minimum number of connected pixels to be considered an
        independent source.
    contrast : float, optional
        Minimum contrast ratio used to separate blended sources.
        Smaller values result in more aggressive deblending.
    levels : int, optional
        Number of deblending thresholds.
    deblend_mode : {'exponential', 'linear', 'sinh'}, optional
        Multi-threshold spacing mode used by Photutils.
        Ignored when ``deblend_tool='sep'``.
    connect_mode : {4, 8}, optional
        Pixel connectivity used by Photutils during deblending.
        Ignored when ``deblend_tool='sep'``.
    detection_threshold : float, optional
        Detection threshold in units of the background noise level
        (sigma). Only required when ``deblend_tool='sep'``.
    verbose : bool, optional
        Whether to display warnings and diagnostic messages.

    Returns
    -------
    photutils.segmentation.SegmentationImage
        Deblended segmentation map. If deblending fails, the input
        segmentation map is returned unchanged.

    Notes
    -----
    To maintain consistent results when using 'sep', if a filtered
    image was used for the initial detection, the same filtered image and
    detection_threshold should be used here.

    For the SEP backend, the deblended segmentation map is constrained to
    the footprint of the input segmentation map and is post-processed with
    ``separate_segments()`` to remove fuzzy boundaries between labels.
    """

    if deblend_tool not in ["sep", "phut"]:
        if verbose:
            warnings.warn("Not a valid deblend_tool. Defaulting to 'sep'", UserWarning, stacklevel=2)
        deblend_tool = "sep"

    if deblend_tool == "phut":
        seg_map_deb = deblend_sources(image, seg_map, npixels=npixels, nlevels=levels, contrast=contrast,
                                      progress_bar=False, mode=deblend_mode, connectivity=connect_mode)

    elif deblend_tool == "sep":
        threshold = detect_threshold(image, n_sigma=detection_threshold)
        try:
            _, seg_map_deb_data = sep.extract(image.astype(np.float32), thresh=threshold.mean(), segmentation_map=True,
                                              minarea=npixels, deblend_nthresh=levels, deblend_cont=contrast)
            if seg_map is not None:
                seg_map_mask = seg_map.make_source_mask()
                seg_map_deb_data[~seg_map_mask] = 0.
            seg_map_deb_data = separate_segments(seg_map_deb_data)
            seg_map_deb = SegmentationImage(seg_map_deb_data)
        except Exception as e:
            print(f"Could not perform deblending: {e}")
            return seg_map

    return seg_map_deb

def analyze_deblended_map(seg_map, main_label, use_redshift = True, redshift = None, max_delta_z = None,
                          z_sources = None, use_photo_type = True, phot_sources = None, keep_undecided_labels = True,
                          show_plots = False, verbose = False):

    """
    Analyze a deblended segmentation map using redshift and photometric
    source information.

    Segments can be classified as safe or discarded based on nearby
    redshifts and/or photometric source types (where redshift classification
    has priority). Remaining undecided segments are either kept or discarded
    according to ``keep_undecided_labels``. Safe segments touching the main
    source are merged back into the final segmentation map.

    Parameters
    ----------
    seg_map : photutils.segmentation.SegmentationImage
        Deblended segmentation map.
    main_label : int
        Label of the main source.
    use_redshift : bool, optional
        Enable redshift-based classification.
    redshift : float, optional
        Redshift of the main source.
    max_delta_z : float, optional
        Maximum allowed redshift difference for association.
    z_sources : astropy.table.Table, optional
        Table of sources with spectroscopic redshifts in the image region.
    use_photo_type : bool, optional
        Enable photometric type-based classification.
    phot_sources : astropy.table.Table, optional
        Table of photometric source types in the image region.
    keep_undecided_labels : bool, optional
        Whether labels not classified by any criterion are retained.

    Returns
    -------
    photutils.segmentation.SegmentationImage
        Final cleaned segmentation map.
    """

    if seg_map is None:
        warnings.warn("Could not perform deblending: invalid segmentation map",
                      UserWarning, stacklevel=2)
        return seg_map

    safe_label_list = []
    discard_label_list = []

    if use_redshift:
        if redshift is None:
            if verbose:
                warnings.warn("Omitting the redshift-based deblending: The provided redshift is not valid.",
                              UserWarning, stacklevel=2)
        elif max_delta_z is None:
            if verbose:
                warnings.warn("Omitting the redshift-based deblending: The provided max_delta_z is not valid.",
                              UserWarning, stacklevel=2)
        elif z_sources is None or (len(z_sources) < 1):
            if verbose:
                warnings.warn("Omitting the redshift-based deblending: The provided z_sources table appears to be empty.",
                              UserWarning, stacklevel=2)
        else:
            safe_label_list, discard_label_list = _check_deblended_redshifts(seg_map, main_label, redshift,
                                                                             max_delta_z, z_sources, safe_label_list,
                                                                             discard_label_list, show_plots=False,
                                                                             verbose=verbose)

    else:
        if verbose:
            print("Redshift-based deblending is not activated")

    safe_label_list_z, discard_label_list_z = safe_label_list.copy(), discard_label_list.copy()

    if use_photo_type:
        if phot_sources is None or (len(phot_sources) < 1):
            if verbose:
                warnings.warn("Omitting the photometric type-based deblending: The provided phot_sources table appears to be empty.",
                              UserWarning, stacklevel=2)
        else:
            safe_label_list, discard_label_list = _check_deblended_photo_sources(seg_map, main_label, phot_sources, safe_label_list,
                                                         discard_label_list, show_plots = False, verbose = verbose)
    else:
        if verbose:
            print("Photometric type-based deblending is not activated")

    safe_label_list_phot = [label for label in safe_label_list if label not in safe_label_list_z]
    discard_label_list_phot = [label for label in discard_label_list if label not in discard_label_list_z]

    exclude = [main_label] + safe_label_list + discard_label_list

    undecided_labels = [
        label for label in seg_map.labels
        if label not in exclude
    ]

    if len(undecided_labels) > 0:
        if keep_undecided_labels:
            safe_label_list += undecided_labels
        else:
            discard_label_list += undecided_labels

    # Ultimately, only safe segments that are in contact with the main segment are kept.
    seg_map_out = seg_map.copy()
    seg_map_out = _merge_touching_segments(seg_map_out, main_label, safe_label_list)

    if show_plots:
        safe_label = int(np.max(seg_map_out.labels) + 1)
        discard_label = safe_label + 1
        seg_map_z = seg_map.copy()
        if safe_label_list_z:
            seg_map_z.reassign_labels(labels=safe_label_list_z, new_label=safe_label)
        if discard_label_list_z:
            seg_map_z.reassign_labels(labels=discard_label_list_z, new_label=discard_label)
        seg_map_phot = seg_map_z.copy()
        if safe_label_list_phot:
            seg_map_phot.reassign_labels(labels=safe_label_list_phot, new_label=safe_label)
        if discard_label_list_phot:
            seg_map_phot.reassign_labels(labels=discard_label_list_phot, new_label=discard_label)

        _plot_deblended_maps(seg_map, seg_map_z, seg_map_phot, seg_map_out, redshift, z_sources, phot_sources)

    return seg_map_out

def remove_deblended_objects(image, seg_map, main_label, overlap_portion_threshold = 0.1, mask_radius_factor=1.0, pix_scale = 1):
    """
    Remove secondary/deblended sources from an astronomical image.

    Identifies all segmented sources other than the specified main source,
    processes them in order of decreasing distance from the main source
    centroid, and replaces their contribution in the image with background
    noise and/or interpolated values.

    Secondary sources are handled differently depending on their spatial
    relationship with the main source:

    - Sources that do not touch the main source are removed by replacing
      their mask with background noise.
    - Sources fully contained within the main source are inpainted and
      filled with background noise.
    - Sources that overlap the main source are treated according to their
      estimated overlapping fraction. Weakly overlapping sources are
      removed and the resulting edge is smoothed, while strongly
      overlapping sources are removed using contour-based interpolation.

    The size of the mask used to remove each secondary source is estimated
    from its number of pixels using an empirical relation calibrated for
    DESI Legacy Survey images. The resulting radius can be scaled with
    ``mask_radius_factor``.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional astronomical image containing the sources to be
        removed.

    seg_map : photutils.segmentation.SegmentationImage
        Segmentation map identifying the sources in ``image``. Source labels
        are expected to be positive integers, with background pixels having
        no source label.

    main_label : int
        Label identifying the primary source that should be preserved.
        Secondary sources are removed relative to this source.

    overlap_portion_threshold : float, optional
        Threshold on the estimated overlapping fraction used to distinguish
        weakly from strongly overlapping secondary sources. Sources with
        an overlapping portion smaller than this value are treated as
        weakly overlapping. Default is 0.1.

    mask_radius_factor : float, optional
        Multiplicative factor applied to the empirically estimated mask
        radius used when growing each secondary-source mask. A value of
        ``1`` uses the nominal calibrated radius. Values greater than or
        less than ``1`` increase or decrease the size of the removal
        region, respectively. The resulting radius is always at least
        one pixel. Default is 1.

    pix_scale : float, optional
        The scale of the artificial noise pixels added when removing contaminating
        sources. If the image is correctly sampled w.r.t. the resolution, this value
        should always be 1.

    Returns
    -------
    numpy.ndarray
        The image after removal of the secondary/deblended sources.

    Notes
    -----
    The secondary sources are processed from the furthest to the closest
    relative to the centroid of the main source. This ordering is intended
    to ensure that more distant sources are removed before sources that
    are closer to, or overlapping with, the main source.

    The mask radius for each secondary source is calculated from its number
    of segmented pixels using the empirical relation

    ``r = 0.7419 + 0.0066 * N_pix``,

    which was calibrated for DESI Legacy Survey images. The radius is then
    multiplied by ``mask_radius_factor`` and rounded to the nearest integer.

    """

    all_sources_mask = seg_map.make_source_mask()

    catalog = SourceCatalog(image, seg_map)

    sec_labels = seg_map.labels
    sec_labels = sec_labels[sec_labels != main_label]
    main_source_mask = seg_map.data == main_label

    # First we sort the objects by proximity to main source (the furthest first)
    main_source_segment = catalog.select_label(main_label)
    main_source_centroid = main_source_segment.centroid
    distances_to_main_centroid = []
    for source_segment in catalog:
        if source_segment.label == main_label:
            continue
        sec_centroid = source_segment.centroid
        distance = np.hypot(sec_centroid[0] - main_source_centroid[0], sec_centroid[1] - main_source_centroid[1])
        distances_to_main_centroid.append(distance)
    distances_to_main_centroid = np.array(distances_to_main_centroid)
    sort_idx = np.argsort(distances_to_main_centroid)[::-1]
    sec_labels = sec_labels[sort_idx]

    # Now we treat each object
    background_lvl, background_std = sigma_clipping(image, 3)
    for this_source_label in sec_labels:

        this_source_mask = seg_map.data == this_source_label

        all_but_this_source_mask = all_sources_mask & ~this_source_mask
        target_area = np.sum(this_source_mask)

        # mask_radius_cal = 0.74 + 0.0066 * target_area
        mask_radius_cal = -2.93 + 0.35 * np.sqrt(target_area) # Approx. radius for source profile to reach background + sigma level. Calibrated for DESI Legacy images
        mask_radius = round(mask_radius_cal * mask_radius_factor)
        mask_radius = max(1, mask_radius)
        # print(mask_radius)

        # Grow the region making sure it does not invade any other regions besides the background
        this_source_mask_adjusted = grow_mask(this_source_mask, mask_radius) & ~all_but_this_source_mask
        in_contact_with_main = _are_regions_in_contact(seg_map.data, main_label, this_source_label)

        if not in_contact_with_main:
            image = replace_mask_with_noise(image, this_source_mask_adjusted, n_pixels=mask_radius, scale=pix_scale)
            continue
        secondary_is_contained = _check_if_region_is_contained(seg_map.data, this_source_label, main_label)
        if secondary_is_contained:
            image = interpolate_image(image, this_source_mask, method = "inpaint")
            image = add_noise_to_region(image, this_source_mask, noise_center = 0, noise_std = background_std,
                                        scale=pix_scale)
        else:
            overlapping_portion = estimate_overlapping_portion(seg_map.data, main_label, this_source_label)
            if overlapping_portion < overlap_portion_threshold:
                image = replace_mask_with_noise(image, this_source_mask_adjusted, n_pixels=mask_radius, sigma_factor=5,
                                                scale = pix_scale, edge_smoothing_factor = 0)
                smoothing_layers = round(np.sqrt(np.sum(main_source_mask)) / (2*np.pi))
                image = smooth_region_edge(image, main_source_mask, this_source_mask_adjusted, smoothing_layers,
                                           background_lvl, background_std, pix_scale = pix_scale)

            else:
                influence_radius = mask_radius * overlapping_portion
                image = replace_mask_with_noise(image, this_source_mask_adjusted, sigma_factor=5, scale = pix_scale,
                                                edge_smoothing_factor = 0.5)
                layer_noise_sample = np.random.normal(loc=background_lvl, scale=background_std,
                                                      size=np.sum(this_source_mask_adjusted))
                image = contour_interpolation(this_source_mask_adjusted, image, connectivity=8,
                                              noise_sample=layer_noise_sample, decay_threshold=0.15,
                                              influence_radius=influence_radius)

    return image

def _check_deblended_redshifts(seg_map, main_label, redshift, max_delta_z, nearby_z_sources = None,
                               safe_labels = None, discard_labels = None, show_plots = False, verbose = False):
    """
    Classify deblended segments using nearby redshift information.

    Segments containing at least one source within ``max_delta_z`` of the
    target redshift are assigned the ``safe_label``. Segments containing only
    incompatible redshifts are assigned the ``discard_label``.

    Parameters
    ----------
    seg_map : photutils.segmentation.SegmentationImage
        Input segmentation map
    main_label : int
        The label of the primary object
    redshift : float
        Target redshift of the main source.
    max_delta_z : float
        Maximum allowed redshift difference for association.
    nearby_z_sources: astropy.table.Table, optional
        Table of sources with spectroscopic redshifts in the image region
    safe_labels : list
        The list of labels that are already classified as "safe" (these labels will not be affected by this function)
    discard_labels : list
        The list of labels that are already classified as "discard" (these labels will not be affected by this function)

    Returns
    -------
    list, list
        Updated safe_labels and discard_labels lists.
    """

    if safe_labels is None:
        safe_labels = []
    if discard_labels is None:
        discard_labels = []

    non_main_segmap = seg_map.copy()
    non_main_segmap.remove_label(main_label)

    if not len(non_main_segmap.labels) > 0:
        if verbose:
            print("No labels found besides main. Skipping redshift check for deblending")
        return seg_map
    if nearby_z_sources is None or (len(nearby_z_sources) < 1):
        if verbose:
            warnings.warn("List of nearby sources appears to be empty.", UserWarning, stacklevel=2)
        return safe_labels, discard_labels

    image_h, image_w = seg_map.data.shape

    nearby_z_sources_pd = nearby_z_sources.to_pandas()
    zs_x = np.clip(np.rint(nearby_z_sources_pd['x_pos'].values).astype(int), 0, image_w - 1)
    zs_y = np.clip(np.rint(nearby_z_sources_pd['y_pos'].values).astype(int), 0, image_h - 1)
    zs_z = nearby_z_sources_pd['z'].values
    zs_plot = {}

    for seg_label in non_main_segmap.labels:
        if seg_label in discard_labels or seg_label in safe_labels:
            continue
        seg_mask = non_main_segmap.data == seg_label
        inside = seg_mask[zs_y, zs_x]
        if np.any(inside):
            z_seps = np.abs(zs_z[inside] - redshift)
            if np.any(z_seps <= max_delta_z):
                safe_labels.append(seg_label)
            else:
                discard_labels.append(seg_label)

            zs_plot[seg_label] = {"pos_x": zs_x[inside],
                                  "pos_y": zs_y[inside],
                                  "z": np.round(zs_z[inside],4)}

    if show_plots:
        import matplotlib.pyplot as plt
        seg_map_plot = seg_map.copy()
        safe_label = int(np.max(seg_map_plot.labels) + 1)
        discard_label = safe_label + 1

        if len(safe_labels) > 0:
            seg_map_plot.reassign_labels(labels=safe_labels, new_label=safe_label)
        if len(discard_labels) > 0:
            seg_map_plot.reassign_labels(labels=discard_labels, new_label=discard_label)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(seg_map.data, origin="lower")
        ax[1].imshow(seg_map_plot.data, origin="lower")
        for lab in zs_plot.keys():
            for zx, zy, zz in zip(zs_plot[lab]["pos_x"], zs_plot[lab]["pos_y"], zs_plot[lab]["z"]):
                ax[0].text(zx, zy, f"{zz:.4f}", color = "red", ha = "center", va = "center", fontsize = 7)
        plt.show()

    return safe_labels, discard_labels

def _check_deblended_photo_sources(seg_map, main_label, phot_sources = None, safe_labels = None, discard_labels = None,
                                  show_plots = False, verbose = False):
    """
    Assign the object type to the segments based on the photometric which of the
    objects within each segment is closest to the centroid. Segments assigned as
    artifact (``ATF``) or star-like (``PSF``) are assigned the ``discard_label``.

    Parameters
    ----------
    seg_map : photutils.segmentation.SegmentationImage
        Input segmentation map
    main_label : int
        The label of the primary object
    safe_labels : list
        The list of labels that are already classified as "safe" (these labels will not be affected by this function)
    discard_labels : list
        The list of labels that are already classified as "discard" (these labels will not be affected by this function)
    phot_sources : astropy.table.Table, optional
        Table of photometric source types in the image region

    Returns
    -------
    list, list
        Updated safe_labels and discard_labels lists.
    """

    if safe_labels is None:
        safe_labels = []
    if discard_labels is None:
        discard_labels = []

    non_main_segmap = seg_map.copy()
    non_main_segmap.remove_label(main_label)

    if len(non_main_segmap.labels) == 0:
        if verbose:
            print("No labels found besides main. Skipping redshift check for deblending")
        return seg_map

    image_h, image_w = seg_map.data.shape

    ps_x = np.clip(np.rint(phot_sources['x_centroid']).astype(int), 0, image_w - 1)
    ps_y = np.clip(np.rint(phot_sources['y_centroid']).astype(int), 0, image_h - 1)
    ps_type = phot_sources['type']

    ps_plot = {}
    for seg_label in non_main_segmap.labels:
        if seg_label in discard_labels or seg_label in safe_labels:
            continue
        seg_mask = non_main_segmap.data == seg_label
        seg_centroid = get_image_centroid(seg_mask)
        inside = seg_mask[ps_y, ps_x]
        if np.any(inside):
            dx = ps_x[inside] - seg_centroid[0]
            dy = ps_y[inside] - seg_centroid[1]
            dist2 = dx**2 + dy**2
            idx_closest = np.argmin(dist2)
            closest_type = ps_type[inside][idx_closest]
            if str(closest_type).strip() in {"ATF", "PSF"}:
                discard_labels.append(seg_label)
            ps_plot[seg_label] = {"pos_x":ps_x[inside][idx_closest],
                                  "pos_y":ps_y[inside][idx_closest],
                                  "type": closest_type}


    if show_plots:
        import matplotlib.pyplot as plt
        seg_map_plot = seg_map.copy()
        safe_label = int(np.max(seg_map_plot.labels) + 1)
        discard_label = safe_label + 1
        if len(discard_labels) > 0:
            seg_map_plot.reassign_labels(labels=discard_labels, new_label=discard_label)

        fig, axo = plt.subplots(1, 2)
        axo[0].imshow(seg_map.data, origin="lower")
        axo[1].imshow(seg_map_plot.data, origin="lower")
        for lab in ps_plot.keys():
            axo[0].text(ps_plot[lab]["pos_x"], ps_plot[lab]["pos_y"], ps_plot[lab]["type"], color = "red",
                        ha = "center", va = "center", fontsize = 7)
        plt.show()

    return safe_labels, discard_labels

def _check_if_region_is_contained(segmap, inner_label, outer_label, connectivity=8):
    """
    Checks if all pixels of inner_val are fully contained inside outer_val.

    Parameters
    ----------
    segmap : ndarray
        Integer label array.
    inner_label : int
        The value of the supposed inner region.
    outer_label : int
        The value of the supposed outer region.
    connectivity : int, optional
        4 or 8 — connectivity for defining containment.

    Returns
    -------
    bool
        True if inner_val region is completely inside outer_val, False otherwise.
    """
    # Mask for inner region
    inner_mask = (segmap == inner_label)
    if not np.any(inner_mask):
        return False  # no inner region present

    # Define structure for connectivity
    if connectivity == 4:
        structure = generate_binary_structure(2, 1)
    elif connectivity == 8:
        structure = generate_binary_structure(2, 2)
    else:
        raise ValueError("connectivity must be 4 or 8")

    # Dilate inner region by 1 pixel
    dilated_inner = binary_dilation(inner_mask, structure=structure)

    # The ring around inner_val after dilation
    border_mask = dilated_inner & ~inner_mask

    # Condition 1: border must be entirely outer_val
    border_values = segmap[border_mask]
    if not np.all(border_values == outer_label):
        return False

    # Condition 2: inner region must not touch image border
    rows, cols = segmap.shape
    r_idx, c_idx = np.where(inner_mask)
    if (0 in r_idx) or (rows - 1 in r_idx) or (0 in c_idx) or (cols - 1 in c_idx):
        return False

    return True

def _are_regions_in_contact(array, region1_values, region2_values):
    """
    Checks whether any pixel from region1 is adjacent to any pixel from region2.

    Parameters:
    - array: 2D numpy array of values.
    - region1_values: set or list of values defining the first region.
    - region2_values: set or list of values defining the second region.

    Returns:
    - True if any pixel from region1 is adjacent to any pixel from region2, False otherwise.
    """
    region1_mask = np.isin(array, region1_values)
    region2_mask = np.isin(array, region2_values)

    region1_coords = np.argwhere(region1_mask)

    for y, x in region1_coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < array.shape[0] and 0 <= nx < array.shape[1]:
                    if region2_mask[ny, nx]:
                        return True
    return False

def _merge_touching_segments(seg_map, main_label, sec_labels=None, connectivity=8):
    """
    Merge secondary regions connected to a main region.

    ```
    Identifies the regions in ``sec_labels`` that are connected to
    ``main_label`` through a chain of touching regions. A secondary region
    is merged if it directly touches ``main_label`` or touches another
    secondary region that is itself connected to ``main_label``.

    Adjacency is determined by the specified pixel connectivity. With
    8-connectivity, diagonal neighbors are considered touching; with
    4-connectivity, only horizontal and vertical neighbors are considered
    touching.

    The input segmentation map is not modified; a copy is returned with
    all connected secondary regions reassigned to ``main_label``.

    Parameters
    ----------
    seg_map
        Segmentation map containing the labeled regions.
    main_label
        Label of the main region from which connectivity is evaluated.
    sec_labels : sequence of int, optional
        Labels of secondary regions eligible for merging. If ``None`` or
        empty, the input segmentation map is returned unchanged.
    connectivity : {4, 8}, default=8
        Pixel connectivity used to determine whether regions are touching.
        ``4`` considers horizontal and vertical neighbors only, while
        ``8`` also considers diagonal neighbors.

    Returns
    -------
    SegmentationMap
        A copy of ``seg_map`` in which every secondary region connected
        to ``main_label`` through a chain of eligible touching regions
        has been reassigned to ``main_label``.

    """
    if not sec_labels:
        return seg_map

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    elif connectivity == 4:
        structure = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            dtype=bool,
        )
    else:
        raise ValueError("connectivity must be 4 or 8")

    seg_map_out = seg_map.copy()
    data = seg_map.data

    # Labels that have already been reached from the main region.
    connected_labels = {main_label}
    remaining_labels = set(sec_labels)

    # Start with the main region.
    current_mask = data == main_label

    while remaining_labels:
        dilated = binary_dilation(current_mask, structure=structure)

        touching_labels = {
            label
            for label in remaining_labels
            if np.any(dilated & (data == label))
        }

        if not touching_labels:
            break

        connected_labels.update(touching_labels)
        remaining_labels -= touching_labels

        # Expand the region to include the newly connected labels.
        for label in touching_labels:
            current_mask |= data == label

    touching_labels = connected_labels - {main_label}

    if touching_labels:
        seg_map_out.reassign_labels(
            labels=list(touching_labels),
            new_label=main_label,
        )

    return seg_map_out

def _plot_deblended_maps(seg_map_initial, seg_map_z, seg_map_phot, seg_map_final, redshift = None,
                        z_sources = None, phot_sources = None):
    import matplotlib.pyplot as plt
    c_map = "viridis"
    for this_seg_map in (seg_map_initial, seg_map_z, seg_map_phot, seg_map_final):
        this_seg_map.relabel_consecutive()
    fg, axo = plt.subplots(2, 2, figsize=(8, 10))
    fg.suptitle('Deblending maps')
    axo[0, 0].imshow(seg_map_initial, origin="lower", cmap = c_map)
    axo[0, 0].set_title("initial")

    axo[0, 1].imshow(seg_map_z, origin="lower", cmap = c_map)
    axo[0, 1].set_title("deblend z")
    if redshift is not None:
        axo[0, 1].text(round(seg_map_z.data.shape[1]*0.02),round(seg_map_z.data.shape[0]*0.98), f'main z = {redshift:.4f}', color="lime", fontsize=10, ha='left',va='top')

    if z_sources is not None:
        for row in z_sources:
            axo[0, 1].text(int(row['x_pos']), int(row['y_pos']), f'{row['z']:.4f}', color="red", fontsize=7, ha='center', va='center')

    axo[1, 0].imshow(seg_map_phot.data, origin="lower", cmap = c_map)
    axo[1, 0].set_title("deblend phot")
    if phot_sources is not None:
        for x, y, type in phot_sources['x_centroid', 'y_centroid', 'type'].iterrows():
            axo[1, 0].text(int(x), int(y), f'{type}', color="red", fontsize=6, ha='center', va='center')

    axo[1, 1].imshow(seg_map_final.data, origin="lower", cmap = c_map)
    axo[1, 1].set_title("final")

    plt.show()