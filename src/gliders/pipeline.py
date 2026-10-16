import numpy as np
from astropy.table import Table, vstack
from photutils.segmentation import SourceCatalog, detect_threshold, detect_sources
from .utils import convolve_with_gaussian_kernel, sigma_clipping, get_main_source_label, check_edge_contact, remove_low_outliers
from .background_suppressor import suppress_background
from .stars import detect_and_mask_stars, star_contamination_check
from .deblending import deblend_and_mask_sources
from .artifacts import detect_and_clean_artifacts
from .others import flatten_background, mask_secondary_detections, clear_background, segmentation_map_smoothing
import traceback
import warnings

def update_sources(sources_table, artifact_mask):
    """
    Update the sources table with detected artifacts.

    Parameters:
    -----------
    sources_table : astropy.table.Table
        Table containing detected sources.
    artifact_mask : numpy.ndarray
        Boolean mask indicating the location of detected artifacts.

    Returns:
    --------
    astropy.table.Table
        Updated sources table with detected artifacts added.
    """
    if (artifact_mask == True).any():
        artifact_segmap = detect_sources(artifact_mask, 0.5, 1, connectivity=8)
        artifact_table = SourceCatalog(artifact_mask, artifact_segmap).to_table()
        artifact_table = artifact_table[['x_centroid', 'y_centroid']]
        artifact_table['type'] = 'ATF'
        sources_table = vstack([sources_table, artifact_table])
    return sources_table

def update_detections(image, config = None, source_min_pix = 1, source_pos = None, use_filter = True, verbose = False):
    """
    Update the detections in the image by convolving with a Gaussian kernel,
    detecting sources, and extracting the main source segment.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image in which to detect sources.
    config : Config, optional
        Configuration object containing parameters for source detection.
        If None, the default configuration will be used.
    source_min_pix : int, optional
        Minimum number of pixels for a source to be detected. Default is 1.
    source_pos : tuple or list, optional
        The (x, y) position of the main source. If None, the center of the image will be used.
    use_filter : bool, optional
        Whether to convolve the image with a Gaussian kernel before detection. Default is True.
    verbose : bool, optional
        Whether to print progress messages. Default is False.

    Returns:
    --------
    tuple
        A tuple containing:
        - image_for_detection : numpy.ndarray
            The image used for source detection (possibly convolved).
        - seg_map : photutils.segmentation.SegmentationImage
            The segmentation map of detected sources.
        - main_source_segmap : photutils.segmentation.SegmentationImage
            The segmentation map of the main source.
    """

    if config is None:
        if verbose:
            print("No config provided. Using default config.")
        from .cleaning_configs import configs
        config = configs["default"]
    if source_pos is None:
        ny, nx = image.shape
        source_pos = np.array((int(nx / 2), int(ny / 2)))

    if use_filter:
        image_for_detection = convolve_with_gaussian_kernel(image, kernel_std=config.psf_std)
    else:
        image_for_detection = image

    threshold = detect_threshold(image_for_detection, n_sigma=config.galaxy_detection_threshold)
    seg_map = detect_sources(image_for_detection, threshold, source_min_pix, connectivity=config.connect_mode)
    if not seg_map:
        raise Exception("No sources were detected in the image.")

    main_label = get_main_source_label(seg_map, source_pos = source_pos)
    main_source_segmap = seg_map.copy()
    main_source_segmap.keep_label(label=main_label)
    return image_for_detection, seg_map, main_source_segmap

def clean_image(image, image_psf = None, config = None, source_pos = None, redshift = None, z_sources = None,
                query_z_sources = True, wcs = None, phot_sources = None, clean_artifacts = True, skip_flagged_artifacts = True,
                reduce_noise_level = False, subtract_background = False, mask_sec_objects = True, mask_stars = True,
                deblend_sources = True, deblend_with_redshift = True, deblend_delta_z = None, keep_undecided_segments = True,
                deblend_with_photo_type = True, robust_background = True, sync_z_query = False,
                clear_all_but_segment = False, smooth_segmap = True, pix_scale = 1, verbose = True, show_plots = False):
    """
    Clean an astronomical image while preserving the main target source.

    This function applies a configurable sequence of image-cleaning
    operations to reduce contamination from overexposure artifacts, stars,
    secondary sources, and overlapping objects. It detects the main source,
    optionally deblends neighboring sources, removes unwanted detections,
    and refines the main-source segmentation map.

    The main source is identified using ``source_pos``. If the supplied
    position is missing, non-finite, or outside the image boundaries, the
    image center is used instead.

    The cleaning pipeline can include the following operations:

    1. Validate the image and initialize the cleaning configuration.
    2. Optionally detect and clean overexposure artifacts.
    3. Optionally subtract the background and suppress background noise.
    4. Detect sources and identify the main-source segment.
    5. Optionally detect and mask contaminating stars.
    6. Optionally deblend neighboring sources using redshift and/or
       photometric-type information.
    7. Optionally mask secondary detections.
    8. Optionally smooth the main-source segmentation map.
    9. Optionally remove everything outside the main-source segment.
    10. Optionally remove low-intensity outliers.
    11. Evaluate source size, image-edge contact, and positional offset.

    A Boolean cleaning mask records pixels marked as cleaned by the
    artifact-removal, star-masking, deblending, and secondary-object
    masking operations. The function also returns a list of integer flags
    summarizing potential issues identified during processing.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional input image containing the main astronomical source
        and any contaminating sources or artifacts. The image is processed
        sequentially and may be modified or replaced by intermediate
        cleaning operations.

        An image containing only zero-valued pixels is rejected.

    image_psf : numpy.ndarray or None, optional
        Reference point spread function (PSF) image passed to the star
        detection and masking procedure. Used to help identify and validate
        stellar sources. Default is ``None``.

    config : object or None, optional
        Configuration object controlling source detection, artifact
        cleaning, masking, deblending, and interpolation parameters.

        If ``None``, the default configuration is loaded from
        ``cleaning_configs.configs["default"]``. The function prints a
        message indicating that the default configuration is being used.

        The configuration must provide the attributes required by the
        enabled processing routines, including ``source_min_pix``,
        ``artifact_candidate_ratio``, ``galaxy_detection_threshold``,
        ``big_contaminator_pix_ratio``, and ``z_query_timeout``.
        Additional attributes may be required by downstream routines.
        Default is ``None``.

    source_pos : array-like or None, optional
        Approximate position of the main source, specified as
        ``(x, y)`` in image pixel coordinates.

        If the position is ``None``, contains non-finite values, or falls
        outside the image boundaries, the image center is used. Otherwise,
        both coordinates are rounded to the nearest integer pixel.
        Default is ``None``.

    redshift : float or None, optional
        Redshift estimate for the main source. Passed to the deblending
        procedure when ``deblend_sources=True`` and used according to the
        configured redshift-based source-selection criteria. Default is
        ``None``.

    z_sources : object or None, optional
        Additional source-redshift information supplied to the deblending
        procedure. The expected format depends on
        ``deblend_and_mask_sources``. Default is ``None``.

    query_z_sources : bool, optional
        If ``True``, allow the deblending procedure to query external
        redshift information when supported. The exact behavior depends
        on ``deblend_and_mask_sources``. Default is ``True``.

    wcs : object or None, optional
        World Coordinate System object passed to the deblending procedure
        for coordinate transformations or source matching. Default is
        ``None``.

    phot_sources : object or None, optional
        Photometric source catalog used by the star-detection and
        deblending procedures.

        If ``None``, an empty Astropy table with ``'type'``,
        ``'x_centroid'``, and ``'y_centroid'`` columns is initialized.
        Default is ``None``.

    clean_artifacts : bool, optional
        If ``True``, check for and clean potential overexposure artifacts.
        Artifact detection is attempted only if the ratio of the maximum
        image intensity to the estimated background level exceeds
        ``config.artifact_candidate_ratio``. Default is ``True``.

    skip_flagged_artifacts : bool, optional
        If ``True``, instruct the artifact-cleaning procedure to skip
        artifacts that meet its flagged-artifact criteria. The exact
        behavior is defined by ``detect_and_clean_artifacts``.
        Default is ``True``.

    reduce_noise_level : bool, optional
        If ``True``, suppress background noise using the logit-based
        ``suppress_background`` method. Default is ``False``.

    subtract_background : bool, optional
        If ``True``, flatten the image background using
        ``flatten_background`` with the configured source-detection
        threshold and minimum source-pixel count. Default is ``False``.

    mask_sec_objects : bool, optional
        If ``True``, mask secondary detections after deblending and other
        source-cleaning operations. The corresponding mask is added to the
        overall cleaning mask. Default is ``True``.

    mask_stars : bool, optional
        If ``True``, detect and mask contaminating stars using
        ``detect_and_mask_stars``. Also evaluates whether the main source
        is significantly contaminated by stars. Default is ``True``.

    deblend_sources : bool, optional
        If ``True``, deblend overlapping sources and mask unwanted
        contributions using ``deblend_and_mask_sources``. The segmentation
        map and convolved image are subsequently updated. Default is
        ``True``.

    deblend_with_redshift : bool, optional
        If ``True``, enable redshift-based criteria in the deblending
        procedure. The actual selection behavior depends on the
        implementation of ``deblend_and_mask_sources``. Default is ``True``.

    deblend_delta_z : float or None, optional
        Redshift difference threshold supplied to the deblending procedure
        for distinguishing sources according to their redshifts.
        The precise interpretation depends on ``deblend_and_mask_sources``.
        Default is ``None``.

    keep_undecided_segments : bool, optional
        If ``True``, instruct the deblending procedure to retain segments
        whose association with the main source cannot be decided confidently.
        The exact behavior depends on ``deblend_and_mask_sources``.
        Default is ``True``.

    deblend_with_photo_type : bool, optional
        If ``True``, enable photometric-type information in the deblending
        procedure when determining which sources or segments to retain.
        Default is ``True``.

    robust_background : bool, optional
        If ``True`` and ``clear_all_but_segment=False``, remove low-intensity
        outliers using ``remove_low_outliers`` after the other cleaning
        operations. This step is skipped when ``clear_all_but_segment=True``.
        Default is ``True``.

    sync_z_query : bool, optional
        If ``True``, request synchronous redshift-query behavior from the
        deblending procedure. If ``show_plots=True``, this parameter is
        automatically set to ``True``. Default is ``False``.

    clear_all_but_segment : bool, optional
        If ``True``, clear the image outside the final main-source mask
        using ``clear_background``. Default is ``False``.

    smooth_segmap : bool, optional
        If ``True``, smooth the main-source segmentation map before
        generating the final source mask. Default is ``True``.

    pix_scale : float, optional
        Pixel-scale factor passed to applicable artifact, star-masking,
        deblending, and background-clearing routines. Some operations may
        use their own fixed scale values. Default is 1.

    verbose : bool, optional
        If ``True``, print progress messages and warnings during processing.
        Also controls the verbosity passed to several downstream routines.
        Default is ``True``.

    show_plots : bool, optional
        If ``True``, enable plots in supported downstream routines.
        Also forces ``sync_z_query=True``. Default is ``False``.

    Returns
    -------
    image : numpy.ndarray
        The cleaned image after the enabled artifact-removal, background
        processing, star masking, deblending, secondary-object masking,
        and outlier-removal operations.

    main_source_segment : object
        Final segmentation-map object corresponding to the main source.
        May have been smoothed if ``smooth_segmap=True``. Its source mask
        can be obtained using ``main_source_segment.make_source_mask()``.

    cleaning_mask : numpy.ndarray
        Boolean array with the same shape as the input image, indicating
        pixels marked by the artifact-cleaning, star-masking, deblending,
        and secondary-object masking operations.

        The mask is initialized to all ``False``. It is updated by
        combining the masks returned by the relevant cleaning procedures.
        Background subtraction, noise suppression, segmentation smoothing,
        low-outlier removal, and inner-star interpolation are not necessarily
        represented in this mask.

    cleaning_flags : list of int
        List of six integer flags, each equal to 0 or 1, in the following
        order:

        1. ``unsafe_artifact_flag``:
           Set if the artifact-cleaning procedure reports unsafe artifacts.
        2. ``offset_source_flag``:
           Set if the main-source position is sufficiently far from the
           image center or the final source mask touches an image boundary.
        3. ``deblended_flag``:
           Set if the deblending procedure reports that deblending occurred.
        4. ``deblend_bad_z_query_flag``:
           Set if the deblending procedure reports a redshift-query problem.
        5. ``big_contaminator_flag``:
           Set if the star-contamination check detects significant
           contamination according to the configured threshold.
        6. ``small_source_flag``:
           Set if the final main-source mask contains fewer pixels than
           ``config.source_min_pix``.

    Raises
    ------
    Exception
        If the input image contains no nonzero pixels.

    Exception
        If no source is detected at the expected source position, as
        determined by the segmentation map.

    Other exceptions
        Exceptions raised by configuration access, source detection,
        artifact cleaning, masking, deblending, or other processing steps
        are printed with their traceback and re-raised.

    Notes
    -----
    The source position is represented in ``(x, y)`` order, whereas
    NumPy image indexing uses ``(row, column)`` order. The function
    accounts for this distinction when checking the segmentation map.

    The estimated background level and standard deviation are computed
    at the beginning of processing. The background level is used for the
    artifact-candidate check, while the background standard deviation is
    not directly used elsewhere in this function.

    If ``show_plots=True``, redshift queries are forced into synchronous
    mode, regardless of the supplied value of ``sync_z_query``.

    The offset-source flag is evaluated using the selected source position
    and the final main-source mask. The positional criterion compares the
    source position with the image center using a threshold equal to one
    quarter of the image diagonal.

    The function prints a full traceback and re-raises any exception
    encountered during processing. It does not return a partial cleaning
    result when an exception occurs.

    """

    try:
        # ------------ Handling input parameters ------------
        if not np.any(image.astype(bool)):
            raise Exception("Image appears to be empty.")
        if config is None:
            if verbose:
                warnings.warn("No configuration object was provided. Will use default configuration.",
                              UserWarning, stacklevel=2)
            from .cleaning_configs import configs
            config = configs['default']
        if phot_sources is None:
            phot_sources = Table(names=['type','x_centroid','y_centroid'], dtype=('str', 'float', 'float'))
        ny, nx = image.shape
        img_center = np.array((int(nx / 2), int(ny / 2)))
        img_bkg_lvl, img_bkg_lvl_std = sigma_clipping(image)
        invalid_pos = (source_pos is None or not np.all(np.isfinite(source_pos)) or not (0 <= source_pos[0] < nx) or not (0 <= source_pos[1] < ny))
        if invalid_pos:
            source_pos = img_center
        else:
            source_pos = np.array((round(source_pos[0]),round(source_pos[1])))
        if show_plots:
            sync_z_query = True

        source_min_pix = config.source_min_pix

        # Initializing default flags
        unsafe_artifact_flag = False
        big_contaminator_flag = False
        offset_source_flag = False
        deblended_flag = False
        deblend_bad_z_query_flag = False
        small_source_flag = False
        # ---------------------------------------------------


        cleaning_mask = np.full(image.shape, False) # Initializing the overall cleaning mask

        # Detecting and conditionally removing over exposure artifacts
        if clean_artifacts:
            if verbose:
                print("Detecting and cleaning overexposure artifacts...")
            # Non modified images with artifacts generally have pixels with values >> background, so first we filter based on that
            if (np.max(image)/img_bkg_lvl) > config.artifact_candidate_ratio:
                image, artifact_mask, unsafe_artifact_flag = detect_and_clean_artifacts(image, config, img_bkg_lvl,
                                                                                        pix_scale = pix_scale, main_source_pos = source_pos,
                                                                                        skip_flagged_artifacts = skip_flagged_artifacts,
                                                                                        verbose = verbose, show_plots=show_plots)
                cleaning_mask = artifact_mask | cleaning_mask
                phot_sources = update_sources(phot_sources, artifact_mask) # Adding detected artifacts to photo source table


        if subtract_background:
            image = flatten_background(image, src_detection_thresh = config.galaxy_detection_threshold,
                                       src_min_pixels = source_min_pix, show_plots=False)

        if reduce_noise_level:
            image = suppress_background(image, method ="logit", kernel = None, logit_alpha = 1.0)
            # smooth_noise_curve must be False, otherwise sep deblending may not work correctly


        # First detection of sources
        if verbose:
            print("Detecting sources in the image...")

        image_convolved, seg_map, main_source_segment = update_detections(image, config, source_min_pix, source_pos, verbose)
        seg_map_mask = seg_map.make_source_mask()
        if not seg_map_mask[source_pos[1], source_pos[0]]:
            raise Exception("No sources were detected in the expected location. Either the source is too faint or source_pos does not coincide with the x,y source position")

        source_cat = SourceCatalog(image, seg_map)
        if verbose:
            print("Detected", len(source_cat), "sources before cleaning.")


        # Detecting and masking stars in the image
        if mask_stars:
            # image, main_source_segment, cleaning_mask, cleaning_flags
            if verbose:
                print("Detecting and masking stars...")
            main_source_mask = main_source_segment.make_source_mask()
            image, masked_stars_mask, all_star_apertures = detect_and_mask_stars(image, main_source_mask, config,
                                                                                 source_pos, phot_sources, image_psf,
                                                                                 pix_scale, show_plots = show_plots)

            cleaning_mask = masked_stars_mask | cleaning_mask
            big_contaminator_flag = star_contamination_check(main_source_mask, all_star_apertures, config.big_contaminator_pix_ratio)

            # Updating the convolved image and segmentation map if there was any detection
            if all_star_apertures is not None:
                if (len(all_star_apertures["inner"])+len(all_star_apertures["edge"])+len(all_star_apertures["outer"])) > 0:
                    image_convolved, seg_map, main_source_segment = update_detections(image, config, source_min_pix,
                                                                                      source_pos, verbose)

        # Deblendnig and removing sources
        if deblend_sources:
            if verbose:
                print("Deblending sources...")
            z_timeout = config.z_query_timeout

            deblend_res = deblend_and_mask_sources(image, image_convolved, seg_map, source_pos, config,
                                                   deblend_with_redshift, redshift, deblend_delta_z, z_sources,
                                                   query_z_sources, wcs, z_timeout, sync_z_query,
                                                   deblend_with_photo_type, phot_sources, keep_undecided_segments,
                                                   pix_scale, show_plots = show_plots, verbose=verbose)

            image, deb_seg_map, deblending_mask, deblended_flag, deblend_bad_z_query_flag = deblend_res

            cleaning_mask = deblending_mask | cleaning_mask
            # Updating the convolved image and segmentation map of the post-deblend main source
            image_convolved, seg_map, main_source_segment = update_detections(image, config, source_min_pix, source_pos, verbose)

        # Removing all detected sources except the main one.
        if mask_sec_objects:
            if verbose:
                print("Removing secondary objects...")
            image, secondary_objects_mask = mask_secondary_detections(image, seg_map=seg_map, source_pos=source_pos,
                                                                      pix_scale=pix_scale, verbose=verbose)
            cleaning_mask = secondary_objects_mask | cleaning_mask
            # update_detections() not necessary from here on.


        # Smoothing the main source segmentation map
        if smooth_segmap:
            main_source_segment = segmentation_map_smoothing(main_source_segment)

        main_source_mask = main_source_segment.make_source_mask()

        # Remove all but main source
        if clear_all_but_segment:
            image, cleaning_mask = clear_background(image, main_source_mask, pix_scale)

        if robust_background & ~clear_all_but_segment:
            image = remove_low_outliers(image)

        source_npixels_final = np.sum(main_source_mask)
        if source_npixels_final < source_min_pix:
            small_source_flag = True
            if verbose:
                print(f"Warning, the detected source is very small (< {source_npixels_final} pixels)")

        # Checking if source extends beyond image
        edge_contact = check_edge_contact(main_source_mask)
        offset_source_pos = np.linalg.norm(source_pos - img_center) > np.hypot(*image.shape) / 4
        if offset_source_pos or edge_contact:
            if verbose:
                print("WARNING: Galaxy might be partially out of the image")
            offset_source_flag = True

        cleaning_flags = [int(unsafe_artifact_flag), int(offset_source_flag), int(deblended_flag), int(deblend_bad_z_query_flag), int(big_contaminator_flag), int(small_source_flag)]

        if verbose:
            print("Image cleaning finished!")

        return image, main_source_segment, cleaning_mask, cleaning_flags

    except Exception:
        print(traceback.format_exc())
        raise

if __name__ == "__main__":

    quit()
