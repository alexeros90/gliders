import numpy as np
from .utils import interpolate_image, grow_mask, add_noise_to_region, sigma_clipping, shrink_mask, get_profile_gradient, get_profile_from_border
from photutils.segmentation import SourceCatalog, detect_sources


def detect_and_clean_artifacts(image, config = None, img_bkg_lvl = None, pix_scale = 1, main_source_pos = None,
                               skip_flagged_artifacts = False, verbose = False,  show_plots = False):
    """
    Detect and optionally remove overexposure artifacts from an image.

    Artifact candidates are first identified using
    :func:`detect_artifacts`, which searches for bright structures with
    characteristic intensity profiles. Detected artifacts are then
    classified as either safe or unsafe according to their size and
    proximity to the target source through
    :func:`get_artifact_flag`.

    If artifacts are detected and cleaning is allowed, the affected
    regions are replaced using :func:`clean_artifacts`, including an
    additional mask to account for bleeding around the artifact.

    Parameters
    ----------
    image : ndarray
        Input image to be inspected and cleaned.

    config : object, optional
        Configuration object containing the artifact-detection and
        cleaning parameters. If None, the default cleaning configuration
        is loaded from ``cleaning_configs``.

    img_bkg_lvl : float, optional
        Background level of the image. If None, it is estimated using
        sigma clipping.

    pix_scale : float, optional
        Pixel scale of the image. Used when cleaning artifacts. Default is 1.

    main_source_pos : tuple(float, float) or None, optional
        Pixel coordinates (x, y) of the primary target source. Used to
        determine whether a detected artifact lies within the protected
        central region of the image. If None, the image center is used.

    skip_flagged_artifacts : bool, optional
        If True, artifacts classified as unsafe are not cleaned and the
        original image is returned unchanged. This can be useful to
        preserve images where an artifact significantly affects the
        target source. Default is False.

    show_plots : bool, optional
        If True, display diagnostic plots showing the original image,
        cleaned image, detected artifact mask, and final cleaning mask.
        Default is False.

    Returns
    -------
    clean_image : ndarray
        The cleaned image. If no artifacts are detected, or if cleaning
        is skipped, this will be identical to the input image.

    cleaning_mask : ndarray of bool
        Boolean mask identifying pixels affected by the artifact
        cleaning process. This includes both the detected artifact
        pixels and any additional bleeding region considered during
        cleaning.

    unsafe_artifact_flag : bool
        True if a detected artifact overlaps the protected central
        region and exceeds the configured relative size threshold.
        False otherwise.

    Notes
    -----
    The behavior of the detector and cleaner is controlled by the
    following configuration parameters:

    - ``npixels_artifact``
    - ``artifact_delta_slope_threshold``
    - ``artifact_segment_slack``
    - ``artifact_candidate_ratio``
    - ``artifact_rel_size_thresh``
    - ``artifact_safe_region_radius``
    - ``artifact_bleeding_factor``

    When ``skip_flagged_artifacts=True``, images containing unsafe
    artifacts are not modified, although the artifact mask and flag are
    still returned.
    """

    if config is None:
        from .cleaning_configs import configs
        config = configs["default"]
    if img_bkg_lvl is None:
        img_bkg_lvl, _ = sigma_clipping(image)

    artifact_segmap = detect_artifacts(image, npixels_artifact=config.npixels_artifact,
                                     delta_slope_threshold=config.artifact_delta_slope_threshold,
                                     segment_slack=config.artifact_segment_slack,
                                     hi_pass_threshold=img_bkg_lvl * config.artifact_candidate_ratio)

    if artifact_segmap is None:
        if verbose:
            print("No overexposure artifacts were detected in the image.")
        cleaning_mask = np.zeros_like(image, dtype=bool)
        return image, cleaning_mask, False

    artifact_mask = artifact_segmap.make_source_mask()
    unsafe_artifact_flag = get_artifact_flag(artifact_mask, main_source_pos, config.artifact_rel_size_thresh, config.artifact_safe_region_radius, show_plots = show_plots)

    if artifact_mask.any() and not (skip_flagged_artifacts and unsafe_artifact_flag):
        clean_image, cleaning_mask = clean_artifacts(image, artifact_segmap, config.artifact_bleeding_factor,
                                                     config.artifact_interpolation_radius, pix_scale)
    else:
        clean_image = image
        cleaning_mask = artifact_mask

    if show_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(image, norm = "log", origin = "lower")
        ax[0,0].set_title('Input image')
        ax[0,1].imshow(clean_image, norm = "log", origin = "lower")
        ax[0,1].set_title('Treated image')
        ax[1,0].imshow(artifact_mask, origin = "lower")
        ax[1,0].set_title('Detected artifacts')
        ax[1,1].imshow(cleaning_mask, origin = "lower")
        ax[1,1].set_title('Detected artifacts + bleeding')
        plt.tight_layout()
        plt.show()

    return clean_image, cleaning_mask, unsafe_artifact_flag

def clean_artifacts(image, artifact_segmap = None, bleeding_factor = 1.0, interpolation_radius = 1, pix_scale = 1):

    """Remove artifacts based on interpolation over segmented artifacts.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    artifact_segmap : segmentation map, optional
        Artifact segmentation map. If None, the image is returned unchanged.
    bleeding_factor : float
        Multiplier controlling how far the artifact mask is expanded.
    interpolation_radius : float
        Radius of the circle containing the pixels to be considered in the interpolation.
    pix_scale : float
        Scale factor of the noise pixels added.

    Returns
    -------
    image : np.ndarray
        Cleaned image.
    full_cleaning_mask : np.ndarray or None
        Mask of pixels modified by the cleaning procedure.
    """

    if artifact_segmap is None or (len(artifact_segmap.segments) < 1):
        return image, np.zeros(image.shape, dtype=bool)

    if bleeding_factor < 0:
        raise ValueError("bleeding_factor must be non-negative")

    if interpolation_radius < 1:
        raise ValueError("interpolation_radius must be equal or greater than 1")
    all_artifacts_mask = artifact_segmap.make_source_mask()
    full_cleaning_mask = all_artifacts_mask.copy()
    for artifact in artifact_segmap.segments:
        artifact_mask = artifact_segmap.data == artifact.label
        artifact_area = artifact_mask.sum()
        bleeding_decay = 2 * artifact_area**(1/3) # Rounded from empiric formula based on DESI Legacy images.
        bleeding_radius = round(bleeding_factor * bleeding_decay)
        # Expanding detection mask to cover artifact bleeding
        cleaning_mask = grow_mask(artifact_mask, pixels=bleeding_radius)
        full_cleaning_mask |= cleaning_mask

    image = interpolate_image(image, full_cleaning_mask, interpolation_radius, method="inpaint")
    _, img_bkg_lvl_std = sigma_clipping(image)
    image = add_noise_to_region(image, full_cleaning_mask, noise_center=0, noise_std=img_bkg_lvl_std, scale=pix_scale)
    return image, full_cleaning_mask

def detect_artifacts(image, npixels_artifact=100, delta_slope_threshold = 2.7, segment_slack = 5, hi_pass_threshold = 1):
    """
    Detect candidate overexposure artifacts based on their radial intensity profile.

    The algorithm first identifies bright connected regions above a fixed
    intensity threshold. For each candidate region, a series of concentric
    layers is constructed around the detected segment and the mean pixel value
    within each layer is measured. The resulting radial profile is analyzed
    through its gradient and second-order variations (delta slopes).

    Regions showing an abrupt change in profile slope are classified as
    artifacts.

    Parameters
    ----------
    image : ndarray
        2D image to analyze.

    npixels_artifact : int, optional
        Minimum number of connected pixels required for a region to be
        considered as an artifact candidate.

    delta_slope_threshold : float, optional
        Minimum peak value in the profile delta-slope sequence required to
        classify a candidate as an artifact. Larger values make the detection
        more conservative.

    segment_slack : int, optional
        Number of pixels by which each candidate segment is expanded when
        computing the surrounding radial profile. Also sets the number of
        profile layers evaluated.

    hi_pass_threshold : float, optional
        Pixel intensity threshold used to generate the initial binary mask of
        artifact candidates.

    Returns
    -------
    ndarray of bool
        Segmentation map of all detected artifacts.

    Notes
    -----
    Artifact identification relies on the assumption that overexposed features
    often exhibit sharp transitions in their radial intensity profile, unlike
    typical astronomical sources whose profiles vary more smoothly.
    """

    hi_pass_mask = image > hi_pass_threshold
    artifacts_segmap = detect_sources(hi_pass_mask, 0.5, npixels_artifact, connectivity=8)
    if artifacts_segmap is None:
        return artifacts_segmap

    candidate_artifacts_mask = artifacts_segmap.data > 0

    labels_to_remove = []
    artifacts_cat = SourceCatalog(candidate_artifacts_mask, artifacts_segmap)

    for artifact in artifacts_cat:
        segment_mask = artifact.segment.astype(bool)

        pad_top = min(segment_slack, artifact.bbox.iymin)
        pad_bottom = min(segment_slack, image.shape[0] - artifact.bbox.iymax)
        pad_left = min(segment_slack, artifact.bbox.ixmin)
        pad_right = min(segment_slack, image.shape[1] - artifact.bbox.ixmax)
        segment_mask_padded = np.pad(segment_mask, ((pad_top, pad_bottom), (pad_left, pad_right)),
                                     mode="constant", constant_values=False)

        segment_cutout = image[max(0, artifact.bbox.iymin - segment_slack):min(image.shape[0],
                                                                               artifact.bbox.iymax + segment_slack),
                         max(0, artifact.bbox.ixmin - segment_slack):min(image.shape[1],
                                                                         artifact.bbox.ixmax + segment_slack)]

        mean_profile = get_profile_from_border(segment_cutout, segment_mask_padded, segment_slack)

        if len(mean_profile) < 3:
            labels_to_remove.append(artifact.label)
            continue
        _, profile_delta_slopes = get_profile_gradient(mean_profile)
        max_delta_slope = np.max(profile_delta_slopes)
        is_artifact = max_delta_slope >= delta_slope_threshold
        if not is_artifact:
            labels_to_remove.append(artifact.label)
    artifacts_segmap.remove_labels(labels_to_remove)

    return artifacts_segmap

def get_artifact_flag(artifact_mask, main_source_pos = None, relative_size_thresh = 0.005, safe_region_radius = 0.25, show_plots = False):
    """
        Determine whether a detected artifact should be considered critical.

        An artifact is flagged when both of the following conditions are met:

        1. At least one artifact pixel lies inside a central circular region
           of the image.
        2. The total artifact area exceeds a minimum fraction of the area
           of that central region.

        The central region is defined as a circle centered on
        ``main_source_pos``. If no valid source position is provided, the
        image center is used instead. The circle radius is computed as
        ``safe_region_radius * min(image_shape)``.

        Parameters
        ----------
        artifact_mask : ndarray of bool
            Boolean mask containing the detected artifact pixels.
            True values indicate artifact locations.

        main_source_pos : tuple(float, float) or None, optional
            Pixel coordinates (x, y) of the target source. If None or if
            the coordinates are not finite, the geometric center of the
            image is used.

        relative_size_thresh : float, optional
            Minimum ratio between the total artifact area and the area of
            the central circular region required to trigger the flag.
            Default is 0.005.

        safe_region_radius : float, optional
            Radius of the central circular region expressed as a fraction of
            the smallest image dimension. Values are clipped to the interval
            [0, 1]. Default is 0.25.

        show_plots : bool, optional
            If True and a critical artifact is detected, display a diagnostic
            plot showing the artifact mask and the central region. Default is
            False.

        Returns
        -------
        bool
            True if the artifact overlaps the central region and is larger
            than the specified relative size threshold. False otherwise.

        Notes
        -----
        The artifact size is computed using the total number of pixels in
        ``artifact_mask`` and is not restricted to the portion lying within
        the central region.
        """
    ny, nx = artifact_mask.shape
    if (main_source_pos is None or not np.all(np.isfinite(main_source_pos))):
        center = (nx // 2, ny // 2)
    else:
        center = main_source_pos
    safe_reg_prop_radius = np.clip(safe_region_radius, 0, 1)
    radius = int(round(min((ny, nx)) * safe_reg_prop_radius))
    # Create a mask for the central circular region
    y, x = np.ogrid[:ny, :nx]
    circle_mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2)

    artifact_size = np.sum(artifact_mask)
    inner_region_size = np.sum(circle_mask)
    if inner_region_size == 0:
        return False
    big_artifact = (artifact_size / inner_region_size >= relative_size_thresh)
    artifact_inside_region = np.any(artifact_mask & circle_mask)
    artifact_flag = artifact_inside_region and big_artifact

    if show_plots and artifact_flag:
        import matplotlib.pyplot as plt
        plt.imshow(artifact_mask.astype(int) + circle_mask.astype(int), origin = "lower")
        plt.title("Artifact not-safe region")
        plt.show()

    return artifact_flag

