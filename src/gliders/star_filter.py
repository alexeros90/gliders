import numpy as np
from photutils.profiles import RadialProfile
from scipy.stats import pearsonr
from astropy.modeling import models, fitting
from .utils import get_image_centroid, sigma_clipping, grow_pixel_circular, get_profile_gradient, extract_profile


def _get_profile_average(profile):
    """
    Calculate the average of a profile by averaging its two halves.
    """
    if (len(profile) % 2) == 0:
        mid_offset = 0
    else:
        mid_offset = 1
    half_1 = profile[int(len(profile) // 2):]
    half_2 = profile[:int(len(profile) // 2 + mid_offset)][::-1]
    return (half_1 + half_2) / 2

def _get_mean_profile_symmetry_ratio(profile_1, profile_2, null_value):
    """
    Calculate the mean symmetry ratio between two profiles.

    Parameters
    ----------
    profile_1 : array-like
        The first profile to compare.
    profile_2 : array-like
        The second profile to compare.
    null_value : float
        The value representing missing or invalid data in the profiles.

    Returns
    -------
    float or None
        The mean symmetry ratio between the two profiles, or None if either profile contains the null_value
    """

    if null_value in profile_1 or null_value in profile_2:
        return None
    profile_1_av = _get_profile_average(profile_1)
    profile_2_av = _get_profile_average(profile_2)
    max_extension = np.min((len(profile_1_av), len(profile_2_av)))
    profile_2_av = profile_2_av[:max_extension]
    profile_1_av = profile_1_av[:max_extension]
    deviation = np.sum(np.pow(profile_1_av - profile_2_av, 2)) / max_extension
    return _get_symmetry_ratio(profile_1, profile_2, deviation)

def _get_symmetry_ratio(profile_1, profile_2, deviation):
    """
    Calculate the symmetry ratio between two profiles based on their deviation and value range.
    """
    min_value = np.min((np.min(profile_1), np.min(profile_2)))
    max_value = np.max((np.max(profile_1), np.max(profile_2)))
    similarity_tolerance = np.pow((max_value - min_value), 2)
    return np.sqrt(deviation / similarity_tolerance)

def _patch_profile(profile, null_value=-999):
    """
    Patch a profile by replacing null values with the corresponding values from the opposite end of the profile.
    """
    idxs = np.where(profile == null_value)[0]
    for this_idx in idxs:
        if profile[-this_idx - 1] != -999:
            profile[this_idx] = profile[-this_idx - 1]
    return profile

def _get_distance_to_closest_edge(image_shape, position):
    """
    Calculate the distance from a given position to the closest edge of the image.
    """
    edges = (image_shape[0] - 1, image_shape[1] - 1)
    distance = np.min([position[0], edges[0] - position[0], position[1], edges[1] - position[1]])
    return distance

def _compare_profiles(profile_1, profile_2):
    """
    Compare two profiles (equal length vectors).

    Returns:
    --------
        pearson_corr : float
            Pearson correlation coefficient
        p_value_corr : float
            Two-tailed p-value for correlation significance test
        euclidean_dist : float
            Euclidean distance between the profiles
        cos_dist : float
            Cosine distance between the profiles
    """

    profile_1 = np.asarray(profile_1, dtype=float)
    profile_2 = np.asarray(profile_2, dtype=float)

    if profile_1.shape != profile_2.shape:
        raise ValueError("profile_2s must have the same length.")

    # Pearson correlation + p-value
    pearson_corr, p_value_corr = pearsonr(profile_1, profile_2)

    # Euclidean distance
    euclidean_dist = np.linalg.norm(profile_1 - profile_2)

    # Cosine distance
    cos_dist = 1-np.sum(profile_1 * profile_2)/(np.sqrt(np.sum(np.power(profile_1,2)))*np.sqrt(np.sum(np.power(profile_2,2))))

    return pearson_corr, p_value_corr, euclidean_dist, cos_dist

def _star_processing_parallel_wrapper(i, star, image, params):
    """
    Wrapper function for parallel processing of stars.
    """
    final_radius, final_std = _process_one_star(star,image,params)

    return i, final_radius, final_std

def _process_one_star(this_star, image, params):
    """
    Determine the valid radius of a star using its radial intensity profile
    and validate the star against profile, symmetry, and Gaussian-shape
    criteria.

    The function iteratively computes the star's radial profile over
    increasing radii, starting at ``min_radius`` and ending at ``max_radius``
    or earlier if a stopping criterion is met. It tracks the last radius
    for which the radial profile satisfies the selection criteria.

    If the star is close to an image boundary, its effective centroid may
    be relocated using a circular mask and an intensity-weighted centroid
    estimate.

    After the radius search, the candidate is validated using the following
    tests, where applicable:

    - Radial-profile similarity to a reference PSF profile.
    - Validity of a fitted two-dimensional Gaussian.
    - Symmetry of the profiles extracted along the Gaussian's principal axes.
    - Ratio of the Gaussian's fitted standard deviations.

    A star that fails any required post-processing test is rejected.

    Parameters
    ----------
    this_star : dict
        Dictionary describing the candidate star. Must contain the following
        keys:

        - ``'y_centroid'`` : float
            Estimated centroid row coordinate in the image.
        - ``'x_centroid'`` : float
            Estimated centroid column coordinate in the image.
        - ``'type'`` : str
            Star classification. If the value is not ``"PSF"`` and a
            reference PSF profile is provided, profile similarity is tested.

    image : numpy.ndarray
        Two-dimensional image array containing the candidate star. Pixel
        values are used to compute the radial profile, estimate a relocated
        centroid when necessary, and fit a two-dimensional Gaussian.

    params : tuple
        Tuple containing the parameters required for radius selection and
        star validation, in the following order:

        min_radius : int
            Minimum radius, in pixels, considered during the search.
        max_radius : int
            Maximum radius, in pixels, considered during the search.
        rp_psf : object or None
            Reference radial PSF profile. Its ``profile`` attribute is used
            for comparison with the candidate's radial profile. The
            comparison is skipped for stars classified as ``"PSF"`` or
            when this value is ``None``.
        image_background : float
            Background intensity threshold. The search stops if any radial
            profile value is less than or equal to this value.
        gaussian_fitter : callable
            Callable used to fit the initialized two-dimensional Gaussian
            model to the image cutout. It must return a fitted model with
            ``x_stddev``, ``y_stddev``, and ``theta`` parameters.
        std_convergence_threshold : float
            Relative change threshold used to determine whether the fitted
            radial-profile standard deviation has converged.
        gaussian_fit_max_sigma : float
            Maximum permitted ratio of the current search radius to the
            previous fitted radial-profile standard deviation once
            convergence has been detected.
        slope_threshold : float
            Minimum slope threshold used by the radial-profile stopping
            criterion.
        delta_slope_threshold : float
            Maximum slope-change threshold used by the radial-profile
            stopping criterion.
        profile_cos_distance_threshold : float
            Maximum permitted cosine distance between the reference PSF
            profile and the candidate's radial profile. The candidate is
            rejected when the distance exceeds this threshold.
        symmetry_ratio_threshold : float
            Maximum permitted mean symmetry ratio between profiles extracted
            along the fitted Gaussian's principal axes.
        gaussian_ratio_threshold : float
            Maximum permitted ratio between the fitted Gaussian standard
            deviations. The ratio is defined as the larger standard
            deviation divided by the smaller one.

    Returns
    -------
    tuple of (int, float)
        A tuple ``(final_radius, final_std)`` containing:

        final_radius : int
            Selected radius, in pixels. Returns ``0`` if no valid radius
            is found or if the candidate fails a post-processing test.
        final_std : float
            Fitted standard deviation of the radial profile associated
            with the selected radius. Returns ``0`` when the candidate is
            rejected or no valid radius is found.

    Notes
    -----
    The radius search terminates early if any of the following conditions
    is encountered:

    - The radius exceeds half the smaller image dimension.
    - The candidate is sufficiently close to an image boundary and the
      relocation procedure does not subsequently allow further processing
      to continue normally.
    - The radial profile has zero total absolute intensity.
    - Any radial profile value is less than or equal to the background
      threshold.
    - The fitted radial-profile standard deviation converges and the
      radius exceeds the configured multiple of that standard deviation.
    - The radial-profile slope and slope-change criteria indicate that
      the profile is stabilizing.

    The effective centroid may differ from the original centroid when
    relocation is performed near an image boundary.

    The Gaussian is fitted to a square image cutout centered approximately
    on the effective centroid and extending to the selected radius, clipped
    to the image boundaries.

    A successful return indicates that the candidate passed the implemented
    validation criteria; it does not guarantee that the candidate is
    astrophysically a star.

    Notes
    -----
    This function depends on the following external helpers and objects:
    ``_get_distance_to_closest_edge``, ``grow_pixel_circular``,
    ``get_image_centroid``, ``RadialProfile``, ``get_profile_gradient``,
    ``_compare_profiles``, ``extract_profile``,
    and ``_get_mean_profile_symmetry_ratio``, as well as the Gaussian
    model implementation provided by ``models.Gaussian2D``.

    """

    (
        min_radius, max_radius, rp_psf, image_background, gaussian_fitter, std_convergence_threshold,
        gaussian_fit_max_sigma, slope_threshold, delta_slope_threshold,
        profile_cos_distance_threshold, symmetry_ratio_threshold,
        gaussian_ratio_threshold
    ) = params

    y_cen, x_cen = this_star['y_centroid'], this_star['x_centroid']
    candidate_pos = (int(round(y_cen)), int(round(x_cen)))
    candidate_eff_pos = candidate_pos
    # Clip centroid so it is always within the image.
    candidate_eff_pos = (np.clip(candidate_eff_pos[0], 0, image.shape[0] - 1),
                          np.clip(candidate_eff_pos[1], 0, image.shape[1] - 1))
    distance_to_closest_edge = _get_distance_to_closest_edge(image.shape, candidate_eff_pos)
    relocated = False
    final_radius = 0
    final_profile = None
    final_std = 0
    profile_std = None
    half_image_min = np.min(image.shape) / 2

    # ----------------------------------------------------------
    # MAIN RADIUS LOOP
    # ----------------------------------------------------------
    for radius in range(min_radius, max_radius + 1):
        if radius > half_image_min:
            break
        if distance_to_closest_edge < radius / 2 and not relocated:
            mask = grow_pixel_circular(image.shape, candidate_pos, radius)
            new_cc = get_image_centroid(image * mask)
            if new_cc is not None:
                new_c_x, new_c_y = new_cc
                candidate_eff_pos = (new_c_y, new_c_x)
                relocated = True

        edge_radii = np.arange(radius + 1)
        rp = RadialProfile(image, (candidate_eff_pos[1], candidate_eff_pos[0]),
                           edge_radii, error=None, mask=None, method="center")

        if np.sum(np.abs(rp.profile)) == 0:
            break
        if np.any(rp.profile <= image_background):
            break

        new_profile_std = rp.gaussian_fit.stddev.value
        if profile_std is not None:
            if profile_std == 0:
                diff = std_convergence_threshold + 1
            else:
                diff = abs(profile_std - new_profile_std) / profile_std
            if diff < std_convergence_threshold:
                if radius > profile_std * gaussian_fit_max_sigma:
                    break
        profile_std = new_profile_std

        if len(rp.profile) < 2:
            continue

        profile_slopes, profile_delta_slopes = get_profile_gradient(rp.profile)
        max_delta_slope = np.max(profile_delta_slopes)

        if len(profile_delta_slopes) > 1:
            slope_last = profile_slopes[-1]
            slope_prev = profile_slopes[-2]
            delta_last = profile_delta_slopes[-1]
            delta_prev = profile_delta_slopes[-2]

            # Cond 1: The slope is increasing and above the threshold.
            if slope_last > slope_prev and np.max(profile_slopes) > slope_threshold:
                # Cond 2: The slope change is decreasing (slopes are stabilizing) and below the threshold
                if delta_last < delta_prev and delta_last < delta_slope_threshold:
                    break

        if max_delta_slope > 0: # If the slope started to increase at any point we consider the radius as valid.
            final_radius = radius
            final_profile = rp.profile
            final_std = profile_std


    # ----------------------------------------------------------
    # POST-PROCESSING IF STAR PASSED RADIUS LOOP
    # ----------------------------------------------------------
    if final_radius <= 0:
        return final_radius, final_std
    else:
        if this_star['type'] != "PSF" and rp_psf is not None:
            psf_prof = rp_psf.profile[:len(final_profile)]
            if len(psf_prof) < len(final_profile):
                tail = np.full((len(final_profile) - len(psf_prof),), psf_prof[-1])
                psf_prof = np.append(psf_prof, tail)
            corr, pval, euc_dist, cos_dist = _compare_profiles(psf_prof, final_profile)
            if cos_dist > profile_cos_distance_threshold:
                return 0, 0

        # symmetry tests
        y0, x0 = candidate_eff_pos
        ny, nx = image.shape
        cutout = image[
            max(y0 - final_radius, 0):min(y0 + final_radius + 1, ny),
            max(x0 - final_radius, 0):min(x0 + final_radius + 1, nx)
        ]

        cutout_ny, cutout_nx  = cutout.shape
        cutout_y, cutout_x = np.indices(cutout.shape)
        init = models.Gaussian2D(amplitude=cutout.max(),
                                 x_mean=cutout_nx / 2, y_mean=cutout_ny / 2,
                                 x_stddev=final_radius / 2, y_stddev=final_radius / 2)
        init.x_mean.bounds = (0, cutout_nx)
        init.y_mean.bounds = (0, cutout_ny)

        fit = gaussian_fitter(init, cutout_x, cutout_y, cutout)
        x_std = fit.x_stddev.value
        y_std = fit.y_stddev.value
        theta = fit.theta.value
        theta_deg = 180 * theta / np.pi

        if (not np.isfinite(x_std) or not np.isfinite(y_std) or x_std <= 0 or y_std <= 0):
            return 0, 0
        gaussian_ratio = max(x_std / y_std, y_std / x_std)

        # Getting the profiles over the gaussian ellipse axes
        distance_a, profile_a = extract_profile(cutout, cutout_nx / 2, cutout_ny / 2, 2 * final_radius - 1, theta_deg)
        distance_b, profile_b = extract_profile(cutout, cutout_nx / 2, cutout_ny / 2, 2 * final_radius - 1,
                                                theta_deg + 90)
        
        symmetry_ratio = _get_mean_profile_symmetry_ratio(profile_a, profile_b, np.nan)

        if (symmetry_ratio > symmetry_ratio_threshold) or (gaussian_ratio > gaussian_ratio_threshold):
            return 0, 0

        # Success — update and return the result
        return final_radius, final_std

def star_detection_filtering(image, stars_detected, min_radius=2, max_radius=15, slope_threshold = -0.008,
                             delta_slope_threshold = 0.008, image_psf = None, profile_cos_distance_threshold = 0.045,
                             symmetry_ratio_threshold = 0.07, gaussian_ratio_threshold = 1.2, std_convergence_threshold = 0.01,
                             gaussian_fit_max_sigma = 3, do_parallel = False, show_plots = False):
    """
    Filter detected stars using radial-profile and shape-based criteria.

    Each candidate star is processed to estimate a valid radius from its
    radial intensity profile. Candidates are then validated using criteria
    based on radial-profile behavior, optional similarity to a reference
    point spread function (PSF), and the symmetry and shape of a fitted
    two-dimensional Gaussian.

    The estimated radius and radial-profile standard deviation are stored
    in the input ``stars_detected`` array in the ``'r'`` and ``'std'``
    fields, respectively. Candidates with a positive estimated radius are
    retained in the returned array.

    Star processing can be performed sequentially or in parallel using a
    thread pool. An optional plot can be displayed to visualize the image
    and the estimated radius of each detected candidate.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional image containing the detected stars. Used to
        estimate the background level, calculate radial profiles, and
        validate the candidates.

    stars_detected : numpy.ndarray or None
        Structured NumPy array containing the detected star candidates.
        Each entry must include at least the following fields:

        - ``'x_centroid'`` : float
            Estimated centroid column coordinate.
        - ``'y_centroid'`` : float
            Estimated centroid row coordinate.
        - ``'id'`` : object
            Candidate identifier, required when ``show_plots=True``.

        The array is modified in place by adding or updating the ``'r'``
        and ``'std'`` fields. These fields must be supported by the array's
        dtype.

    min_radius : int, optional
        Minimum radius, in pixels, considered when determining the valid
        radius of a candidate. Must be at least 1. Default is 2.

    max_radius : int, optional
        Maximum radius, in pixels, considered during the radius search.
        Must be greater than or equal to ``min_radius``. Default is 15.

    slope_threshold : float, optional
        Slope threshold used by the radial-profile stopping criterion.
        Default is -0.008.

    delta_slope_threshold : float, optional
        Slope-change threshold used to identify stabilization of the radial
        intensity profile. Default is 0.008.

    image_psf : numpy.ndarray or None, optional
        Reference image representing the point spread function (PSF).
        If provided, a radial profile is computed from the image center and
        used to evaluate the similarity of eligible candidates to the
        reference profile. Candidates classified as ``"PSF"`` are not
        subjected to this comparison by ``_process_one_star``.
        If ``None``, PSF profile comparison is skipped. Default is ``None``.

    profile_cos_distance_threshold : float, optional
        Maximum permitted cosine distance between a candidate's radial
        profile and the reference PSF profile. Candidates exceeding this
        threshold are rejected. Default is 0.045.

    symmetry_ratio_threshold : float, optional
        Maximum permitted mean symmetry ratio between profiles extracted
        along the principal axes of the fitted Gaussian. Candidates
        exceeding this threshold are rejected. Default is 0.07.

    gaussian_ratio_threshold : float, optional
        Maximum permitted ratio between the fitted Gaussian's standard
        deviations along its two principal axes. The ratio is calculated
        as the larger standard deviation divided by the smaller one.
        Candidates exceeding this threshold are rejected. Default is 1.2.

    std_convergence_threshold : float, optional
        Relative change threshold used to determine convergence of the
        fitted radial-profile standard deviation during the radius search.
        Default is 0.01.

    gaussian_fit_max_sigma : float, optional
        Maximum permitted ratio of the current search radius to the
        previous fitted radial-profile standard deviation once convergence
        has been detected. Default is 3.

    do_parallel : bool, optional
        If ``True``, process candidates concurrently using a
        ``ThreadPoolExecutor`` with up to 30 worker threads. If ``False``,
        process candidates sequentially. Default is ``False``.

    show_plots : bool, optional
        If ``True``, display the image using a logarithmic intensity
        normalization and overlay a circle for each detected candidate,
        centered on its original centroid and with radius equal to its
        estimated radius. Candidate identifiers are also displayed.
        Rejected candidates are shown with a zero-radius circle.
        Default is ``False``.

    Returns
    -------
    numpy.ndarray or None
        A filtered view of ``stars_detected`` containing only candidates
        whose estimated radius is greater than zero. The returned array
        retains the original structured dtype and all existing fields,
        including the newly assigned ``'r'`` and ``'std'`` fields.

        Returns ``None`` if ``stars_detected`` is empty or evaluates to
        ``False``.

    Raises
    ------
    ValueError
        If ``min_radius`` is less than 1, or if ``max_radius`` is less
        than ``min_radius``.

    Notes
    -----
    The image background level and its standard deviation are estimated
    using ``sigma_clipping``. The background level is used during candidate
    processing; the estimated background standard deviation is calculated
    but is not otherwise used in this function.

    When ``image_psf`` is provided, its radial profile is calculated using
    the image center and radii from zero up to, but excluding, half the
    image width, rounded down. The resulting profile is used as the
    reference for candidate validation.

    Candidate processing is delegated to ``_process_one_star`` in
    sequential mode and to ``_star_processing_parallel_wrapper`` in
    parallel mode.

    The input array is modified in place, even though the returned array
    contains only accepted candidates. Consequently, rejected candidates
    remain in ``stars_detected`` with their computed ``'r'`` and ``'std'``
    values, typically both zero.

    The returned filtered array is a view rather than an independent copy.
    Modifications to its elements may therefore affect the corresponding
    elements in ``stars_detected``.

    Notes
    -----
    This function depends on ``sigma_clipping``, ``RadialProfile``,
    ``_process_one_star``, and, when parallel processing is enabled,
    ``_star_processing_parallel_wrapper``. Plotting requires Matplotlib.

    """

    if not stars_detected:
        return None
    if min_radius < 1:
        raise ValueError("Minimum radius cannot be less than 1.")
    if max_radius < min_radius:
        raise ValueError("Maximum radius cannot be less than minimum radius.")

    image_background, image_background_std = sigma_clipping(image)
    stars_detected['r'] = 0
    stars_detected['std'] = 0.0
    if image_psf is not None:
        rp_psf = RadialProfile(image_psf, (image_psf.shape[1]//2, image_psf.shape[0]//2),
                               np.arange(image_psf.shape[1]//2), error=None, mask=None, method = "center")
    else:
        rp_psf = None
    gaussian_fitter = fitting.TRFLSQFitter()

    params = (
        min_radius, max_radius, rp_psf,
        image_background, gaussian_fitter,
        std_convergence_threshold, gaussian_fit_max_sigma,
        slope_threshold, delta_slope_threshold,
        profile_cos_distance_threshold, symmetry_ratio_threshold,
        gaussian_ratio_threshold
    )

    if do_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [
                executor.submit(_star_processing_parallel_wrapper, i, star, image, params)
                for i, star in enumerate(stars_detected)
            ]
            for future in as_completed(futures):
                i, final_radius, final_std = future.result()
                stars_detected[i]['r'] = final_radius
                stars_detected[i]['std'] = final_std
    else:
        for this_star in stars_detected:
            final_radius, final_std = _process_one_star(this_star, image, params)
            this_star['r'] = final_radius
            this_star['std'] = final_std

    final_stars = stars_detected[stars_detected['r'] > 0]

    if show_plots:
        import matplotlib.pyplot as plt
        plt.imshow(image, norm='log', origin='lower', cmap='grey')
        for fstr in stars_detected:
            circle = plt.Circle((fstr['x_centroid'], fstr['y_centroid']), fstr['r'], color="orange", fill=False,
                                linewidth=1.5)
            plt.gca().add_artist(circle)
            plt.text(fstr['x_centroid'], fstr['y_centroid'], s=f"{fstr['id']}")
        plt.show()

    return final_stars

if __name__ == "__main__":
    quit()
