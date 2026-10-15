import numpy as np
from photutils.profiles import RadialProfile
from scipy.stats import pearsonr
from astropy.modeling import models, fitting
from .utils import get_image_centroid, sigma_clipping, grow_pixel_circular, get_profile_gradient


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

def _extract_centered_profile(image, mask, center, direction, radius, fill_value=-999):
    """
    Extract a profile in the specified direction from the circular mask,
    centered at `center`, preserving image order and inserting `fill_value`
    where valid mask pixels would fall outside the image.
    """
    h, w = image.shape
    y, x = center
    deltas = {
        'horizontal': (0, 1),
        'vertical': (1, 0),
        'diagonal_main': (1, 1),  # \
        'diagonal_anti': (1, -1),  # /
    }

    dy, dx = deltas[direction]
    profile = []
    # Sample from -radius to +radius
    for step in range(-radius, radius + 1):
        ny = y + dy * step
        nx = x + dx * step

        if 0 <= ny < h and 0 <= nx < w:
            if mask[ny, nx]:
                profile.append(image[ny, nx])
        else:
            # Pixel lies outside image, but would be part of circular mask
            dist2 = (dy * step) ** 2 + (dx * step) ** 2
            if dist2 <= radius ** 2:
                profile.append(fill_value)

    return np.array(profile)

def _extract_all_profiles(image, mask, center, radius, fill_value=-999):
    """
    Extract profiles in all four directions from the circular mask centered at `center`.
    """
    return {
        'horizontal': _extract_centered_profile(image, mask, center, 'horizontal', radius, fill_value),
        'vertical': _extract_centered_profile(image, mask, center, 'vertical', radius, fill_value),
        'diagonal_main': _extract_centered_profile(image, mask, center, 'diagonal_main', radius, fill_value),
        'diagonal_anti': _extract_centered_profile(image, mask, center, 'diagonal_anti', radius, fill_value),
    }

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

def _get_mean_profile_symmetry_ratio_sum(profiles, fill_value):
    """
    Calculate the sum of mean symmetry ratios for orthogonal and diagonal profile pairs.
    """
    orth_ratio = _get_mean_profile_symmetry_ratio(profiles['horizontal'], profiles['vertical'], fill_value)
    diag_ratio = _get_mean_profile_symmetry_ratio(profiles['diagonal_main'], profiles['diagonal_anti'], fill_value)
    if orth_ratio is None:
        orth_ratio = 100
    if diag_ratio is None:
        diag_ratio = 100
    return orth_ratio + diag_ratio

def _star_processing_parallel_wrapper(i, star, image, params):
    """
    Wrapper function for parallel processing of stars.
    """
    final_radius, final_std = process_one_star(star,image,params)

    return i, final_radius, final_std

def process_one_star(this_star, image, params):
    """
    Process a single star to determine its radius and profile standard deviation.

    Parameters
    ----------
    this_star : dict
        A dictionary containing information about the star, including its centroid coordinates.
    image : 2D array
        The image in which the star is located.
    params : tuple
        A tuple containing various parameters for processing, including:
        - min_radius : int
        - max_radius : int
        - rp_psf : RadialProfile or None
        - rp_error : 2D array
        - image_background : float
        - gaussian_fitter : astropy.modeling.fitting.Fitter
        - std_convergence_threshold : float
            The relative difference threshold for the convergence of the fitted gaussian profile std. Determines if the
            star radius is compared to the fitted gaussian std (see gaussian_fit_max_sigma).
        - gaussian_fit_max_sigma : float
            The star radius cannot be larger than the fitted gaussian profile std * gaussian_fit_max_sigma. This is
            only checked if the relative difference between the profiles of two consecutive radii is lower than
            std_convergence_threshold
        - slope_threshold : float
        - delta_slope_threshold : float
        - profile_cos_distance_threshold : float
        - mean_symmetry_ratio_threshold : float
        - gaussian_ratio_threshold : float
        - fill_value : float

    Returns
    -------
    final_radius : int
        The determined radius of the star. Returns 0 if the star does not pass the filtering criteria.
    final_std : float
        The standard deviation of the star's profile. Returns 0 if the star does not pass the filtering criteria.
    """

    (
        min_radius, max_radius, rp_psf, rp_error,
        image_background, gaussian_fitter, std_convergence_threshold, gaussian_fit_max_sigma,
        slope_threshold, delta_slope_threshold,
        profile_cos_distance_threshold, mean_symmetry_ratio_threshold,
        gaussian_ratio_threshold, fill_value
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
            # new_cc = get_image_centroid(image * mask) # TODO: Check and test if this is better than the below method
            new_cc = get_image_centroid(image * image * mask)
            if new_cc is not None:
                new_c_x, new_c_y = new_cc
                candidate_eff_pos = (new_c_y, new_c_x)
                relocated = True

        edge_radii = np.arange(radius + 1)
        rp = RadialProfile(image, (candidate_eff_pos[1], candidate_eff_pos[0]),
                           edge_radii, error=rp_error, mask=None, method="center")

        if np.sum(np.abs(rp.profile)) == 0:
            break
        if np.any(rp.data_profile <= image_background):
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

        print("rp.data_profile")
        print(rp.data_profile)
        if len(rp.data_profile) < 2:
            continue

        profile_slopes, profile_delta_slopes = get_profile_gradient(rp.profile)
        max_delta_slope = np.max(profile_delta_slopes)

        if len(profile_delta_slopes) > 1:
            slope_last = profile_slopes[-1]
            slope_prev = profile_slopes[-2]
            delta_last = profile_delta_slopes[-1]
            delta_prev = profile_delta_slopes[-2]

            if slope_last > slope_prev and np.max(profile_slopes) > slope_threshold: # TODO: Check why this.
                if delta_last < delta_prev and delta_last < delta_slope_threshold:
                    break

        if max_delta_slope > 0: # TODO: Check why this.
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

        if (not np.isfinite(x_std) or not np.isfinite(y_std) or x_std <= 0 or y_std <= 0):
            return 0, 0
        gaussian_ratio = max(x_std / y_std, y_std / x_std)

        mask = grow_pixel_circular(image.shape, candidate_eff_pos, final_radius)
        profiles = _extract_all_profiles(image, mask, candidate_eff_pos, final_radius, fill_value)
        for key in profiles:
            profiles[key] = _patch_profile(profiles[key], fill_value)

        mean_symmetry_ratio = _get_mean_profile_symmetry_ratio_sum(profiles, fill_value)
        if (mean_symmetry_ratio > mean_symmetry_ratio_threshold or
                gaussian_ratio > gaussian_ratio_threshold):
            return 0, 0

        # Success — update and return the result
        return final_radius, final_std

def star_detection_filtering(image, stars_detected, min_radius=2, max_radius=15, slope_threshold = -0.008,
                             delta_slope_threshold = 0.008, image_psf = None, profile_cos_distance_threshold = 0.045,
                             mean_symmetry_ratio_threshold = 0.07, gaussian_ratio_threshold = 1.2, std_convergence_threshold = 0.01,
                             gaussian_fit_max_sigma = 3, do_parallel = False, show_plots = False):
    """
        Filtering of detected stars based on radial profile analysis and measuring of their radius.
        Returns a table of stars that passed the filtering with their measured radius and profile stddev.

        Parameters
        ----------
        image : 2D array
            The image that will be analyzed.
        stars_detected: table
            The table of detected stars
        min_radius: int
            Minimum radius in pixels for a star to be accepted
        max_radius: int
            Maximum radius in pixels for a star to be accepted
        slope_threshold: float
            Threshold for the slope of the profile. When this threshold and delta_slope_threshold are met, we define the star's radius
        delta_slope_threshold: float
            Threshold for the change in slope of the profile. When this threshold and delta_slope_threshold are met, we define the star's radius
        image_psf: 2D array
            The image of the PSF to compare the profiles with
        profile_cos_distance_threshold: float
            The maximum cosine distance allowed between the star profile and the PSF profile
        mean_symmetry_ratio_threshold: float
            The maximum mean symmetry ratio allowed for the star profiles
        gaussian_ratio_threshold: float
            The maximum accepted ratio between the x and y sigma of the 2D gaussian fit to the star cutout
        std_convergence_threshold: float
            The threshold for the convergence of the stddev of the fitted gaussian. When the stddev converges, we define the star's radius
        gaussian_fit_max_sigma: float
            The maximum number of sigmas of the fitted gaussian at which we accept the radius.
        do_parallel: bool
            Process stars with parallel computing
        show_plots
            Show the image with the apertures overlaid
        """

    if not stars_detected:
        return None
    if min_radius < 1:
        raise ValueError("Minimum radius cannot be less than 1.")
    if max_radius < min_radius:
        raise ValueError("Maximum radius cannot be less than minimum radius.")

    fill_value = -999
    image_background, image_background_std = sigma_clipping(image)
    rp_error = np.full_like(image, image_background_std)
    # rp_error = np.full_like(image, image_background) #TODO: Check if this is better than the above line
    stars_detected['r'] = 0
    stars_detected['std'] = 0.0
    if image_psf is not None:
        psf_background, psf_background_std = sigma_clipping(image_psf)
        rp_psf_error = np.zeros_like(image_psf) + psf_background
        rp_psf = RadialProfile(image_psf, (image_psf.shape[1]//2, image_psf.shape[0]//2),
                               np.arange(image_psf.shape[1]//2), error=rp_psf_error, mask=None, method = "center")
    else:
        rp_psf = None
    gaussian_fitter = fitting.TRFLSQFitter()

    params = (
        min_radius, max_radius, rp_psf, rp_error,
        image_background, gaussian_fitter,
        std_convergence_threshold, gaussian_fit_max_sigma,
        slope_threshold, delta_slope_threshold,
        profile_cos_distance_threshold, mean_symmetry_ratio_threshold,
        gaussian_ratio_threshold, fill_value
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
            final_radius, final_std = process_one_star(this_star, image, params)
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
