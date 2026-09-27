import numpy as np
from photutils.profiles import RadialProfile
from scipy.stats import pearsonr
from astropy.modeling import models, fitting
from .utils import get_image_centroid, sigma_clipping, grow_pixel_circular, get_profile_gradient


def get_profile_average(profile):
    if (len(profile) % 2) == 0:
        mid_offset = 0
    else:
        mid_offset = 1
    half_1 = profile[int(len(profile) // 2):]
    half_2 = profile[:int(len(profile) // 2 + mid_offset)][::-1]
    return (half_1 + half_2) / 2

def get_mean_profile_symmetry_ratio(profile_1, profile_2, null_value):
    if null_value in profile_1 or null_value in profile_2:
        return None
    profile_1_av = get_profile_average(profile_1)
    profile_2_av = get_profile_average(profile_2)
    max_extension = np.min((len(profile_1_av), len(profile_2_av)))
    profile_2_av = profile_2_av[:max_extension]
    profile_1_av = profile_1_av[:max_extension]
    deviation = np.sum(np.pow(profile_1_av - profile_2_av, 2)) / (max_extension)
    return get_symmetry_ratio(profile_1, profile_2, deviation)

def get_symmetry_ratio(profile_1, profile_2, deviation):
    min_value = np.min((np.min(profile_1), np.min(profile_2)))
    max_value = np.max((np.max(profile_1), np.max(profile_2)))
    similarity_tolerance = np.pow((max_value - min_value), 2)
    return np.sqrt(deviation / similarity_tolerance)

def extract_centered_profile(image, mask, center, direction, radius, fill_value=-999):
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

def extract_all_profiles(image, mask, center, radius, fill_value=-999):
    return {
        'horizontal': extract_centered_profile(image, mask, center, 'horizontal', radius, fill_value),
        'vertical': extract_centered_profile(image, mask, center, 'vertical', radius, fill_value),
        'diagonal_main': extract_centered_profile(image, mask, center, 'diagonal_main', radius, fill_value),
        'diagonal_anti': extract_centered_profile(image, mask, center, 'diagonal_anti', radius, fill_value),
    }

def patch_profile(profile, null_value=-999):
    idxs = np.where(profile == null_value)[0]
    for this_idx in idxs:
        if profile[-this_idx - 1] != -999:
            profile[this_idx] = profile[-this_idx - 1]
    return profile

def get_distance_to_closest_edge(image_shape, position):
    edges = (image_shape[0] - 1, image_shape[1] - 1)
    distance = np.min([position[0], edges[0] - position[0], position[1], edges[1] - position[1]])
    return distance

def compare_profiles(reference, profile):
    """
    Compare two profiles (equal length vectors).

    Returns:
        pearson_corr : float
            Pearson correlation coefficient
        p_value_corr : float
            Two-tailed p-value for correlation significance test
        euclidean_dist : float
            Euclidean distance between the profiles
        chi2 : float
            Chi-square statistic
        chi2_red : float
            Reduced chi-square statistic
        p_value_chi2 : float
            P-value for chi-square test
    """

    reference = np.asarray(reference, dtype=float)
    profile = np.asarray(profile, dtype=float)

    if reference.shape != profile.shape:
        raise ValueError("Profiles must have the same length.")

    # Pearson correlation + p-value
    pearson_corr, p_value_corr = pearsonr(reference, profile)

    # Euclidean distance
    euclidean_dist = np.linalg.norm(reference - profile)

    # Cosine distance
    cos_dist = 1-np.sum(reference * profile)/(np.sqrt(np.sum(np.power(reference,2)))*np.sqrt(np.sum(np.power(profile,2))))

    return pearson_corr, p_value_corr, euclidean_dist, cos_dist

def get_mean_profile_symmetry_ratio_sum(profiles, fill_value):
    orth_ratio = get_mean_profile_symmetry_ratio(profiles['horizontal'], profiles['vertical'], fill_value)
    diag_ratio = get_mean_profile_symmetry_ratio(profiles['diagonal_main'], profiles['diagonal_anti'], fill_value)
    if isinstance(orth_ratio, type(None)):
        orth_ratio = 100
    if isinstance(diag_ratio, type(None)):
        diag_ratio = 100
    return orth_ratio + diag_ratio

def process_one_star(this_star, image, params):
    (
        min_radius, max_radius, half_image_min, rp_psf,
        rp_error, prof_method, image_background, gaussian_fitter,
        std_convergence_threshold, gaussian_fit_max_sigma,
        slope_threshold, delta_slope_threshold,
        image_psf, profile_cos_distance_threshold,
        image_size_y, image_size_x,
        mean_symmetry_ratio_threshold, gaussian_ratio_threshold,
        fill_value
    ) = params

    y_cen, x_cen = this_star['y_centroid'], this_star['x_centroid']
    candidate_pos = (int(round(y_cen)), int(round(x_cen)))
    candidate_centroid = candidate_pos
    # Clip centroid so it is always within the image.
    candidate_centroid = (np.clip(candidate_centroid[0], 0, image.shape[0] - 1),
                          np.clip(candidate_centroid[1], 0, image.shape[1] - 1))
    distance_to_closest_edge = get_distance_to_closest_edge(image.shape, candidate_centroid)
    relocated = False
    final_radius = 0
    final_profile = None
    final_std = None
    profile_std = None

    # ----------------------------------------------------------
    # MAIN RADIUS LOOP
    # ----------------------------------------------------------
    for radius in range(min_radius, max_radius + 1):
        if radius > half_image_min:
            break
        if distance_to_closest_edge < radius / 2 and not relocated:
            mask = grow_pixel_circular(image.shape, candidate_pos, radius)
            new_cc = get_image_centroid(image * image * mask)
            if new_cc is not None:
                candidate_centroid = (new_cc[1], new_cc[0])
                relocated = True

        edge_radii = np.arange(radius + 1)
        rp = RadialProfile(image, (candidate_centroid[1], candidate_centroid[0]),
                           edge_radii, error=rp_error, mask=None, method=prof_method)

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

        profile_slopes, profile_delta_slopes = get_profile_gradient(rp.profile)
        max_delta_slope = np.max(profile_delta_slopes)

        if len(profile_delta_slopes) > 1:
            slope_last = profile_slopes[-1]
            slope_prev = profile_slopes[-2]
            delta_last = profile_delta_slopes[-1]
            delta_prev = profile_delta_slopes[-2]

            if slope_last > slope_prev and np.max(profile_slopes) > slope_threshold:
                if delta_last < delta_prev and delta_last < delta_slope_threshold:
                    break

        if max_delta_slope > 0:
            final_radius = radius
            final_profile = rp.profile
            final_std = profile_std


    # ----------------------------------------------------------
    # POST-PROCESSING IF STAR PASSED RADIUS LOOP
    # ----------------------------------------------------------
    if final_radius <= 0:
        return final_radius, final_std
    else:
        if this_star['type'] != "PSF" and image_psf is not None:
            psf_prof = rp_psf.profile[:len(final_profile)]
            if len(psf_prof) < len(final_profile):
                tail = np.full((len(final_profile) - len(psf_prof),), psf_prof[-1])
                psf_prof = np.append(psf_prof, tail)
            corr, pval, euc_dist, cos_dist = compare_profiles(psf_prof, final_profile)
            if cos_dist > profile_cos_distance_threshold:
                return 0, 0

        # symmetry tests
        y0, x0 = candidate_centroid
        cutout = image[max(y0 - final_radius, 0):min(y0 + final_radius, image_size_y),
                 max(x0 - final_radius, 0):min(x0 + final_radius, image_size_x)]
        cutout_y, cutout_x = np.indices(cutout.shape)
        init = models.Gaussian2D(amplitude=cutout.max(),
                                 x_mean=cutout.shape[1] / 2, y_mean=cutout.shape[0] / 2,
                                 x_stddev=final_radius / 2, y_stddev=final_radius / 2)
        init.x_mean.bounds = (0, cutout.shape[1])
        init.y_mean.bounds = (0, cutout.shape[0])

        fit = gaussian_fitter(init, cutout_x, cutout_y, cutout)
        x_std = fit.x_stddev.value
        y_std = fit.y_stddev.value
        gaussian_ratio = np.exp(abs(np.log(x_std / y_std)))

        mask = grow_pixel_circular(image.shape, candidate_centroid, final_radius)
        profiles = extract_all_profiles(image, mask, candidate_centroid, final_radius, fill_value)
        for key in profiles:
            profiles[key] = patch_profile(profiles[key], fill_value)

        mean_symmetry_ratio = get_mean_profile_symmetry_ratio_sum(profiles, fill_value)
        if (mean_symmetry_ratio > mean_symmetry_ratio_threshold or
                gaussian_ratio > gaussian_ratio_threshold):
            return 0, 0

        # Success — update and return the result
        return final_radius, final_std

def parallel_wrapper(i, star, image, params):
    """We wrap so the executor knows the index and can return it."""
    final_radius, final_std = process_one_star(star,image,params)

    return i, final_radius, final_std

def star_detection_filtering(image, stars_detected, min_radius=2, max_radius=15, slope_threshold = -0.008,
                             delta_slope_threshold = 0.008, image_psf = None, profile_cos_distance_threshold = 0.045,
                             mean_symmetry_ratio_threshold = 0.07, gaussian_ratio_threshold = 1.2, std_convergence_threshold = 0.01,
                             gaussian_fit_max_sigma = 3, do_parallel = True, show_plots = False):
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
            The threshold for the convergence of the profile stddev. If the stddev changes less than this value between two consecutive radii, we consider it converged
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
    image_size_y, image_size_x = image.shape
    rp_error = np.full_like(image, image_background_std)
    stars_detected['r'] = 0
    stars_detected['std'] = 0.0
    half_image_min = np.min(image.shape) / 2
    prof_method = "center"
    if image_psf is not None:
        psf_background, psf_background_std = sigma_clipping(image_psf)
        rp_psf_error = np.zeros_like(image_psf) + psf_background
        rp_psf = RadialProfile(image_psf, (image_psf.shape[1]//2, image_psf.shape[0]//2),
                               np.arange(image_psf.shape[1]//2), error=rp_psf_error, mask=None, method = prof_method)
    else:
        rp_psf_error = None
        rp_psf = None
    gaussian_fitter = fitting.TRFLSQFitter()

    params = (
        min_radius, max_radius, half_image_min, rp_psf,
        rp_error, prof_method, image_background, gaussian_fitter,
        std_convergence_threshold, gaussian_fit_max_sigma,
        slope_threshold, delta_slope_threshold,
        image_psf, profile_cos_distance_threshold,
        image_size_y, image_size_x,
        mean_symmetry_ratio_threshold, gaussian_ratio_threshold,
        fill_value
    )

    if do_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [
                executor.submit(parallel_wrapper, i, star, image, params)
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
            thecol = "orange"
            circle = plt.Circle((fstr['x_centroid'], fstr['y_centroid']), fstr['r'], color=thecol, fill=False,
                                linewidth=1.5)
            plt.gca().add_artist(circle)
            plt.text(fstr['x_centroid'], fstr['y_centroid'], s=f"{fstr['id']}")
        plt.show()

    return final_stars

if __name__ == "__main__":
    quit()
