import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import asyncio
import cv2
from collections import Counter
from scipy.ndimage import label, generate_binary_structure, binary_dilation, binary_erosion
from scipy.ndimage import convolve as sci_convolve
from skimage.draw import polygon
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.interpolate import LinearNDInterpolator
import astropy.units as u
from astropy.coordinates import SkyCoord
import time

from sympy.physics.units import newton


def smooth_region_edge(image, region_mask, bordering_mask = None, layers = 5, background = 0, background_std = 0,
                       pix_scale = 1, linear_weights = False):
    if bordering_mask is None:
        bordering_mask = ~region_mask
    contact_region = grow_mask(region_mask, 1) & bordering_mask
    current_layer = contact_region.copy()
    previous_layer = region_mask.copy()
    if linear_weights:
        weights = np.linspace(1, 0, layers)
    else:
        weights = np.geomspace(1, 0.05, layers)
    for layer_index in range(layers):
        layer_indices = np.argwhere(current_layer)
        for i, j in layer_indices:
            i0, i1 = max(i - 1, 0), min(i + 2, image.shape[0])
            j0, j1 = max(j - 1, 0), min(j + 2, image.shape[1])
            previous_layer_mask = previous_layer[i0:i1, j0:j1]
            neighborhood = image[i0:i1, j0:j1]
            neighbor_vals = neighborhood[previous_layer_mask]
            neighbor_mean = np.mean(neighbor_vals)
            image[i, j] = image[i, j] * weights[layer_index] + neighbor_mean * (1 - weights[layer_index])
        previous_layer = current_layer | previous_layer
        current_layer = grow_mask(current_layer, 1)

    image = add_noise_to_region(image, previous_layer & ~region_mask, noise_center=background,
                                noise_std=background_std, scale=pix_scale)

    return image

def get_aspect_ratio(mask):
    from skimage.measure import regionprops
    props = regionprops(mask.astype(int))[0]
    length = props.major_axis_length
    width = props.minor_axis_length
    return width/length

def check_edge_contact(mask):
    edges = np.concatenate([
        mask[0, :],
        mask[-1, :],
        mask[1:-1, 0],
        mask[1:-1, -1]
    ])
    return np.any(edges)

def get_main_source_label(seg_map, source_pos=(0, 0), table = []):
    main_label = seg_map.data[source_pos[1],source_pos[0]]
    if main_label > 0:
        return main_label
    else:
        closest_dist = np.sqrt(pow(seg_map.shape[0], 2) + pow(seg_map.shape[1], 2))
        for src in table:
            this_dist = np.sqrt(pow(source_pos[0] - src['xcentroid'], 2) + pow(source_pos[1] - src['ycentroid'], 2))
            if this_dist < closest_dist:
                main_label = src['label']
                closest_dist = this_dist
        return main_label

def contour_contact_endpoints(contour, label_image, target_value, connectivity=4, return_indices=False):
    """
    Given a traced contour (list of (row, col)), find the first and last points
    along the contour that touch a pixel with `target_value`.

    Parameters
    ----------
    contour : list of (row, col)
        Ordered contour coordinates.
    label_image : 2D np.ndarray
    target_value : int
    connectivity : int (4 or 8)
    return_indices : bool
        If True, also return indices of these points in contour.

    Returns
    -------
    (first_point, last_point) or (first_point, last_point, first_idx, last_idx)
    """
    if connectivity == 4:
        neighbor_offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    elif connectivity == 8:
        neighbor_offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    contact_indices = []
    contact_positions = []
    for idx, (r, c) in enumerate(contour):
        for dr, dc in neighbor_offsets:
            nr, nc = r+dr, c+dc
            if 0 <= nr < label_image.shape[0] and 0 <= nc < label_image.shape[1]:
                if label_image[nr, nc] == target_value and (r,c) not in contact_positions:
                    contact_indices.append(idx)
                    contact_positions.append((r,c))
                    break  # no need to check more neighbors

    if not contact_indices:
        return None if not return_indices else (None, None, None, None)

    first_idx = contact_indices[0]
    last_idx = contact_indices[-1]
    first_point = contour[first_idx]
    last_point = contour[last_idx]

    if return_indices:
        return first_point, last_point, first_idx, last_idx
    else:
        return first_point, last_point

def sort_contour_coordinates(mask, connectivity=4):
    """
    Trace the contour of a binary mask in order (Moore-Neighbor tracing).
    Returns a list of (row, col) coordinates in walking order.
    Works with narrow 1-pixel-wide structures and avoids early termination.
    """
    if connectivity == 4:
        # Right-hand rule (clockwise search)
        directions = [(-1,0), (0,1), (1,0), (0,-1)]
    elif connectivity == 8:
        directions = [(-1,0), (-1,1), (0,1), (1,1),
                      (1,0), (1,-1), (0,-1), (-1,-1)]
    else:
        raise ValueError("connectivity must be 4 or 8")


    # Find starting boundary pixel: topmost, then leftmost
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []
    start = (rows.min(), cols[rows.argmin()])

    # Start facing "west" (so that we search clockwise around the shape)
    current = start
    dir_idx = 0  # index into directions
    contour = [current]
    first_time = True

    while True:
        # Turn left until you find a neighbor inside the mask
        for i in range(len(directions)):
            check_idx = (dir_idx + i) % len(directions)
            dr, dc = directions[check_idx]
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                if mask[nr, nc]:
                    # Found next boundary pixel
                    current = (nr, nc)
                    contour.append(current)
                    dir_idx = (check_idx - 1) % len(directions)  # Turn left relative to movement
                    break

        if current == start and not first_time:
            break
        first_time = False

    return contour

def shift_contour_to_starting_point(contour_pixels, starting_point):
    # Shift the list to start from the starting_point
    try:
        start_idx = contour_pixels.index(starting_point)
    except ValueError:
        raise ValueError("Starting point not found in contour pixels.")

    shifted = contour_pixels[start_idx:] + contour_pixels[:start_idx]

    # Remove consecutive duplicates
    cleaned = [shifted[0]]
    cleaned.extend(p for i, p in enumerate(shifted[1:], start=1) if p != shifted[i - 1])

    # Ensure the contour is closed
    if cleaned[0] != cleaned[-1]:
        cleaned.append(cleaned[0])

    return cleaned

def estimate_overlapping_portion(segmap, main_label, target_label):
    main_region_mask = segmap == main_label
    contour_pixels = sort_contour_coordinates(main_region_mask)
    starting_point = get_starting_point(contour_pixels, segmap, target_label)
    contour_pixels = shift_contour_to_starting_point(contour_pixels, starting_point)
    endpoints = contour_contact_endpoints(contour_pixels,segmap,target_label, 4, return_indices=True)
    if isinstance(endpoints[2], type(None)) or isinstance(endpoints[3], type(None)):
        return 0
    overlapping_pixels = count_target_in_closed_segment(contour_pixels, segmap, target_label, endpoints[2], endpoints[3])
    target_pixels = np.sum(segmap == target_label)
    # print(overlapping_pixels)
    # print(target_pixels)
    return overlapping_pixels/target_pixels

def get_starting_point(contour_pixels, labels, target_value, connectivity=8):
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    rows, cols = labels.shape

    for pix in contour_pixels:
        if all(
            not (0 <= pix[0] + dr[0] < rows and 0 <= pix[1] + dr[1] < cols) or
            labels[pix[0] + dr[0], pix[1] + dr[1]] != target_value
            for dr in directions
        ):
            return pix

    return contour_pixels[0]  # fallback if no good candidate found

def count_target_in_closed_segment(contour, label_image, target_value, first_idx, last_idx):
    """
    Count pixels with value == target_value that are enclosed by:
    - contour[first_idx:last_idx+1]
    - straight line from contour[last_idx] to contour[first_idx]

    Parameters
    ----------
    contour : list of (row, col)
    label_image : 2D numpy array
    target_value : int
    first_idx, last_idx : int
        Indices in contour that mark the endpoints.

    Returns
    -------
    count : int
        Number of enclosed pixels with target_value.
    """

    # extract contour segment
    if first_idx <= last_idx:
        segment = contour[first_idx:last_idx + 1]
    else:
        # wrap around contour list
        segment = contour[first_idx:] + contour[:last_idx + 1]

    # add straight line between last -> first
    r0, c0 = segment[-1]
    r1, c1 = segment[0]
    # Bresenham-style integer line
    line_rr, line_cc = np.linspace(r0, r1, num=max(abs(r1 - r0), abs(c1 - c0)) + 1, dtype=int), \
        np.linspace(c0, c1, num=max(abs(r1 - r0), abs(c1 - c0)) + 1, dtype=int)
    line_coords = list(zip(line_rr, line_cc))

    closed_coords = segment + line_coords

    # polygon fill
    poly_r = [p[0] for p in closed_coords]
    poly_c = [p[1] for p in closed_coords]
    rr, cc = polygon(poly_r, poly_c, shape=label_image.shape)

    # count pixels with target value inside polygon
    mask = (label_image[rr, cc] == target_value)
    return int(np.sum(mask))

def get_box_bounds(wcs, ref_image_shape):
    nx, ny = ref_image_shape
    corners = np.array([[0, 0], [0, ny], [nx, 0], [nx, ny]])
    sky = wcs.all_pix2world(corners, 0)
    ra = sky[:, 0]
    dec = sky[:, 1]
    ramin, ramax = ra.min(), ra.max()
    decmin, decmax = dec.min(), dec.max()
    return ((ramin, ramax), (decmin, decmax))

def convolve_with_psf(image_data, kernel_std=2.0):
    if kernel_std <= 0:
        return image_data
    kernel = Gaussian2DKernel(kernel_std)  # FWHM = 3.0
    kernel.normalize()  # make sure kernel adds up to 1
    psf = kernel.array  # we only need the numpy array
    convolved_image = convolve(image_data, psf)
    return convolved_image

def grow_mask(input_mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    Expands the `True` regions of a 2D boolean mask by a given number of pixels.

    Parameters:
    - mask: 2D numpy array of bools
    - pixels: Number of pixels to expand in all directions

    Returns:
    - Expanded boolean mask
    """
    if pixels <= 0:
        return input_mask

    # Create a square structuring element (disk) for isotropic dilation

    y, x = np.ogrid[-pixels:pixels + 1, -pixels:pixels + 1]
    structure = x ** 2 + y ** 2 <= pixels ** 2

    # Apply dilation
    expanded_mask = binary_dilation(input_mask, structure=structure)
    return expanded_mask

def shrink_mask(input_mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    Shrinks the `True` regions of a 2D boolean mask by a given number of pixels.

    Parameters:
    - mask: 2D numpy array of bools
    - pixels: Number of pixels to shrink in all directions

    Returns:
    - Shrunk boolean mask
    """
    if pixels <= 0:
        return input_mask

    # Create a square structuring element (disk) for isotropic erosion
    y, x = np.ogrid[-pixels:pixels + 1, -pixels:pixels + 1]
    structure = x ** 2 + y ** 2 <= pixels ** 2

    # Apply erosion
    shrunk_mask = binary_erosion(input_mask, structure=structure)
    return shrunk_mask

def normalize_data(data, robust = False):
    # Input must be a numpy array
    if robust:
        data_min = np.percentile(data, 1)
        data_max = np.percentile(data, 99)
    else:
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)

    if data_max == data_min:
        data_max = data_min + 1e-6
    data = (data - data_min) / (data_max - data_min)
    return np.clip(data, 0, 1)

def get_profile(image, center_pos = None, radius = 1):
    if center_pos is None:
        center_pos = image.shape[0] // 2, image.shape[1] // 2
    current_mask = np.zeros_like(image, dtype=bool)
    current_mask[center_pos] = True
    profile = []
    for layer in range(1, radius + 1):
        new_mask = np.zeros_like(image, dtype=bool)
        new_mask[center_pos] = True
        new_mask = grow_mask(new_mask, layer)
        profile_mask = np.logical_xor(new_mask, current_mask)
        mean_value = float(np.mean(image[profile_mask]))
        profile.append(mean_value)
        current_mask = new_mask
    return profile

def get_profile_from_border(image, mask=None, n_layers=1):
    if n_layers < 0:
        raise ValueError("n_layers must be >= 0")

    if mask is None:
        mask = np.zeros(image.shape[:2], dtype=bool)
        center = (image.shape[0] // 2, image.shape[1] // 2)
        mask[center] = True

    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {image.shape[:2]}"
        )

    max_layers = min(image.shape[:2]) // 2
    n_layers = min(n_layers, max_layers)

    profile = []

    previous = shrink_mask(mask, 1)
    profile_mean = float(np.mean(image[mask ^ previous]))
    profile.append(profile_mean)

    previous = mask
    for layer_idx in range(1, n_layers + 1):
        current = grow_mask(mask, layer_idx)
        profile_mean = float(np.mean(image[current ^ previous]))
        profile.append(profile_mean)
        previous = current

    return profile

def get_profile_gradient(profile):
    # This function assumes the x values increase by 1.
    profile = np.asarray(profile)
    slopes = np.diff(profile)
    slope_deltas = np.diff(slopes, prepend=0)
    return slopes, slope_deltas

def add_noise_to_region(image, mask, noise_sample = None, noise_center = None, noise_std = None, scale = 1):
    if not isinstance(noise_sample, type(None)):
        noise_center = np.mean(noise_sample)
        noise_img = np.random.choice(noise_sample, size=(image.shape[0], image.shape[1]))
    elif noise_center is None or noise_std is None:
        raise Exception("Must provide either noise_sample or both noise_center and noise_std.")
    else:
        noise_img = np.random.normal(loc=noise_center, scale=noise_std, size=(image.shape[0], image.shape[1]))

    if scale > 1:
        noise_img = rescale_image(noise_img, scale)

    image[mask] = image[mask] + noise_img[mask] - noise_center
    return image

def rescale_image(image, scale=1.0):
    """
    Will downgrade an image resolution by scale. Recommended mostly for noise.

    Parameters
    ----------
    image : 2D array
        Input image (noise or any array).
    scale : float
        Factor for diagonal neighbors: weight = scaling_factor*(scale-1)/2

    Returns
    -------
    new_image : 2D array
        Image after applying scaling_factor blending with neighbors.
    """

    if scale < 1:
        print("Scale cannot be less than 1. Forcing scale to 1")
        scale = 1
    scaling_factor = 2 * (scale - 1) / np.power(scale, 2)

    original_mean = np.mean(image)
    # Define neighbor kernels
    k_orth = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=float)
    k_diag = np.array([[1, 0, 1],
                       [0, 0, 0],
                       [1, 0, 1]], dtype=float)

    # Sum of neighbor values
    sum_orth = sci_convolve(image, k_orth, mode='reflect')
    sum_diag = sci_convolve(image, k_diag, mode='reflect')

    # Compute weighted sum of neighbors
    neighbor_sum = scaling_factor * sum_orth + (scaling_factor * (scale - 1) / 2.0) * sum_diag

    # Compute total neighbor weights
    weight_orth = sci_convolve(np.ones_like(image), k_orth, mode='reflect') * scaling_factor
    weight_diag = sci_convolve(np.ones_like(image), k_diag, mode='reflect') * (scaling_factor * (scale - 1) / 2.0)
    neighbor_weight = weight_orth + weight_diag

    # Weighted average
    neighbor_mean = neighbor_sum / neighbor_weight

    # Update pixels with central weight = 1 - scaling_factor
    new_image = (1 - scaling_factor) * image + neighbor_mean
    new_image -= (np.mean(new_image) - original_mean)
    return new_image

def smooth_blend(noise, image, mask, blend_width=10):
    from scipy.ndimage import distance_transform_edt
    """
    Blend image into noise smoothly inside mask.

    Parameters
    ----------
    noise : np.ndarray
        Background array.
    image : np.ndarray
        Foreground array of same shape as noise.
    mask : np.ndarray (bool)
        Region of interest.
    blend_width : int
        Width (in pixels) of the transition zone.
    """
    # Distance inside mask (to nearest edge)
    dist_in = distance_transform_edt(mask)
    # Distance outside mask (to nearest edge)
    dist_out = distance_transform_edt(~mask)

    # Weight of image:
    #  - 1 deep inside mask
    #  - 0 outside mask
    #  - smooth transition in between
    w = np.clip((blend_width - dist_out) / blend_width, 0, 1)

    # Blend
    blended = w * image + (1 - w) * noise
    return blended

def replace_mask_with_noise(input_image, mask, noise_factor=3, base_line_factor = 0, use_median=False,
                             smoothing_factor = 0.2, linear_image=True,
                             n_pixels=3, edge_smoothing_factor=0, noise_sample = None, scale = 1):
    """
        Replace pixels inside a mask with noise and optionally smooth them with neighbor pixels.

        Parameters
        ----------
        input_image : 2D array
            Original image.
        mask : 2D bool array
            True for pixels to replace/smooth.
        noise_factor : float
            Range factor for noise sampling.
        base_line_factor : float
            Additional baseline added to background.
        use_median : bool
            Whether to use median in sigma-clipping.
        smoothing_factor : float
            Weight of equal contribution smoothing of all neighbors (0–1), so that noise looks a bit more natural
        linear_image : bool
            Whether image is linear (affects background estimation).
        n_pixels : int
            Number of edge layers for edge smoothing.
        edge_smoothing_factor : float
            Weight for edge pixels blending.
        noise_sample : array or None
            Precomputed noise values. If None, sampled automatically.
        scale : float
            Factor for scaling neighbor contribution. scale is used to get the scaling_factor, which is the same as smoothing factor but with the purpose of making the image look as if it had lower resolution. scale = 1 does nothing on the image

        Returns
        -------
        image : 2D array
            Modified image.
        """

    if scale < 1:
        print("Scale cannot be less than 1. Forcing scale to 1")
        scale = 1
    scaling_factor = 2 * (scale - 1) / np.power(scale, 2)
    smoothing_factor = np.clip(smoothing_factor, 0, 1)
    edge_smoothing_factor = np.clip(edge_smoothing_factor, 0, 1)
    n_pixels = max(n_pixels, 0)

    image = input_image.copy()
    if noise_sample is None:

        image_chk = image[~np.isnan(image)]
        background, background_std = sigma_clipping(image, use_median = use_median, iteration_limit=100)

        if not linear_image:
            q_img = input_image[input_image <= background + background_std / 2]
            background = np.median(q_img)
            background_std = np.std(q_img, ddof=1)
        background += np.abs(background) * base_line_factor

        # Generate noise
        noise_sample = image_chk[
            (image_chk >= background - background_std * noise_factor) &
            (image_chk <= background + background_std * noise_factor)
            ]

    # Fill mask pixels with noise
    num_noise_pixels = np.count_nonzero(mask)
    noise_iter = iter(np.random.choice(noise_sample, size=num_noise_pixels))
    image[mask] = np.fromiter(noise_iter, dtype=image.dtype, count=num_noise_pixels)

    if smoothing_factor > 0 or scaling_factor > 0:

        pre_smoothing_mask_mean = np.mean(image[mask])
        # Kernels
        k_orth = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=float)

        # Diagonal (corners)
        k_diag = np.array([[1, 0, 1],
                           [0, 0, 0],
                           [1, 0, 1]], dtype=float)

        # All 8 neighbors
        k_all = np.ones((3, 3), dtype=float)
        k_all[1, 1] = 0

        masked_img = np.where(mask, image, 0.0)
        masked_mask = mask.astype(float)

        # Sums of neighbors
        sum_orth = sci_convolve(masked_img, k_orth, mode='constant', cval=0.0)
        sum_diag = sci_convolve(masked_img, k_diag, mode='constant', cval=0.0)
        sum_all = sci_convolve(masked_img, k_all, mode='constant', cval=0.0)

        # counts (for smoothing_factor neighbors — only count neighbors inside mask)
        count_all = sci_convolve(masked_mask, k_all, mode='constant', cval=0.0)
        mean_all = np.zeros_like(image, dtype=float)
        valid_all = (count_all > 0) & mask
        mean_all[valid_all] = sum_all[valid_all] / count_all[valid_all]

        # Final update
        new_img = image.copy()
        valid = mask.copy()
        scaling_factor_diag = scaling_factor * (scale - 1) / 2.0
        scaling_factor_total = scaling_factor + scaling_factor_diag
        new_img[valid] = (
                (1 - scaling_factor_total) * (1 - smoothing_factor) * image[valid]
                + scaling_factor * sum_orth[valid]
                + (scaling_factor * (scale - 1) / 2.0) * sum_diag[valid]
                + smoothing_factor * mean_all[valid]
        )
        image = new_img
        # Correcting mean for artificial increase due to averaging
        image[mask] -= (np.mean(image[mask]) - pre_smoothing_mask_mean)


    if edge_smoothing_factor > 0:
        current_mask = mask.copy()
        prev_layer_mask = ~mask  # First outer reference is outside the mask
        prev_layer_values = image.copy()
        for layer_index in range(n_pixels):
            eroded = binary_erosion(current_mask)
            layer_mask = current_mask ^ eroded  # Pixels in current layer only
            layer_indices = np.argwhere(layer_mask)
            for i, j in layer_indices:
                i0, i1 = max(i - 1, 0), min(i + 2, image.shape[0])
                j0, j1 = max(j - 1, 0), min(j + 2, image.shape[1])
                neighborhood = prev_layer_values[i0:i1, j0:j1]
                prev_layer_neighbor_mask = prev_layer_mask[i0:i1, j0:j1]
                neighbor_vals = neighborhood[prev_layer_neighbor_mask]
                if neighbor_vals.size > 0:
                    influence = np.mean(neighbor_vals)
                    image[i, j] = (
                            (1 - edge_smoothing_factor) * image[i, j] +
                            edge_smoothing_factor * influence
                    )

            # Update for next layer inward
            current_mask = eroded
            prev_layer_mask = layer_mask.copy()
            prev_layer_values = image.copy()

            if not np.any(current_mask):
                break  # No more layers inside the mask

    return image

def sigma_clipping(image, sigma_factor=3, convergence_tolerance=0.001, iteration_limit=50, use_median=False,
                   robust=True):
    # Will return mean and standard deviation or median and iqr depending on use_median
    # If convergence_tolerance, the clipping will go on until there is no change in the central measurement or the iteration_limit is exceeded

    iter_count = 0
    mask = np.ones(image.shape, dtype=bool)

    # Initial stats
    masked_image = image[mask]
    if use_median:
        central_measure = np.median(masked_image)
        q25, q75 = np.percentile(masked_image, [25, 75])
        sigma = q75 - q25
    else:
        central_measure = np.mean(masked_image)
        sigma = np.std(masked_image, ddof=0)

    # Initial threshold
    threshold = central_measure + sigma_factor * sigma
    new_mask = image < threshold

    while iter_count < iteration_limit:
        masked_image = image[new_mask]
        if robust:
            q_lo, q_hi = np.percentile(masked_image, [5, 95])
            robust_mask = (masked_image >= q_lo) & (masked_image <= q_hi)
            trimmed = masked_image[robust_mask]
        else:
            trimmed = masked_image

        if use_median:
            new_central = np.median(trimmed)
            q25, q75 = np.percentile(trimmed, [25, 75])
            new_sigma = q75 - q25
        else:
            new_central = np.mean(trimmed)
            new_sigma = np.std(trimmed, ddof=0)

        # Check convergence
        convergence_threshold = np.abs(sigma) * convergence_tolerance
        mean_convergence = np.abs(new_central - central_measure) <= convergence_threshold
        std_convergence = np.abs(new_sigma - sigma) <= convergence_threshold
        if mean_convergence and std_convergence:
            break

        if new_mask.sum() == mask.sum():
            break

        central_measure = new_central
        sigma = new_sigma
        mask = new_mask
        threshold = central_measure + sigma_factor * sigma
        new_mask = image < threshold
        iter_count += 1

    return central_measure, sigma

def linear_kernel(type=8, center_weight=2.0):
    if type == 8:
        kernel = np.ones((3, 3))  # includes center
    elif type == 4:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        kernel = np.array(
            [[0.25, 0.5, 0.75, 0.5, 0.25], [0.5, 0.75, 1, 0.75, 0.5], [0.75, 1, 1, 1, 0.75], [0.5, 0.75, 1, 0.75, 0.5],
             [0.25, 0.5, 0.75, 0.5, 0.25]])

    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = center_weight
    # normalize kernel so sum of neighbor weights = 1 (optional)
    kernel = kernel / kernel.sum()
    return kernel

def get_image_centroid(image):
    x_coords = np.linspace(0, image.shape[1] - 1, image.shape[1]).astype('int')
    y_coords = np.linspace(0, image.shape[0] - 1, image.shape[0]).astype('int')
    x_mean = np.mean(image, axis=0)
    y_mean = np.mean(image, axis=1)
    x_mean_sum = np.sum(x_mean)
    y_mean_sum = np.sum(y_mean)
    if x_mean_sum <= 0 or y_mean_sum <= 0:
        return None
    x_centroid = int(np.sum(x_mean * x_coords) / x_mean_sum)
    y_centroid = int(np.sum(y_mean * y_coords) / y_mean_sum)
    return (x_centroid, y_centroid)

def grow_pixel_circular(mask_shape, center, radius):
    """
    Grow a pixel in a circular region around the center.

    Parameters:
    - mask_shape: tuple, shape of the output mask (height, width)
    - center: tuple, (row, col) coordinate of the pixel to grow
    - radius: int, radius of the circular growth

    Returns:
    - A boolean mask of the same shape with True in the circular region
    """
    rows, cols = np.ogrid[:mask_shape[0], :mask_shape[1]]
    cy, cx = center
    dist_sq = (rows - cy)**2 + (cols - cx)**2
    circle_mask = dist_sq <= radius**2
    return circle_mask

def adjust_image_resolution_to_scale(image, scale = 1):
    if scale < 1:
        print("Scale cannot be less than 1. Forcing scale to 1")
        scale = 1
    scaling_factor = 2 * (scale - 1) / np.power(scale, 2)
    pre_smoothing_image_mean = np.mean(image)
    # Kernels
    k_orth = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=float)

    # Diagonal (corners)
    k_diag = np.array([[1, 0, 1],
                       [0, 0, 0],
                       [1, 0, 1]], dtype=float)

    # Sums of neighbors
    sum_orth = sci_convolve(image, k_orth, mode='constant', cval=0.0)
    sum_diag = sci_convolve(image, k_diag, mode='constant', cval=0.0)

    # Final update
    scaling_factor_diag = scaling_factor * (scale - 1) / 2.0
    scaling_factor_total = scaling_factor + scaling_factor_diag
    new_image = (
            (1 - scaling_factor_total) * image
            + scaling_factor * sum_orth
            + (scaling_factor * (scale - 1) / 2.0) * sum_diag
    )
    image = new_image
    # Correcting mean for artificial increase due to averaging
    image -= (np.mean(image) - pre_smoothing_image_mean)
    return image

def has_false_neighbor(mask: np.ndarray, region_mask: np.ndarray, connect_type = 4) -> bool:
    """
    Returns True if any cell in region_mask has a False neighbor in mask.
    """
    # Create a structuring element to define 4- or 8-connected neighborhood
    if connect_type == 8:
        structure = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
    else:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])

    # Dilate the region to get the region + its neighbors
    dilated = binary_dilation(region_mask, structure=structure)

    # Exclude original region to keep only the border/neighbor cells
    neighbors = dilated & ~region_mask

    # Now check if any of the neighbors are False in the main mask
    return np.any(~mask & neighbors)

def interpolate_image(image_data, mask_data, mask_radius = 3, method = "inpaint"):
    interpolated_img = image_data.copy()
    if method == "inpaint":
        inpaint_r = np.max((round(mask_radius/2,0),2))
        img_max = np.max(image_data)
        img_min = np.min(image_data)
        mask_data_cv = (mask_data.astype(np.uint8)) * 255
        norm_img = 255*((image_data - img_min) / (img_max - img_min)).astype(np.float32)
        inpainted = cv2.inpaint(norm_img, mask_data_cv, inpaintRadius=inpaint_r, flags=cv2.INPAINT_NS)
        interp_img = inpainted/255 * (img_max - img_min) + img_min
        interpolated_img[mask_data] = interp_img[mask_data]

    else:
        non_masked_coords = []
        non_masked_values = []
        for i in range(mask_data.shape[0]):
            for j in range(mask_data.shape[1]):
                if mask_data[i, j]:
                    continue
                non_masked_coords.append((i, j))
                non_masked_values.append(image_data[i, j])
        interp_img = LinearNDInterpolator(non_masked_coords, non_masked_values)
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                if mask_data[i, j]:
                    interpolated_img[i, j] = interp_img(i, j)

    return interpolated_img

def separate_segments(labels, omit_background=True):
    labels = np.array(labels, copy=True)
    h, w = labels.shape
    structure = generate_binary_structure(2, 1)  # 4-connectivity

    for lbl in np.unique(labels):
        if lbl == -1:
            continue  # skip special labels if desired
        if omit_background and lbl == 0:
            continue  # skip background if requested

        # Binary mask for this label
        mask = (labels == lbl)

        # Label connected components inside this label's mask
        labeled, num = label(mask, structure=structure)

        if num <= 1:
            continue  # only one region, nothing to do

        # Find largest component ID
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background of component labeling
        largest_id = sizes.argmax()

        for comp_id in range(1, num + 1):
            if comp_id == largest_id:
                continue  # keep main region

            # Coordinates of this small component
            coords = np.argwhere(labeled == comp_id)

            # Find neighboring labels
            neighbor_labels = []
            for x, y in coords:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if labels[nx, ny] != lbl:
                            neighbor_labels.append(labels[nx, ny])

            if neighbor_labels:
                # Most common neighbor label
                new_lbl = Counter(neighbor_labels).most_common(1)[0][0]
                labels[labeled == comp_id] = new_lbl

    return labels

def coords_to_pixels(sources, wcs):
    sources['x_pos'] = -1.
    sources['y_pos'] = -1.
    for zsrc_id, zsrc_ra, zsrc_dec in zip(sources["object_id"], sources["ra"], sources["dec"]):
        z_source_coords = SkyCoord(zsrc_ra * u.deg, zsrc_dec * u.deg, frame='icrs')
        z_source_x_pix, z_source_y_pix = wcs.world_to_pixel(z_source_coords)
        sources.loc[sources['object_id'] == zsrc_id, 'x_pos'] = z_source_x_pix
        sources.loc[sources['object_id'] == zsrc_id, 'y_pos'] = z_source_y_pix
    sources = sources[(sources['x_pos'] >= 0) & (sources['y_pos'] >= 0)]
    return (sources)

def check_ra_range_leap(ra_range):
    dir_dif = abs(ra_range[1] - ra_range[0])
    skip_dif = abs(ra_range[1] + 360 - ra_range[0])
    if dir_dif > skip_dif and (ra_range[1]<ra_range[0]):
        return True
    else:
        return False

async def _get_nearby_z_sources_async(image_bounds, timeout=60, max_attempts=2, retry_wait=2):
    from gliders.z_queries.DESI_Query import get_desi_sources
    from gliders.z_queries.SDSS_Query import get_sdss_square_region
    from gliders.z_queries.NED_Query import get_ned_square_region
    from gliders.z_queries.GAMA_Query import get_gama_sources

    failed_query = False

    tasks = {
        "DESI": _retry_async(lambda: asyncio.to_thread(get_desi_sources, image_bounds[0], image_bounds[1], timeout),
            max_attempts, retry_wait),
        "GAMA": _retry_async(lambda: asyncio.to_thread(get_gama_sources, image_bounds[0], image_bounds[1], timeout),
            max_attempts, retry_wait),
        "SDSS": _retry_async(lambda: asyncio.to_thread(get_sdss_square_region, image_bounds[0], image_bounds[1], 19, timeout),
            max_attempts, retry_wait),
        "NED": _retry_async(lambda: asyncio.to_thread(get_ned_square_region, image_bounds[0], image_bounds[1], True, timeout),
            max_attempts, retry_wait),
    }

    completed = await asyncio.gather(
        *tasks.values(),
        return_exceptions=True
    )

    results = {}
    for name, result in zip(tasks.keys(), completed):
        if isinstance(result, Exception):
            print(f"{name} failed: {result}")
            results[name] = None
            failed_query = True
        else:
            results[name] = result

    return _combine_query_results(results), failed_query

async def _retry_async_OLD(func, max_attempts, wait_seconds = 2):
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(wait_seconds)
    return last_exception


#TEST MEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
async def _retry_async(func, max_attempts, wait_seconds=2):
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception:
            if attempt < max_attempts - 1:
                await asyncio.sleep(wait_seconds)
            else:
                raise

def _retry_sync(func, max_attempts, wait_seconds = 2):
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(wait_seconds)
    raise last_exception

def _get_nearby_z_sources_sync(image_bounds, timeout=60, max_attempts=2, retry_wait=2):
    from gliders.z_queries.DESI_Query import get_desi_sources
    from gliders.z_queries.SDSS_Query import get_sdss_square_region
    from gliders.z_queries.NED_Query import get_ned_square_region
    from gliders.z_queries.GAMA_Query import get_gama_sources

    failed_query = False

    results = {}

    query_functions = {
        "DESI": lambda: get_desi_sources(image_bounds[0], image_bounds[1], timeout),
        "GAMA": lambda: get_gama_sources(image_bounds[0], image_bounds[1], timeout),
        "SDSS": lambda: get_sdss_square_region(image_bounds[0], image_bounds[1], 19, timeout),
        "NED":  lambda: get_ned_square_region(image_bounds[0], image_bounds[1], True, timeout),
    }

    for name, func in query_functions.items():
        try:
            results[name] = _retry_sync(func, max_attempts, retry_wait)
        except Exception as e:
            print(f"{name} failed: {e}")
            results[name] = None
            failed_query = True

    return _combine_query_results(results), failed_query

def _combine_query_results(results_dict):
    """
    Combine the dictionary of query results into a cleaned dataframe.
    """

    valid_dfs = [
        df[["object_id", "ra", "dec", "z"]]
        for df in results_dict.values()
        if df is not None and not df.empty
    ]

    if valid_dfs:
        all_z_sources = pd.concat(valid_dfs, ignore_index=True)
    else:
        all_z_sources = pd.DataFrame(
            columns=["object_id", "ra", "dec", "z"]
        )

    all_z_sources = all_z_sources.dropna(subset=["z"])
    all_z_sources["object_id"] = all_z_sources["object_id"].astype(str)

    return all_z_sources

def get_nearby_z_sources(image_bounds, timeout=60, synchronous=False, attempts = 2):

    if synchronous:
        return _get_nearby_z_sources_sync(image_bounds, timeout, max_attempts = attempts, retry_wait = 2)
    else:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop → safe in scripts
            return asyncio.run(_get_nearby_z_sources_async(image_bounds, timeout, max_attempts = attempts, retry_wait = 2) )
        else:
            raise RuntimeError(
                "Active event loop detected. Use:\n\n"
                "    await get_nearby_z_sources_async(...)"
            )

def contour_interpolation(region_mask, image, connectivity=8, add_noise=True, noise_sample=None, decay_threshold=0.1,
                          influence_radius=5):
    # Will get the parameters to then perform an outside-in layer averaging based interpolation with layered_contour_averaging()
    if add_noise and noise_sample is None:
        background, background_std = sigma_clipping(image, iteration_limit=50, use_median=False)
        noise_sample = np.random.normal(loc=0, scale=background_std, size=np.sum(region_mask))
    elif not add_noise:
        noise_sample = None
    neighbor_initial_weight = np.pow(decay_threshold, (1 / (influence_radius)))
    image = layered_contour_averaging(region_mask, image, neighbor_weight=neighbor_initial_weight,
                                      connectivity=connectivity, max_layers=None, noise_sample=noise_sample)
    return image

def layered_contour_averaging(region_mask, image, neighbor_weight=0.5, max_layers=None,
                              connectivity=4, noise_sample=None):

    assert 0 <= neighbor_weight <= 1, "neighbor_weight must be between 0 and 1"
    assert connectivity in (4, 8), "connectivity must be either 4 or 8"

    averaged_image = image.copy()
    current_mask = region_mask.copy()
    previous_contour_mask = binary_dilation(region_mask) & ~region_mask

    # Neighbor offsets depending on connectivity
    if connectivity == 4:
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8-connectivity
        neighbor_offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if
                            not (dy == 0 and dx == 0)]
    if noise_sample is not None:
        noise_mean = np.mean(noise_sample)
    else:
        noise_mean = 0

    layer_count = 0
    while np.any(current_mask):
        if max_layers is not None and layer_count >= max_layers:
            break
        layer_count += 1

        contour_mask = current_mask & ~binary_erosion(current_mask)
        contour_coords = np.argwhere(contour_mask)

        for y, x in contour_coords:
            neighbors = []
            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    if previous_contour_mask[ny, nx]:
                        neighbors.append(averaged_image[ny, nx])
            if neighbors:
                if noise_sample is not None:
                    noise_pix = np.random.choice(noise_sample)
                else:
                    noise_pix = 0
                averaged_image[y, x] = ((1 - neighbor_weight) * averaged_image[y, x] +
                                         neighbor_weight * np.mean(neighbors)
                                       ) + noise_pix - noise_mean

        current_mask = binary_erosion(current_mask)
        previous_contour_mask = contour_mask.copy()

    return averaged_image
