import numpy as np
import asyncio
import cv2
from collections import Counter
from scipy.ndimage import label, generate_binary_structure, binary_dilation, binary_erosion
from scipy.ndimage import convolve as sci_convolve
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.interpolate import LinearNDInterpolator
import astropy.units as u
from astropy.coordinates import SkyCoord


# ------------------------------------------------
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

def list_rounding(input_list, decimals=3):
    output_list = []
    for i_item in input_list:
        if isinstance(i_item, float):
            o_item = round(i_item, decimals)
        else:
            o_item = i_item
        output_list.append(o_item)
    return output_list

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

def GetAxesDims(NoOfPlots, StartHorizontal = True, bias = 0):
    BaseAxes = [1,1]
    Enough = False
    if StartHorizontal:
        Dim1, Dim2 = 1,0
        h_bias = bias
        v_bias = 0
    else:
        Dim1, Dim2 = 0,1
        v_bias = bias
        h_bias = 0
    while not Enough:
        if (BaseAxes[Dim1] * BaseAxes[Dim2]) >= NoOfPlots:
            Enough = True
        else:
            if BaseAxes[Dim1] - h_bias > BaseAxes[Dim2] - v_bias:
                BaseAxes[Dim2] += 1
            else:
                BaseAxes[Dim1] += 1
    return BaseAxes

def GetNextAxis(Current_Axis = [0,0], shape = [1,1], StartHorizontal = True):
    if StartHorizontal:
        if Current_Axis[1] >= shape[1] - 1:
            Next_Axis = [Current_Axis[0] + 1, 0]
        else:
            Next_Axis = [Current_Axis[0], Current_Axis[1]+1]
    else:
        if Current_Axis[0] >= shape[0] - 1:
            Next_Axis = [0, Current_Axis[1] + 1]
        else:
            Next_Axis = [Current_Axis[0] + 1, Current_Axis[1]]
    return Next_Axis

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

def ScientificNotationFormat(number=2.5465168e-20, decimals = 2):
    new_number = str(number)
    if "e" in new_number:
        new_number = str(f'{number:.{decimals}e}')
        new_number = new_number.split("e")[0] + " \\times 10^{" + new_number.split("e")[1] + "}"
    else:
        new_number = str(f'{number:.{decimals}}')
    return new_number

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


def center_crop_around_centroid(data, centroid):
    """
    Crop the image centered around the centroid, ensuring the shape stays even and valid.
    """
    cx, cy = map(int, centroid)
    h, w = data.shape

    y_extent = min(cy, h - cy)
    x_extent = min(cx, w - cx)

    return data[cy - y_extent:cy + y_extent, cx - x_extent:cx + x_extent]

def replace_mask_with_noise(input_image, mask, noise_factor=3, base_line_factor = 0, use_median=False,
                             smoothing_factor = 0.2,linear_image=True,
                             n_pixels=3, edge_smoothing_factor=0.5, noise_sample = None, scale = 1):
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
    if isinstance(noise_sample, type(None)):

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
            neighbor_mask = prev_layer_mask[i0:i1, j0:j1]
            neighbor_vals = neighborhood[neighbor_mask]

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

def sigma_clipping(image, sigma_factor=3, convergence_tolerance=0.01,
                   iteration_limit=50, use_median=False, robust=True):
    # Will return mean and standard deviation or median and iqr depending on use_median

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
        sigma = np.std(masked_image, ddof=0)  # Use ddof=0 for full population std

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

def sigma_clipping_asymmetryc(image, sigma_factor=3, convergence_tolerance=0.01, iteration_limit=50, robust=True):
    # Will return mean and standard deviation or median and iqr depending on use_median
    iter_count = 0
    mask = np.ones(image.shape, dtype=bool)

    # Initial stats
    masked_image = image[mask]
    central_measure = np.median(masked_image)
    q25, q75 = np.percentile(masked_image, [25, 75])
    sigma = (q25, q75)
    upper_limit_distance = q75 - central_measure

    # Initial threshold
    threshold = central_measure + sigma_factor * upper_limit_distance
    new_mask = image < threshold

    # quit()
    while iter_count < iteration_limit:
        masked_image = image[new_mask]
        if robust:
            q_lo, q_hi = np.percentile(masked_image, [5, 95])
            robust_mask = (masked_image >= q_lo) & (masked_image <= q_hi)
            trimmed = masked_image[robust_mask]
        else:
            trimmed = masked_image

        new_central = np.median(trimmed)
        q25, q75 = np.percentile(trimmed, [25, 75])
        sigma = (q25, q75)
        upper_limit_distance = q75 - central_measure

        # Check convergence
        if np.abs(new_central - central_measure) < (np.abs(central_measure) * convergence_tolerance):
            break

        if new_mask.sum() == mask.sum():
            break

        central_measure = new_central
        mask = new_mask
        threshold = central_measure + sigma_factor * upper_limit_distance
        new_mask = image < threshold
        iter_count += 1

    return central_measure, sigma

def pad_with_slack(data, slack_factor=2, fill_value=0):
    h, w = data.shape
    if isinstance(data[0, 0], float):
        fill_value = float(fill_value)
    new_h, new_w = h * slack_factor, w * slack_factor
    padded = np.full((new_h, new_w), fill_value)
    y_start = (new_h - h) // 2
    x_start = (new_w - w) // 2
    y_end = y_start + h
    x_end = x_start + w
    padded[y_start:y_end, x_start:x_end] = data
    return padded

def check_contact(segmap, label_1, label_2, connectivity=8):
    from scipy.ndimage import binary_dilation
    """
    Check if two labeled segments are touching in a segmentation map.

    Parameters
    ----------
    segmap : 2D array-like (int)
        Segmentation map with integer labels.
    label_1 : int
        Label of the main segment.
    label_2 : int
        Label of the secondary segment to test.
    connectivity : {4, 8}, optional
        Pixel connectivity. If 8, consider diagonal touching as contact.

    Returns
    -------
    bool
        True if label_2 touches label_1; otherwise False.
    """
    # Get boolean masks
    mask_1 = segmap == label_1
    mask_2 = segmap == label_2

    if not np.any(mask_1) or not np.any(mask_2):
        return False  # one or both segments not present

    # Define the structure element for dilation
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    elif connectivity == 4:
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    else:
        raise ValueError("connectivity must be 4 or 8")

    # Dilate main region slightly and check for overlap
    dilated_main = binary_dilation(mask_1, structure=structure)
    contact = np.any(dilated_main & mask_2)

    return bool(contact)

def get_image_centroid(image):
    x_coords = np.linspace(0, image.shape[1] - 1, image.shape[1]).astype('int')
    y_coords = np.linspace(0, image.shape[0] - 1, image.shape[0]).astype('int')
    x_mean = np.mean(image, axis=0)
    y_mean = np.mean(image, axis=1)
    x_mean_sum = np.sum(x_mean)
    y_mean_sum = np.sum(y_mean)
    if x_mean_sum <= 0 or y_mean_sum <= 0:
        return None
    x_centroid = int(np.sum(x_mean * x_coords) / np.sum(x_mean))
    y_centroid = int(np.sum(y_mean * y_coords) / np.sum(y_mean))
    return (x_centroid, y_centroid)

def area_under_the_curve(x, y, limits = (0,1)):
    """
    Compute the area under the curve y = f(x) for a portion of the x domain
    specified by normalized limits in [0, 1].

    Parameters:
    - x: 1D numpy array of x values (must be sorted in ascending order)
    - y: 1D numpy array of y values corresponding to x
    - limits: tuple (low, high) with values between 0 and 1, indicating
              the fractional range of the x domain to integrate over

    Returns:
    - area: float, area under the curve between the specified limits
    """

    if not (0 <= limits[0] < limits[1] <= 1):
        raise ValueError("Limits must be a tuple (low, high) with 0 <= low < high <= 1")

    x = np.asarray(x)
    y = np.asarray(y)

    # Normalize bounds
    x_min, x_max = x[0], x[-1]
    x_low = x_min + limits[0] * (x_max - x_min)
    x_high = x_min + limits[1] * (x_max - x_min)

    # Combine original x and y with interpolated endpoints if needed
    x_new = []
    y_new = []

    for bound in [x_low, x_high]:
        if bound in x:
            x_new.append(bound)
            y_new.append(y[np.where(x == bound)[0][0]])
        else:
            # Linear interpolation between the two points that surround the bound
            idx = np.searchsorted(x, bound)
            if idx == 0 or idx == len(x):
                raise ValueError("Integration bounds out of x range.")
            x1, x2 = x[idx - 1], x[idx]
            y1, y2 = y[idx - 1], y[idx]
            y_interp = y1 + (y2 - y1) * (bound - x1) / (x2 - x1)
            x_new.append(bound)
            y_new.append(y_interp)

    # Create full x and y arrays in the integration region
    mask = (x > x_low) & (x < x_high)
    x_region = np.concatenate(([x_low], x[mask], [x_high]))
    y_region = np.concatenate(([y_new[0]], y[mask], [y_new[1]]))

    # Integrate
    area = np.trapezoid(y_region, x_region)
    return area

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
# ------------------------------------------------


def has_false_neighbor(mask: np.ndarray, region_mask: np.ndarray) -> bool:
    """
    Returns True if any cell in region_mask has a False neighbor in mask.
    """
    # Create a structuring element to define 4- or 8-connected neighborhood
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])  # 8-connected. Use [[0,1,0],[1,1,1],[0,1,0]] for 4-connected.

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


def _run_async_fallback(coro):
    import threading

    result_container = {}

    def runner():
        result_container["result"] = asyncio.run(coro)

    t = threading.Thread(target=runner)
    t.start()
    t.join()

    return result_container["result"]

async def _get_nearby_z_sources_async(image_bounds, timeout):
    from DESI_Query import get_desi_sources
    from SDSS_Query import get_sdss_square_region
    from NED_Query import get_ned_square_region
    from GAMA_Query import get_gama_sources

    loop = asyncio.get_running_loop()

    tasks = {
        "DESI": loop.run_in_executor(None, get_desi_sources,image_bounds[0], image_bounds[1], timeout),
        "GAMA": loop.run_in_executor(None, get_gama_sources,image_bounds[0], image_bounds[1], timeout),
        "SDSS": loop.run_in_executor(None, get_sdss_square_region,image_bounds[0], image_bounds[1], 19, timeout),
        "NED": loop.run_in_executor(None, get_ned_square_region,image_bounds[0], image_bounds[1],True, timeout),
    }

    results = {}
    completed = await asyncio.gather(*tasks.values(),return_exceptions=True)

    for name, result in zip(tasks.keys(), completed):
        if isinstance(result, Exception):
            print(f"Function {name} failed: {result}")
            results[name] = None
        else:
            results[name] = result

    return results

def get_nearby_z_sources(image_bounds, timeout=60):
    """
    Synchronous wrapper that runs async queries safely.
    """
    try:
        results = asyncio.run(_get_nearby_z_sources_async(image_bounds, timeout))
    except RuntimeError:
        # In case an event loop already exists (very rare here, but safe)
        results = _run_async_fallback(_get_nearby_z_sources_async(image_bounds, timeout))

    all_dfs = []
    for key, df in results.items():
        if df is not None and not df.empty:
            all_dfs.append(df[["object_id", "ra", "dec", "z"]])

    if all_dfs:
        all_z_sources = pd.concat(all_dfs, ignore_index=True)
    else:
        all_z_sources = pd.DataFrame(columns=["object_id", "ra", "dec", "z"])

    all_z_sources = all_z_sources.dropna(subset=["z"])
    all_z_sources["object_id"] = all_z_sources["object_id"].astype(str)
    return all_z_sources

def contour_interpolation(region_mask, image, connectivity=8, add_noise=True, noise_sample=None, decay_threshold=0.1,
                          influence_radius=5):
    # Will get the parameters to then perform an outside-in layer averaging based interpolation with layered_contour_averaging()
    if add_noise and isinstance(noise_sample, type(None)):
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
                averaged_image[y, x] = (
                                               (1 - neighbor_weight) * averaged_image[y, x] +
                                               neighbor_weight * np.mean(neighbors)
                                       ) + noise_pix

        current_mask = binary_erosion(current_mask)
        previous_contour_mask = contour_mask.copy()

    return averaged_image
