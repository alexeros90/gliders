from photutils.segmentation import deblend_sources, detect_threshold, SegmentationImage
from utils import separate_segments, get_nearby_z_sources, coords_to_pixels, get_image_centroid
from skimage.draw import polygon
import sep

def get_deblended_map(image, seg_map, npixels, levels, config, deblend_tool = 'sep'):

    if deblend_tool == "phut":
        segmap_deblend_obj = deblend_sources(image, seg_map, npixels=npixels,
                                             nlevels=levels,
                                             contrast=config.deblend_contrast, progress_bar=False,
                                             mode=config.deblend_mode,
                                             connectivity=config.connect_mode)

    else:
        threshold = detect_threshold(image, nsigma=config.galaxy_detection_threshold)
        sources_sxt_table, segmap_deblend = sep.extract(image.astype(np.float32),
                                                        thresh=threshold[0][0], segmentation_map=True,
                                                        minarea=npixels, deblend_nthresh=levels,
                                                        deblend_cont=config.deblend_contrast)

        seg_map_mask = seg_map.make_source_mask()
        segmap_deblend[~seg_map_mask] = 0.
        # if use_shape:
        segmap_deblend = separate_segments(segmap_deblend)
        segmap_deblend_obj = SegmentationImage(segmap_deblend)

    return segmap_deblend_obj

def analyze_deblended_map(segmap_deblend, main_label, phot_sources, redshift, z_interaction, wcs = None,
                          use_profile = False, show_plots = False):
    if isinstance(wcs, type(None)):
        print("World Coordinate System (WCS) object must be provided to find nearby redshifts")
        return segmap_deblend

    if show_plots:
        fg, axo = plt.subplots(1, 3)
        axo[0].imshow(segmap_deblend, origin="lower")

    non_main_segmap = segmap_deblend.copy()
    non_main_segmap.remove_label(main_label)
    safe_label = int(np.max(segmap_deblend.labels) + 1)
    image_bounds = get_box_bounds(wcs, segmap_deblend.shape)
    image_h, image_w = segmap_deblend.data.shape
    print("Looking for nearby sources with available redshift")
    nearby_sources = get_nearby_z_sources(image_bounds)
    nearby_sources = coords_to_pixels(nearby_sources, wcs)
    # print(nearby_sources.to_string())
    for seg_label in non_main_segmap.labels:
        seg_mask = non_main_segmap.data == seg_label
        source_is_safe = True
        source_is_safe_confirmed = False
        for zs_pos_x, zs_pos_y, zs_z in zip(nearby_sources['x_pos'], nearby_sources['y_pos'],
                                            nearby_sources['z']):
            zs_pos_x_pix = int(np.clip(round(zs_pos_x), 0, image_w - 1))
            zs_pos_y_pix = int(np.clip(round(zs_pos_y), 0, image_h - 1))
            if seg_mask[zs_pos_y_pix, zs_pos_x_pix]:
                z_sep = np.abs(zs_z - redshift)
                if show_plots:
                    axo[0].text(zs_pos_x, zs_pos_y, f"{zs_z:.4f}")
                if z_sep > z_interaction:
                    source_is_safe = False
                else:
                    source_is_safe_confirmed = True
                    source_is_safe = True
                    break
        if source_is_safe and not source_is_safe_confirmed and use_profile:
            current_dist_to_centroid = np.max(segmap_deblend.shape)
            seg_centroid = get_image_centroid(seg_mask)
            closest_type = None
            for ts_pos_x, ts_pos_y, ts_type in zip(phot_sources['xcentroid'], phot_sources['ycentroid'],
                                                   phot_sources['type']):
                ts_pos_x_pix = int(np.clip(round(ts_pos_x), 0, image_w - 1))
                ts_pos_y_pix = int(np.clip(round(ts_pos_y), 0, image_h - 1))
                if seg_mask[ts_pos_y_pix, ts_pos_x_pix]:
                    if show_plots:
                        axo[0].text(ts_pos_x, ts_pos_y, ts_type)
                    dist_to_centroid = np.hypot(seg_centroid[0] - ts_pos_x, seg_centroid[1] - ts_pos_y)
                    if dist_to_centroid < current_dist_to_centroid:
                        current_dist_to_centroid = dist_to_centroid
                        closest_type = ts_type
            if closest_type == "ATF" or closest_type == "PSF":
                source_is_safe = False

        if source_is_safe:
            # Moving safe segment to temporary label (will merge later)
            segmap_deblend.reassign_label(label=seg_label, new_label=safe_label)

    if show_plots:
        axo[1].imshow(segmap_deblend, origin="lower")
    segmap_deblend = merge_touching_regions(segmap_deblend, main_label, safe_label)
    if show_plots:
        axo[2].imshow(segmap_deblend, origin="lower")
        plt.show()

    return segmap_deblend

def remove_deblended_objects(image, segmap_deblend, main_label, overlapping_portion_threshold, pix_scale):

    background, background_std = sigma_clipping(image, 3)
    all_sources_mask = segmap_deblend.make_source_mask()
    sec_labels = segmap_deblend.labels
    sec_labels = sec_labels[sec_labels != main_label]
    # First we sort the objects by proximity to main source (the furthest first)
    main_obj_centroid = get_image_centroid(segmap_deblend.data == main_label)
    distances_to_main_centroid = []
    for this_source_label in sec_labels:
        sec_centroid = get_image_centroid(segmap_deblend.data == this_source_label)
        distances_to_main_centroid.append(np.sqrt((sec_centroid[0] - main_obj_centroid[0]) ** 2 + (sec_centroid[1] - main_obj_centroid[1]) ** 2))
    distances_to_main_centroid = np.array(distances_to_main_centroid)
    sort_idx = np.argsort(distances_to_main_centroid)
    sort_idx = sort_idx[::-1]
    sec_labels = sec_labels[sort_idx]

    # Now we treat each object
    for this_source_label in sec_labels:
        this_source_mask = grow_mask(segmap_deblend.data == this_source_label, 2)
        all_but_this_source_mask = all_sources_mask & ~this_source_mask
        target_pix_number = np.sum(this_source_mask)
        mask_radius = round(np.sqrt(target_pix_number) / 2)
        mask_growth = round(mask_radius / 2)
        # We grow the region but make sure that it does not invade any other regions besides the background so we only mask the most outer pixels
        this_source_mask_adjusted = grow_mask(this_source_mask, mask_growth) & ~all_but_this_source_mask
        in_contact_with_main = are_regions_in_contact(segmap_deblend.data, main_label, this_source_label)
        if not in_contact_with_main:
            image = replace_mask_with_noise(image, this_source_mask_adjusted, n_pixels=mask_growth, scale=pix_scale)
            continue
        secondary_is_contained = check_if_region_is_contained(segmap_deblend.data, this_source_label,
                                                              main_label)
        if secondary_is_contained:
            image = interpolate_image(image, this_source_mask, method="inpaint")
            image = add_noise_to_region(image, this_source_mask, noise_center=0,
                                        noise_std=background_std, scale=pix_scale)
        else:
            overlapping_portion = estimate_overlapping_portion(segmap_deblend.data, main_label,
                                                               this_source_label)
            if overlapping_portion < overlapping_portion_threshold:
                image = replace_mask_with_noise(image, this_source_mask_adjusted, n_pixels=mask_growth, noise_factor=5, scale = pix_scale, use_median=False, edge_smoothing_factor = 0.4)
            else:
                influence_radius = mask_radius * overlapping_portion * 2
                image = replace_mask_with_noise(image, this_source_mask_adjusted, noise_factor=5, scale = pix_scale)
                layer_noise_sample = np.random.normal(loc=background, scale=background_std,
                                                      size=np.sum(this_source_mask_adjusted))
                image = contour_interpolation(this_source_mask_adjusted, image, connectivity=8,
                                              noise_sample=layer_noise_sample, decay_threshold=0.1,
                                              influence_radius=influence_radius)

    return image

def estimate_overlapping_portion(segmap, main_label, target_label):
    main_region_mask = segmap == main_label
    contour_pixels = sort_contour_coordinates(main_region_mask)
    starting_point = get_starting_point(contour_pixels, segmap, target_label)
    contour_pixels = shift_contour_to_starting_point(contour_pixels, starting_point)
    endpoints = contour_contact_endpoints(contour_pixels,segmap,target_label, 4, return_indices=True)
    if endpoints[2] == None or endpoints[3] == None:
        return 0
    overlapping_pixels = count_target_in_closed_segment(contour_pixels, segmap, target_label, endpoints[2], endpoints[3])
    target_pixels = np.sum(segmap == target_label)
    # print(overlapping_pixels)
    # print(target_pixels)
    return overlapping_pixels/target_pixels

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

def check_if_region_is_contained(segmap, inner_label, outer_label, connectivity=8):
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

def are_regions_in_contact(array, region1_values, region2_values):
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

def merge_touching_regions(segm, main_label, seg_label, connectivity=8):
    """
    Check all disjoint regions with the same label (seg_label) and merge
    only those that touch the main_label region into main_label.

    Parameters
    ----------
    segm : photutils.segmentation.SegmentationImage
        Input segmentation map.
    main_label : int
        Label of the main segment.
    seg_label : int
        Label to test for contact against main_label.
    connectivity : {4, 8}, optional
        Pixel connectivity definition (default=8).

    Returns
    -------
    new_segm : photutils.segmentation.SegmentationImage
        Updated segmentation image with touching regions merged into main_label.
    """

    data = segm.data.copy()
    structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else np.array(
        [[0,1,0],[1,1,1],[0,1,0]], dtype=bool
    )

    main_mask = data == main_label
    seg_mask = data == seg_label

    # Label disjoint subregions within seg_label
    subregions, n_subregions = label(seg_mask, structure=structure)

    if n_subregions == 0:
        return segm.copy()

    dilated_main = binary_dilation(main_mask, structure=structure)

    new_data = data.copy()

    for i in range(1, n_subregions + 1):
        sub_mask = subregions == i
        # If any pixel in the subregion touches main_label, merge it
        if np.any(dilated_main & sub_mask):
            new_data[sub_mask] = main_label

    return SegmentationImage(new_data)
