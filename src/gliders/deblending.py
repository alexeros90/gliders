import numpy as np
from photutils.segmentation import SourceCatalog, deblend_sources, detect_threshold, SegmentationImage
from scipy.ndimage import label, generate_binary_structure, binary_dilation
import sep
import warnings
from .utils import (get_main_source_label,separate_segments, get_nearby_z_sources, coords_to_pixels, get_image_centroid, get_box_bounds,
                    sigma_clipping, grow_mask, replace_mask_with_noise, interpolate_image, add_noise_to_region,
                    estimate_overlapping_portion, contour_interpolation, get_aspect_ratio, smooth_region_edge)


def deblend_and_mask_sources(image, image_convolved = None, main_source_segment = None, source_pos = None, config = None,
                             use_redshift = True, ref_redshift = None, max_delta_z = 0.005, use_photo_type = True,
                             phot_sources = None, keep_undecided_segments = True, wcs = None, pix_scale=1,
                             synch_z_query = True, z_query_timeout = 60, show_plots=False, verbose=False):

    # Determining the minimum number of pixels for de-blending
    deblended_flag = False
    failed_z_query = False
    deblending_mask = np.full_like(image, False, dtype=bool)
    if main_source_segment is None:
        if verbose:
            warnings.warn("Invalid main source segment, returning unmodified image and default values.",
            UserWarning, stacklevel = 2)
        return image, deblending_mask, deblended_flag, failed_z_query
    if image_convolved is None:
        image_convolved = image
    if config is None:
        if verbose:
            warnings.warn("No configuration object was provided. Will use default configuration.",
                          UserWarning, stacklevel=2)
        from .cleaning_configs import configs
        config = configs['default']
    if source_pos is None:
        # If no source position is provided, we assume the source is located at the imag center.
        ny, nx = image.shape
        source_pos = (nx // 2, ny // 2)

    if use_redshift:
        if ref_redshift is None:
            use_redshift = False
            if verbose:
                warnings.warn("No reference redshift was provided. Redshift-based deblending will be omitted.",
                              UserWarning, stacklevel=2)
        elif max_delta_z is None: # and ref_redshift:
            max_delta_z = config.deblend_max_delta_z
            if verbose:
                warnings.warn("Non valid input for max_delta_z. Will use value from configuration.",
                              UserWarning, stacklevel=2)
            if max_delta_z is None:
                use_redshift = False
                if verbose:
                    warnings.warn("Non valid value for max_delta_z in configuration. Redshift-based deblending will be omitted.",
                                  UserWarning, stacklevel=2)

    main_source_mask = main_source_segment.make_source_mask()
    source_npixels = main_source_mask.sum()

    npixels_deblend = max(int(source_npixels * config.npixels_deblend_factor), 1)
    # Getting the deblended candidates. We use the convolved image to smoothen the de-blending edges.

    segmap_deblend = get_deblended_map(image_convolved, main_source_segment, config.deblend_tool,
                                       npixels_deblend, config.deblend_contrast, config.deblend_levels,
                                       config.deblend_mode, config.connect_mode, config.galaxy_detection_threshold)
    if segmap_deblend is None:
        if verbose:
            warnings.warn("No sources detected after de-blending, returning unmodified image and default values.",
                UserWarning, stacklevel=2)
        return image, deblending_mask, deblended_flag, failed_z_query

    elif len(segmap_deblend.labels) > 1:
        deb_cat = SourceCatalog(image, segmap_deblend)
        deb_main_label = get_main_source_label(segmap_deblend, source_pos=source_pos, table=deb_cat.to_table())
        segmap_deblend, failed_z_query = analyze_deblended_map(segmap_deblend, deb_main_label, phot_sources, use_redshift, ref_redshift,
                                               max_delta_z, use_photo_type, keep_undecided_segments, wcs, synch_z_query,
                                               z_query_timeout, show_plots, verbose)

        if len(segmap_deblend.labels) > 1:
            image = remove_deblended_objects(image, segmap_deblend, deb_main_label,
                                             overlap_portion_threshold=config.overlap_portion_threshold,
                                             pix_scale=pix_scale)
            deblended_flag = True
            removal_segmap = segmap_deblend.copy()
            removal_segmap.remove_label(deb_main_label)
            deblending_mask = removal_segmap.make_source_mask()

    if verbose:
        print("Detected", len(segmap_deblend.labels), "sources after de-blending.")

    return image, deblending_mask, deblended_flag, failed_z_query


def get_deblended_map(image, seg_map, deblend_tool = 'sep', npixels = None, contrast = 0.005, levels = 32,
                      deblend_mode = 'exponential', connect_mode = 8, detection_threshold = 5):

    if npixels is None:
        npixels = int(image.size*0.005)

    if deblend_tool == "phut":
        segmap_deblend_obj = deblend_sources(image, seg_map, npixels=npixels,
                                             nlevels=levels,
                                             contrast=contrast, progress_bar=False,
                                             mode=deblend_mode,
                                             connectivity=connect_mode)

    else:
        threshold = detect_threshold(image, nsigma=detection_threshold)
        try:
            sources_sxt_table, segmap_deblend = sep.extract(image.astype(np.float32),
                                                            thresh=threshold[0][0], segmentation_map=True,
                                                            minarea=npixels, deblend_nthresh=levels,
                                                            deblend_cont=contrast)
            seg_map_mask = seg_map.make_source_mask()
            segmap_deblend[~seg_map_mask] = 0.
            segmap_deblend = separate_segments(segmap_deblend)
            segmap_deblend_obj = SegmentationImage(segmap_deblend)
        except Exception as e:
            print("Could not perform deblending:")
            print(e)
            return seg_map


    return segmap_deblend_obj

def analyze_deblended_map(segmap_deblend, main_label, phot_sources = None, use_redshift = True, redshift = None,
                          max_delta_z = None, use_photo_type = True, keep_undecided_labels = True, wcs = None,
                          synch_z_query = True, z_query_timeout = 60, show_plots = False, verbose = False):
    #TODO: if use_photo_type is True but no phot_sources list is given, it should be a warning and move on, instead of error as it is now
    failed_z_query = False
    if segmap_deblend is None:
        print("Could not perform deblending: invalid segmentaion map")
        return segmap_deblend, failed_z_query

    segmap_deblend_initial = segmap_deblend.copy()
    safe_label = int(np.max(segmap_deblend.labels) + 1)
    discard_label = safe_label + 1
    z_sources = None
    if use_redshift:
        segmap_deblend, z_sources, failed_z_query = check_deblended_redshifts(segmap_deblend, main_label, safe_label, discard_label, redshift,
                                                                              max_delta_z, wcs = wcs, synch_z_query = synch_z_query, return_segmap_only = False,
                                                                              z_query_timeout = z_query_timeout, show_plots = False, verbose = verbose)
    else:
        if verbose:
            print("Deblending by source redshift is not activated.")
    if show_plots:
        segmap_deblend_z = segmap_deblend.copy()

    if use_photo_type:
        segmap_deblend = check_deblended_photo_sources(segmap_deblend, main_label, safe_label, discard_label,
                                                       phot_sources, show_plots = False, verbose = verbose)
    else:
        if verbose:
            print("Deblending by source photometric type is not activated.")
    if show_plots:
        segmap_deblend_phot = segmap_deblend.copy()

    undecided_segmap = segmap_deblend.copy()

    exclude = [main_label, safe_label, discard_label]

    undecided_labels = [
        label for label in undecided_segmap.labels
        if label not in exclude
    ]

    if len(undecided_labels) > 0:
        if keep_undecided_labels:
            segmap_deblend.reassign_labels(labels=undecided_labels, new_label=safe_label)
        else:
            segmap_deblend.reassign_labels(labels=undecided_labels, new_label=discard_label)

    segmap_deblend = merge_touching_regions(segmap_deblend, main_label, safe_label)
    segmap_deblend_split = segmap_deblend.data.copy()
    segmap_deblend_split[segmap_deblend_split==discard_label] = segmap_deblend_initial.data[segmap_deblend_split==discard_label]
    segmap_deblend = SegmentationImage(segmap_deblend_split)

    if show_plots:
        plot_deblended_maps(segmap_deblend_initial, segmap_deblend_z, segmap_deblend_phot, segmap_deblend, redshift,
                            z_sources, phot_sources)

    return segmap_deblend, failed_z_query

def check_deblended_redshifts(segmap_deblend, main_label, safe_label, discard_label, redshift, max_delta_z, wcs = None,
                              synch_z_query = True, return_segmap_only = False, z_query_timeout = 60, show_plots = False,
                              verbose = False):
    if redshift is None:
        print("The provided redshift is not valid. Redshift-based deblending will be omitted.")
        return segmap_deblend

    if wcs is None:
        print("World Coordinate System (WCS) object must be provided to find nearby redshifts")
        return segmap_deblend

    non_main_segmap = segmap_deblend.copy()
    non_main_segmap.remove_label(main_label)

    if not len(non_main_segmap.labels) > 0:
        if verbose:
            print("No labels found besides main and background.")
        return segmap_deblend

    image_bounds = get_box_bounds(wcs, segmap_deblend.shape)
    image_h, image_w = segmap_deblend.data.shape
    if verbose:
        print("Looking for nearby sources with available redshift")
    nearby_sources, failed_z_query = get_nearby_z_sources(image_bounds, timeout=z_query_timeout, synchronous = synch_z_query, attempts = 2)

    if not len(nearby_sources)>0:
        if verbose:
            print("No nearby redshift sources detected.")
        if return_segmap_only:
            return segmap_deblend
        else:
            return segmap_deblend, nearby_sources, failed_z_query

    nearby_sources = coords_to_pixels(nearby_sources, wcs)
    # print(nearby_sources.to_string())

    if show_plots:
        import matplotlib.pyplot as plt
        fg, axo = plt.subplots(1, 2)
        axo[0].imshow(segmap_deblend, origin="lower")

    labels_to_save = []
    labels_to_discard = []

    zs_x = np.clip(np.rint(nearby_sources['x_pos'].values).astype(int), 0, image_w - 1)
    zs_y = np.clip(np.rint(nearby_sources['y_pos'].values).astype(int), 0, image_h - 1)
    zs_z = nearby_sources['z'].values
    for seg_label in non_main_segmap.labels:
        if seg_label == discard_label or seg_label == safe_label:
            continue
        seg_mask = non_main_segmap.data == seg_label
        inside = seg_mask[zs_y, zs_x]
        if np.any(inside):
            z_seps = np.abs(zs_z[inside] - redshift)
            if np.any(z_seps <= max_delta_z):
                labels_to_save.append(seg_label)
            else:
                labels_to_discard.append(seg_label)
            if show_plots:
                for x, y, z in zip(zs_x[inside], zs_y[inside], zs_z[inside]):
                    axo[0].text(x, y, f"{z:.4f}")

    if len(labels_to_save) > 0:
        segmap_deblend.reassign_labels(labels=labels_to_save, new_label=safe_label)
    if len(labels_to_discard) > 0:
        segmap_deblend.reassign_labels(labels=labels_to_discard, new_label=discard_label)

    if show_plots:
        axo[1].imshow(segmap_deblend, origin="lower")
        plt.show()
    if return_segmap_only:
        return segmap_deblend
    else:
        return segmap_deblend, nearby_sources, failed_z_query

def check_deblended_photo_sources(segmap_deblend, main_label, safe_label, discard_label, phot_sources,
                                  show_plots = False, verbose = False):
    if show_plots:
        import matplotlib.pyplot as plt
        fg, axo = plt.subplots(1, 2)
        axo[0].imshow(segmap_deblend, origin="lower")

    non_main_segmap = segmap_deblend.copy()
    non_main_segmap.remove_label(main_label)

    if not len(non_main_segmap.labels) > 0:
        if verbose:
            print("No labels found besides main and background.")
        return segmap_deblend

    if phot_sources is None:
        if verbose:
            print("No photometric sources found in the image region.")
    elif not len(phot_sources) > 0:
        if verbose:
            print("No photometric sources found in the image region.")
        return segmap_deblend

    image_h, image_w = segmap_deblend.data.shape
    labels_to_discard = []

    ts_x = np.clip(np.rint(phot_sources['xcentroid']).astype(int), 0, image_w - 1)
    ts_y = np.clip(np.rint(phot_sources['ycentroid']).astype(int), 0, image_h - 1)
    ts_type = phot_sources['type']

    for seg_label in non_main_segmap.labels:
        if seg_label == discard_label or seg_label == safe_label:
            continue
        seg_mask = non_main_segmap.data == seg_label
        seg_centroid = get_image_centroid(seg_mask)
        inside = seg_mask[ts_y, ts_x]
        if np.any(inside):
            dx = ts_x[inside] - seg_centroid[0]
            dy = ts_y[inside] - seg_centroid[1]
            dist2 = dx * dx + dy * dy
            idx_closest = np.argmin(dist2)
            closest_type = ts_type[inside][idx_closest]
            if closest_type == "ATF":
                labels_to_discard.append(seg_label)
            elif closest_type == "PSF":
                # We make sure that the segment shape is roughly consistent with a PSF object (star)
                aspect_ratio = get_aspect_ratio(seg_mask)
                if aspect_ratio >= 0.3:
                    labels_to_discard.append(seg_label)
            if show_plots:
                axo[0].text(ts_x[inside][idx_closest], ts_y[inside][idx_closest], closest_type)

    if len(labels_to_discard) > 0:
        segmap_deblend.reassign_labels(labels=labels_to_discard, new_label=discard_label)
    if show_plots:
        axo[1].imshow(segmap_deblend, origin="lower")
        plt.show()
    return segmap_deblend


def remove_deblended_objects(image, segmap_deblend, main_label, overlap_portion_threshold, pix_scale):

    background, background_std = sigma_clipping(image, 3)
    all_sources_mask = segmap_deblend.make_source_mask()
    sec_labels = segmap_deblend.labels
    sec_labels = sec_labels[sec_labels != main_label]
    main_source_mask = segmap_deblend.data == main_label
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
        # this_source_mask = grow_mask(segmap_deblend.data == this_source_label, 2)
        this_source_mask = segmap_deblend.data == this_source_label
        all_but_this_source_mask = all_sources_mask & ~this_source_mask
        target_pix_number = np.sum(this_source_mask)
        mask_radius = round(np.sqrt(target_pix_number) / 2)
        mask_growth = round(mask_radius / 2)
        if mask_growth < 1:
            mask_growth = 1
        # We grow the region but make sure that it does not invade any other regions besides the background so we only mask the most outer pixels
        this_source_mask_adjusted = grow_mask(this_source_mask, mask_growth) & ~all_but_this_source_mask
        in_contact_with_main = are_regions_in_contact(segmap_deblend.data, main_label, this_source_label)
        if not in_contact_with_main:
            image = replace_mask_with_noise(image, this_source_mask_adjusted, n_pixels=mask_growth, scale=pix_scale)
            continue
        secondary_is_contained = check_if_region_is_contained(segmap_deblend.data, this_source_label, main_label)
        if secondary_is_contained:
            image = interpolate_image(image, this_source_mask, method = "inpaint")
            image = add_noise_to_region(image, this_source_mask, noise_center = 0, noise_std = background_std,
                                        scale=pix_scale)
        else:
            overlapping_portion = estimate_overlapping_portion(segmap_deblend.data, main_label, this_source_label)
            if overlapping_portion < overlap_portion_threshold:
                image = replace_mask_with_noise(image, this_source_mask_adjusted, n_pixels=mask_growth, noise_factor=5,
                                                scale = pix_scale, use_median=False, edge_smoothing_factor = 0)
                smoothing_layers = round(np.sqrt(np.sum(main_source_mask)) / (2*np.pi))
                image = smooth_region_edge(image, main_source_mask, this_source_mask_adjusted, smoothing_layers,
                                           background, background_std, pix_scale = pix_scale)
            else:
                influence_radius = mask_radius * overlapping_portion * 2
                image = replace_mask_with_noise(image, this_source_mask_adjusted, noise_factor=5, scale = pix_scale,
                                                edge_smoothing_factor = 0.5)
                layer_noise_sample = np.random.normal(loc=background, scale=background_std,
                                                      size=np.sum(this_source_mask_adjusted))
                image = contour_interpolation(this_source_mask_adjusted, image, connectivity=8,
                                              noise_sample=layer_noise_sample, decay_threshold=0.15,
                                              influence_radius=influence_radius)

    return image

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

def merge_touching_regions(segment, main_label, seg_label, connectivity=8):
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

    data = segment.data.copy()
    structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else np.array(
        [[0,1,0],[1,1,1],[0,1,0]], dtype=bool
    )

    main_mask = data == main_label
    seg_mask = data == seg_label

    # Label disjoint subregions within seg_label
    subregions, n_subregions = label(seg_mask, structure=structure)

    if n_subregions == 0:
        return segment.copy()

    dilated_main = binary_dilation(main_mask, structure=structure)

    new_data = data.copy()

    for i in range(1, n_subregions + 1):
        sub_mask = subregions == i
        # If any pixel in the subregion touches main_label, merge it
        if np.any(dilated_main & sub_mask):
            new_data[sub_mask] = main_label

    return SegmentationImage(new_data)


def plot_deblended_maps(segmap_deblend_initial, segmap_deblend_z, segmap_deblend_phot, segmap_deblend, redshift = None,
                        z_sources = None, phot_sources = None):
    import matplotlib.pyplot as plt
    sdi_reset = segmap_deblend_initial.copy()
    sdz_reset = segmap_deblend_z.copy()
    sdp_reset = segmap_deblend_phot.copy()
    sd_reset = segmap_deblend.copy()
    for seg_map in (sdi_reset, sdz_reset, sdp_reset, sd_reset):
        seg_map.relabel_consecutive()
    fg, axo = plt.subplots(2, 2, figsize=(6, 7))
    fg.suptitle('Deblending maps')
    axo[0, 0].imshow(sdi_reset, origin="lower")
    axo[0, 0].set_title("initial")
    axo[0, 1].imshow(sdz_reset, origin="lower")
    axo[0, 1].set_title("deblend z")
    if redshift is not None:
        axo[0, 1].text(round(sdi_reset.data.shape[1]*0.02),round(sdi_reset.data.shape[0]*0.98), f'main z = {redshift:.4f}', color="lime", fontsize=10, ha='left',va='top')
    # print(z_sources)
    if z_sources is not None:
        for row in z_sources[['x_pos', 'y_pos', 'z']].itertuples(index=False):
            axo[0, 1].text(int(row.x_pos), int(row.y_pos), f'{row.z:.4f}', color="red", fontsize=7, ha='center', va='center')
    axo[1, 0].imshow(sdp_reset.data, origin="lower")
    axo[1, 0].set_title("deblend phot")
    if phot_sources is not None:
        for x, y, type in phot_sources['xcentroid', 'ycentroid', 'type'].iterrows():
            axo[1, 0].text(int(x), int(y), f'{type}', color="red", fontsize=6, ha='center', va='center')
    axo[1, 1].imshow(sd_reset.data, origin="lower")
    axo[1, 1].set_title("final")
    plt.show()