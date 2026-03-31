import numpy as np
from astropy.table import Table, vstack
from photutils.segmentation import SourceCatalog, detect_threshold, detect_sources
from .utils import convolve_with_psf, sigma_clipping, get_main_source_label, check_edge_contact
from .noise_pusher import diminish_noise_level
from .stars import detect_and_mask_stars, star_contamination_check
from .deblending import deblend_and_mask_sources
from .artifacts import detect_and_clean_artifacts
from .others import background_subtraction, mask_secondary_detections, clear_background, segmentation_map_smoothing, make_background_robust
import traceback
import time

def update_sources(sources_table, artifact_mask):
    if (artifact_mask == True).any():
        artifact_segmap = detect_sources(artifact_mask, 0.5, 1, connectivity=8)
        artifact_table = SourceCatalog(artifact_mask, artifact_segmap).to_table()
        artifact_table = artifact_table[['xcentroid', 'ycentroid']]
        artifact_table['type'] = 'ATF'
        sources_table = vstack([sources_table, artifact_table])
    return sources_table

def update_detections(image, config, source_min_pix, source_pos, verbose):
    image_convolved = convolve_with_psf(image, kernel_std=config.psf_std)
    threshold = detect_threshold(image_convolved, nsigma=config.galaxy_detection_threshold)
    seg_map = detect_sources(image_convolved, threshold, source_min_pix, connectivity=config.connect_mode)
    if not seg_map:
        if verbose:
            print("No sources detected in the convolved image, trying with the non-convolved image.")
        seg_map = detect_sources(image, threshold, source_min_pix, connectivity=config.connect_mode)
        if not seg_map:
            raise Exception("No sources were detected in the image.")
    source_cat = SourceCatalog(image, seg_map)
    sources_table = source_cat.to_table()
    main_label = get_main_source_label(seg_map, source_pos = source_pos, table = sources_table)
    main_source_segmap = seg_map.copy()
    main_source_segmap.keep_label(label=main_label)
    return image_convolved, seg_map, main_source_segmap

def clean_image(image, image_psf = None, config = None, source_pos = None, redshift = None, phot_sources = None,
                wcs = None, return_clean_image_only = False, clean_artifacts = True, skip_flagged_artifacts = True,
                reduce_noise_level = False, subtract_background = False, mask_sec_objects = True, mask_stars = True,
                deblend_sources = True, deblend_with_redshift = True, deblend_delta_z = None,
                deblend_with_photo_type = True, robust_background = True, sync_z_query = False,
                clear_all_but_segment = False, smooth_segmap = True, pix_scale = 1, verbose = True, show_plots = False):

    try:
        # ------------ Handling input parameters ------------
        if not np.any(image.astype(bool)):
            raise Exception("Image appears to be empty.")
        if config is None:
            print("No config provided. Using default config.")
            from .cleaning_configs import configs
            config = configs["default"]
        if phot_sources is None:
            phot_sources = Table(names=['type','xcentroid','ycentroid'], dtype=('str', 'float', 'float'))
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
        artifact_flag = False
        deblended_flag = False
        deblend_bad_z_query_flag = False
        big_contaminator_flag = False
        offset_flag = False
        small_source_flag = False
        # ---------------------------------------------------

        # times["initial_images"] = time.perf_counter()

        cleaning_mask = np.full(image.shape, False) # Initializing the overall cleaning mask

        # Detecting and conditionally removing over exposure artifacts
        if clean_artifacts:
            if verbose:
                print("Detecting and cleaning overexposure artifacts...")
            # Non modified images with artifacts generally have pixels with values >> background, so first we filter based on that
            if (np.max(image)/img_bkg_lvl) > config.artifact_candidate_ratio:
                image, artifact_mask, artifact_flag = detect_and_clean_artifacts(image, config, img_bkg_lvl,
                                                                                 pix_scale = pix_scale, main_source_pos=source_pos,
                                                                                 skip_flagged_artifacts = skip_flagged_artifacts, show_plots=show_plots)
                cleaning_mask = artifact_mask | cleaning_mask
                phot_sources = update_sources(phot_sources, artifact_mask) # Adding detected artifacts to photo source table
                # times["artifact_cleaning"] = time.perf_counter()

        if reduce_noise_level:
            image = diminish_noise_level(image, method = "logit", kernel_type = 16, kernel_center_weight = 3.0, smooth_noise_curve = False)
            # smooth_noise_curve must be False, otherwise sep deblending may not work correctly

        if subtract_background:
            image = background_subtraction(image, show_plots=False, normalize=False, detection_threshold=3, psf_kernel_std = config.psf_std)
            # times["background_subtraction"] = time.perf_counter()

        # times["psf_convolution"] = time.perf_counter()

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

        # times["first_detection"] = time.perf_counter()

        # Detecting and masking stars in the image
        if mask_stars:
            # image, main_source_segment, cleaning_mask, cleaning_flags
            if verbose:
                print("Detecting and masking stars...")
            main_source_mask = main_source_segment.make_source_mask()
            image, masked_stars_mask, all_star_apertures = detect_and_mask_stars(image, main_source_mask, source_pos, phot_sources, config, image_psf, pix_scale, show_plots = show_plots)
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
            image, deblending_mask, deblended_flag, deblend_bad_z_query_flag = deblend_and_mask_sources(image, image_convolved, main_source_segment,
                                                                              source_pos, config, deblend_with_redshift,
                                                                              redshift, deblend_delta_z, deblend_with_photo_type,
                                                                              phot_sources, config.keep_undecided_segments,
                                                                              wcs, pix_scale, sync_z_query,show_plots = show_plots)
            cleaning_mask = deblending_mask | cleaning_mask
            # Updating the convolved image and segmentation map of the post-deblend main source
            image_convolved, seg_map, main_source_segment = update_detections(image, config, source_min_pix, source_pos, verbose)

        # Removing all detected sources except the main one.
        if mask_sec_objects:
            if verbose:
                print("Removing secondary objects...")
            main_source_mask = main_source_segment.make_source_mask()
            image, secondary_objects_mask = mask_secondary_detections(image, image_convolved, main_source_mask,
                                                                      config.galaxy_detection_threshold, pix_scale)
            cleaning_mask = secondary_objects_mask | cleaning_mask
            # update_detections() not necessary from here on.

        # times["deblending"] = time.perf_counter()

        # Smoothing the main source segmentation map
        if smooth_segmap:
            main_source_segment = segmentation_map_smoothing(main_source_segment)

        main_source_mask = main_source_segment.make_source_mask()

        # Remove all but main source
        if clear_all_but_segment:
            image, cleaning_mask = clear_background(image, main_source_mask, pix_scale)

        if robust_background & ~clear_all_but_segment:
            image = make_background_robust(image)

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
            offset_flag = True

        cleaning_flags = [int(artifact_flag), int(offset_flag), int(deblended_flag), int(deblend_bad_z_query_flag), int(big_contaminator_flag), int(small_source_flag)]

        if verbose:
            print("Image cleaning finished!")

        if return_clean_image_only:
            return image
        else:
            return image, main_source_segment, cleaning_mask, cleaning_flags

    except Exception as e:
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":

    quit()
