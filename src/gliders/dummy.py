import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from photutils.segmentation import SourceCatalog, detect_threshold, detect_sources, SegmentationImage
# from utils import convolve_with_psf, list_rounding, grow_mask, shrink_mask, adjust_image_resolution_to_scale, sigma_clipping, replace_mask_with_noise, add_noise_to_region, get_profile_gradient, normalize_data, get_image_centroid, get_box_bounds
from image_background_diminisher import diminish_noise_level
from stars import detect_and_mask_stars, star_contamination_check
from deblending import get_deblended_map, analyze_deblended_map
from artifacts import detect_and_clean_artifacts
from others import subtract_background
import traceback
import time

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

def get_clean_image_and_segment(image, image_psf = None, config = None, source_pos = None, redshift = None,
                              z_interaction=0.0035, tractor_sources = None, wcs = None,
                              pix_scale = 1, do_artifact_cleaning = True, do_reduce_noise_level = False,
                              do_bkg_subtraction = False, do_mask_sec_objects = True, do_mask_stars = True,
                              do_deblending = True, estimate_source_pixels = True, auto_safe_masking_region=True,
                              clear_all_but_segment = True, use_z_for_deblending = True,
                              smooth_segmap = True, verbose=True, show_plots = False, min_pixels = 0):

    void_return = None, None, None, [None, None, None, None]
    try:
        # print(len(tractor_sources[tractor_sources['type'] == "PSF"]))
        # plt.imshow(image , norm = "log", origin = "lower")
        # plt.show()
        # return void_return
        # ------------ Handling input parameters ------------
        if not np.any(image.astype(bool)):
            raise Exception("Image appears to be empty.")
        if config is None:
            print("No config provided. Using default config.")
            from cleaning_configs import configs
            config = configs["default"]
        if not isinstance(tractor_sources, type(None)):
            phot_sources = tractor_sources.copy()
        else:
            phot_sources = Table(names=['type','xcentroid','ycentroid'], dtype=('str', 'float', 'float'))
        img_center = np.array((int(image.shape[1] / 2), int(image.shape[0] / 2)))
        img_bkg_lvl, img_bkg_lvl_std = sigma_clipping(image)
        if source_pos == None:
            source_pos = img_center
        else:
            source_pos = np.array((round(source_pos[0]),round(source_pos[1])))
        if estimate_source_pixels:
            if not redshift == None:
                z_pix = redshift
            else:
                z_pix = 0
            source_min_pix = int(2 * np.pow(z_pix, -1)) # Estimating source_pixels based on the redshift
        else:
            source_min_pix = config.source_min_pix

        # Initializing default flags
        artifact_flag = False
        deblended_flag = False
        big_contaminator_flag = False
        offset_flag = False
        # ---------------------------------------------------

        # times["initial_images"] = time.perf_counter()

        removal_mask = np.full(image.shape, False) # Here we keep all regions that have been modified in the image

        # Non modified images with artifacts generally have pixels with values >> background, so first we filter based on that
        if do_artifact_cleaning:
            if verbose:
                print("Detecting and cleaning overexposure artifacts.")
            if (np.max(image)/img_bkg_lvl) > config.artifact_candidate_ratio:
                image, artifact_mask, artifact_flag = detect_and_clean_artifacts(image, config, img_bkg_lvl, pix_scale = pix_scale)
                removal_mask = artifact_mask | removal_mask
                img_bkg_lvl, img_bkg_lvl_std = sigma_clipping(image)  # Updating the image background
                phot_sources = update_sources(phot_sources, artifact_mask)
                # times["artifact_cleaning"] = time.perf_counter()

        if do_reduce_noise_level:
            image = diminish_noise_level(image, method = "logit", kernel_type = 16, kernel_center_weight=3.0, smooth_noise_curve = False)
            # smooth_noise_curve must be False, otherwise sep deblending may not work correctly
            img_bkg_lvl, img_bkg_lvl_std = sigma_clipping(image) # Update background level and deviation

        if do_bkg_subtraction:
            image = subtract_background(image, show_plots=False, normalize=False, detection_threshold=3, psf_kernel_std = config.psf_std)
            # times["background_subtraction"] = time.perf_counter()

        # times["psf_convolution"] = time.perf_counter()

        # First detection of sources
        if verbose:
            print("Detecting sources in the image...")
        image_convolved, seg_map, main_source_segment = update_detections(image, config, source_min_pix, source_pos, verbose)
        seg_map_mask = seg_map.make_source_mask()
        main_source_mask = main_source_segment.make_source_mask()
        if not seg_map_mask[source_pos[1], source_pos[0]]:
            raise Exception("No sources were detected in the expected location.")
        # Checking how far the source is from the image center
        if np.linalg.norm(source_pos - img_center) > np.hypot(*image.shape) / 4:
            if verbose:
                print("WARNING: Galaxy might be partially out of the image")
            offset_flag = True

        source_cat = SourceCatalog(image, seg_map)
        if verbose:
            print("Detected", len(source_cat), "sources before cleaning.")

        # Getting the main source segment
        # sources_table = source_cat.to_table()
        # main_label = get_main_source_label(seg_map, source_pos=source_pos, table=sources_table)
        # times["first_detection"] = time.perf_counter()

        # Determining the number of pixels for de-blending and safe star masking radius
        source_npixels = np.sum(main_source_mask)
        npixels_deblend = int(source_npixels * config.npixels_deblend_factor)

        # Removing all objects that do not overlap with the main galaxy
        # if do_mask_sec_objects:
        #     print("Removing secondary detected objects")
        #     secondary_objects_mask = (image_convolved > threshold) & ~main_source_mask
        #     secondary_objects_mask = grow_mask(secondary_objects_mask,3) & ~main_source_mask
        #     image = replace_mask_with_noise(image, secondary_objects_mask, noise_factor=3, use_median=False, scale=pix_scale, smoothing_factor=0)
        #     removal_mask = secondary_objects_mask | removal_mask
        #     # Updating the convolved image and segmentation map
        #     image_convolved = convolve_with_psf(image, kernel_std=config.psf_std)
        #     threshold = detect_threshold(image_convolved, nsigma=config.galaxy_detection_threshold)
        #     seg_map = detect_sources(image_convolved, threshold, source_pixels, connectivity=config.connect_mode)

        # Detecting and masking remaining stars in the image
        if do_mask_stars:
            if verbose:
                print("Detecting and masking stars")
            if auto_safe_masking_region:
                safe_star_masking_radius = round(np.sqrt(source_npixels * config.safe_star_masking_prop_radius / np.pi))
                if safe_star_masking_radius < 1: safe_star_masking_radius = 1
            else:
                safe_star_masking_radius = config.safe_star_masking_radius
            image, all_star_apertures, masked_stars_mask = detect_and_mask_stars(image, main_source_mask, source_pos, tractor_sources, config, safe_star_masking_radius, image_psf, pix_scale, show_plots = show_plots)
            removal_mask = masked_stars_mask | removal_mask
            big_contaminator_flag = star_contamination_check(main_source_mask, all_star_apertures, config.safe_pix_ratio)
            # Updating the convolved image and segmentation map if there was any detection
            if (len(all_star_apertures["inner"])+len(all_star_apertures["edge"])+len(all_star_apertures["outer"])) > 0:
                image_convolved, seg_map, main_source_segment = update_detections(image, config, source_min_pix,
                                                                                  source_pos, verbose)
                seg_map_mask = seg_map.make_source_mask()
                main_source_mask = main_source_segment.make_source_mask()

        # De-blendnig sources
        if do_deblending:
            if verbose:
                print("Deblending sources...")
            # De-blending the overlapping sources and creating a table with all detected sources
            # We use psf convolved version of the image to smooth the de-blending edges
            segmap_deblend = get_deblended_map(image_convolved, main_source_segment, npixels_deblend, config.deblend_levels, config, config.deblend_tool)
            sec_objects = len(segmap_deblend.labels)

            if verbose:
                print("Detected", sec_objects, "sources after de-blending")

            if sec_objects > 1:
                deblended_flag = True
                deb_cat = SourceCatalog(image, segmap_deblend)
                deb_main_label = get_main_source_label(segmap_deblend, source_pos=source_pos,
                                                       table=deb_cat.to_table())
                if use_z_for_deblending:
                    segmap_deblend = analyze_deblended_map(segmap_deblend, deb_main_label, phot_sources, redshift, z_interaction, wcs, use_profile = True, show_plots = show_plots)

                if len(segmap_deblend.labels) > 1:
                    removal_segmap = segmap_deblend.copy()
                    removal_segmap.remove_label(deb_main_label)
                    deblend_removal_mask = removal_segmap.make_source_mask()
                    removal_mask = deblend_removal_mask | removal_mask
                    image = remove_deblended_objects(image, segmap_deblend, deb_main_label, overlapping_portion_threshold = config.overlapping_portion_threshold, pix_scale = pix_scale)
                    # Updating the segmentation map and label of the post-deblending main source
                    image_convolved, seg_map, main_source_segment = update_detections(image, config, source_min_pix, source_pos, verbose)
                    seg_map_mask = seg_map.make_source_mask()
                    main_source_mask = main_source_segment.make_source_mask()

            # axo[1,1].imshow(image, norm = 'log', origin = "lower")
            # plt.show()

        else:
            if verbose:
                print("De-blending omitted")

        if do_mask_sec_objects:
            if verbose:
                print("Removing secondary objects")
            threshold = detect_threshold(image_convolved, nsigma=config.galaxy_detection_threshold)
            secondary_objects_mask = (image_convolved > threshold) & ~(main_source_mask)
            secondary_objects_mask = grow_mask(secondary_objects_mask,3) & ~(main_source_mask)
            image = replace_mask_with_noise(image, secondary_objects_mask, noise_factor=3, use_median=False, scale=pix_scale, smoothing_factor=0)
            removal_mask = secondary_objects_mask | removal_mask
            # image_convolved = convolve_with_psf(image, kernel_std=config.psf_std)

        # times["deblending"] = time.perf_counter()

        # Smoothing the main source segmentation map
        if smooth_segmap:
            smoothed_segmap_data = convolve_with_psf(main_source_mask.astype(int), kernel_std=2)
            smoothed_segmap_data = np.round(smoothed_segmap_data,0)
            main_source_segment = SegmentationImage(smoothed_segmap_data.astype(int))
            main_source_mask = main_source_segment.make_source_mask()

        if clear_all_but_segment:
            # Remove all but main source
            influence_radius = round(np.sqrt(np.sum(main_source_mask))/4)
            img_bkg_lvl, img_bkg_lvl_std = sigma_clipping(image)
            noise_sample = np.random.normal(loc=img_bkg_lvl, scale=img_bkg_lvl_std, size=np.sum(main_source_mask))
            noise_image = np.random.choice(noise_sample, size=image.shape)
            noise_image = adjust_image_resolution_to_scale(noise_image, pix_scale)
            image = smooth_blend(noise_image, image, main_source_mask, blend_width=influence_radius)
            removal_mask = ~grow_mask(main_source_mask,2)

        source_npixels_final = np.sum(main_source_mask)
        if source_npixels_final < min_pixels:
            return void_return
        cleaning_flags = [int(artifact_flag), int(offset_flag), int(deblended_flag), int(big_contaminator_flag)]
        if verbose:
            print("Image cleaning finished!")
        return image, main_source_segment, removal_mask, cleaning_flags
    except Exception as e:
        print(traceback.format_exc())
        raise e




# clean_image, seg_map, image_mask, flags = get_clean_image_and_segment(image, image_psf, config, object_pix_position,
#                                                                       redshift, max_z_sep, tractor_sources,
#                                                                       wcs=wcs,
#                                                                       do_artifact_cleaning=do_artifact_cleaning,
#                                                                       do_mask_stars=do_mask_stars,
#                                                                       do_mask_sec_objects=do_mask_sec_objects,
#                                                                       do_deblending=do_deblending,
#                                                                       do_reduce_noise_level=do_reduce_noise_level,
#                                                                       do_bkg_subtraction=False,
#                                                                       smooth_segmap=False,
#                                                                       clear_all_but_segment=clear_all_but_segment,
#                                                                       verbose=verbose, pix_scale=pix_scale,
#                                                                       show_plots=show_plots,
#                                                                       min_pixels=min_pixels)



if __name__ == "__main__":

    quit()
