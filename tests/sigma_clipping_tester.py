import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import gliders as gl
from astropy.wcs import WCS
def extract_psf_data(hdu_list, verbose=True, bands_to_analyze = ['r']):
    all_psf = {}
    for hdu in hdu_list:
        try:
            header = hdu._header
            if verbose:
                print(header)
            band = header.get("BAND")

            if band is None:
                continue  # Skip if essential header info is missing
            if band not in bands_to_analyze:
                continue
            if band not in all_psf:
                all_psf[band] = hdu.copy().data

        except Exception as e:
            print(f"Skipped HDU due to error: {e}")

    return all_psf

def extract_band_data(hdu_list, verbose=True, bands_to_analyze = ['r']):
    all_bands = {}
    all_img_diffs = {}
    all_img_areas = {}
    all_weights = {}
    all_wgh_diffs = {}
    all_wgh_areas = {}
    ref_hdu = None
    for hdu in hdu_list:
        try:
            header = hdu._header
            if verbose:
                print(header)
            band = header.get("BAND")
            if band not in bands_to_analyze: continue
            image_type = header.get("IMAGETYP")
            imagew = header.get("IMAGEW")
            imageh = header.get("IMAGEH")

            if band is None or image_type is None or imagew is None or imageh is None:
                continue  # Skip if essential header info is missing

            img_dim_diff = abs(imagew - imageh)
            img_area = imagew * imageh

            if image_type == "image":
                if band not in all_bands or img_area > all_img_areas[band] or (img_area == all_img_areas[band] and img_dim_diff < all_img_diffs[band]):
                    all_bands[band] = hdu.copy().data
                    all_img_areas[band] = img_area
                    all_img_diffs[band] = img_dim_diff
                    ref_hdu = hdu.header

            elif image_type == "invvar":
                w_map = hdu.copy().data
                w_map[w_map == 0] += 1  # Avoid division by zero
                weight = 1 / np.sqrt(w_map)
                if band not in all_weights or img_area > all_wgh_areas[band] or (img_area == all_wgh_areas[band] and img_dim_diff < all_wgh_diffs[band]):
                    all_weights[band] = weight
                    all_wgh_areas[band] = img_area
                    all_wgh_diffs[band] = img_dim_diff

        except Exception as e:
            print(f"Skipped HDU due to error: {e}")

    return all_bands, all_weights, ref_hdu

file_dir = "C:\\Astrostuff\\PhD\\Data\\DESI FITS"
# file_name = "39633321931571417_r.fits"
file_name = "ngc6505_cutout_mer.fits"

file_path = os.path.join(file_dir, file_name)
# with fits.open(file_path) as hdu_list:
#     hdu_list.info()
#     # all_bands, all_weights, ref_header = extract_band_data(hdu_list, verbose=True, bands_to_analyze=bands_to_analyze)
#     all_psf = extract_psf_data(hdu_list, verbose=False, bands_to_analyze=bands_to_analyze)

with fits.open(file_path) as hdu_list:
    hdu_list.info()
    image = hdu_list[0].data

plt.imshow(image)
plt.show()
results = gl.clean_image(image, return_clean_image_only=False)

# results = gl.clean_image(image, image_psf = None, config = None, source_pos = None, redshift = None, phot_sources = None,
#                 wcs = None, return_clean_image_only = False, clean_artifacts = True, skip_flagged_artifacts = True,
#                 reduce_noise_level = False, subtract_background = False, mask_sec_objects = True, mask_stars = True,
#                 deblend_sources = True, deblend_with_redshift = True, deblend_delta_z = None, deblend_with_photo_type = True,
#                 sync_z_query = False, clear_all_but_segment = False, smooth_segmap = True, pix_scale = 1, verbose = True,
#                 show_plots = False)

print(results)
# image = results[0]
# flags = results[3]
# print(flags)
# plt.imshow(image, norm = 'log', origin = "lower")
# plt.show()

