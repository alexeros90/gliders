import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gliders import clean_image
from astropy.wcs import WCS

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
# file_dir = "D:\\Astrostuff\\PhD\\Data\\DESI FITS"
file_dir = "C:\\DESI_fits_images\\DESI_dwarf_AGN_fits"
object_id = "39633419155538773"
obj_coords = (192.04242,62.67725)
redshift = 0.0821
bands = ['r']
file_name = f"{object_id}_{bands[0]}.fits"
sources_name = f"{object_id}_sources.fits"


file_path = os.path.join(file_dir, file_name)
sources_path = os.path.join(file_dir, sources_name)
with fits.open(file_path) as hdu_list:
    hdu_list.info()
    all_bands, all_weights, ref_header = extract_band_data(hdu_list, verbose=True,
                                                               bands_to_analyze=bands)
from astropy.table import Table
with fits.open(sources_path) as hdul:
    data = hdul[1].data
    as_table = Table(data)
wcs = WCS(ref_header)
tractor_x_pix, tractor_y_pix = wcs.all_world2pix(as_table['ra'], as_table['dec'], 0)  # origin=0 for Python indexing
as_table['xcentroid'] = tractor_x_pix
as_table['ycentroid'] = tractor_y_pix
desired_cols = as_table['type','xcentroid','ycentroid']
pix_scale = 1.72
# plt.imshow(all_bands['r'], norm = "log")
# plt.show()
clean_r, seg,removal_mask, cleaning_flags = clean_image(all_bands['r'], redshift=redshift, wcs=wcs, mask_stars = True,
                                                        show_plots=True, deblend_sources = True, mask_sec_objects = True,
                                                        clean_artifacts = True, phot_sources = desired_cols,
                                                        pix_scale = pix_scale)
plt.imshow(clean_r, norm = "log", origin = "lower")
plt.show()