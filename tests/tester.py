import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import gliders as gl
from astropy.wcs import WCS

import os

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

file_dir = ""
object_info = (39627847656019106,150.98807,2.41803,0.02661) # ID, RA, DEC, redshift
# object_info = (39633244085289292,173.47971,49.84688, 0.0821)
# object_info = (39628516576202568,144.50186,31.63659,0.02251)

object_id = object_info[0]
obj_coords = (object_info[1],object_info[2])
redshift = object_info[3]
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
source_x_pix, source_y_pix = wcs.all_world2pix(obj_coords[0], obj_coords[1], 0)  # origin=0 for Python indexing
source_pos = (np.round(source_x_pix), np.round(source_y_pix))
as_table['x_centroid'] = tractor_x_pix
as_table['y_centroid'] = tractor_y_pix
phot_sources = as_table['type','x_centroid','y_centroid']
# pix_scale = 1.72 # For BASS images
pix_scale = 1
raw_image = all_bands['r']

results = gl.clean_image(raw_image, redshift=redshift, wcs=wcs, mask_stars = True, show_plots=False, deblend_sources = True,
                         mask_sec_objects = True, clean_artifacts = True, phot_sources = phot_sources, source_pos=source_pos,
                         pix_scale = pix_scale, verbose = True, skip_flagged_artifacts = False, deblend_delta_z = 0.01,
                         deblend_with_redshift = False, deblend_with_photo_type = True, sync_z_query = False, smooth_segmap = False)

clean_image, seg_map, removal_mask, cleaning_flags = results

print(cleaning_flags)
fig, ax = plt.subplots(2,2, figsize = (6,6))
ax[0,0].imshow(raw_image-np.min(raw_image)+0.05, norm = "log", origin = "lower", cmap = "magma")
ax[0,0].set_title("Raw image")
ax[0,1].imshow(clean_image-np.min(clean_image)+0.005, norm = "log", origin = "lower", cmap = "magma")
ax[0,1].set_title("Clean image")
ax[1,0].imshow(seg_map, origin = "lower")
ax[1,0].set_title("Segmentation map")
ax[1,1].imshow(removal_mask, origin = "lower")
ax[1,1].set_title("Removal mask")
plt.tight_layout()
plt.show()