#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from gliders.utils import sigma_clipping, grow_mask, get_profile_from_border, normalize_data
from gliders.artifacts import detect_artifacts, get_artifact_flag, clean_artifacts, detect_and_clean_artifacts
from gliders.background_suppressor import suppress_background, show_all_methods
from scipy.stats import norm
from scipy.signal import convolve2d
from astropy.convolution import Gaussian2DKernel

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

file_dir = "C:/Astrostuff/PhD/Data/DESI FITS/"
artif_file = "D:/Astrostuff/PhD/Data/My_data/DESI_New_mergers.csv"
artif_catalog = Table.read(artif_file, format = "csv", converters = {"object_id":str})
print(len(artif_catalog))
print(artif_catalog)
#%%
for ii, object_info in enumerate(artif_catalog[140:]):
    object_id = object_info[0]
    print(object_id)
    obj_coords = (object_info[1],object_info[2])
    redshift = object_info[3]
    bands = ['r'] #,'g']
    file_name = f"{object_id}_{"".join(bands)}.fits"
    sources_name = f"{object_id}_sources.fits"

    file_path = os.path.join(file_dir, file_name)
    sources_path = os.path.join(file_dir, sources_name)
    try:
        with fits.open(file_path) as hdu_list:
            # hdu_list.info()
            all_bands, all_weights, ref_header = extract_band_data(hdu_list, verbose=False, bands_to_analyze=bands)
    except:
        continue

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
    image = all_bands['r']


    if obj_coords[1] > 32.375:
        pix_scale = 1.72
    else:
        pix_scale = 1

    # if ii < 2:
    #     continue


    radius_deg = 1.5/60
    print(radius_deg)

    ra = obj_coords[0]
    dec = obj_coords[1]
    ra_bounds = (ra - 0.001, ra + 0.001)
    dec_bounds = (dec - 0.001, dec + 0.001)

    from gliders.z_queries.z_fetcher import get_nearby_z_sources
    sources, z_fail_flag = get_nearby_z_sources((ra_bounds, dec_bounds), 20, False)
    print(sources)
    print(z_fail_flag)
    # Query NED around that position
    # Ned.clear_cache()
    # result = Ned.query_region(pos, radius=0.1*u.arcmin)
    # print(result.pprint_all())
    if ii > 4:
        quit()

