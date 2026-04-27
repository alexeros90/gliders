import os
from astropy.io import fits
import numpy as np
from astropy.table import Table
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


file_dir = "C:\\Astrostuff\\PhD\\Data\\DESI FITS"
# object_info = (39633244085289292,173.47971,49.84688, 0.0821)
object_info = (39633293687129181,212.82599,52.89456,0.07559)
# object_info = (39627884997902878,218.32299,3.9033,0.02958)
# object_info = (39627842752876903,218.68445,2.20796,0.06553)

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

import matplotlib.pyplot as plt
raw_image = all_bands['r']
from gliders.noise_pusher import show_all_methods, get_kernel, get_p_logit,weighted_image,get_p_meta
# plt.imshow(raw_image, norm = "log")
# plt.show()
from gliders.utils import sigma_clipping
from scipy.stats import norm

sig = sigma_clipping(raw_image)
print(sig)
#p_map = norm.sf(raw_image, loc=sig[0], scale=sig[1])
raw_image_or = raw_image.copy()
for i in range(3):
    p_map = norm.cdf(raw_image, loc=sig[0], scale=sig[1])
    kernel = get_kernel(type=16, center_weight=1.50)
    #p_logit = get_p_meta(p_map, kernel)
    p_logit = get_p_logit(p_map, kernel)
    im_w_logit, w = weighted_image(raw_image, 1 - p_logit, sigma=sig[1])
    out_im = im_w_logit * np.max(raw_image) / np.max(im_w_logit)
    ratio = out_im/raw_image
    out_im_2 = out_im.copy()
    out_im_2[ratio < 0.75] = out_im_2[ratio < 0.75] * 0.5 + raw_image[ratio < 0.75] * 0.5
    raw_image = out_im_2.copy()
fg, ax = plt.subplots(1,3, layout = "tight")
ax[0].imshow(raw_image_or-np.min(raw_image_or), norm = "log", cmap = "plasma")
ax[1].imshow(out_im-np.min(out_im), norm = "log", cmap = "plasma")
ax[2].imshow(out_im_2-np.min(out_im_2), norm = "log", cmap = "plasma")
plt.show()
#show_all_methods(raw_image, sig[1], p_map, kernel_type = 8, cmap = "plasma", show_distributions = False, smooth_noise_curve = False)
