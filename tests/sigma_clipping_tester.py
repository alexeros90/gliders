import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gliders import clean_image
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

file_dir = "D:\\Astrostuff\\PhD\\Data\\DESI FITS"
# file_name = "39633321931571417_r.fits"
# file_name = "11834-12703_gr_psf.fits"
file_name = "ngc6505_cutout_mer.fits"

bands_to_analyze = ['r','g']
file_path = os.path.join(file_dir, file_name)
# with fits.open(file_path) as hdu_list:
#     hdu_list.info()
#     # all_bands, all_weights, ref_header = extract_band_data(hdu_list, verbose=True, bands_to_analyze=bands_to_analyze)
#     all_psf = extract_psf_data(hdu_list, verbose=False, bands_to_analyze=bands_to_analyze)

with fits.open(file_path) as hdu_list:
    hdu_list.info()
    image = hdu_list[0].data

def get_aspect_ratio(mask):
    from skimage.measure import regionprops
    props = regionprops(mask.astype(int))[0]
    length = props.major_axis_length
    width = props.minor_axis_length
    return width/length


print(np.geomspace(1,0.1,10))
quit()
# image = hdu_list
image[:,:] = 0
image = image.astype(bool)
image[30:50,30:35] = True
image[45:50,10:40] = True
image[30:50,15:20] = True
rt = get_aspect_ratio(image)
plt.imshow(image)
plt.show()
quit()
results = clean_image(image, return_clean_image_only=False)
image = results[0]
flags = results[3]
print(flags)
plt.imshow(image, norm = 'log')
plt.show()

quit()
image = all_psf['r']
flat = image.copy().flatten()
# plt.hist(flat, bins = np.linspace(np.min(flat),np.max(flat),20), log = True)
plt.hist(flat, bins = np.linspace(np.min(flat),0.0003,30), log = True)
# plt.imshow(image)
plt.show()

quit()
from gliders.utils import sigma_clipping
from astropy.stats import sigma_clip
from scipy.stats import sigmaclip
import time
times = {}
times[0] = time.perf_counter()
them_vals = sigma_clipping(image, iteration_limit = 1000, robust = False, convergence_tolerance=0.0001)
times[1] = time.perf_counter()
them_vals2 = sigma_clip(image, sigma_upper = 3, sigma_lower=10, maxiters = 100, cenfunc = "mean")
times[2] = time.perf_counter()
them_vals3, a, b = sigmaclip(image, high = 3)
times[3] = time.perf_counter()
print(them_vals)
print(np.mean(them_vals2),np.std(them_vals2))
print(np.mean(them_vals3),np.std(them_vals3))
# for ks in range(1,len(times)):
#     print(times[ks]-times[ks-1])
