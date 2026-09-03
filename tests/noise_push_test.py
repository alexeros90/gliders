import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import gliders as gl
from astropy.wcs import WCS

from gliders.utils import sigma_clipping, grow_mask
from gliders.artifacts import detect_artifacts, get_artifact_flag, clean_artifacts, detect_and_clean_artifacts
from gliders.utils import grow_mask, get_profile_from_border, normalize_data
from gliders.background_suppressor import suppress_background
from scipy.stats import norm
from scipy.signal import convolve2d
from gliders.background_suppressor import get_kernel


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

file_dir = "C:\\Astrostuff\\PhD\\Data\\DESI FITS\\DESI_dwarf_AGN_fits"
objects = [(39633451355210156, 267.63085, 65.63547, 0.04221), (39633336229953972, 244.6546, 56.01365, 0.09504), (39633311303206191, 175.89012, 54.17397, 0.08386), (39633270966585030, 169.38493, 51.43146, 0.00939)]
objects += [(39633438612917306, 269.841, 64.47956, 0.08708), (39633259579050725, 172.01626, 50.63296, 0.05905), (39633465871697510, 268.98092, 67.33692, 0.05093), (39633149646341168, 240.32645, 43.9398, 0.02835), (39633453855018348, 270.78025, 66.01194, 0.04011), (39633300964245553, 214.7485, 53.58666, 0.04963), (39633411303802862, 267.07812, 62.08441, 0.06833), (39633453834044162, 267.3559, 66.08475, 0.08667), (39633136430091031, 235.32035, 43.3674, 0.05734), (39633456321268631, 272.30384, 66.21461, 0.05186), (39633408476842376, 271.05605, 61.80761, 0.02767)]
objects += [(39627782275205894, 213.87222, -0.25671, 0.02578), (39633405591160319, 270.4193, 61.41409, 0.0447), (39633162636103789, 241.39688, 44.75577, 0.06423), (39633304177083919, 176.40406, 53.77439, 0.05134), (39633402680316031, 269.4657, 61.18638, 0.08599), (39627927024832299, 210.5733, 5.67949, 0.07674), (39633304109974560, 169.58651, 53.75272, 0.03607), (39633336229953980, 244.65552, 56.01383, 0.09522), (39627793834708039, 182.89745, 0.18125, 0.07771), (39627781767695886, 183.68573, -0.23029, 0.02098), (39633328701180963, 170.76138, 55.42655, 0.06752), (39633278944151489, 215.00241, 51.88165, 0.02923), (39627788436636224, 221.01972, 0.05113, 0.07332), (39633414168514317, 271.29734, 62.36193, 0.05279), (39627087811715001, 61.47254, -30.3324, 0.05933), (39633318307693663, 171.95932, 54.65036, 0.07166), (39632929848036547, 133.24705, 32.3808, 0.07455)]
# objects = [(39633451355210156,267.63085,65.63547,0.04221)]

areas = []
intersections = []
from astropy.table import Table
# for
all_artifs = []
artif_file = "D:/Astrostuff/PhD/Data/My_Data/images_with_artifacts_DESI.csv"
artif_catalog = Table.read(artif_file, format = "csv")
for object_info in artif_catalog:
# try:
    object_id = object_info[0]
    obj_coords = (object_info[1],object_info[2])
    redshift = object_info[3]
    bands = ['r','g']
    file_name = f"{object_id}_{"".join(bands)}.fits"
    sources_name = f"{object_id}_sources.fits"

    file_path = os.path.join(file_dir, file_name)
    sources_path = os.path.join(file_dir, sources_name)
    with fits.open(file_path) as hdu_list:
        # hdu_list.info()
        all_bands, all_weights, ref_header = extract_band_data(hdu_list, verbose=False, bands_to_analyze=bands)

    from astropy.table import Table
    with fits.open(sources_path) as hdul:
        data = hdul[1].data
        as_table = Table(data)
    wcs = WCS(ref_header)
    tractor_x_pix, tractor_y_pix = wcs.all_world2pix(as_table['ra'], as_table['dec'], 0)  # origin=0 for Python indexing
    source_x_pix, source_y_pix = wcs.all_world2pix(obj_coords[0], obj_coords[1], 0)  # origin=0 for Python indexing
    source_pos = (np.round(source_x_pix), np.round(source_y_pix))
    as_table['xcentroid'] = tractor_x_pix
    as_table['ycentroid'] = tractor_y_pix
    phot_sources = as_table['type','xcentroid','ycentroid']
    image = all_bands['r']


    if obj_coords[1] > 32.375:
        pix_scale = 1.72
    else:
        pix_scale = 1

    artifacts_segmap = detect_artifacts(image, npixels_artifact=100, delta_slope_threshold=2.7, segment_slack=5,
                         hi_pass_threshold=1)
    artifacts_mask = artifacts_segmap.make_source_mask()
    segment_slack = 15
    sigmas = 10
    test_bleed = 30
    bleeding_factor = 0.4
    noise_mean, noise_std = sigma_clipping(image)
    noise_lvl = noise_mean + sigmas * noise_std
    clean_image, clean_mask = clean_artifacts(image, artifacts_segmap, bleeding_factor = 2.5, interpolation_radius = 2, pix_scale = pix_scale)
    clean_image_noise_mean, clean_image_noise_std = sigma_clipping(image)

    kernel = get_kernel(type=8, center_weight=1.0)
    # p_map = norm.sf(clean_image.flatten(), loc=clean_image_noise_mean, scale=clean_image_noise_std)  # sf = 1 - cdf = survival function
    # p_map = norm.sf(clean_image.flatten(), loc=clean_image_noise_mean, scale=clean_image_noise_std)  # sf = 1 - cdf = survival function
    xdata = clean_image
    # xdata = np.linspace(-0.01,0.01,100)
    p_map = norm.sf(xdata, loc=clean_image_noise_mean, scale=clean_image_noise_std)  # sf = 1 - cdf = survival function
    p_map_1 = norm.cdf(xdata, loc=clean_image_noise_mean, scale=clean_image_noise_std)  # sf = 1 - cdf = survival function
    p_map_cnv = convolve2d(p_map_1, kernel, mode='same', boundary='symm')
    p_out = p_map_1 * p_map_cnv
    sigma = clean_image_noise_std
    eps=1e-12
    # weighted_image(image, p_noise, sigma=1.0, eps=1e-12):
    # Effective weight = (1 - p_noise) / sigma^2
    # w = np.clip(1.0 - p_map_1, 0.0, 1.0)
    w = np.clip(p_out, 0.0, 1.0)
    # w = 1/(p_out+eps)
    # if np.isscalar(sigma):
    #     w = w / (sigma**2)
    # else:
    #     w = w / (np.clip(sigma, eps, None) ** 2)
    clean_image_noise_mean_REDU = clean_image - 3*clean_image_noise_std
    clean_image_shift = clean_image.min()
    # image_weighted = (clean_image - clean_image_shift) * w + clean_image_shift
    # image_weighted = (clean_image-clean_image_shift) * np.exp(w) + clean_image_shift
    image_weighted = clean_image*w - clean_image_noise_mean_REDU*(1-w)

    # image_weighted = image_weighted * clean_image.sum() / image_weighted.sum()  # matching totals (weighting reduces the total, so it needs to be compensated)
    # image_weighted = image_weighted * clean_image.max() / image_weighted.max()
    # # p_map = 1 - p_map
    # print(p_map)
    # print(p_map_1)
    clean_image_norm = normalize_data(clean_image) + 0.001
    image_weighted_norm = normalize_data(image_weighted) + 0.001

    clean_image_norm_noise = sigma_clipping(clean_image_norm)
    lim = clean_image_norm_noise[0] + 5*clean_image_norm_noise[1]
    clean_image_flat = clean_image_norm.flatten()
    clean_image_flat = clean_image_flat[clean_image_flat<lim]
    weighted_image_flat = image_weighted_norm.flatten()
    weighted_image_flat = weighted_image_flat[weighted_image_flat<lim]

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax[0,0].imshow(clean_image_norm, norm = "log") #, bins = 30)
    ax[0,1].imshow(p_map_cnv, norm = "log") #, bins = 30)
    ax[1,0].hist(clean_image_flat, bins = 40)
    ax[1,1].hist(weighted_image_flat, bins = 40)
    # plt.xlim(clean_image_noise_mean - 5*clean_image_noise_std,clean_image_noise_mean + 5*clean_image_noise_std)
    # plt.imshow(p_map - p_map_1)
    plt.show()
    # p_map = norm.sf(clean_image, loc=clean_image_noise_mean, scale=clean_image_noise_std)  # sf = 1 - cdf = survival function
    # p_map = 1 - p_map
    quit()

