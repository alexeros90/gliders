import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.signal import convolve2d
from .utils import sigma_clipping, normalize_data


def show_all_methods(ref_image, img_sigma, p_map, kernel_type = 16, cmap = "plasma", show_distributions = False, smooth_noise_curve = True):
    kernel = get_kernel(type=kernel_type, center_weight=2.0)
    p_logit = get_p_logit(p_map, kernel)
    p_map_product = get_p_product(p_map, kernel)
    p_map_meta = get_p_meta(p_map, kernel, p_map_product)
    p_map_meta_2 = get_p_meta_2(p_map, kernel, p_map_product)

    # Normalize, apply and de-normalize
    alt_image_totes = np.sum(ref_image)
    im_w_dir, w = weighted_image(ref_image, 1 - p_map, sigma=img_sigma)
    im_w_meta, w = weighted_image(ref_image, 1 - p_map_meta, sigma=img_sigma)
    im_w_logit, w = weighted_image(ref_image, 1 - p_logit, sigma=img_sigma)
    im_w_prod, w = weighted_image(ref_image, 1 - p_map_product, sigma=img_sigma)
    im_w_meta_2, w = weighted_image(ref_image, 1 - p_map_meta_2, sigma=img_sigma)

    images = {
        'im_w_dir': im_w_dir,
        'im_w_meta': im_w_meta,
        'im_w_logit': im_w_logit,
        'im_w_prod': im_w_prod,
        'im_w_meta_2': im_w_meta_2
    }

    image_bias = 0.0  # -np.min(ref_image)*1.1
    ref_image_hist = ref_image.flatten()
    bins = np.logspace(np.log10(np.min(ref_image_hist[ref_image_hist>0])),np.log10(np.max(ref_image_hist)),100)
    im_orig = ref_image + image_bias
    im_orig_max = np.max(im_orig)
    im_orig_min = np.min(im_orig)
    # im_orig = normalize_data(ref_image) + image_bias
    # im_orig_den = (im_orig_n)*(im_orig_max - im_orig_min) + im_orig_min

    for key in images:
        images[key] = images[key] * alt_image_totes / np.sum(
            images[key])  # matching totals (weighting reduces the total, so it needs to be compensated)
        if smooth_noise_curve:
            images[key] = noise_profile_smoothing(images[key], 10, -0.9)
        images[key] = images[key] * (im_orig_max/np.max(images[key])) # Adjusting to the original image range
        # images[key] = images[key]*0.5 + ref_image*0.5

        if show_distributions:
            plt.hist(images[key].flatten(), bins = bins, alpha = 0.8, log = True, histtype='step', label = key, linewidth = 3, density = False)
    # im_orig = (im_orig)*(im_orig_max - im_orig_min) + im_orig_min

    if show_distributions:
        plt.hist(im_orig.flatten(), bins = bins, alpha = 0.8, log = True, histtype='step', label = "orig", linewidth = 3, density = False)
        plt.xscale("log")
        plt.legend()
        plt.show()

    vis_norm = "log"
    fegs, axo = plt.subplots(2, 5, layout="tight")
    axo[0, 0].imshow(p_map, origin="lower")
    axo[0, 0].set_title("dir")
    axo[0, 1].imshow(p_map_meta, origin="lower")
    axo[0, 1].set_title("meta")
    axo[1, 0].imshow(p_logit, origin="lower")
    axo[1, 0].set_title("logit")
    axo[1, 1].imshow(p_map_product, origin="lower")
    axo[1, 1].set_title("prod")
    axo[0, 2].imshow(im_orig, origin="lower", norm=vis_norm, cmap=cmap)
    axo[0, 2].set_title("original")
    axo[0, 3].imshow(images["im_w_dir"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[0, 3].set_title("dir")
    axo[0, 4].imshow(images["im_w_meta"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[0, 4].set_title("meta")
    axo[1, 2].imshow(images["im_w_logit"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[1, 2].set_title("logit")
    axo[1, 3].imshow(images["im_w_prod"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[1, 3].set_title("prod")
    axo[1, 4].imshow(images["im_w_meta_2"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[1, 4].set_title("meta 2")
    plt.show()

def noise_profile_smoothing(image, sigma_factor = 7, w_bias = 0.1, outlier_fraction = 0.01):
    image_range = np.max(image) - np.min(image)
    image_min = np.min(image)
    image_bckg = sigma_clipping(image, robust=False, use_median=True)
    img_hist = image.flatten()
    img_hist_cut = img_hist[img_hist < image_bckg[0] + (sigma_factor) * image_bckg[1]]
    norm_dist = np.random.normal(loc=image_bckg[0], scale=image_bckg[1], size = len(img_hist_cut))
    weight = norm.cdf(image, loc=image_bckg[0], scale=image_bckg[1])  # sf = 1 - cdf = survival function
    sigmas = sigma_factor * image_bckg[1]
    limit = image_bckg[0] + sigmas

    image_replace = (np.random.choice(norm_dist, size=np.sum(image<limit), replace=True)*np.clip(weight[image<limit]+w_bias,0,1) + image[image<limit]*np.clip(1-weight[image<limit]-w_bias,0,1))
    ratio = np.max(img_hist_cut)/np.max(image_replace)
    image[image<limit] = image_replace * ratio
    # outlier_limit = image_bckg[0]*outlier_fraction
    # for _ in range(5):
    #     image[image<outlier_limit] = (np.random.choice(norm_dist, size=np.size(image[image<outlier_limit]), replace = True) + image_bckg[0])/2
    # image[image<image] = (np.random.choice(norm_dist[norm_dist>=image_min], size=np.size(image[image<image_min]), replace = True) + image_bckg[0])/2
    # bins = np.logspace(np.log10(np.min(img_hist[img_hist>0])),np.log10(np.max(img_hist)),100)
    # img_hist_n = image.flatten()
    # plt.hist(img_hist, bins = bins , log=True)
    # plt.hist(img_hist_n, bins = bins , log=True, alpha = 0.7)
    # plt.hist(norm_dist, bins = bins , alpha = 0.3, histtype='step')
    # plt.xscale('log')
    # plt.show()
    image = image * image_range + image_min
    return image

def get_kernel(type=8, center_weight=2.0):
    if type == 8:
        kernel = np.ones((3, 3))  # includes center
    elif type == 4:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        # custom
        # kernel_custom = np.array([[0.125,0.25,0.5,0.25,0.125],[0.25,0.75,1,0.75,0.25],[0.5,1,1,1,0.5],[0.25,0.75,1,0.75,0.25],[0.125,0.25,0.5,0.25,0.125]])
        kernel = np.array(
            [[0.25, 0.5, 0.75, 0.5, 0.25], [0.5, 0.75, 1, 0.75, 0.5], [0.75, 1, 1, 1, 0.75], [0.5, 0.75, 1, 0.75, 0.5],
             [0.25, 0.5, 0.75, 0.5, 0.25]])

    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = center_weight
    # normalize kernel so sum of neighbor weights = 1 (optional)
    kernel = kernel / kernel.sum()
    return kernel

def weighted_image(image, p_noise, sigma=1.0, eps=1e-12):
    # Effective weight = (1 - p_noise) / sigma^2
    w = np.clip(1.0 - p_noise, 0.0, 1.0)
    if np.isscalar(sigma):
        w = w / (sigma**2)
    else:
        w = w / (np.clip(sigma, eps, None) ** 2)
    return image * w, w

def get_p_logit(p_map, kernel, alpha = 1.0):
    # optionally scale strength by alpha (alpha>1 strengthens neighborhood influence)
    p_map[p_map >= 1] = 1-1e-9
    logit = np.log(p_map / (1 - p_map))
    L = convolve2d(logit, kernel, mode='same', boundary='symm')  # weighted sum of logit
    p_out = 1.0 / (1.0 + np.exp(-alpha * L))
    return p_out

def get_p_product(p_map, kernel):
    # Perform convolution (sum of weighted neighbors)
    p_map_cnv = convolve2d(p_map, kernel, mode='same', boundary='symm')
    p_out = p_map * p_map_cnv
    return p_out

def get_p_meta(p_map, kernel, p_map_product = None):
    if isinstance(p_map_product,type(None)):
        p_map_product = get_p_product(p_map, kernel)
    p_bck = sigma_clipping(p_map_product, robust = False)
    p_map_meta = norm.sf(p_map_product, loc=p_bck[0], scale=p_bck[1])  # sf = 1 - cdf = survival function
    p_map_meta = 1-p_map_meta
    return normalize_data(p_map_meta)

def get_p_meta_2(p_map, kernel, p_map_product = None):
    if isinstance(p_map_product,type(None)):
        p_map_product = get_p_product(p_map, kernel)
    p_bck = sigma_clipping(p_map_product, robust = False)
    p_map_meta = norm.sf(p_map_product, loc=p_bck[0], scale=p_bck[1])  # sf = 1 - cdf = survival function
    p_map_meta = 1-p_map_meta
    p_map_meta = get_p_product(p_map_meta, kernel)
    return normalize_data(p_map_meta)


if __name__ == "__main__":
    quit()


def diminish_noise_level(ref_image, method = "direct", kernel_type = 16, kernel_center_weight = 2.0, smooth_noise_curve = True):
    # This function assumes that the image noise follows a normal-like distribution
    image_background = sigma_clipping(ref_image, robust=False)

    # Probability of obtaining x or higher:
    p_map = norm.sf(ref_image, loc=image_background[0], scale=image_background[1])  # sf = 1 - cdf = survival function
    p_map = 1 - p_map

    kernel = get_kernel(type=kernel_type, center_weight=kernel_center_weight)

    if method == "logit":
        p_map = get_p_logit(p_map, kernel)
    elif method == "product":
        p_map = get_p_product(p_map, kernel)
    elif method == "meta":
        p_map_product = get_p_product(p_map, kernel)
        p_map = get_p_meta(p_map, kernel, p_map_product)
    elif method == "meta_prod":
        p_map_product = get_p_product(p_map, kernel)
        p_map = get_p_meta_2(p_map, kernel, p_map_product)
    else:
        p_map = p_map

    ref_image_totes = np.sum(ref_image)
    image_weighted, w = weighted_image(ref_image, 1 - p_map, sigma=image_background[1])
    image_weighted = image_weighted * ref_image_totes / np.sum(image_weighted)  # matching totals (weighting reduces the total, so it needs to be compensated)
    if smooth_noise_curve:
        image_weighted = noise_profile_smoothing(image_weighted, 10, -0.9)
    image_weighted = image_weighted * (np.max(ref_image) / np.max(image_weighted)) # Adjusting to the original image range

    return image_weighted
