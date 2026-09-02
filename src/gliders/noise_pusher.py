import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.signal import convolve2d
from .utils import sigma_clipping, normalize_data
from astropy.convolution import Gaussian2DKernel

def show_all_methods(ref_image, kernel = None, cmap = "plasma", show_distributions = False):
    if kernel is None:
        kernel = Gaussian2DKernel(2.0)
    image_noise_lvl, image_noise_std = sigma_clipping(ref_image, robust=False)
    p_map = norm.cdf(ref_image, loc=image_noise_lvl, scale=image_noise_std)  # sf = 1 - cdf = survival function

    p_simple = get_p_simple(p_map, kernel)
    p_logit = get_p_logit(p_map, kernel)
    p_layers = get_p_layers(ref_image, p_map, kernel)
    norm_bias = 0.001
    # Normalize, apply and de-normalize

    bkg_reduced_image_simple = apply_p_map(ref_image, p_simple)
    bkg_reduced_image_logit = apply_p_map(ref_image, p_logit)
    bkg_reduced_image_layers = apply_p_map(ref_image, p_layers)

    images = {
        'im_raw': ref_image,
        'im_simple': bkg_reduced_image_simple,
        'im_logit': bkg_reduced_image_logit,
        'im_layers': bkg_reduced_image_layers,
    }

    ref_image_hist = ref_image.flatten()
    bins = np.logspace(np.log10(np.min(ref_image_hist[ref_image_hist>0])),np.log10(np.max(ref_image_hist)),100)

    if show_distributions:
        for key in images:
            plt.hist(images[key].flatten(), bins = bins, alpha = 0.8, log = True, histtype='step', label = key, linewidth = 3, density = False)
        plt.xscale("log")
        plt.legend()
        plt.show()

    for key in images:
        images[key] = normalize_data(images[key]) + norm_bias
    vis_norm = "log"
    fegs, axo = plt.subplots(2, 4, layout="tight")
    axo[1, 0].imshow(p_map, origin="lower")
    axo[1, 0].set_title("raw p_map")
    axo[0, 0].imshow(images["im_raw"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[0, 0].set_title("original")

    axo[1, 1].imshow(p_simple, origin="lower")
    axo[1, 1].set_title("simple")
    axo[0, 1].imshow(images["im_simple"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[0, 1].set_title("simple")

    axo[1, 2].imshow(p_logit, origin="lower")
    axo[1, 2].set_title("logit")
    axo[0, 2].imshow(images["im_logit"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[0, 2].set_title("logit")

    axo[1, 3].imshow(p_layers, origin="lower")
    axo[1, 3].set_title("layers")
    axo[0, 3].imshow(images["im_layers"], origin="lower", norm=vis_norm, cmap=cmap)
    axo[0, 3].set_title("layers")
    plt.show()

def apply_p_map(image, p_map):
    image_p_map = image * p_map
    image_p_map_norm = image_p_map * image.sum() / image_p_map.sum()  # matching totals (weighting reduces the total, so it needs to be compensated)
    image_p_map_leveled = image_p_map_norm * image.max() / image_p_map_norm.max()  # Adjusting to the original image range
    return image_p_map_leveled

def get_p_simple(noise_p_map, kernel):
    p_out = convolve2d(noise_p_map, kernel, mode='same', boundary='symm')
    return p_out

def get_p_logit(p_map, kernel, alpha = 1.0):
    # optionally scale strength by alpha (alpha>1 strengthens neighborhood influence)
    p_map[p_map >= 1] = 1-1e-9
    logit = np.log(p_map / (1 - p_map))
    L = convolve2d(logit, kernel, mode='same', boundary='symm')  # weighted sum of logit
    p_out = 1.0 / (1.0 + np.exp(-alpha * L))
    return p_out

def get_p_layers(image, p_map, kernel):
    p_map_cnv = convolve2d(p_map, kernel, mode='same', boundary='symm')
    image_p_map_cnv = image * p_map_cnv
    image_p_map_cnv_noise = sigma_clipping(image_p_map_cnv, robust=False)
    p_map_new_layer = norm.cdf(image_p_map_cnv, loc=image_p_map_cnv_noise[0], scale=image_p_map_cnv_noise[1])
    p_map_new_layer_cnv = convolve2d(p_map_new_layer, kernel, mode='same', boundary='symm')
    p_out = p_map_cnv * p_map_new_layer_cnv
    return p_out


if __name__ == "__main__":
    quit()


def diminish_noise_level(image, method = "simple", kernel = None, logit_alpha = 1.0):

    if kernel is None:
        kernel = Gaussian2DKernel(2.0) # Larger kernels yield higher noise level, but more symmetric noise distribution

    # This function assumes that the image noise follows a normal-like distribution
    image_noise_lvl, image_noise_std = sigma_clipping(image, robust=False)

    # Probability of obtaining x or lower:
    noise_p_map = norm.cdf(image, loc=image_noise_lvl, scale=image_noise_std)  # sf = 1 - cdf = survival function

    if method == "logit":
        p_map = get_p_logit(noise_p_map, kernel, alpha = logit_alpha)
    elif method == "layers":
        p_map = get_p_layers(image, noise_p_map, kernel)
    else: #simple
        p_map = get_p_simple(noise_p_map, kernel)

    bkg_reduced_image_leveled = apply_p_map(image, p_map)

    return bkg_reduced_image_leveled
    # return bkg_reduced_image
