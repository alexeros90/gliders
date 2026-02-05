from utils import convolve_with_psf, normalize_data
from photutils.background import Background2D

def subtract_background(image, show_plots=False, normalize=False, detection_threshold=2,
                        box_divisor=14, bkg_min_pixels=5, psf_kernel_std=2.0):
    image_convolved = convolve_with_psf(image, psf_kernel_std)
    bkg_threshold = detect_threshold(image_convolved, nsigma=detection_threshold)
    bkg_segmap = detect_sources(image_convolved, bkg_threshold, bkg_min_pixels, connectivity=8)
    if bkg_segmap is None:
        bkg_mask = np.full(image.shape, False)  # TEMP
    else:
        bkg_mask = bkg_segmap.make_source_mask(size=6)
    valid_background = False
    box_size = (int(np.floor(image.shape[0] / box_divisor)), int(np.floor(image.shape[1] / box_divisor)))
    percentile = 10
    while not valid_background and box_divisor > 1:
        try:
            Background = Background2D(image, box_size=box_size, mask=bkg_mask, filter_size=3,
                                      exclude_percentile=percentile)
            valid_background = True
        except:
            percentile += 5

    if box_divisor <= 1:
        raise Exception("An error occurred while performing back-subtraction.")
    background_subtracted_image = image - Background.background
    if normalize:
        background_subtracted_image = normalize_data(background_subtracted_image)
    if show_plots:
        masked_img = image * (np.invert(bkg_mask) + 1 - 1) + np.max(image) / 1000
        fig_bck, ax_bck = plt.subplots(ncols=3, nrows=2, figsize=(8, 6), layout="tight")
        ax_bck[0, 0].imshow(image - np.min(image), cmap='gray', origin='lower', norm="log")
        ax_bck[0, 0].set_title("Image")
        ax_bck[0, 1].imshow(bkg_mask, origin='lower')
        ax_bck[0, 1].set_title("Detected sources")
        ax_bck[1, 0].imshow(masked_img, cmap='gray', origin='lower', norm="log")
        ax_bck[1, 0].set_title("Masked image")
        ax_bck[1, 1].imshow(Background.background, origin='lower')
        ax_bck[1, 1].set_title("Background")
        ax_bck[1, 2].imshow(Background.background_rms, origin='lower')
        ax_bck[1, 2].set_title("Background RMS")
        ax_bck[0, 2].imshow(background_subtracted_image - np.min(background_subtracted_image), origin='lower',
                            cmap='gray', norm="log")
        ax_bck[0, 2].set_title("Subtracted image")
        plt.show()

    return background_subtracted_image
