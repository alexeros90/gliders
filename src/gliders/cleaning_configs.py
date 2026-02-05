from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    bands_to_analyze: str # e.g., 'griz'
    do_background_subtraction: bool # Whether to perform background subtraction
    do_masking: bool # Whether to apply masking to the images
    do_mask_stars: bool
    interpolate_star_mask: bool # Whether to interpolate the star mask
    do_deblending: bool # Whether to perform deblending of the images
    do_artifacts_cleaning: bool # Whether to clean artifacts from the images
    print_results: bool # Whether to print results to the console
    write_results: bool # Whether to write results to a file
    write_input_parameters: bool # Whether to write input parameters to a file
    delete_downloaded_files: bool # Whether to delete downloaded files after processing
    notify_crash: bool # Whether to notify in case of a crash
    notify_finish: bool # Whether to notify when processing is finished
    masking_method: str # Method used for masking, e.g., 'precise', 'aperture', etc.
    gain: float # Gain value for the images when not using weight maps

    # Artifact detection parameters
    artifact_candidate_ratio: float # The maximum pixel value of the image should be at least this amount of times the image background to search for over exposure artifacts
    delta_slope_threshold: float # The slope delta threshold to distinguish artifacts from other objects
    artifact_segment_slack: float # The distance in pixels at which the artifact profile will be used
    npixels_artifact: int # The minimum amount of pixels that a source must have to be considered an artifact

    # Star detection parameters
    star_detector: str # Method used for star detection, e.g., 'dao', 'iraf', etc.
    star_detection_threshold: float # Threshold for star detection
    star_combination_method: str # Method used for combining images for star detection, e.g., 'anti-exp-product', 'average', etc.
    star_roundness: float # Roundness threshold for star detection
    star_sharpness_low: float # Lower threshold for sharpness in star detection
    star_fwhm_span: float # The span within which the FWHM will be used to detect stars at each psf kernel size
    star_filter_psf_std: float # The std for the kernel used to convolve the image with a psf to filter the detected stars
    safe_star_masking_radius: float # Radius within which masking is not allowed
    sigma_radius: float # sigma_radius for DAO and IRAF star finders
    interpolation_extent: float # When interpolating, will consider a square area of 'interpolation_extent' times the star's radius
    safe_star_masking_prop_radius: float # The safe region will have a radius that corresponds to a circle with an area that is this times the number of pixels of the source
    safe_pix_ratio: float # If the star to galaxy size ratio is larger than this value, a warning is triggered and the galaxy is flagged
    star_min_radius: float # Minimum radius fro star to be detected
    star_max_radius: float # Maximum radius fro star to be detected
    star_filter_slope_thresh: float # threshold for the slope of the profile. When this threshold and delta_slope_threshold are met, we define the star's radius
    star_filter_delta_slope_thres: float # threshold for the change in slope of the profile. When this threshold and delta_slope_threshold are met, we define the star's radius
    star_filter_prof_cos_dist_thresh: float # the maximum cosine distance allowed between the star profile and the PSF profile
    star_filter_mean_symmetry_ratio_thresh: float # the maximum mean symmetry ratio allowed for the star profiles
    star_filter_gaussian_ratio_thresh: float # the maximum accepted ratio between the x and y sigma of the 2D gaussian fit to the star cutout
    star_filter_std_convergence_threshold: float # the threshold for the convergence of the profile stddev. If the stddev changes less than this value between two consecutive radii, we consider it converged
    star_filter_gaussian_fit_max_sigma: float # the maximum number of sigmas of the fitted gaussian at which we accept the radius.



    # Galaxy detection parameters
    source_min_pix: int # The minimum number of pixels that a source should have to be detected
    auto_safe_masking_region: bool # overrides safe_masking_reduction, which will be calculated based on the galaxy's redshift
    galaxy_detection_threshold: float
    connect_mode: int # Either 4 or 8

    # De-blending parameters
    deblend_mode: str
    deblend_levels: int
    deblend_contrast: float
    npixels_deblend_factor: float # When deblending, the minimum amount of pixels for detection will be number of pixels of the main source times this
    deblend_tool: str # Which tool will be used if performing de-blending
    overlapping_portion_threshold: float # The overlapping portion to determine the masking method on touching segments

    combination_method: str
    background_percentile: float # For artifact cleaning
    extreme_percentile: float # For artifact cleaning
    psf_std: float
    star_psf_stds: List[float] = field(default_factory=list)   # PSF kernel sizes for star detection

# Change settings cause big ass star is not being detected!
# Config instances
# Default configuration for morphology preprocessing
default_config = Config(bands_to_analyze = "r",
    do_background_subtraction = False, do_masking = True, interpolate_star_mask = True, do_mask_stars = True,
    do_deblending = False, do_artifacts_cleaning = True, print_results = True, write_results = True,
    write_input_parameters = True, delete_downloaded_files = True, notify_crash = True, notify_finish = True,
    background_percentile = 0.5, extreme_percentile = 0.99, masking_method = "precise", gain = 3.0, delta_slope_threshold = 2.7,
    artifact_segment_slack = 5, npixels_artifact = 20, star_detector = "dao", star_detection_threshold = 40, star_combination_method = "average",
    star_roundness = 0.2, star_sharpness_low = 0.4, star_min_radius = 3, star_max_radius = 90, safe_star_masking_radius = 20,
    interpolation_extent = 5, star_psf_stds = [0,1,2,3,4], sigma_radius = 1.5, artifact_candidate_ratio = 3000,
    source_min_pix = 10, auto_safe_masking_region = True, star_filter_psf_std = 1.0,
    galaxy_detection_threshold = 5, safe_star_masking_prop_radius = 0.005, deblend_mode = 'exponential', deblend_tool='sep',
    overlapping_portion_threshold = 0.15, safe_pix_ratio = 0.03,
    star_filter_slope_thresh = -0.008, star_filter_delta_slope_thres = 0.008, star_filter_prof_cos_dist_thresh = 0.045,
    star_filter_mean_symmetry_ratio_thresh = 0.07, star_filter_gaussian_ratio_thresh = 1.2,
    star_filter_std_convergence_threshold = 0.01, star_filter_gaussian_fit_max_sigma = 4,
    deblend_levels = 32, deblend_contrast = 0.005, npixels_deblend_factor = 0.005, connect_mode = 8,
    combination_method = "anti-exp-product", psf_std = 1.0, star_fwhm_span = 1.0)

dagn_config = Config(bands_to_analyze = "r",
    do_background_subtraction = False, do_masking = True, interpolate_star_mask = True, do_mask_stars = True,
    do_deblending = False, do_artifacts_cleaning = True, print_results = True, write_results = True,
    write_input_parameters = True, delete_downloaded_files = True, notify_crash = True, notify_finish = True,
    background_percentile = 0.5, extreme_percentile = 0.99, masking_method = "precise", gain = 3.0, delta_slope_threshold = 2.7,
    artifact_segment_slack = 5, npixels_artifact = 20, star_detector = "dao", star_detection_threshold = 40, star_combination_method = "average",
    star_roundness = 0.2, star_sharpness_low = 0.4, star_min_radius = 3, star_max_radius = 90, safe_star_masking_radius = 20,
    interpolation_extent = 5, star_psf_stds = [0,1,2,3,4], sigma_radius = 1.5, artifact_candidate_ratio = 3000,
    source_min_pix = 10, auto_safe_masking_region = True, star_filter_psf_std = 1.0,
    galaxy_detection_threshold = 7.5, safe_star_masking_prop_radius = 0.005, deblend_mode = 'exponential', deblend_tool='sep',
    overlapping_portion_threshold = 0.15, safe_pix_ratio = 0.03,
    star_filter_slope_thresh = -0.008, star_filter_delta_slope_thres = 0.008, star_filter_prof_cos_dist_thresh = 0.045,
    star_filter_mean_symmetry_ratio_thresh = 0.07, star_filter_gaussian_ratio_thresh = 1.2,
    star_filter_std_convergence_threshold = 0.01, star_filter_gaussian_fit_max_sigma = 4,
    deblend_levels = 32, deblend_contrast = 0.005, npixels_deblend_factor = 0.005, connect_mode = 8,
    combination_method = "anti-exp-product", psf_std = 1.0, star_fwhm_span = 1.0)

iraf_config = Config(bands_to_analyze = "g-r",
    do_background_subtraction = False, do_masking = True, interpolate_star_mask = True, do_mask_stars = True,
    do_deblending = True, do_artifacts_cleaning = True, print_results = True, write_results = True,
    write_input_parameters = True, delete_downloaded_files = True, notify_crash = True, notify_finish = True,
    background_percentile = 0.5, extreme_percentile = 0.99, masking_method = "precise", gain = 3.0, delta_slope_threshold = 2.7,
    artifact_segment_slack = 5, npixels_artifact = 50, star_detector = "iraf", star_detection_threshold = 20, star_combination_method = "anti-exp-product",
    star_roundness = 0.15, star_sharpness_low = 0.55, star_min_radius = 3, star_max_radius = 80, safe_star_masking_radius = 20,
    interpolation_extent = 5, star_psf_stds = [0,1,2,3], sigma_radius = 2,artifact_candidate_ratio = 3000,
    source_min_pix = 10, auto_safe_masking_region = True, star_filter_psf_std = 1.0,
    galaxy_detection_threshold = 10, safe_star_masking_prop_radius = 0.04, deblend_mode = 'exponential', deblend_tool='sep',
    overlapping_portion_threshold = 0.2, safe_pix_ratio = 0.03,
    star_filter_slope_thresh = -0.008, star_filter_delta_slope_thres = 0.008, star_filter_prof_cos_dist_thresh = 0.045,
    star_filter_mean_symmetry_ratio_thresh = 0.07, star_filter_gaussian_ratio_thresh = 1.2,
    star_filter_std_convergence_threshold = 0.01, star_filter_gaussian_fit_max_sigma = 4,
    deblend_levels = 30, deblend_contrast = 0.01, npixels_deblend_factor = 0.01, connect_mode = 4,
    combination_method = "anti-exp-product", psf_std = 2.0, star_fwhm_span = 2.5)


non_processing_config = Config(bands_to_analyze = "g-r",
    do_background_subtraction = False, do_masking = False, interpolate_star_mask = False, do_mask_stars = False,
    do_deblending = False, do_artifacts_cleaning = False, print_results = False, write_results = True,
    write_input_parameters = True, delete_downloaded_files = True, notify_crash = False, notify_finish = False,
    background_percentile = 0.5, extreme_percentile = 0.99, masking_method = "precise", gain = 3.0, delta_slope_threshold = 2.7,
    artifact_segment_slack = 5, npixels_artifact = 50, star_detector = "dao", star_detection_threshold = 100, star_combination_method = "average",
    star_roundness = 0.15, star_sharpness_low = 0.45, star_min_radius = 3, star_max_radius = 80, safe_star_masking_radius = 20,
    interpolation_extent = 5, star_psf_stds = [0,1,2,3], sigma_radius = 2,artifact_candidate_ratio = 3000,
    source_min_pix = 10, auto_safe_masking_region = False, star_filter_psf_std = 1.0,
    galaxy_detection_threshold = 10, safe_star_masking_prop_radius = 0.03, deblend_mode = 'exponential', deblend_tool='sep',
    overlapping_portion_threshold = 0.2, safe_pix_ratio = 0.03,
    star_filter_slope_thresh = -0.008, star_filter_delta_slope_thres = 0.008, star_filter_prof_cos_dist_thresh = 0.045,
    star_filter_mean_symmetry_ratio_thresh = 0.07, star_filter_gaussian_ratio_thresh = 1.2,
    star_filter_std_convergence_threshold = 0.01, star_filter_gaussian_fit_max_sigma = 4,
    deblend_levels = 10, deblend_contrast = 0.01, npixels_deblend_factor = 0.01, connect_mode = 4,
    combination_method = "average", psf_std = 2.0, star_fwhm_span = 2.5)

# prod_config = Config(param1=20, param2="production", debug=False)
# test_config = Config(param1=0, param2="testing", debug=True)

# Dictionary to hold configurations
configs = {
    "default": default_config,
    "non_processing": non_processing_config,
    "iraf": iraf_config,
    "dagn": dagn_config
}