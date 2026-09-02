from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Artifact detection parameters
    artifact_candidate_ratio: float # The maximum pixel value of the image should be at least this amount of times the image background to search for over exposure artifacts
    artifact_segment_slack: float # The distance in pixels at which the artifact profile will be used
    artifact_rel_size_thresh: float # Should be between 0 and 1. If the size of an artifact with respect to the image is greater than this, the artifact flag might be set to True (depending on artifact_safe_region_radius)
    artifact_delta_slope_threshold: float # The slope delta threshold to distinguish artifacts from other objects
    artifact_safe_region_radius: float # Should be between 0 and 1. The radius of the safe region proportional to the image size. If an artifact is (partially) within this region, the artifact flag might be set to True (depending on artifact_rel_size_thresh)
    artifact_bleeding_factor: float # Factor to extend or reduce the bleeding radius (i.e. number of pixel-thin layers to expand the artifact mask) used for artifact cleaning. Set to 0 to neglect artifact bleeding.
    artifact_interpolation_radius: float # Radius if the circle containing the pixels to be considered in the interpolation.
    npixels_artifact: int # The minimum amount of pixels that a source must have to be considered an artifact


    # Star detection parameters
    star_detector: str # Method used for star detection, e.g., 'dao', 'iraf', etc.
    star_detection_threshold: float # Threshold for star detection
    star_roundness: float # Roundness threshold for star detection
    star_sharpness_low: float # Lower threshold for sharpness in star detection
    star_fwhm_span: float # The span within which the FWHM will be used to detect stars at each psf kernel size
    auto_safe_masking_region: bool # Whether to automatically estimate the safe_star_masking_radius. If False, the value set in safe_star_masking_radius will be used.
    safe_star_masking_radius: float # Radius within which masking is not allowed.
    sigma_radius: float # sigma_radius for DAO and IRAF star finders
    interpolation_extent: float # When interpolating, will consider a square area of 'interpolation_extent' times the star's radius
    safe_star_masking_prop_radius: float # The safe region will have a radius that corresponds to a circle with an area that is this times the number of pixels of the source
    big_contaminator_pix_ratio: float # If the star to galaxy size ratio is larger than this value, a warning is triggered and the galaxy is flagged
    star_min_radius: float # Minimum radius fro star to be detected
    star_max_radius: float # Maximum radius fro star to be detected
    star_filter_psf_std: float # The std for the kernel used to convolve the image with a psf to filter the detected stars
    star_filter_slope_thresh: float # threshold for the slope of the profile. When this threshold and delta_slope_threshold are met, we define the star's radius
    star_filter_delta_slope_thres: float # threshold for the change in slope of the profile. When this threshold and delta_slope_threshold are met, we define the star's radius
    star_filter_prof_cos_dist_thresh: float # the maximum cosine distance allowed between the star profile and the PSF profile
    star_filter_mean_symmetry_ratio_thresh: float # the maximum mean symmetry ratio allowed for the star profiles
    star_filter_gaussian_ratio_thresh: float # the maximum accepted ratio between the x and y sigma of the 2D gaussian fit to the star cutout
    star_filter_gaussian_fit_max_sigma: float # the maximum number of sigmas of the fitted gaussian at which we accept the radius.


    # Galaxy detection parameters
    source_min_pix: int # The minimum number of pixels that a source should have to be detected
    galaxy_detection_threshold: float # Sources with background level times this value will be detected. See detect_threshold() from photutils
    connect_mode: int # Connecting mode for source detection. Either 4 or 8


    # De-blending parameters
    deblend_mode: str # Deblending mode for photutils/sep
    deblend_levels: int # Deblending levels for photutils/sep
    deblend_contrast: float # Deblending contrast for photutils/sep
    deblend_max_delta_z: float # A segment with a redshift difference higher than this with respect to the main segment will be deblended
    npixels_deblend_factor: float # The minimum amount of pixels for deblending detections will be the number of pixels of the main source times this
    deblend_tool: str # Which tool will be used if performing de-blending
    # keep_undecided_segments: bool # If True the segments that were not confirmed nor discarded in the deblending analysis will be kept, otherwise they will be discarded.
    overlap_portion_threshold: float # The overlapping portion to determine the masking method on touching segments
    z_query_timeout: float # The times (in seconds) after which the redshift queries will be cancelled.

    psf_std: float # When convolving the image, if no psf is provided a gaussian kernel with this std will be used by default
    star_psf_stds: List[float] = field(default_factory=list) # PSF kernel sizes for star detection

# Change settings cause big ass star is not being detected!
# Config instances

# Default configuration for morphology preprocessing, optimized for DESI
default_config = Config(artifact_delta_slope_threshold = 2.7, artifact_segment_slack = 5, npixels_artifact = 20,
                        star_detector = "dao", star_detection_threshold = 40, star_roundness = 0.2,
                        star_sharpness_low = 0.4, star_min_radius = 3, star_max_radius = 90, auto_safe_masking_region = True,
                        safe_star_masking_radius = 20, interpolation_extent = 5, star_psf_stds = [0,1,2,3,4],
                        sigma_radius = 1.5, artifact_candidate_ratio = 2500, source_min_pix = 40,
                        star_filter_psf_std = 1.0, galaxy_detection_threshold = 5,
                        safe_star_masking_prop_radius = 0.005, deblend_mode = 'exponential', deblend_tool='sep',
                        overlap_portion_threshold = 0.1, big_contaminator_pix_ratio = 0.03, star_filter_slope_thresh = -0.008,
                        star_filter_delta_slope_thres = 0.008, star_filter_prof_cos_dist_thresh = 0.045,
                        star_filter_mean_symmetry_ratio_thresh = 0.07, star_filter_gaussian_ratio_thresh = 1.2,
                        deblend_levels = 32, deblend_contrast = 0.005, npixels_deblend_factor = 0.005,
                        deblend_max_delta_z = 0.0035, connect_mode = 8,
                        psf_std = 1.0, star_fwhm_span = 1.0, star_filter_gaussian_fit_max_sigma = 4, artifact_rel_size_thresh = 0.005,
                        artifact_safe_region_radius = 0.25, artifact_bleeding_factor = 1.14, artifact_interpolation_radius = 1.0,
                        z_query_timeout = 60)


dagn_config = Config(artifact_delta_slope_threshold = 2.7, artifact_segment_slack = 5, npixels_artifact = 20,
                     star_detector = "dao", star_detection_threshold = 40, star_roundness = 0.2,
                     star_sharpness_low = 0.4, star_min_radius = 3, star_max_radius = 90, safe_star_masking_radius = 20,
                     interpolation_extent = 5, star_psf_stds = [0,1,2,3,4], sigma_radius = 1.5,
                     artifact_candidate_ratio = 2500, source_min_pix = 40, star_filter_psf_std = 1.0,
                     auto_safe_masking_region = True, galaxy_detection_threshold = 7.5, safe_star_masking_prop_radius = 0.005,
                     deblend_mode = 'exponential', deblend_tool='sep', overlap_portion_threshold = 0.15,
                     big_contaminator_pix_ratio = 0.03, star_filter_slope_thresh = -0.008, star_filter_delta_slope_thres = 0.008,
                     star_filter_prof_cos_dist_thresh = 0.045, star_filter_mean_symmetry_ratio_thresh = 0.07,
                     star_filter_gaussian_ratio_thresh = 1.2, deblend_levels = 32, deblend_contrast = 0.005,
                     npixels_deblend_factor = 0.005, deblend_max_delta_z = 0.0035,
                     connect_mode = 8, psf_std = 1.0, star_fwhm_span = 1.0, star_filter_gaussian_fit_max_sigma = 4,
                     artifact_rel_size_thresh = 0.01, artifact_safe_region_radius = 0.25, artifact_bleeding_factor = 1.14,
                     artifact_interpolation_radius = 1.0, z_query_timeout = 30)

iraf_config = Config(artifact_delta_slope_threshold = 2.7, artifact_segment_slack = 5, npixels_artifact = 50, star_detector = "iraf",
                     star_detection_threshold = 20, star_roundness = 0.15, star_sharpness_low = 0.55, star_min_radius = 3, star_max_radius = 80, safe_star_masking_radius = 20,
    interpolation_extent = 5, star_psf_stds = [0,1,2,3], sigma_radius = 2,artifact_candidate_ratio = 3000,
    source_min_pix = 40, star_filter_psf_std = 1.0, auto_safe_masking_region = True,
    galaxy_detection_threshold = 10, safe_star_masking_prop_radius = 0.04, deblend_mode = 'exponential', deblend_tool='sep',
    overlap_portion_threshold = 0.2, big_contaminator_pix_ratio = 0.03,
    star_filter_slope_thresh = -0.008, star_filter_delta_slope_thres = 0.008, star_filter_prof_cos_dist_thresh = 0.045,
    star_filter_mean_symmetry_ratio_thresh = 0.07, star_filter_gaussian_ratio_thresh = 1.2,
    deblend_levels = 30, deblend_contrast = 0.01, npixels_deblend_factor = 0.01, deblend_max_delta_z = 0.0035,
    connect_mode = 4, psf_std = 2.0, star_fwhm_span = 2.5, star_filter_gaussian_fit_max_sigma = 4, artifact_rel_size_thresh = 0.01,
                     artifact_safe_region_radius = 0.25, artifact_bleeding_factor = 1.14, artifact_interpolation_radius = 1.0
                     , z_query_timeout = 60)


non_processing_config = Config(artifact_delta_slope_threshold = 2.7, artifact_segment_slack = 5, npixels_artifact = 50,
                               star_detector = "dao", star_detection_threshold = 100,
    star_roundness = 0.15, star_sharpness_low = 0.45, star_min_radius = 3, star_max_radius = 80, safe_star_masking_radius = 20,
    interpolation_extent = 5, star_psf_stds = [0,1,2,3], sigma_radius = 2,artifact_candidate_ratio = 3000,
    source_min_pix = 40, star_filter_psf_std = 1.0, auto_safe_masking_region = True,
    galaxy_detection_threshold = 10, safe_star_masking_prop_radius = 0.03, deblend_mode = 'exponential', deblend_tool='sep',
    overlap_portion_threshold = 0.2, big_contaminator_pix_ratio = 0.03,
    star_filter_slope_thresh = -0.008, star_filter_delta_slope_thres = 0.008, star_filter_prof_cos_dist_thresh = 0.045,
    star_filter_mean_symmetry_ratio_thresh = 0.07, star_filter_gaussian_ratio_thresh = 1.2,
    deblend_levels = 10, deblend_contrast = 0.01, npixels_deblend_factor = 0.01, deblend_max_delta_z = 0.0035,
    connect_mode = 4, psf_std = 2.0, star_fwhm_span = 2.5,
    star_filter_gaussian_fit_max_sigma = 4, artifact_rel_size_thresh = 0.01, artifact_safe_region_radius = 0.25,
                               artifact_bleeding_factor = 1.14, artifact_interpolation_radius = 1.0
                               , z_query_timeout = 60)

# prod_config = Config(param1=20, param2="production", debug=False)
# test_config = Config(param1=0, param2="testing", debug=True)

# Dictionary to hold configurations
configs = {
    "default": default_config,
    "non_processing": non_processing_config,
    "iraf": iraf_config,
    "dagn": dagn_config
}