import numpy as np
from .utils import interpolate_image, grow_mask, add_noise_to_region, sigma_clipping, shrink_mask, get_profile_gradient
from photutils.segmentation import SourceCatalog, detect_sources


def detect_and_clean_artifacts(image, config, img_bkg_lvl, pix_scale=1, skip_flagged_artifacts = False,
                               return_image_only = False):
    artifact_mask = detect_artifacts(image, npixels_artifact=config.npixels_artifact,
                                     delta_slope_threshold=config.artifact_delta_slope_threshold,
                                     segment_slack=config.artifact_segment_slack,
                                     hi_pass_threshold=img_bkg_lvl * config.artifact_candidate_ratio)
    artifact_flag = get_artifact_flag(artifact_mask, config)
    if (artifact_mask == True).any() and not (skip_flagged_artifacts and artifact_flag):
        growth = round(np.sqrt(np.sum(artifact_mask))*0.15)
        artifact_mask = grow_mask(artifact_mask, pixels=growth)
        image = interpolate_image(image, artifact_mask, 1, method="inpaint")
        img_bkg_lvl, img_bkg_lvl_std = sigma_clipping(image)
        image = add_noise_to_region(image, artifact_mask, noise_center=0, noise_std=img_bkg_lvl_std, scale=pix_scale)
    if return_image_only:
        return image
    else:
        return image, artifact_mask, artifact_flag


def detect_artifacts(image, npixels_artifact=100, delta_slope_threshold = 2.7, segment_slack = 5, hi_pass_threshold = 1):
    # This function detects artifacts in the image data. If there are artifacts detected within the circle of radius =image_size/4 it will flag the image.
    # If there are no artifacts detected, it will mask the candidates outside the circle and replace them with noise so the image is still usable.

    hi_pass_mask = image > hi_pass_threshold
    candidate_artifacts_segmap = detect_sources(hi_pass_mask, 0.5, npixels_artifact, connectivity=8)
    if isinstance(candidate_artifacts_segmap, type(None)):
        candidate_artifacts_mask = np.full(image.shape, False, dtype=bool)
        return candidate_artifacts_mask

    detected_artifacts = candidate_artifacts_segmap.copy()
    candidate_artifacts_segmap.data[candidate_artifacts_segmap.data != 0] = 1
    candidate_artifacts_mask = candidate_artifacts_segmap.data.astype(bool)

    # Here we check the shape of each artifact candidate
    labels_to_remove = []
    detected_artifacts_cat = SourceCatalog(candidate_artifacts_mask, detected_artifacts)

    for artifact in detected_artifacts_cat:
        segment_mask = artifact.segment.astype(bool)
        segment_mask_padded = np.pad(segment_mask, ((min(segment_slack, artifact.bbox.iymin),
                                                     min(segment_slack,
                                                         image.shape[0] - artifact.bbox.iymax)),
                                                    (min(segment_slack, artifact.bbox.ixmin),
                                                     min(segment_slack,
                                                         image.shape[1] - artifact.bbox.ixmax))),
                                     mode='constant', constant_values=False)
        segment_cutout = image[max(0, artifact.bbox.iymin - segment_slack):min(image.shape[0],
                                                                               artifact.bbox.iymax + segment_slack),
                         max(0, artifact.bbox.ixmin - segment_slack):min(image.shape[1],
                                                                         artifact.bbox.ixmax + segment_slack)]
        current_mask = segment_mask_padded
        mean_values = []
        initial_mask = shrink_mask(segment_mask_padded, 1)
        profile_mask = np.logical_xor(current_mask,initial_mask)
        mean_values.append(float(np.mean(segment_cutout[profile_mask])))
        for grow_layer in range(1, segment_slack + 1):
            new_mask = grow_mask(segment_mask_padded, grow_layer)
            profile_mask = np.logical_xor(new_mask, current_mask)
            mean_values.append(float(np.mean(segment_cutout[profile_mask])))
            current_mask = new_mask
        profile_slopes, profile_delta_slopes = get_profile_gradient(mean_values)
        max_delta_slope = np.max(profile_delta_slopes)
        # An artifact will be considered as such if it has a max_delta_slope > delta_slope_threshold
        if max_delta_slope < delta_slope_threshold:
            labels_to_remove.append(artifact.label)
    detected_artifacts.remove_labels(labels_to_remove)
    detected_artifacts.data[detected_artifacts.data != 0] = 1
    detected_artifacts_mask = detected_artifacts.data.astype(bool)

    return detected_artifacts_mask

def get_artifact_flag(artifact_mask, config):
    center = (artifact_mask.shape[1] // 2, artifact_mask.shape[0] // 2)
    radius = min(artifact_mask.shape) // 4
    # Now we make a boolean mask if the circle around the center is not empty
    circle_mask = np.zeros(artifact_mask.shape, dtype=bool)
    y, x = np.ogrid[:artifact_mask.shape[0], :artifact_mask.shape[1]]
    circle_mask[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = True

    artifact_size = np.sum(artifact_mask)
    inner_region_size = np.sum(circle_mask)
    # print((circle_mask & artifact_mask).any())
    # print(artifact_size,inner_region_size,artifact_size/inner_region_size)
    if (artifact_size/inner_region_size) > config.artifact_rel_size_thresh:
        big_artifact = True
    else:
        big_artifact = False

    if (circle_mask & artifact_mask).any() and big_artifact:
        artifact_flag = True
    else:
        artifact_flag = False
    return artifact_flag
