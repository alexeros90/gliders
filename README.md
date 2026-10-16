# GLIDErS (GaLaxy Image DEcontamination Scripts)

**GLIDErS** is a Python package for cleaning astronomical images while preserving the main source of interest, such as a galaxy, and reducing contamination from foreground stars, neighboring objects, and overexposure artifacts.

## Features

- **Artifact removal:** Detects and cleans potential overexposure artifacts.
- **Star masking:** Identifies contaminating stars and applies masking or interpolation based on their overlap with the main source.
- **Source deblending:** Separates overlapping sources using configurable criteria, including redshift and photometric-type information.
- **Background subtraction:** Supports background subtraction.
- **Background suppression:** Reduces the background noise level improving the signal-to-noise ratio for detecting faint sources or features.
- **Quality flags:** Reports potential issues such as significant stellar contamination, source truncation at image boundaries, failed redshift queries, and insufficient source size.

## Getting Started

The main entry point is `clean_image`, which combines the available cleaning operations into a configurable workflow.

```python
from gliders.pipeline import clean_image

cleaned_image, main_source_segment, cleaning_mask, cleaning_flags = clean_image(
    image,
    source_pos=(x, y),
    mask_stars=True,
    deblend_sources=True,
    clean_artifacts=True,
    show_plots=False
)
```

Here, `image` is a two-dimensional astronomical image, and `source_pos` specifies the approximate pixel coordinates `(x, y)` of the main source.

The function returns the cleaned image, the final main-source segmentation, a mask identifying pixels marked by selected cleaning operations, and flags summarizing potential issues encountered during processing.

GLIDErS supports configurable cleaning strategies, allowing individual operations to be enabled or disabled according to the scientific application.

## Configuration

GLIDErS uses a configuration object to control source detection, masking, and deblending parameters. When no configuration is provided, the package loads its default configuration.

## License

See the repository's `LICENSE` file for licensing information.
