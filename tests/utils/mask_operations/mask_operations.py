import numpy as np


def segment_horizontally_based_on_pixel_density(
    mask: np.ndarray, pixel_density_threshold: float = 0.1, start_x: int | None = 0
) -> np.ndarray:
    """
    Traverses a binary mask from left to right, leaving only the region from start_x to
    the column where the pixel density drops below a given threshold.

    Parameters
    ----------
    mask
        A binary mask as a 2D numpy array.
    pixel_density_threshold
        Minimum fraction of pixels in a column that must be non-zero to continue. Default is 0.1 (10%).
    start_x
        Column index to start traversing from. Default is 0 (leftmost column).
    """

    assert len(mask.shape) == 2, "Mask must be a 2D array"
    height, width = mask.shape

    if start_x is None:
        start_x = 0

    assert 0 <= start_x < width, "start_x must be within the width of the mask"

    for x in range(start_x, width):
        column = mask[:, x]
        non_zero_count = np.count_nonzero(column)
        print(non_zero_count)
        pixel_density = non_zero_count / height

        if pixel_density < pixel_density_threshold:
            # Zero out all columns to the right of this point
            mask[:, x:] = 0
            break

    return mask
