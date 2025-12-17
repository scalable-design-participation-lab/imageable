import numpy as np


def get_mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """
    Compute the centroid of a binary mask.

    Parameters
    ----------
    mask
        A binary mask array.

    Returns
    -------
    centroid
        A tuple (x,y) representing the centroid coordinates.
    """
    n = np.size(mask, 0)
    m = np.size(mask, 1)

    centroid_x = 0
    centroid_y = 0
    count = 0
    for i in range(n):
        for j in range(m):
            if mask[i, j] == 1:
                centroid_x += j
                centroid_y += i
                count += 1

    centroid_x = centroid_x / count
    centroid_y = centroid_y / count

    return (centroid_x, centroid_y)


def get_mask_area(mask: np.ndarray) -> int:
    """
    Get the area (number of pixels) of a binary mask.

    Parameters
    ----------
    mask
        Binary mask array.

    Returns
    -------
    area
        The area (number of pixels) of the mask
    """
    return np.sum(mask)


def get_mask_limits(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Get the bounding box of a binary mask.

    Parameters
    ----------
    mask
        Binary mask array

    Returns
    -------
    bounds
        A tuple (min_x, min_y, max_x, max_y)
    """
    n = np.size(mask, 0)
    m = np.size(mask, 1)

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for i in range(n):
        for j in range(m):
            if mask[i, j] == 1:
                min_x = min(min_x, j)
                min_y = min(min_y, i)
                max_x = max(max_x, j)
                max_y = max(max_y, i)

    if min_x == float("inf"):
        return None

    return (min_x, min_y, max_x, max_y)


def segment_horizontally_based_on_pixel_density(
    mask: np.ndarray, pixel_density_threshold: float = 0.15, start_x: int | None = 0
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
    assert 0 <= start_x < width, "start_x must be within the width of the mask"
    final_column = width - 1
    for x in range(start_x, width):
        column = mask[:, x]
        non_zero_count = np.count_nonzero(column)
        pixel_density = non_zero_count / height

        if pixel_density < pixel_density_threshold:
            # Zero out all columns to the right of this point
            mask[:, x:] = 0
            final_column = x
            break

    return mask, final_column
