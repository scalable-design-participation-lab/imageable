import numpy as np
from imageable._utils.masks.mask_operations import get_mask_area, get_mask_centroid, get_mask_limits, segment_horizontally_based_on_pixel_density


def _get_mock_mask():
    mask = np.zeros((640,640), dtype = np.uint8)

    x_lim_1 = 200
    x_lim_2 = 400

    mask[:, x_lim_1:x_lim_2] = 1

    return x_lim_1, x_lim_2,mask

def _get_mock_mask_with_density():
    mask = np.zeros((640,640), dtype = np.uint8)

    # Left segment: High density
    max_pixels = 640
    high_density_pixels = int(0.9 * max_pixels)
    mask[0:high_density_pixels, 0:150] = 1

    # Middle segment: high density
    medium_density_pixels = int(0.5 * max_pixels)
    mask[0:medium_density_pixels, 150:400] = 1

    # Right segment: medium density
    low_density_pixels = int(0.05 * max_pixels)
    mask[0:low_density_pixels, 400:580] = 1

    return mask


def test_mask_operations(): 
    x_lim_1, x_lim_2, mask = _get_mock_mask()

    area = get_mask_area(mask)
    area_expected = (x_lim_2 - x_lim_1) * mask.shape[0]


    centroid = get_mask_centroid(mask)

    x_min, y_min, x_max,y_max = get_mask_limits(mask)

    assert area == area_expected
    assert centroid[0]+0.5 == (x_lim_1 + x_lim_2) / 2
    assert centroid[1]+0.5 == mask.shape[0] / 2
    assert x_min == x_lim_1
    assert x_max == x_lim_2 - 1
    assert y_min == 0
    assert y_max == mask.shape[0]-1


def test_segment_horizontally_based_on_pixel_density():
    mask = _get_mock_mask_with_density()

    _, final_column = segment_horizontally_based_on_pixel_density(
        mask
    )

    assert final_column == 400
