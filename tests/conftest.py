"""
Shared pytest fixtures for imageable tests.

This module provides common fixtures used across multiple test files.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon


@pytest.fixture
def simple_polygon():
    """Create a simple rectangular polygon in WGS84 coordinates."""
    return Polygon([
        (-71.0589, 42.3601),
        (-71.0585, 42.3601),
        (-71.0585, 42.3605),
        (-71.0589, 42.3605),
    ])


@pytest.fixture
def complex_polygon():
    """Create a more complex L-shaped polygon."""
    return Polygon([
        (-71.059, 42.360),
        (-71.058, 42.360),
        (-71.058, 42.361),
        (-71.057, 42.361),
        (-71.057, 42.362),
        (-71.059, 42.362),
    ])


@pytest.fixture
def neighbor_polygons():
    """Create a list of neighboring building polygons."""
    return [
        Polygon([
            (-71.0579, 42.3601),
            (-71.0575, 42.3601),
            (-71.0575, 42.3605),
            (-71.0579, 42.3605),
        ]),
        Polygon([
            (-71.0599, 42.3601),
            (-71.0595, 42.3601),
            (-71.0595, 42.3605),
            (-71.0599, 42.3605),
        ]),
        Polygon([
            (-71.0589, 42.3611),
            (-71.0585, 42.3611),
            (-71.0585, 42.3615),
            (-71.0589, 42.3615),
        ]),
    ]


@pytest.fixture
def mock_rgb_image():
    """Create a mock RGB image for testing."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def mock_building_mask():
    """Create a mock binary mask for building segmentation."""
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:400, 100:400] = True
    return mask


@pytest.fixture
def mock_window_mask():
    """Create a mock mask with window regions."""
    mask = np.zeros((512, 512), dtype=bool)
    # Two windows
    mask[150:200, 150:200] = True
    mask[150:200, 300:350] = True
    return mask


@pytest.fixture
def mock_door_mask():
    """Create a mock mask with door region."""
    mask = np.zeros((512, 512), dtype=bool)
    mask[350:450, 220:280] = True
    return mask


@pytest.fixture
def solid_red_image():
    """Create a solid red test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    return img


@pytest.fixture
def white_image():
    """Create a white test image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


@pytest.fixture
def black_image():
    """Create a black test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def rectangular_mask():
    """Create a rectangular test mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return mask


@pytest.fixture
def api_key():
    """Provide a placeholder API key for tests that mock API calls."""
    return "test_api_key_placeholder"
