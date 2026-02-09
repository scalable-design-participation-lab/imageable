"""
Comprehensive tests for the ImageCalculator class.

Tests cover:
- Color features (RGB, brightness, vividness)
- Shape features (area, length, complexity, vertices)
- Facade features (window/door positions and counts)
"""

import numpy as np
import pytest
from PIL import Image

from imageable._extraction.image import ImageCalculator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def solid_red_image():
    """Create a solid red image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Red channel
    return img


@pytest.fixture
def solid_green_image():
    """Create a solid green image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 1] = 255  # Green channel
    return img


@pytest.fixture
def solid_blue_image():
    """Create a solid blue image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # Blue channel
    return img


@pytest.fixture
def white_image():
    """Create a white image (high brightness)."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


@pytest.fixture
def black_image():
    """Create a black image (low brightness)."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def gradient_image():
    """Create a gradient image for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img[i, :, :] = int(i * 2.55)
    return img


@pytest.fixture
def rectangular_mask():
    """Create a rectangular mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return mask


@pytest.fixture
def circular_mask():
    """Create a circular mask."""
    mask = np.zeros((100, 100), dtype=bool)
    y, x = np.ogrid[:100, :100]
    center = (50, 50)
    radius = 30
    mask[(x - center[0])**2 + (y - center[1])**2 <= radius**2] = True
    return mask


@pytest.fixture
def complex_mask():
    """Create a more complex polygon mask (L-shape)."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:50] = True  # Vertical part
    mask[60:90, 10:90] = True  # Horizontal part
    return mask


@pytest.fixture
def window_mask():
    """Create a mask with two window regions."""
    mask = np.zeros((100, 100), dtype=bool)
    # Window 1
    mask[20:35, 20:35] = True
    # Window 2
    mask[20:35, 65:80] = True
    return mask


@pytest.fixture
def door_mask():
    """Create a mask with a door region."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[60:95, 40:60] = True
    return mask


# =============================================================================
# Tests for Color Features
# =============================================================================


class TestColorFeatures:
    """Tests for color feature extraction."""

    def test_red_channel_solid_red(self, solid_red_image, rectangular_mask):
        """Test red channel on solid red image."""
        calc = ImageCalculator(solid_red_image, rectangular_mask)
        assert calc.average_red_channel_value() == 255.0

    def test_red_channel_solid_blue(self, solid_blue_image, rectangular_mask):
        """Test red channel on solid blue image (should be 0)."""
        calc = ImageCalculator(solid_blue_image, rectangular_mask)
        assert calc.average_red_channel_value() == 0.0

    def test_green_channel_solid_green(self, solid_green_image, rectangular_mask):
        """Test green channel on solid green image."""
        calc = ImageCalculator(solid_green_image, rectangular_mask)
        assert calc.average_green_channel_value() == 255.0

    def test_blue_channel_solid_blue(self, solid_blue_image, rectangular_mask):
        """Test blue channel on solid blue image."""
        calc = ImageCalculator(solid_blue_image, rectangular_mask)
        assert calc.average_blue_channel_value() == 255.0

    def test_brightness_white_image(self, white_image, rectangular_mask):
        """Test brightness on white image (should be 100%)."""
        calc = ImageCalculator(white_image, rectangular_mask)
        brightness = calc.average_brightness()
        assert brightness == pytest.approx(100.0, rel=0.01)

    def test_brightness_black_image(self, black_image, rectangular_mask):
        """Test brightness on black image (should be 0%)."""
        calc = ImageCalculator(black_image, rectangular_mask)
        brightness = calc.average_brightness()
        assert brightness == pytest.approx(0.0, abs=0.01)

    def test_vividness_saturated(self, solid_red_image, rectangular_mask):
        """Test vividness on saturated color."""
        calc = ImageCalculator(solid_red_image, rectangular_mask)
        vividness = calc.average_vividness()
        assert vividness == pytest.approx(100.0, rel=0.01)

    def test_vividness_gray(self, rectangular_mask):
        """Test vividness on grayscale image (should be 0%)."""
        gray_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        calc = ImageCalculator(gray_img, rectangular_mask)
        vividness = calc.average_vividness()
        assert vividness == pytest.approx(0.0, abs=1.0)

    def test_color_features_no_mask(self, solid_red_image):
        """Test color features without mask (uses entire image)."""
        calc = ImageCalculator(solid_red_image, building_mask=None)
        assert calc.average_red_channel_value() == 255.0
        assert calc.average_green_channel_value() == 0.0
        assert calc.average_blue_channel_value() == 0.0


# =============================================================================
# Tests for Shape Features
# =============================================================================


class TestShapeFeatures:
    """Tests for shape feature extraction."""

    def test_mask_area_rectangular(self, white_image, rectangular_mask):
        """Test mask area on rectangular mask."""
        calc = ImageCalculator(white_image, rectangular_mask)
        expected_area = 60 * 60  # 60x60 rectangle
        assert calc.mask_area() == expected_area

    def test_mask_area_no_mask(self, white_image):
        """Test mask area without mask (entire image)."""
        calc = ImageCalculator(white_image, building_mask=None)
        expected_area = 100 * 100
        assert calc.mask_area() == expected_area

    def test_mask_length_rectangular(self, white_image, rectangular_mask):
        """Test mask length (perimeter) on rectangular mask."""
        calc = ImageCalculator(white_image, rectangular_mask)
        length = calc.mask_length()
        # Perimeter should be approximately 4 * 60 = 240
        assert length == pytest.approx(240.0, rel=0.1)

    def test_mask_complexity_rectangular(self, white_image, rectangular_mask):
        """Test complexity on rectangular mask."""
        calc = ImageCalculator(white_image, rectangular_mask)
        complexity = calc.mask_complexity()
        # Complexity = perimeter / area
        expected = 240.0 / (60 * 60)
        assert complexity == pytest.approx(expected, rel=0.1)

    def test_number_of_edges_rectangular(self, white_image, rectangular_mask):
        """Test number of edges on rectangular mask."""
        calc = ImageCalculator(white_image, rectangular_mask)
        edges = calc.number_of_edges()
        # Rectangle should have approximately 4 edges
        assert edges >= 4

    def test_number_of_vertices_rectangular(self, white_image, rectangular_mask):
        """Test number of vertices (same as edges for closed polygon)."""
        calc = ImageCalculator(white_image, rectangular_mask)
        vertices = calc.number_of_vertices()
        edges = calc.number_of_edges()
        assert vertices == edges

    def test_circular_mask_complexity(self, white_image, circular_mask):
        """Test that circular mask has lower complexity than rectangle."""
        calc_rect = ImageCalculator(white_image, np.zeros((100, 100), dtype=bool))
        rect_mask = np.zeros((100, 100), dtype=bool)
        rect_mask[20:80, 20:80] = True
        calc_rect = ImageCalculator(white_image, rect_mask)

        calc_circle = ImageCalculator(white_image, circular_mask)

        # Circle should have different complexity than rectangle
        assert calc_rect.mask_complexity() != calc_circle.mask_complexity()

    def test_empty_mask(self, white_image):
        """Test features with empty mask."""
        empty_mask = np.zeros((100, 100), dtype=bool)
        calc = ImageCalculator(white_image, empty_mask)

        assert calc.mask_area() == 0
        assert calc.mask_length() == 0.0
        assert calc.mask_complexity() == 0.0


# =============================================================================
# Tests for Facade Features
# =============================================================================


class TestFacadeFeatures:
    """Tests for facade feature extraction (windows, doors)."""

    def test_window_position_single(self, white_image, rectangular_mask):
        """Test window position with single window."""
        # Single window at center
        single_window = np.zeros((100, 100), dtype=bool)
        single_window[40:60, 40:60] = True

        calc = ImageCalculator(white_image, rectangular_mask)

        avg_x = calc.average_window_x(single_window)
        avg_y = calc.average_window_y(single_window)

        # Center of mass should be at center of window
        assert avg_x == pytest.approx(50.0, rel=0.1)
        assert avg_y == pytest.approx(50.0, rel=0.1)

    def test_window_position_two_windows(self, white_image, rectangular_mask, window_mask):
        """Test window position with two windows."""
        calc = ImageCalculator(white_image, rectangular_mask)

        avg_x = calc.average_window_x(window_mask)
        avg_y = calc.average_window_y(window_mask)

        # With two windows on left and right, x should be somewhere in the middle
        assert avg_x > 20 and avg_x < 80
        assert avg_y > 0

    def test_door_position(self, white_image, rectangular_mask, door_mask):
        """Test door position calculation."""
        calc = ImageCalculator(white_image, rectangular_mask)

        avg_x = calc.average_door_x(door_mask)
        avg_y = calc.average_door_y(door_mask)

        # Door should be around x=50, y=77.5 (center of door mask)
        assert avg_x == pytest.approx(50.0, rel=0.1)
        assert avg_y > 60

    def test_number_of_windows_two(self, white_image, rectangular_mask, window_mask):
        """Test counting two distinct windows."""
        calc = ImageCalculator(white_image, rectangular_mask)

        n_windows = calc.number_of_windows(window_mask, max_k=10)
        # Should detect approximately 2 clusters
        assert n_windows >= 1 and n_windows <= 5

    def test_number_of_doors_one(self, white_image, rectangular_mask, door_mask):
        """Test counting one door."""
        calc = ImageCalculator(white_image, rectangular_mask)

        n_doors = calc.number_of_doors(door_mask, max_k=5)
        assert n_doors >= 1

    def test_empty_window_mask(self, white_image, rectangular_mask):
        """Test window features with empty mask."""
        empty_mask = np.zeros((100, 100), dtype=bool)
        calc = ImageCalculator(white_image, rectangular_mask)

        avg_x = calc.average_window_x(empty_mask)
        avg_y = calc.average_window_y(empty_mask)

        assert avg_x == 0.0
        assert avg_y == 0.0


# =============================================================================
# Tests for extract_all_features
# =============================================================================


class TestExtractAllFeatures:
    """Tests for the convenience extract_all_features method."""

    def test_extract_all_basic(self, white_image, rectangular_mask):
        """Test extracting all features without facade masks."""
        calc = ImageCalculator(white_image, rectangular_mask)
        features = calc.extract_all_features()

        # Check all basic features are present
        assert "average_red_channel_value" in features
        assert "average_green_channel_value" in features
        assert "average_blue_channel_value" in features
        assert "average_brightness" in features
        assert "average_vividness" in features
        assert "mask_area" in features
        assert "mask_length" in features
        assert "mask_complexity" in features
        assert "number_of_edges" in features
        assert "number_of_vertices" in features

        # Facade features should NOT be present without masks
        assert "average_window_x" not in features
        assert "number_of_windows" not in features

    def test_extract_all_with_facades(self, white_image, rectangular_mask, window_mask, door_mask):
        """Test extracting all features including facade features."""
        calc = ImageCalculator(white_image, rectangular_mask)
        features = calc.extract_all_features(
            window_mask=window_mask,
            door_mask=door_mask,
        )

        # Check facade features are present
        assert "average_window_x" in features
        assert "average_window_y" in features
        assert "number_of_windows" in features
        assert "average_door_x" in features
        assert "average_door_y" in features
        assert "number_of_doors" in features

    def test_extract_all_values_reasonable(self, solid_red_image, rectangular_mask):
        """Test that extracted values are reasonable."""
        calc = ImageCalculator(solid_red_image, rectangular_mask)
        features = calc.extract_all_features()

        # RGB values should be in valid range
        assert 0 <= features["average_red_channel_value"] <= 255
        assert 0 <= features["average_green_channel_value"] <= 255
        assert 0 <= features["average_blue_channel_value"] <= 255

        # Percentages should be 0-100
        assert 0 <= features["average_brightness"] <= 100
        assert 0 <= features["average_vividness"] <= 100

        # Area and other metrics should be non-negative
        assert features["mask_area"] >= 0
        assert features["mask_length"] >= 0
        assert features["mask_complexity"] >= 0


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_pixel_mask(self, white_image):
        """Test with single pixel mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True

        calc = ImageCalculator(white_image, mask)

        assert calc.mask_area() == 1
        # Single pixel has no perimeter in traditional sense
        assert calc.mask_length() >= 0

    def test_full_image_mask(self, white_image):
        """Test with mask covering entire image."""
        mask = np.ones((100, 100), dtype=bool)
        calc = ImageCalculator(white_image, mask)

        assert calc.mask_area() == 10000

    def test_uint8_mask(self, white_image):
        """Test with uint8 mask (non-boolean)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        calc = ImageCalculator(white_image, mask)

        # Should still work - converts to boolean
        assert calc.mask_area() > 0

    def test_float_mask(self, white_image):
        """Test with float mask."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0

        calc = ImageCalculator(white_image, mask)
        assert calc.mask_area() > 0

    def test_random_image(self, rectangular_mask):
        """Test with random noise image."""
        random_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        calc = ImageCalculator(random_img, rectangular_mask)

        features = calc.extract_all_features()

        # All values should be valid
        for key, value in features.items():
            assert not np.isnan(value)
            assert not np.isinf(value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
