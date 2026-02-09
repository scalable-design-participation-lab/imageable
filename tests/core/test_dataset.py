"""Tests for core.dataset module - high-level dataset extraction API."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from shapely.geometry import Polygon

from imageable.core.dataset import get_dataset
from imageable._extraction.building import BuildingProperties


@pytest.fixture
def simple_polygon():
    """Create a simple rectangular polygon for testing."""
    return Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])


class TestGetDataset:
    """Tests for the get_dataset function."""

    @patch("imageable.core.dataset.extract_building_properties")
    def test_returns_building_properties(self, mock_extract, simple_polygon):
        """Test that get_dataset returns BuildingProperties."""
        mock_props = BuildingProperties(building_id="test")
        mock_props.projected_area = 100.0
        mock_extract.return_value = mock_props

        result = get_dataset("api_key", simple_polygon)

        assert isinstance(result, BuildingProperties)
        assert result.projected_area == 100.0

    @patch("imageable.core.dataset.extract_building_properties")
    def test_passes_parameters_correctly(self, mock_extract, simple_polygon):
        """Test that parameters are passed to extract function."""
        mock_extract.return_value = BuildingProperties()

        get_dataset(
            "test_key",
            simple_polygon,
            neighbor_radius=200.0,
            crs=3857,
            verbose=True,
        )

        mock_extract.assert_called_once()
        call_kwargs = mock_extract.call_args[1]
        assert call_kwargs["building_id"] == "test_key"
        assert call_kwargs["neighbor_radius"] == 200.0
        assert call_kwargs["crs"] == 3857

    @patch("imageable.core.dataset.extract_building_properties")
    def test_single_building_mode(self, mock_extract, simple_polygon):
        """Test that all_buildings is empty for single building mode."""
        mock_extract.return_value = BuildingProperties()

        get_dataset("key", simple_polygon)

        call_kwargs = mock_extract.call_args[1]
        assert call_kwargs["all_buildings"] == []

    @patch("imageable.core.dataset.extract_building_properties")
    def test_passes_image_if_provided(self, mock_extract, simple_polygon):
        """Test that pre-fetched image is passed through."""
        mock_extract.return_value = BuildingProperties()
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        get_dataset("key", simple_polygon, image=test_image)

        call_kwargs = mock_extract.call_args[1]
        assert call_kwargs["street_view_image"] is test_image


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
