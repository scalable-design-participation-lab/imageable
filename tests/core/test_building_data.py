"""Tests for core.building_data module - public API for building data extraction."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from imageable.core.building_data import (
    _extract_building_data_core,
    _format_output,
    _load_geojson_to_gdf,
    _load_local_image,
    get_building_data_from_file,
    get_building_data_from_gdf,
    get_building_data_from_geojson,
)
from imageable._extraction.building import BuildingProperties


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_polygon():
    """Create a simple rectangular polygon for testing."""
    return Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])


@pytest.fixture
def simple_gdf(simple_polygon):
    """Create a simple GeoDataFrame with one building."""
    return gpd.GeoDataFrame(
        {"id": ["building_001"], "name": ["Test Building"]},
        geometry=[simple_polygon],
        crs="EPSG:4326",
    )


@pytest.fixture
def multi_building_gdf():
    """Create a GeoDataFrame with multiple buildings."""
    polygons = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
        Polygon([(40, 0), (50, 0), (50, 10), (40, 10)]),
    ]
    return gpd.GeoDataFrame(
        {"building_id": ["b001", "b002", "b003"]},
        geometry=polygons,
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_building_properties():
    """Create a mock BuildingProperties for testing."""
    props = BuildingProperties(building_id="test_001")
    props.projected_area = 100.0
    props.building_height = 15.0
    return props


# =============================================================================
# Tests for _load_geojson_to_gdf
# =============================================================================


class TestLoadGeojsonToGdf:
    """Tests for the _load_geojson_to_gdf helper function."""

    def test_load_from_feature_collection_dict(self, simple_polygon):
        """Test loading from a GeoJSON FeatureCollection dict."""
        geojson_dict = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": simple_polygon.__geo_interface__,
                    "properties": {"name": "Building 1"},
                }
            ],
        }

        gdf = _load_geojson_to_gdf(geojson_dict)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.iloc[0]["name"] == "Building 1"

    def test_load_from_single_feature_dict(self, simple_polygon):
        """Test loading from a single GeoJSON Feature dict."""
        feature_dict = {
            "type": "Feature",
            "geometry": simple_polygon.__geo_interface__,
            "properties": {"name": "Single Building"},
        }

        gdf = _load_geojson_to_gdf(feature_dict)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

    def test_load_from_geometry_dict(self, simple_polygon):
        """Test loading from a bare geometry dict."""
        geometry_dict = simple_polygon.__geo_interface__

        gdf = _load_geojson_to_gdf(geometry_dict)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

    def test_load_from_file_path(self, simple_gdf, tmp_path):
        """Test loading from a GeoJSON file path."""
        geojson_path = tmp_path / "test.geojson"
        simple_gdf.to_file(geojson_path, driver="GeoJSON")

        gdf = _load_geojson_to_gdf(str(geojson_path))

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

    def test_load_from_pathlib_path(self, simple_gdf, tmp_path):
        """Test loading from a pathlib Path object."""
        geojson_path = tmp_path / "test.geojson"
        simple_gdf.to_file(geojson_path, driver="GeoJSON")

        gdf = _load_geojson_to_gdf(geojson_path)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

    def test_load_nonexistent_file_raises(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_geojson_to_gdf("/nonexistent/path/file.geojson")

    def test_load_unsupported_type_raises(self):
        """Test that unsupported GeoJSON type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported GeoJSON type"):
            _load_geojson_to_gdf({"type": "UnsupportedType"})


# =============================================================================
# Tests for _load_local_image
# =============================================================================


class TestLoadLocalImage:
    """Tests for the _load_local_image helper function."""

    def test_load_jpg_image(self, tmp_path):
        """Test loading a JPG image."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "building_001.jpg"
        img.save(img_path)

        result = _load_local_image(tmp_path, "building_001")

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

    def test_load_png_image(self, tmp_path):
        """Test loading a PNG image."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="blue")
        img_path = tmp_path / "building_002.png"
        img.save(img_path)

        result = _load_local_image(tmp_path, "building_002")

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_load_nonexistent_image_returns_none(self, tmp_path):
        """Test that loading nonexistent image returns None."""
        result = _load_local_image(tmp_path, "nonexistent_building")
        assert result is None


# =============================================================================
# Tests for _format_output
# =============================================================================


class TestFormatOutput:
    """Tests for the _format_output helper function."""

    def test_format_as_dict(self, simple_gdf, mock_building_properties):
        """Test formatting output as list of dicts."""
        results = [mock_building_properties]

        output = _format_output(results, simple_gdf, "dict")

        assert isinstance(output, list)
        assert len(output) == 1
        assert output[0]["building_id"] == "test_001"
        assert output[0]["projected_area"] == 100.0

    def test_format_as_geojson(self, simple_gdf, mock_building_properties):
        """Test formatting output as GeoJSON."""
        results = [mock_building_properties]

        output = _format_output(results, simple_gdf, "geojson")

        assert isinstance(output, dict)
        assert output["type"] == "FeatureCollection"
        assert len(output["features"]) == 1
        assert output["features"][0]["properties"]["building_id"] == "test_001"

    def test_format_as_gdf(self, simple_gdf, mock_building_properties):
        """Test formatting output as GeoDataFrame (default)."""
        results = [mock_building_properties]

        output = _format_output(results, simple_gdf, "gdf")

        assert isinstance(output, gpd.GeoDataFrame)
        assert len(output) == 1
        assert output.iloc[0]["building_id"] == "test_001"
        assert output.crs == simple_gdf.crs


# =============================================================================
# Tests for get_building_data_from_gdf
# =============================================================================


class TestGetBuildingDataFromGdf:
    """Tests for the main get_building_data_from_gdf function."""

    def test_raises_type_error_for_non_gdf(self):
        """Test that non-GeoDataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Expected GeoDataFrame"):
            get_building_data_from_gdf(
                {"not": "a geodataframe"},
                "fake_api_key",
            )

    def test_raises_value_error_for_empty_gdf(self):
        """Test that empty GeoDataFrame raises ValueError."""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        with pytest.raises(ValueError, match="GeoDataFrame is empty"):
            get_building_data_from_gdf(empty_gdf, "fake_api_key")

    @patch("imageable.core.building_data._extract_building_data_core")
    def test_calls_core_with_correct_args(self, mock_core, simple_gdf):
        """Test that correct arguments are passed to core function."""
        mock_core.return_value = simple_gdf

        get_building_data_from_gdf(
            simple_gdf,
            "test_api_key",
            id_column="id",
            neighbor_radius=200.0,
            output_format="dict",
            verbose=True,
        )

        mock_core.assert_called_once()
        call_kwargs = mock_core.call_args[1]
        assert call_kwargs["image_key"] == "test_api_key"
        assert call_kwargs["id_column"] == "id"
        assert call_kwargs["neighbor_radius"] == 200.0
        assert call_kwargs["output_format"] == "dict"
        assert call_kwargs["verbose"] is True


# =============================================================================
# Tests for get_building_data_from_geojson
# =============================================================================


class TestGetBuildingDataFromGeojson:
    """Tests for the get_building_data_from_geojson function."""

    @patch("imageable.core.building_data._extract_building_data_core")
    def test_loads_geojson_dict_and_calls_core(self, mock_core, simple_polygon):
        """Test that GeoJSON dict is loaded and processed."""
        geojson_dict = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": simple_polygon.__geo_interface__,
                    "properties": {"name": "Test"},
                }
            ],
        }
        mock_core.return_value = gpd.GeoDataFrame()

        get_building_data_from_geojson(geojson_dict, "test_api_key")

        mock_core.assert_called_once()

    def test_file_not_found_raises(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_building_data_from_geojson("/nonexistent/path.geojson", "api_key")


# =============================================================================
# Tests for get_building_data_from_file
# =============================================================================


class TestGetBuildingDataFromFile:
    """Tests for the get_building_data_from_file function."""

    def test_raises_for_nonexistent_footprints(self, tmp_path):
        """Test that nonexistent footprints file raises FileNotFoundError."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Footprints file not found"):
            get_building_data_from_file(
                tmp_path / "nonexistent.geojson",
                images_dir,
            )

    def test_raises_for_nonexistent_images_dir(self, simple_gdf, tmp_path):
        """Test that nonexistent images directory raises FileNotFoundError."""
        geojson_path = tmp_path / "footprints.geojson"
        simple_gdf.to_file(geojson_path, driver="GeoJSON")

        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            get_building_data_from_file(
                geojson_path,
                tmp_path / "nonexistent_images",
            )

    @patch("imageable.core.building_data._extract_building_data_core")
    def test_loads_files_and_calls_core(self, mock_core, simple_gdf, tmp_path):
        """Test that files are loaded and core function is called."""
        # Create footprints file
        geojson_path = tmp_path / "footprints.geojson"
        simple_gdf.to_file(geojson_path, driver="GeoJSON")

        # Create images directory
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        mock_core.return_value = gpd.GeoDataFrame()

        get_building_data_from_file(
            geojson_path,
            images_dir,
            id_column="id",
        )

        mock_core.assert_called_once()
        call_kwargs = mock_core.call_args[1]
        assert call_kwargs["images_dir"] == images_dir
        assert call_kwargs["id_column"] == "id"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the building data extraction pipeline."""

    @patch("imageable.core.building_data._fetch_street_view_image")
    @patch("imageable.core.building_data._estimate_height")
    @patch("imageable.core.building_data.extract_building_properties")
    def test_full_extraction_pipeline_mocked(
        self,
        mock_extract,
        mock_height,
        mock_image,
        simple_gdf,
    ):
        """Test full extraction pipeline with mocked dependencies."""
        # Setup mocks
        mock_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_height.return_value = 15.5

        mock_props = BuildingProperties(building_id="building_0")
        mock_props.projected_area = 100.0
        mock_props.building_height = 15.5
        mock_extract.return_value = mock_props

        # Run extraction
        result = get_building_data_from_gdf(
            simple_gdf,
            "test_api_key",
            output_format="gdf",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert mock_extract.called

    @patch("imageable.core.building_data._fetch_street_view_image")
    @patch("imageable.core.building_data._estimate_height")
    @patch("imageable.core.building_data.extract_building_properties")
    def test_extraction_with_id_column(
        self,
        mock_extract,
        mock_height,
        mock_image,
        multi_building_gdf,
    ):
        """Test extraction uses provided ID column."""
        mock_image.return_value = None
        mock_height.return_value = None

        mock_props = BuildingProperties()
        mock_extract.return_value = mock_props

        result = get_building_data_from_gdf(
            multi_building_gdf,
            "test_api_key",
            id_column="building_id",
            output_format="dict",
        )

        # Should have called extract 3 times (one per building)
        assert mock_extract.call_count == 3

    @patch("imageable.core.building_data._load_local_image")
    @patch("imageable.core.building_data.extract_building_properties")
    def test_local_image_extraction(
        self,
        mock_extract,
        mock_load_image,
        simple_gdf,
        tmp_path,
    ):
        """Test extraction from local images."""
        # Setup
        geojson_path = tmp_path / "footprints.geojson"
        simple_gdf.to_file(geojson_path, driver="GeoJSON")

        images_dir = tmp_path / "images"
        images_dir.mkdir()

        mock_load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_props = BuildingProperties(building_id="test")
        mock_extract.return_value = mock_props

        result = get_building_data_from_file(
            geojson_path,
            images_dir,
            output_format="dict",
        )

        assert isinstance(result, list)
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
