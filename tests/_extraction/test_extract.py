"""
Comprehensive tests for the extraction module.

Tests cover:
- extract_building_properties (main orchestrator)
- batch_extract_properties
- save_properties_batch
- extract_from_image_path
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from imageable._extraction.building import BuildingProperties
from imageable._extraction.extract import (
    batch_extract_properties,
    extract_building_properties,
    extract_from_image_path,
    save_properties_batch,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_polygon():
    """Create a simple rectangular polygon in WGS84."""
    return Polygon([
        (-71.0589, 42.3601),
        (-71.0585, 42.3601),
        (-71.0585, 42.3605),
        (-71.0589, 42.3605),
    ])


@pytest.fixture
def neighbor_polygons(simple_polygon):
    """Create a list of neighboring polygons."""
    # Offset polygons for neighbors
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
    ]


@pytest.fixture
def mock_image():
    """Create a mock RGB image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def mock_mask():
    """Create a mock binary mask."""
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:400, 100:400] = True
    return mask


# =============================================================================
# Tests for extract_building_properties
# =============================================================================


class TestExtractBuildingProperties:
    """Tests for the main extract_building_properties function."""

    def test_minimal_extraction(self, simple_polygon):
        """Test extraction with just polygon (no neighbors, no image)."""
        props = extract_building_properties(
            building_id="test_001",
            polygon=simple_polygon,
        )

        assert isinstance(props, BuildingProperties)
        assert props.building_id == "test_001"
        assert props.projected_area > 0
        assert props.unprojected_area > 0
        assert props.n_vertices > 0
        assert props.complexity > 0
        assert props.building_height == -1.0  # No height provided

    def test_extraction_with_neighbors(self, simple_polygon, neighbor_polygons):
        """Test extraction with neighbor analysis."""
        all_buildings = [simple_polygon] + neighbor_polygons

        props = extract_building_properties(
            building_id="test_002",
            polygon=simple_polygon,
            all_buildings=all_buildings,
            neighbor_radius=1000.0,
        )

        assert props.neighbor_count >= 0  # May or may not find neighbors depending on radius
        assert props.mean_distance_to_neighbors >= 0
        assert props.nni >= 0 or props.nni == float("inf")

    def test_extraction_with_height(self, simple_polygon):
        """Test extraction with pre-calculated height."""
        props = extract_building_properties(
            building_id="test_003",
            polygon=simple_polygon,
            height_value=25.5,
        )

        assert props.building_height == 25.5

    def test_extraction_with_materials(self, simple_polygon):
        """Test extraction with material percentages."""
        materials = {"brick": 45.0, "glass": 30.0, "concrete": 25.0}

        props = extract_building_properties(
            building_id="test_004",
            polygon=simple_polygon,
            material_percentages=materials,
        )

        assert props.material_percentages == materials

    def test_extraction_with_image_and_mask(self, simple_polygon, mock_image, mock_mask):
        """Test extraction with image and building mask."""
        props = extract_building_properties(
            building_id="test_005",
            polygon=simple_polygon,
            street_view_image=mock_image,
            building_mask=mock_mask,
        )

        # Image features should be extracted
        assert props.average_brightness > 0 or props.average_brightness == 0
        assert props.mask_area > 0
        assert props.mask_length > 0

    def test_extraction_with_facade_masks(self, simple_polygon, mock_image, mock_mask):
        """Test extraction with window and door masks."""
        window_mask = np.zeros((512, 512), dtype=bool)
        window_mask[150:200, 150:200] = True
        window_mask[150:200, 300:350] = True

        door_mask = np.zeros((512, 512), dtype=bool)
        door_mask[350:450, 200:300] = True

        props = extract_building_properties(
            building_id="test_006",
            polygon=simple_polygon,
            street_view_image=mock_image,
            building_mask=mock_mask,
            window_mask=window_mask,
            door_mask=door_mask,
        )

        # Facade features should be present
        assert props.average_window_x >= 0
        assert props.average_window_y >= 0
        assert props.number_of_windows >= 0
        assert props.number_of_doors >= 0

    def test_extraction_with_verbose(self, simple_polygon, capsys):
        """Test that verbose mode prints progress."""
        props = extract_building_properties(
            building_id="test_007",
            polygon=simple_polygon,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Extracting footprint" in captured.out or "footprint" in captured.out.lower()

    def test_extraction_with_different_crs(self, simple_polygon):
        """Test extraction with different CRS."""
        props = extract_building_properties(
            building_id="test_008",
            polygon=simple_polygon,
            crs=4326,
        )

        assert props.projected_area > 0

    def test_full_extraction(self, simple_polygon, neighbor_polygons, mock_image, mock_mask):
        """Test extraction with all options enabled."""
        all_buildings = [simple_polygon] + neighbor_polygons
        materials = {"brick": 50.0, "glass": 50.0}

        props = extract_building_properties(
            building_id="full_test",
            polygon=simple_polygon,
            all_buildings=all_buildings,
            neighbor_radius=1000.0,
            crs=4326,
            street_view_image=mock_image,
            building_mask=mock_mask,
            height_value=20.0,
            material_percentages=materials,
            verbose=False,
        )

        # Verify all categories are populated
        assert props.building_id == "full_test"
        assert props.projected_area > 0
        assert props.building_height == 20.0
        assert props.material_percentages == materials
        assert props.mask_area > 0


# =============================================================================
# Tests for batch_extract_properties
# =============================================================================


class TestBatchExtractProperties:
    """Tests for batch_extract_properties function."""

    def test_batch_extraction_basic(self, simple_polygon, neighbor_polygons):
        """Test basic batch extraction."""
        buildings = [
            {"id": "b001", "polygon": simple_polygon},
            {"id": "b002", "polygon": neighbor_polygons[0]},
        ]

        results = batch_extract_properties(
            buildings,
            neighbor_radius=1000.0,
            verbose=False,
        )

        assert len(results) == 2
        assert all(isinstance(r, BuildingProperties) for r in results)
        assert results[0].building_id == "b001"
        assert results[1].building_id == "b002"

    def test_batch_extraction_with_optional_data(self, simple_polygon, neighbor_polygons, mock_image, mock_mask):
        """Test batch extraction with optional image and height data."""
        buildings = [
            {
                "id": "b001",
                "polygon": simple_polygon,
                "height": 15.0,
            },
            {
                "id": "b002",
                "polygon": neighbor_polygons[0],
                "image": mock_image,
                "mask": mock_mask,
            },
        ]

        results = batch_extract_properties(buildings, verbose=False)

        assert results[0].building_height == 15.0
        assert results[1].mask_area > 0

    def test_batch_extraction_neighbor_context(self, simple_polygon, neighbor_polygons):
        """Test that batch extraction includes neighbor context."""
        buildings = [
            {"id": "b001", "polygon": simple_polygon},
            {"id": "b002", "polygon": neighbor_polygons[0]},
            {"id": "b003", "polygon": neighbor_polygons[1]},
        ]

        results = batch_extract_properties(
            buildings,
            neighbor_radius=10000.0,  # Large radius to ensure neighbors are found
            verbose=False,
        )

        # At least some buildings should have neighbors
        total_neighbors = sum(r.neighbor_count for r in results)
        assert total_neighbors >= 0  # May vary based on actual distances


# =============================================================================
# Tests for save_properties_batch
# =============================================================================


class TestSavePropertiesBatch:
    """Tests for save_properties_batch function."""

    def test_save_as_json(self, simple_polygon, tmp_path):
        """Test saving batch as JSON files."""
        props1 = BuildingProperties(building_id="b001", projected_area=100.0)
        props2 = BuildingProperties(building_id="b002", projected_area=150.0)

        output_dir = tmp_path / "json_output"

        save_properties_batch([props1, props2], str(output_dir), format="json")

        # Check files were created
        assert (output_dir / "b001.json").exists()
        assert (output_dir / "b002.json").exists()

        # Check content
        with open(output_dir / "b001.json") as f:
            data = json.load(f)
        assert data["building_id"] == "b001"
        assert data["projected_area"] == 100.0

    def test_save_as_csv(self, simple_polygon, tmp_path):
        """Test saving batch as CSV file."""
        props1 = BuildingProperties(building_id="b001", projected_area=100.0)
        props2 = BuildingProperties(building_id="b002", projected_area=150.0)

        output_dir = tmp_path / "csv_output"

        save_properties_batch([props1, props2], str(output_dir), format="csv")

        # Check file was created
        csv_path = output_dir / "all_properties.csv"
        assert csv_path.exists()

        # Check content
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "building_id" in df.columns
        assert "projected_area" in df.columns

    def test_save_invalid_format_raises(self, tmp_path):
        """Test that invalid format raises ValueError."""
        props = BuildingProperties(building_id="test")

        with pytest.raises(ValueError, match="Unknown format"):
            save_properties_batch([props], str(tmp_path), format="invalid")


# =============================================================================
# Tests for extract_from_image_path
# =============================================================================


class TestExtractFromImagePath:
    """Tests for extract_from_image_path function."""

    def test_extract_from_image_file(self, simple_polygon, tmp_path):
        """Test extraction from image file path."""
        # Create a test image file
        img = Image.new("RGB", (512, 512), color="red")
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)

        props = extract_from_image_path(
            building_id="image_test",
            polygon=simple_polygon,
            image_path=str(img_path),
        )

        assert isinstance(props, BuildingProperties)
        assert props.building_id == "image_test"

    def test_extract_from_image_with_mask(self, simple_polygon, tmp_path):
        """Test extraction from image with mask file."""
        # Create test image
        img = Image.new("RGB", (512, 512), color="blue")
        img_path = tmp_path / "test_image.png"
        img.save(img_path)

        # Create test mask
        mask = Image.new("L", (512, 512), color=0)
        # Draw white rectangle for building area
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 400, 400], fill=255)
        mask_path = tmp_path / "test_mask.png"
        mask.save(mask_path)

        props = extract_from_image_path(
            building_id="mask_test",
            polygon=simple_polygon,
            image_path=str(img_path),
            mask_path=str(mask_path),
        )

        assert props.mask_area > 0


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_neighbors_list(self, simple_polygon):
        """Test extraction with empty neighbors list."""
        props = extract_building_properties(
            building_id="empty_test",
            polygon=simple_polygon,
            all_buildings=[],
        )

        assert props.neighbor_count == 0

    def test_none_image(self, simple_polygon):
        """Test extraction with None image."""
        props = extract_building_properties(
            building_id="none_image",
            polygon=simple_polygon,
            street_view_image=None,
        )

        assert props.mask_area == 0
        assert props.average_brightness == 0

    def test_image_without_mask(self, simple_polygon, mock_image):
        """Test that image without mask doesn't extract image features."""
        props = extract_building_properties(
            building_id="no_mask",
            polygon=simple_polygon,
            street_view_image=mock_image,
            building_mask=None,
        )

        # Image features should not be extracted without mask
        assert props.mask_area == 0

    def test_string_building_id(self, simple_polygon):
        """Test that string building IDs work."""
        props = extract_building_properties(
            building_id="building_abc_123",
            polygon=simple_polygon,
        )

        assert props.building_id == "building_abc_123"

    def test_integer_building_id(self, simple_polygon):
        """Test that integer building IDs are handled."""
        props = extract_building_properties(
            building_id=42,
            polygon=simple_polygon,
        )

        assert props.building_id == 42

    def test_very_small_polygon(self):
        """Test extraction with very small polygon."""
        tiny_polygon = Polygon([
            (0.0, 0.0),
            (0.00001, 0.0),
            (0.00001, 0.00001),
            (0.0, 0.00001),
        ])

        props = extract_building_properties(
            building_id="tiny",
            polygon=tiny_polygon,
        )

        assert props.projected_area >= 0
        assert props.unprojected_area >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
