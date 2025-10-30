"""
Tests for BuildingProperties dataclass.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path

from imageable.properties.building import BuildingProperties


class TestBuildingPropertiesInitialization:
    """Test initialization and default values."""

    def test_default_initialization(self):
        """Test that all fields have correct default values."""
        props = BuildingProperties()

        # Footprint geometrical
        assert props.unprojected_area == 0.0
        assert props.projected_area == 0.0
        assert props.longitude_difference == 0.0
        assert props.latitude_difference == 0.0
        assert props.n_vertices == 0
        assert props.shape_length == 0.0

        # Footprint engineered
        assert props.complexity == 0.0
        assert props.inverse_average_segment_length == 0.0
        assert props.vertices_per_area == 0.0
        assert props.average_complexity_per_segment == 0.0
        assert props.isoperimetric_quotient == 0.0

        # Footprint contextual
        assert props.neighbor_count == 0
        assert props.mean_distance_to_neighbors == 0.0
        assert props.expected_nearest_neighbor_distance == 0.0
        assert props.nearest_neighbor_distance == 0.0
        assert props.n_size_mean == 0.0
        assert props.n_size_std == 0.0
        assert props.n_size_min == 0.0
        assert props.n_size_max == 0.0
        assert props.n_size_cv == 0.0
        assert props.nni == 0.0

        # Height
        assert props.building_height == -1.0

        # Materials
        assert props.material_percentages == {}

        # Image color
        assert props.average_red_channel_value == 0.0
        assert props.average_green_channel_value == 0.0
        assert props.average_blue_channel_value == 0.0
        assert props.average_brightness == 0.0
        assert props.average_vividness == 0.0

        # Image shape
        assert props.mask_area == 0
        assert props.mask_length == 0.0
        assert props.mask_complexity == 0.0
        assert props.number_of_edges == 0
        assert props.number_of_vertices == 0

        # Image façade
        assert props.average_window_x == 0.0
        assert props.average_window_y == 0.0
        assert props.average_door_x == 0.0
        assert props.average_door_y == 0.0
        assert props.number_of_windows == 0
        assert props.number_of_doors == 0

        # Metadata
        assert props.building_id is None

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        props = BuildingProperties(
            unprojected_area=150.5,
            projected_area=145.2,
            building_id="building_001",
            building_height=15.5,
            neighbor_count=5,
        )

        assert props.unprojected_area == 150.5
        assert props.projected_area == 145.2
        assert props.building_id == "building_001"
        assert props.building_height == 15.5
        assert props.neighbor_count == 5

    def test_initialization_with_materials(self):
        """Test initialization with material percentages."""
        materials = {"brick": 60.0, "glass": 40.0}
        props = BuildingProperties(material_percentages=materials)
        assert props.material_percentages == materials


class TestUpdateMethods:
    """Test update methods for different feature categories."""

    def test_update_footprint_features_all_geometrical(self):
        """Test updating all geometrical footprint features."""
        props = BuildingProperties()

        footprint_data = {
            "unprojected_area": 120.5,
            "projected_area": 115.3,
            "longitude_difference": 0.001,
            "latitude_difference": 0.002,
            "n_vertices": 8,
            "shape_length": 45.2,
        }

        props.update_footprint_features(footprint_data)

        assert props.unprojected_area == 120.5
        assert props.projected_area == 115.3
        assert props.longitude_difference == 0.001
        assert props.latitude_difference == 0.002
        assert props.n_vertices == 8
        assert props.shape_length == 45.2

    def test_update_footprint_features_all_engineered(self):
        """Test updating all engineered footprint features."""
        props = BuildingProperties()

        footprint_data = {
            "complexity": 0.85,
            "inverse_average_segment_length": 0.15,
            "vertices_per_area": 0.069,
            "average_complexity_per_segment": 0.106,
            "isoperimetric_quotient": 0.72,
        }

        props.update_footprint_features(footprint_data)

        assert props.complexity == 0.85
        assert props.inverse_average_segment_length == 0.15
        assert props.vertices_per_area == 0.069
        assert props.average_complexity_per_segment == 0.106
        assert props.isoperimetric_quotient == 0.72

    def test_update_footprint_features_all_contextual(self):
        """Test updating all contextual footprint features."""
        props = BuildingProperties()

        footprint_data = {
            "neighbor_count": 5,
            "mean_distance_to_neighbors": 12.5,
            "expected_nearest_neighbor_distance": 8.3,
            "nearest_neighbor_distance": 7.5,
            "n_size_mean": 110.0,
            "n_size_std": 15.2,
            "n_size_min": 85.0,
            "n_size_max": 135.0,
            "n_size_cv": 0.138,
            "nni": 0.903,
        }

        props.update_footprint_features(footprint_data)

        assert props.neighbor_count == 5
        assert props.mean_distance_to_neighbors == 12.5
        assert props.expected_nearest_neighbor_distance == 8.3
        assert props.nearest_neighbor_distance == 7.5
        assert props.n_size_mean == 110.0
        assert props.n_size_std == 15.2
        assert props.n_size_min == 85.0
        assert props.n_size_max == 135.0
        assert props.n_size_cv == 0.138
        assert props.nni == 0.903

    def test_update_footprint_features_partial(self):
        """Test updating only some footprint features."""
        props = BuildingProperties()

        footprint_data = {"unprojected_area": 100.0, "neighbor_count": 3, "complexity": 0.5}

        props.update_footprint_features(footprint_data)

        assert props.unprojected_area == 100.0
        assert props.neighbor_count == 3
        assert props.complexity == 0.5
        # Other fields remain at defaults
        assert props.projected_area == 0.0
        assert props.n_vertices == 0

    def test_update_footprint_features_empty_dict(self):
        """Test updating with empty dictionary doesn't change values."""
        props = BuildingProperties(unprojected_area=50.0)
        props.update_footprint_features({})
        assert props.unprojected_area == 50.0

    def test_update_height_with_value(self):
        """Test updating height with a valid value."""
        props = BuildingProperties()
        props.update_height(25.5)

        assert props.building_height == 25.5

    def test_update_height_with_zero(self):
        """Test updating height with zero."""
        props = BuildingProperties()
        props.update_height(0.0)

        assert props.building_height == 0.0

    def test_update_height_with_none(self):
        """Test updating height with None (failed calculation)."""
        props = BuildingProperties()
        props.update_height(None)

        assert props.building_height == -1.0

    def test_update_material_percentages(self):
        """Test updating material percentages."""
        props = BuildingProperties()

        materials = {"brick": 45.5, "glass": 30.2, "concrete": 15.8, "metal": 8.5}

        props.update_material_percentages(materials)

        assert props.material_percentages == materials
        # Verify it's a copy, not a reference
        materials["brick"] = 50.0
        assert props.material_percentages["brick"] == 45.5

    def test_update_material_percentages_empty(self):
        """Test updating with empty material dict."""
        props = BuildingProperties()
        props.update_material_percentages({})
        assert props.material_percentages == {}

    def test_update_image_features_all_color(self):
        """Test updating all color features."""
        props = BuildingProperties()

        image_data = {
            "average_red_channel_value": 120.5,
            "average_green_channel_value": 115.3,
            "average_blue_channel_value": 110.2,
            "average_brightness": 135.5,
            "average_vividness": 85.2,
        }

        props.update_image_features(image_data)

        assert props.average_red_channel_value == 120.5
        assert props.average_green_channel_value == 115.3
        assert props.average_blue_channel_value == 110.2
        assert props.average_brightness == 135.5
        assert props.average_vividness == 85.2

    def test_update_image_features_all_shape(self):
        """Test updating all shape features."""
        props = BuildingProperties()

        image_data = {
            "mask_area": 15000,
            "mask_length": 450.5,
            "mask_complexity": 0.75,
            "number_of_edges": 12,
            "number_of_vertices": 12,
        }

        props.update_image_features(image_data)

        assert props.mask_area == 15000
        assert props.mask_length == 450.5
        assert props.mask_complexity == 0.75
        assert props.number_of_edges == 12
        assert props.number_of_vertices == 12

    def test_update_image_features_all_facade(self):
        """Test updating all façade features."""
        props = BuildingProperties()

        image_data = {
            "average_window_x": 250.5,
            "average_window_y": 180.3,
            "average_door_x": 320.0,
            "average_door_y": 420.5,
            "number_of_windows": 8,
            "number_of_doors": 2,
        }

        props.update_image_features(image_data)

        assert props.average_window_x == 250.5
        assert props.average_window_y == 180.3
        assert props.average_door_x == 320.0
        assert props.average_door_y == 420.5
        assert props.number_of_windows == 8
        assert props.number_of_doors == 2

    def test_update_image_features_partial(self):
        """Test updating only some image features."""
        props = BuildingProperties()

        image_data = {"average_brightness": 128.0, "number_of_windows": 5, "mask_area": 12000}

        props.update_image_features(image_data)

        assert props.average_brightness == 128.0
        assert props.number_of_windows == 5
        assert props.mask_area == 12000
        # Others remain at defaults
        assert props.average_red_channel_value == 0.0
        assert props.number_of_doors == 0

    def test_update_image_features_empty_dict(self):
        """Test updating with empty dictionary doesn't change values."""
        props = BuildingProperties(mask_area=1000)
        props.update_image_features({})
        assert props.mask_area == 1000


class TestSerialization:
    """Test serialization and deserialization methods."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        props = BuildingProperties(unprojected_area=150.0, building_height=20.5, building_id="test_001")
        props.update_material_percentages({"brick": 60.0, "glass": 40.0})

        result = props.to_dict()

        assert isinstance(result, dict)
        assert result["unprojected_area"] == 150.0
        assert result["building_height"] == 20.5
        assert result["building_id"] == "test_001"
        assert result["material_percentages"] == {"brick": 60.0, "glass": 40.0}

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields."""
        props = BuildingProperties()
        result = props.to_dict()

        # Check a few key fields exist
        assert "unprojected_area" in result
        assert "building_height" in result
        assert "material_percentages" in result
        assert "average_brightness" in result
        assert "neighbor_count" in result

    def test_to_json_string(self):
        """Test converting to JSON string."""
        props = BuildingProperties(unprojected_area=150.0, building_id="test_001")

        json_str = props.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["unprojected_area"] == 150.0
        assert parsed["building_id"] == "test_001"

    def test_to_json_file(self):
        """Test saving to JSON file."""
        props = BuildingProperties(unprojected_area=150.0, building_height=20.5, building_id="test_001")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            json_str = props.to_json(filepath)

            # Verify file was created
            assert os.path.exists(filepath)

            # Verify content
            with open(filepath, "r") as f:
                content = json.load(f)

            assert content["unprojected_area"] == 150.0
            assert content["building_height"] == 20.5
            assert content["building_id"] == "test_001"
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "unprojected_area": 150.0,
            "projected_area": 145.0,
            "building_height": 20.5,
            "building_id": "test_001",
            "neighbor_count": 5,
            "material_percentages": {"brick": 60.0},
        }

        props = BuildingProperties.from_dict(data)

        assert props.unprojected_area == 150.0
        assert props.projected_area == 145.0
        assert props.building_height == 20.5
        assert props.building_id == "test_001"
        assert props.neighbor_count == 5
        assert props.material_percentages == {"brick": 60.0}

    def test_from_dict_partial(self):
        """Test creating from partial dictionary uses defaults."""
        data = {"unprojected_area": 150.0, "building_id": "test_001"}

        props = BuildingProperties.from_dict(data)

        assert props.unprojected_area == 150.0
        assert props.building_id == "test_001"
        assert props.projected_area == 0.0  # Default
        assert props.building_height == -1.0  # Default

    def test_from_json_string(self):
        """Test creating from JSON string."""
        json_str = json.dumps({"unprojected_area": 150.0, "building_height": 20.5, "building_id": "test_001"})

        props = BuildingProperties.from_json(json_str)

        assert props.unprojected_area == 150.0
        assert props.building_height == 20.5
        assert props.building_id == "test_001"

    def test_from_json_file(self):
        """Test creating from JSON file."""
        data = {"unprojected_area": 150.0, "building_height": 20.5, "building_id": "test_001", "neighbor_count": 3}

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(data, f)
            filepath = f.name

        try:
            props = BuildingProperties.from_json(filepath)

            assert props.unprojected_area == 150.0
            assert props.building_height == 20.5
            assert props.building_id == "test_001"
            assert props.neighbor_count == 3
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_roundtrip_dict(self):
        """Test dictionary serialization roundtrip."""
        original = BuildingProperties(
            unprojected_area=150.0, projected_area=145.0, building_height=20.5, building_id="test_001", neighbor_count=5
        )
        original.update_material_percentages({"brick": 60.0, "glass": 40.0})

        # Roundtrip through dict
        data = original.to_dict()
        restored = BuildingProperties.from_dict(data)

        assert restored.unprojected_area == original.unprojected_area
        assert restored.projected_area == original.projected_area
        assert restored.building_height == original.building_height
        assert restored.building_id == original.building_id
        assert restored.neighbor_count == original.neighbor_count
        assert restored.material_percentages == original.material_percentages

    def test_roundtrip_json_file(self):
        """Test JSON file serialization roundtrip."""
        original = BuildingProperties(unprojected_area=150.0, building_height=20.5, building_id="test_001")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            # Save and load
            original.to_json(filepath)
            restored = BuildingProperties.from_json(filepath)

            assert restored.unprojected_area == original.unprojected_area
            assert restored.building_height == original.building_height
            assert restored.building_id == original.building_id
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestFeatureVector:
    """Test feature vector extraction for ML."""

    def test_get_feature_vector_basic(self):
        """Test getting feature vector without materials."""
        props = BuildingProperties(unprojected_area=150.0, projected_area=145.0, building_height=20.5, neighbor_count=5)

        vector = props.get_feature_vector(exclude_materials=True)

        assert isinstance(vector, np.ndarray)
        assert vector.dtype in [np.float64, np.float32, np.int64, np.int32]
        # 6 geometric + 5 engineered + 10 contextual + 1 height + 5 color + 5 shape + 6 facade = 38
        assert len(vector) == 38

        # Check specific values at correct indices
        assert vector[0] == 150.0  # unprojected_area (index 0)
        assert vector[1] == 145.0  # projected_area (index 1)
        assert vector[21] == 20.5  # building_height (index 21)
        assert vector[11] == 5  # neighbor_count (index 11)

    def test_get_feature_vector_with_materials(self):
        """Test getting feature vector with materials."""
        props = BuildingProperties(unprojected_area=150.0, building_height=20.5)
        props.update_material_percentages({"brick": 45.0, "concrete": 30.0, "glass": 25.0})

        vector = props.get_feature_vector(exclude_materials=False)

        # 38 base features + 3 materials = 41
        assert len(vector) == 41

        # Materials should be sorted alphabetically after building_height (index 21)
        # Index 22: brick, 23: concrete, 24: glass
        assert vector[22] == 45.0  # brick
        assert vector[23] == 30.0  # concrete
        assert vector[24] == 25.0  # glass

    def test_get_feature_vector_order_verification(self):
        """Test that feature vector maintains correct order."""
        props = BuildingProperties()

        # Set specific values to verify ordering
        props.unprojected_area = 1.0
        props.projected_area = 2.0
        props.longitude_difference = 3.0
        props.latitude_difference = 4.0
        props.n_vertices = 5
        props.shape_length = 6.0

        props.complexity = 7.0
        props.inverse_average_segment_length = 8.0
        props.vertices_per_area = 9.0
        props.average_complexity_per_segment = 10.0
        props.isoperimetric_quotient = 11.0

        props.neighbor_count = 12

        vector = props.get_feature_vector(exclude_materials=True)

        assert vector[0] == 1.0
        assert vector[1] == 2.0
        assert vector[2] == 3.0
        assert vector[3] == 4.0
        assert vector[4] == 5
        assert vector[5] == 6.0
        assert vector[6] == 7.0
        assert vector[7] == 8.0
        assert vector[8] == 9.0
        assert vector[9] == 10.0
        assert vector[10] == 11.0
        assert vector[11] == 12

    def test_get_feature_names_basic(self):
        """Test getting feature names without materials."""
        props = BuildingProperties()

        names = props.get_feature_names(exclude_materials=True)

        assert isinstance(names, list)
        assert len(names) == 38
        assert names[0] == "unprojected_area"
        assert names[1] == "projected_area"
        assert names[21] == "building_height"
        assert names[11] == "neighbor_count"
        assert "average_brightness" in names
        assert "number_of_windows" in names

    def test_get_feature_names_with_materials(self):
        """Test getting feature names with materials."""
        props = BuildingProperties()
        props.update_material_percentages({"brick": 45.0, "concrete": 30.0, "glass": 25.0})

        names = props.get_feature_names(exclude_materials=False)

        assert len(names) == 41
        # Check material names appear sorted
        assert "material_brick" in names
        assert "material_concrete" in names
        assert "material_glass" in names

        # Find positions - should be after building_height
        brick_idx = names.index("material_brick")
        concrete_idx = names.index("material_concrete")
        glass_idx = names.index("material_glass")

        assert brick_idx == 22
        assert concrete_idx == 23
        assert glass_idx == 24

    def test_feature_vector_and_names_alignment(self):
        """Test that feature vector and names are properly aligned."""
        props = BuildingProperties(
            unprojected_area=150.0,
            projected_area=145.0,
            building_height=20.5,
            neighbor_count=5,
            average_brightness=128.0,
        )
        props.update_material_percentages({"brick": 60.0, "glass": 40.0})

        vector = props.get_feature_vector(exclude_materials=False)
        names = props.get_feature_names(exclude_materials=False)

        assert len(vector) == len(names)

        # Verify specific alignments
        area_idx = names.index("unprojected_area")
        assert vector[area_idx] == 150.0

        height_idx = names.index("building_height")
        assert vector[height_idx] == 20.5

        brightness_idx = names.index("average_brightness")
        assert vector[brightness_idx] == 128.0

        brick_idx = names.index("material_brick")
        assert vector[brick_idx] == 60.0

        glass_idx = names.index("material_glass")
        assert vector[glass_idx] == 40.0

    def test_feature_vector_all_zeros_by_default(self):
        """Test that default feature vector is mostly zeros/defaults."""
        props = BuildingProperties()
        vector = props.get_feature_vector(exclude_materials=True)

        # Most values should be 0, except building_height which is -1
        zero_count = np.sum(vector == 0.0)
        assert zero_count >= 35  # At least most are zero
        assert vector[21] == -1.0  # building_height default


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_material_percentages(self):
        """Test with no materials."""
        props = BuildingProperties()

        vector = props.get_feature_vector(exclude_materials=False)
        names = props.get_feature_names(exclude_materials=False)

        # Should still work, just without material features
        assert len(vector) == 38
        assert len(names) == 38

    def test_negative_height(self):
        """Test that failed height estimation results in -1.0."""
        props = BuildingProperties()
        props.update_height(None)

        assert props.building_height == -1.0

        vector = props.get_feature_vector()
        height_idx = 21  # Position of building_height
        assert vector[height_idx] == -1.0

    def test_large_material_dict(self):
        """Test with many material types."""
        props = BuildingProperties()
        materials = {f"material_{i}": float(i) for i in range(20)}
        props.update_material_percentages(materials)

        vector = props.get_feature_vector(exclude_materials=False)
        names = props.get_feature_names(exclude_materials=False)

        assert len(vector) == 38 + 20
        assert len(names) == 38 + 20

    def test_type_conversions_footprint(self):
        """Test that footprint update handles type conversions."""
        props = BuildingProperties()

        # Pass integers where floats expected
        footprint_data = {
            "unprojected_area": 150,  # int
            "complexity": 1,  # int
            "n_vertices": 8.5,  # float where int expected
        }

        props.update_footprint_features(footprint_data)

        assert props.unprojected_area == 150.0
        assert props.complexity == 1.0
        assert props.n_vertices == 8

    def test_type_conversions_image(self):
        """Test that image update handles type conversions."""
        props = BuildingProperties()

        image_data = {
            "average_brightness": 128,  # int
            "mask_area": 15000.5,  # float where int expected
            "number_of_windows": 5.9,  # float where int expected
        }

        props.update_image_features(image_data)

        assert props.average_brightness == 128.0
        assert props.mask_area == 15000
        assert props.number_of_windows == 5

    def test_repr(self):
        """Test string representation."""
        props = BuildingProperties(building_id="test_001", projected_area=145.5, building_height=20.3)

        repr_str = repr(props)

        assert "BuildingFeatures" in repr_str
        assert "test_001" in repr_str
        assert "145" in repr_str  # Check number appears
        assert "20" in repr_str  # Check number appears

    def test_zero_values(self):
        """Test that zero values are handled correctly."""
        props = BuildingProperties()
        props.update_footprint_features({"unprojected_area": 0.0})
        props.update_height(0.0)

        assert props.unprojected_area == 0.0
        assert props.building_height == 0.0

    def test_negative_values_allowed(self):
        """Test that negative values can be set (except height with None)."""
        props = BuildingProperties()
        props.update_footprint_features({"longitude_difference": -0.5, "latitude_difference": -0.3})

        assert props.longitude_difference == -0.5
        assert props.latitude_difference == -0.3


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_feature_pipeline(self):
        """Test complete feature extraction pipeline."""
        props = BuildingProperties(building_id="integration_001")

        # Update all feature types
        props.update_footprint_features(
            {"unprojected_area": 150.0, "projected_area": 145.0, "complexity": 0.75, "neighbor_count": 5, "nni": 0.95}
        )

        props.update_height(22.5)

        props.update_material_percentages({"brick": 50.0, "glass": 30.0, "concrete": 20.0})

        props.update_image_features({"average_brightness": 128.0, "number_of_windows": 8, "mask_area": 15000})

        # Get feature vector
        vector = props.get_feature_vector()
        names = props.get_feature_names()

        assert len(vector) == len(names)
        assert props.building_id == "integration_001"

        # Serialize and deserialize
        data = props.to_dict()
        restored = BuildingProperties.from_dict(data)

        restored_vector = restored.get_feature_vector()
        np.testing.assert_array_equal(vector, restored_vector)

    def test_batch_processing_scenario(self):
        """Test scenario for batch processing multiple buildings."""
        buildings = []

        for i in range(5):
            props = BuildingProperties(building_id=f"building_{i:03d}")
            props.update_footprint_features({"unprojected_area": 100.0 + i * 10, "neighbor_count": i + 1})
            props.update_height(15.0 + i * 2)
            buildings.append(props)

        # Get feature matrix
        feature_matrix = np.array([b.get_feature_vector() for b in buildings])

        assert feature_matrix.shape[0] == 5
        assert feature_matrix.shape[1] == 38  # No materials

        # Verify values increase
        assert feature_matrix[0, 0] < feature_matrix[4, 0]  # unprojected_area increases
        assert feature_matrix[0, 21] < feature_matrix[4, 21]  # height increases

    def test_multiple_updates(self):
        """Test multiple successive updates to same features."""
        props = BuildingProperties()

        # First update
        props.update_footprint_features({"unprojected_area": 100.0})
        assert props.unprojected_area == 100.0

        # Second update overwrites
        props.update_footprint_features({"unprojected_area": 150.0})
        assert props.unprojected_area == 150.0

        # Same with height
        props.update_height(10.0)
        assert props.building_height == 10.0

        props.update_height(20.0)
        assert props.building_height == 20.0

        # Materials get replaced
        props.update_material_percentages({"brick": 100.0})
        assert props.material_percentages == {"brick": 100.0}

        props.update_material_percentages({"glass": 100.0})
        assert props.material_percentages == {"glass": 100.0}
        assert "brick" not in props.material_percentages

    def test_feature_vector_changes_with_updates(self):
        """Test that feature vector reflects updates."""
        props = BuildingProperties()

        vector1 = props.get_feature_vector(exclude_materials=True)

        # Update some features
        props.update_footprint_features({"unprojected_area": 100.0})
        props.update_height(15.0)

        vector2 = props.get_feature_vector(exclude_materials=True)

        # Vectors should be different
        assert not np.array_equal(vector1, vector2)

        # Check specific differences
        assert vector2[0] == 100.0  # unprojected_area changed
        assert vector2[21] == 15.0  # height changed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
