"""
Smoke tests for the imageable package.

These tests verify basic import and instantiation works.
"""

import pytest


class TestPackageImport:
    """Tests for basic package import functionality."""

    def test_import_main_package(self):
        """Test that main package can be imported."""
        import imageable
        assert imageable is not None

    def test_version_exists(self):
        """Test that version is accessible."""
        import imageable
        assert hasattr(imageable, "__version__")
        assert isinstance(imageable.__version__, str)
        assert len(imageable.__version__) > 0

    def test_version_info_exists(self):
        """Test that version_info tuple is accessible."""
        import imageable
        assert hasattr(imageable, "__version_info__")
        assert isinstance(imageable.__version_info__, tuple)

    def test_all_public_apis_importable(self):
        """Test that all public API functions can be imported."""
        from imageable import (
            get_building_data_from_gdf,
            get_building_data_from_geojson,
            get_building_data_from_file,
            get_dataset,
            get_image,
            BuildingProperties,
        )

        assert callable(get_building_data_from_gdf)
        assert callable(get_building_data_from_geojson)
        assert callable(get_building_data_from_file)
        assert callable(get_dataset)
        assert callable(get_image)

    def test_building_properties_instantiation(self):
        """Test that BuildingProperties can be instantiated."""
        from imageable import BuildingProperties

        props = BuildingProperties()
        assert props is not None
        assert props.building_id is None
        assert props.building_height == -1.0
        assert props.projected_area == 0.0

    def test_building_properties_with_values(self):
        """Test BuildingProperties with custom values."""
        from imageable import BuildingProperties

        props = BuildingProperties(
            building_id="test_001",
            building_height=25.0,
            projected_area=150.0,
        )

        assert props.building_id == "test_001"
        assert props.building_height == 25.0
        assert props.projected_area == 150.0

    def test_building_properties_to_dict(self):
        """Test BuildingProperties can be converted to dict."""
        from imageable import BuildingProperties

        props = BuildingProperties(building_id="test", projected_area=100.0)
        data = props.to_dict()

        assert isinstance(data, dict)
        assert data["building_id"] == "test"
        assert data["projected_area"] == 100.0


class TestSubmoduleImports:
    """Tests for submodule imports."""

    def test_core_module_import(self):
        """Test that core module can be imported."""
        from imageable import core
        assert hasattr(core, "get_dataset")
        assert hasattr(core, "get_image")

    def test_core_building_data_import(self):
        """Test that core.building_data can be imported."""
        from imageable.core import building_data
        assert hasattr(building_data, "get_building_data_from_gdf")

    def test_core_dataset_import(self):
        """Test that core.dataset can be imported."""
        from imageable.core import dataset
        assert hasattr(dataset, "get_dataset")

    def test_core_image_import(self):
        """Test that core.image can be imported."""
        from imageable.core import image
        assert hasattr(image, "get_image")


class TestInternalModuleStructure:
    """Tests for internal module structure (not public API)."""

    def test_extraction_building_module_exists(self):
        """Test that _extraction.building module exists."""
        from imageable._extraction import building
        assert building is not None
        assert hasattr(building, "BuildingProperties")

    def test_extraction_footprint_module_exists(self):
        """Test that _extraction.footprint module exists."""
        from imageable._extraction import footprint
        assert footprint is not None
        assert hasattr(footprint, "FootprintCalculator")

    def test_camera_parameters_module_exists(self):
        """Test that camera parameters module exists."""
        from imageable._images.camera import camera_parameters
        assert camera_parameters is not None
        assert hasattr(camera_parameters, "CameraParameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
