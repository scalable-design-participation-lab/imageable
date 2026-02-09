"""Tests for imageable public API exports."""

import pytest


class TestPublicAPIExports:
    """Test that all public API items are properly exported."""

    def test_main_api_functions_exported(self):
        """Test that main API functions are accessible from imageable."""
        import imageable

        # Main batch processing API
        assert hasattr(imageable, "get_building_data_from_gdf")
        assert hasattr(imageable, "get_building_data_from_geojson")
        assert hasattr(imageable, "get_building_data_from_file")

        # Single building utilities
        assert hasattr(imageable, "get_dataset")
        assert hasattr(imageable, "get_image")

        # Data classes
        assert hasattr(imageable, "BuildingProperties")

        # Version
        assert hasattr(imageable, "__version__")

    def test_all_contains_expected_items(self):
        """Test that __all__ contains expected public API items."""
        import imageable

        expected = [
            "get_building_data_from_gdf",
            "get_building_data_from_geojson",
            "get_building_data_from_file",
            "get_dataset",
            "get_image",
            "BuildingProperties",
            "__version__",
        ]

        for item in expected:
            assert item in imageable.__all__, f"{item} not in __all__"

    def test_version_is_string(self):
        """Test that version is a valid string."""
        import imageable

        assert isinstance(imageable.__version__, str)
        assert len(imageable.__version__) > 0
        # Should follow semver pattern
        parts = imageable.__version__.split(".")
        assert len(parts) >= 2

    def test_version_info_is_tuple(self):
        """Test that __version_info__ is a tuple of integers."""
        import imageable

        assert hasattr(imageable, "__version_info__")
        assert isinstance(imageable.__version_info__, tuple)
        for part in imageable.__version_info__:
            assert isinstance(part, int)

    def test_building_properties_is_dataclass(self):
        """Test that BuildingProperties can be instantiated."""
        from imageable import BuildingProperties

        props = BuildingProperties()
        assert props is not None
        assert hasattr(props, "building_id")
        assert hasattr(props, "projected_area")
        assert hasattr(props, "building_height")

    def test_functions_are_callable(self):
        """Test that all exported functions are callable."""
        import imageable

        assert callable(imageable.get_building_data_from_gdf)
        assert callable(imageable.get_building_data_from_geojson)
        assert callable(imageable.get_building_data_from_file)
        assert callable(imageable.get_dataset)
        assert callable(imageable.get_image)


class TestImportPaths:
    """Test various import paths work correctly."""

    def test_direct_import(self):
        """Test direct import of main module."""
        import imageable

        assert imageable is not None

    def test_from_import_functions(self):
        """Test from-import of specific functions."""
        from imageable import (
            BuildingProperties,
            get_building_data_from_file,
            get_building_data_from_gdf,
            get_building_data_from_geojson,
            get_dataset,
            get_image,
        )

        assert callable(get_building_data_from_gdf)
        assert callable(get_dataset)
        assert callable(get_image)

    def test_submodule_import(self):
        """Test importing submodules directly."""
        from imageable.core import building_data, dataset, image

        assert hasattr(building_data, "get_building_data_from_gdf")
        assert hasattr(dataset, "get_dataset")
        assert hasattr(image, "get_image")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
