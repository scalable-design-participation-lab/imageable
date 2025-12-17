"""
Tests for FootprintCalculator and footprint property extraction.
"""

import pytest
import numpy as np
from shapely.geometry import Polygon

from imageable._extraction.footprint import FootprintCalculator, extract_footprint_properties, find_neighboring_buildings


# ==================== FIXTURES ====================


@pytest.fixture
def simple_square():
    """Simple 10x10 square polygon in WGS84."""
    return Polygon([(0.0, 0.0), (0.0001, 0.0), (0.0001, 0.0001), (0.0, 0.0001), (0.0, 0.0)])


@pytest.fixture
def simple_rectangle():
    """Simple rectangle polygon in WGS84."""
    return Polygon([(0.0, 0.0), (0.0002, 0.0), (0.0002, 0.0001), (0.0, 0.0001), (0.0, 0.0)])


@pytest.fixture
def complex_polygon():
    """Complex polygon with 8 vertices."""
    return Polygon(
        [
            (0.0, 0.0),
            (0.0001, 0.0),
            (0.00015, 0.00005),
            (0.0002, 0.0001),
            (0.00015, 0.00015),
            (0.0001, 0.0002),
            (0.00005, 0.00015),
            (0.0, 0.0001),
            (0.0, 0.0),
        ]
    )


@pytest.fixture
def triangle():
    """Simple triangle polygon."""
    return Polygon([(0.0, 0.0), (0.0001, 0.0), (0.00005, 0.0001), (0.0, 0.0)])


@pytest.fixture
def building_cluster():
    """Cluster of building polygons for contextual testing."""
    buildings = [
        # Center building
        Polygon([(0.0, 0.0), (0.0001, 0.0), (0.0001, 0.0001), (0.0, 0.0001)]),
        # Neighbor 1 - close by
        Polygon([(0.0002, 0.0), (0.0003, 0.0), (0.0003, 0.0001), (0.0002, 0.0001)]),
        # Neighbor 2 - close by
        Polygon([(0.0, 0.0002), (0.0001, 0.0002), (0.0001, 0.0003), (0.0, 0.0003)]),
        # Neighbor 3 - diagonal
        Polygon([(0.0002, 0.0002), (0.0003, 0.0002), (0.0003, 0.0003), (0.0002, 0.0003)]),
        # Far building - outside 600m radius
        Polygon([(0.01, 0.01), (0.011, 0.01), (0.011, 0.011), (0.01, 0.011)]),
    ]
    return buildings


# ==================== GEOMETRICAL PROPERTIES TESTS ====================


class TestGeometricalProperties:
    """Test basic geometrical property calculations."""

    def test_calculate_unprojected_area_square(self, simple_square):
        """Test unprojected area calculation for square."""
        area = FootprintCalculator.calculate_unprojected_area(simple_square)

        assert area > 0
        assert isinstance(area, float)
        # Should be approximately 0.0001 * 0.0001 = 1e-8
        assert np.isclose(area, 1e-8, rtol=0.1)

    def test_calculate_unprojected_area_rectangle(self, simple_rectangle):
        """Test unprojected area calculation for rectangle."""
        area = FootprintCalculator.calculate_unprojected_area(simple_rectangle)

        assert area > 0
        # Rectangle should be 2x the square's area
        assert area > 1e-8

    def test_calculate_projected_area_square(self, simple_square):
        """Test projected area calculation in meters."""
        area = FootprintCalculator.calculate_projected_area(simple_square)

        assert area > 0
        assert isinstance(area, float)
        # Should be much larger in square meters
        assert area > 100  # Expect something in the range of hundreds of m²

    def test_calculate_projected_area_already_projected(self):
        """Test projected area when already in EPSG:3857."""
        # Create polygon in meters (projected coordinates)
        polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

        area = FootprintCalculator.calculate_projected_area(polygon, original_crs=3857)

        # Should be exactly 10000 m²
        assert np.isclose(area, 10000.0, rtol=0.01)

    def test_longitude_difference_square(self, simple_square):
        """Test longitude difference calculation."""
        lon_diff = FootprintCalculator.longitude_difference(simple_square)

        assert lon_diff == 0.0001

    def test_longitude_difference_rectangle(self, simple_rectangle):
        """Test longitude difference for rectangle."""
        lon_diff = FootprintCalculator.longitude_difference(simple_rectangle)

        assert lon_diff == 0.0002

    def test_latitude_difference_square(self, simple_square):
        """Test latitude difference calculation."""
        lat_diff = FootprintCalculator.latitude_difference(simple_square)

        assert lat_diff == 0.0001

    def test_latitude_difference_rectangle(self, simple_rectangle):
        """Test latitude difference for rectangle."""
        lat_diff = FootprintCalculator.latitude_difference(simple_rectangle)

        assert lat_diff == 0.0001

    def test_n_vertices_square(self, simple_square):
        """Test vertex count for square."""
        n = FootprintCalculator.n_vertices(simple_square)

        assert n == 4

    def test_n_vertices_complex(self, complex_polygon):
        """Test vertex count for complex polygon."""
        n = FootprintCalculator.n_vertices(complex_polygon)

        assert n == 8

    def test_n_vertices_triangle(self, triangle):
        """Test vertex count for triangle."""
        n = FootprintCalculator.n_vertices(triangle)

        assert n == 3

    def test_shape_length_square(self, simple_square):
        """Test perimeter calculation for square."""
        length = FootprintCalculator.shape_length(simple_square)

        assert length > 0
        # Perimeter should be 4 * side_length
        expected = 4 * 0.0001
        assert np.isclose(length, expected, rtol=0.01)

    def test_shape_length_rectangle(self, simple_rectangle):
        """Test perimeter calculation for rectangle."""
        length = FootprintCalculator.shape_length(simple_rectangle)

        # Perimeter should be 2 * (0.0002 + 0.0001)
        expected = 2 * (0.0002 + 0.0001)
        assert np.isclose(length, expected, rtol=0.01)


# ==================== ENGINEERED PROPERTIES TESTS ====================


class TestEngineeredProperties:
    """Test engineered property calculations."""

    def test_complexity_square(self, simple_square):
        """Test complexity calculation for square."""
        comp = FootprintCalculator.complexity(simple_square)

        assert comp > 0
        assert isinstance(comp, float)
        # Complexity = perimeter / area
        # For square: 4*side / side² = 4/side

    def test_complexity_zero_area(self):
        """Test complexity handles zero area."""
        # Degenerate polygon with no area
        polygon = Polygon([(0.0, 0.0), (0.0001, 0.0), (0.0, 0.0)])

        comp = FootprintCalculator.complexity(polygon)

        assert comp == 0.0

    def test_inverse_average_segment_length_square(self, simple_square):
        """Test inverse average segment length for square."""
        iasl = FootprintCalculator.inverse_average_segment_length(simple_square)

        assert iasl > 0
        # All sides equal, so inverse should be 1/side_length
        expected = 1 / 0.0001
        assert np.isclose(iasl, expected, rtol=0.01)

    def test_inverse_average_segment_length_zero(self):
        """Test inverse average segment length handles zero."""
        # Very small polygon
        polygon = Polygon([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])

        iasl = FootprintCalculator.inverse_average_segment_length(polygon)

        assert iasl == 0.0

    def test_vertices_per_area_square(self, simple_square):
        """Test vertices per area for square."""
        vpa = FootprintCalculator.vertices_per_area(simple_square)

        assert vpa > 0
        # 4 vertices / area

    def test_vertices_per_area_zero_area(self):
        """Test vertices per area handles zero area."""
        polygon = Polygon([(0.0, 0.0), (0.0001, 0.0), (0.0, 0.0)])

        vpa = FootprintCalculator.vertices_per_area(polygon)

        assert vpa == 0.0

    def test_average_complexity_per_segment_square(self, simple_square):
        """Test average complexity per segment for square."""
        acps = FootprintCalculator.average_complexity_per_segment(simple_square)

        assert acps > 0
        assert isinstance(acps, float)

    def test_average_complexity_per_segment_zero_area(self):
        """Test average complexity per segment handles zero area."""
        polygon = Polygon([(0.0, 0.0), (0.0001, 0.0), (0.0, 0.0)])

        acps = FootprintCalculator.average_complexity_per_segment(polygon)

        assert acps == 0.0

    def test_isoperimetric_quotient_square(self, simple_square):
        """Test isoperimetric quotient for square."""
        iq = FootprintCalculator.isoperimetric_quotient(simple_square)

        assert iq > 0
        assert iq <= 1.0  # IQ is always <= 1, with 1 being a perfect circle
        # For square, IQ = π/4 ≈ 0.785
        assert np.isclose(iq, np.pi / 4, rtol=0.1)

    def test_isoperimetric_quotient_zero_perimeter(self):
        """Test isoperimetric quotient handles zero perimeter."""
        polygon = Polygon([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])

        iq = FootprintCalculator.isoperimetric_quotient(polygon)

        assert iq == 0.0


# ==================== CONTEXTUAL PROPERTIES TESTS ====================


class TestContextualProperties:
    """Test contextual property calculations."""

    def test_get_footprints_within_radius_basic(self, building_cluster):
        """Test finding footprints within radius."""
        center = building_cluster[0]
        all_buildings = building_cluster

        projected_center, neighbors = FootprintCalculator.get_footprints_within_radius(
            all_buildings, center, radius=600
        )

        assert projected_center is not None
        assert isinstance(neighbors, list)
        # Should find 3 neighbors (excluding center and far building)
        assert len(neighbors) == 3

    def test_get_footprints_within_radius_small_radius(self, building_cluster):
        """Test with small radius finds fewer neighbors."""
        center = building_cluster[0]

        _, neighbors = FootprintCalculator.get_footprints_within_radius(
            building_cluster,
            center,
            radius=10,  # Very small radius
        )

        # Should find very few or no neighbors
        assert len(neighbors) <= 3

    def test_get_footprints_within_radius_large_radius(self, building_cluster):
        """Test with large radius finds all neighbors."""
        center = building_cluster[0]

        _, neighbors = FootprintCalculator.get_footprints_within_radius(
            building_cluster,
            center,
            radius=2000,  # Large radius
        )

        # Should find all other buildings
        assert len(neighbors) == len(building_cluster) - 1

    def test_neighbor_count(self, building_cluster):
        """Test neighbor count calculation."""
        center = building_cluster[0]

        count = FootprintCalculator.neighbor_count(building_cluster, center, radius=600)

        assert count >= 0
        assert isinstance(count, int)
        assert count == 3  # Should find 3 neighbors

    def test_neighbor_count_no_neighbors(self, simple_square):
        """Test neighbor count with no neighbors."""
        count = FootprintCalculator.neighbor_count([simple_square], simple_square, radius=600)

        assert count == 0

    def test_mean_distance_to_neighbors(self, building_cluster):
        """Test mean distance to neighbors calculation."""
        center = building_cluster[0]

        mean_dist = FootprintCalculator.mean_distance_to_neighbors(building_cluster, center, radius=600)

        assert mean_dist > 0
        assert isinstance(mean_dist, float)

    def test_mean_distance_to_neighbors_no_neighbors(self, simple_square):
        """Test mean distance with no neighbors."""
        mean_dist = FootprintCalculator.mean_distance_to_neighbors([simple_square], simple_square, radius=600)

        assert mean_dist == 0.0

    def test_expected_nearest_neighbor_distance(self, building_cluster):
        """Test expected nearest neighbor distance calculation."""
        center = building_cluster[0]

        expected_dist = FootprintCalculator.expected_nearest_neighbor_distance(building_cluster, center, radius=600)

        assert expected_dist > 0
        assert isinstance(expected_dist, float)
        assert expected_dist != float("inf")

    def test_expected_nearest_neighbor_distance_no_neighbors(self, simple_square):
        """Test expected distance with no neighbors."""
        expected_dist = FootprintCalculator.expected_nearest_neighbor_distance(
            [simple_square], simple_square, radius=600
        )

        assert expected_dist == float("inf")

    def test_nearest_neighbor_distance(self, building_cluster):
        """Test nearest neighbor distance calculation."""
        center = building_cluster[0]

        nnd = FootprintCalculator.nearest_neighbor_distance(building_cluster, center, radius=600)

        assert nnd > 0
        assert isinstance(nnd, float)
        assert nnd != float("inf")

    def test_nearest_neighbor_distance_no_neighbors(self, simple_square):
        """Test nearest neighbor distance with no neighbors."""
        nnd = FootprintCalculator.nearest_neighbor_distance([simple_square], simple_square, radius=600)

        assert nnd == float("inf")

    def test_n_size_mean(self, building_cluster):
        """Test mean neighborhood size calculation."""
        center = building_cluster[0]

        mean_size = FootprintCalculator.n_size_mean(building_cluster, center, radius=600)

        assert mean_size > 0
        assert isinstance(mean_size, float)

    def test_n_size_std(self, building_cluster):
        """Test standard deviation of neighborhood sizes."""
        center = building_cluster[0]

        std_size = FootprintCalculator.n_size_std(building_cluster, center, radius=600)

        assert std_size >= 0
        assert isinstance(std_size, float)

    def test_n_size_min(self, building_cluster):
        """Test minimum neighborhood size."""
        center = building_cluster[0]

        min_size = FootprintCalculator.n_size_min(building_cluster, center, radius=600)

        assert min_size > 0
        assert isinstance(min_size, float)

    def test_n_size_max(self, building_cluster):
        """Test maximum neighborhood size."""
        center = building_cluster[0]

        max_size = FootprintCalculator.n_size_max(building_cluster, center, radius=600)

        assert max_size > 0
        assert isinstance(max_size, float)

    def test_n_size_cv(self, building_cluster):
        """Test coefficient of variation for neighborhood sizes."""
        center = building_cluster[0]

        cv = FootprintCalculator.n_size_cv(building_cluster, center, radius=600)

        assert cv >= 0
        assert isinstance(cv, float)

    def test_n_size_cv_zero_mean(self):
        """Test CV handles zero mean correctly."""
        # This is an edge case that shouldn't happen in practice
        # but good to test the zero division handling
        polygon = Polygon([(0.0, 0.0), (0.0001, 0.0), (0.0, 0.0)])

        # With no valid neighbors, mean should be 0
        cv = FootprintCalculator.n_size_cv([polygon], polygon, radius=600)

        assert cv >= 0

    def test_nni(self, building_cluster):
        """Test nearest neighbor index calculation."""
        center = building_cluster[0]

        nni = FootprintCalculator.nni(building_cluster, center, radius=600)

        assert nni > 0
        assert isinstance(nni, float)
        assert nni != float("inf")

    def test_nni_no_neighbors(self, simple_square):
        """Test NNI with no neighbors."""
        nni = FootprintCalculator.nni([simple_square], simple_square, radius=600)

        assert nni == float("inf")


# ==================== CONVENIENCE FUNCTIONS TESTS ====================


class TestConvenienceFunctions:
    """Test convenience functions for property extraction."""

    def test_extract_footprint_properties_without_neighbors(self, simple_square):
        """Test extracting all properties without neighbor data."""
        props = extract_footprint_properties(simple_square)

        # Check all geometrical properties present
        assert "unprojected_area" in props
        assert "projected_area" in props
        assert "longitude_difference" in props
        assert "latitude_difference" in props
        assert "n_vertices" in props
        assert "shape_length" in props

        # Check all engineered properties present
        assert "complexity" in props
        assert "inverse_average_segment_length" in props
        assert "vertices_per_area" in props
        assert "average_complexity_per_segment" in props
        assert "isoperimetric_quotient" in props

        # Check all contextual properties present (with defaults)
        assert "neighbor_count" in props
        assert "mean_distance_to_neighbors" in props
        assert "expected_nearest_neighbor_distance" in props
        assert "nearest_neighbor_distance" in props
        assert "n_size_mean" in props
        assert "n_size_std" in props
        assert "n_size_min" in props
        assert "n_size_max" in props
        assert "n_size_cv" in props
        assert "nni" in props

        # Contextual properties should be defaults
        assert props["neighbor_count"] == 0
        assert props["mean_distance_to_neighbors"] == 0.0
        assert props["nni"] == 0.0

    def test_extract_footprint_properties_with_neighbors(self, building_cluster):
        """Test extracting all properties with neighbor data."""
        center = building_cluster[0]

        props = extract_footprint_properties(center, all_footprints=building_cluster)

        # Check geometrical properties have values
        assert props["unprojected_area"] > 0
        assert props["projected_area"] > 0
        assert props["n_vertices"] == 4

        # Check engineered properties have values
        assert props["complexity"] > 0
        assert props["isoperimetric_quotient"] > 0

        # Check contextual properties were calculated
        assert props["neighbor_count"] > 0
        assert props["mean_distance_to_neighbors"] > 0
        assert props["nearest_neighbor_distance"] > 0
        assert props["n_size_mean"] > 0

    def test_extract_footprint_properties_custom_crs(self, simple_square):
        """Test extraction with custom CRS."""
        props = extract_footprint_properties(simple_square, crs=4326)

        assert "unprojected_area" in props
        assert props["unprojected_area"] > 0

    def test_extract_footprint_properties_custom_radius(self, building_cluster):
        """Test extraction with custom neighbor radius."""
        center = building_cluster[0]

        # Small radius
        props_small = extract_footprint_properties(center, all_footprints=building_cluster, neighbor_radius=100)

        # Large radius
        props_large = extract_footprint_properties(center, all_footprints=building_cluster, neighbor_radius=2000)

        # Large radius should find more neighbors
        assert props_large["neighbor_count"] >= props_small["neighbor_count"]

    def test_extract_footprint_properties_dict_keys(self, simple_square):
        """Test that all expected keys are present in output."""
        props = extract_footprint_properties(simple_square)

        expected_keys = {
            # Geometrical
            "unprojected_area",
            "projected_area",
            "longitude_difference",
            "latitude_difference",
            "n_vertices",
            "shape_length",
            # Engineered
            "complexity",
            "inverse_average_segment_length",
            "vertices_per_area",
            "average_complexity_per_segment",
            "isoperimetric_quotient",
            # Contextual
            "neighbor_count",
            "mean_distance_to_neighbors",
            "expected_nearest_neighbor_distance",
            "nearest_neighbor_distance",
            "n_size_mean",
            "n_size_std",
            "n_size_min",
            "n_size_max",
            "n_size_cv",
            "nni",
        }

        assert set(props.keys()) == expected_keys

    def test_find_neighboring_buildings(self, building_cluster):
        """Test find_neighboring_buildings convenience function."""
        center = building_cluster[0]

        neighbors = find_neighboring_buildings(center, building_cluster, max_distance=600)

        assert isinstance(neighbors, list)
        assert len(neighbors) > 0
        assert center not in neighbors  # Center should be excluded

    def test_find_neighboring_buildings_no_neighbors(self, simple_square):
        """Test finding neighbors when none exist."""
        neighbors = find_neighboring_buildings(simple_square, [simple_square], max_distance=600)

        assert isinstance(neighbors, list)
        assert len(neighbors) == 0

    def test_find_neighboring_buildings_custom_distance(self, building_cluster):
        """Test finding neighbors with custom distance."""
        center = building_cluster[0]

        # Small distance
        neighbors_small = find_neighboring_buildings(center, building_cluster, max_distance=100)

        # Large distance
        neighbors_large = find_neighboring_buildings(center, building_cluster, max_distance=2000)

        # Large distance should find more or equal neighbors
        assert len(neighbors_large) >= len(neighbors_small)


# ==================== EDGE CASES AND INTEGRATION TESTS ====================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_point_polygon(self):
        """Test handling of degenerate single-point polygon."""
        polygon = Polygon([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])

        # Should not crash, return sensible defaults
        props = extract_footprint_properties(polygon)

        assert props["unprojected_area"] == 0.0 or props["unprojected_area"] < 1e-10
        assert props["n_vertices"] >= 0

    def test_very_small_polygon(self):
        """Test handling of very small polygon."""
        polygon = Polygon([(0.0, 0.0), (0.000001, 0.0), (0.000001, 0.000001), (0.0, 0.000001)])

        props = extract_footprint_properties(polygon)

        assert props["unprojected_area"] > 0
        assert props["projected_area"] > 0

    def test_very_large_polygon(self):
        """Test handling of very large polygon."""
        polygon = Polygon([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)])

        props = extract_footprint_properties(polygon)

        assert props["unprojected_area"] > 0
        assert props["projected_area"] > 0

    def test_irregular_polygon(self, complex_polygon):
        """Test handling of irregular polygon."""
        props = extract_footprint_properties(complex_polygon)

        assert props["n_vertices"] == 8
        assert props["unprojected_area"] > 0
        assert props["complexity"] > 0

    def test_all_properties_are_numeric(self, simple_square, building_cluster):
        """Test that all properties are numeric types."""
        props = extract_footprint_properties(simple_square, all_footprints=building_cluster)

        for key, value in props.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric: {type(value)}"

    def test_no_nan_or_none_values(self, simple_square, building_cluster):
        """Test that no NaN or None values are returned."""
        props = extract_footprint_properties(simple_square, all_footprints=building_cluster)

        for key, value in props.items():
            assert value is not None, f"{key} is None"
            if isinstance(value, float):
                assert not np.isnan(value), f"{key} is NaN"


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_pipeline(self, building_cluster):
        """Test complete feature extraction pipeline."""
        results = []

        for building in building_cluster[:3]:  # Test first 3 buildings
            props = extract_footprint_properties(building, all_footprints=building_cluster, neighbor_radius=600)
            results.append(props)

        # All should have same keys
        assert len(results) == 3
        keys_0 = set(results[0].keys())
        assert all(set(r.keys()) == keys_0 for r in results)

        # All should have valid values
        for props in results:
            assert props["unprojected_area"] > 0
            assert props["projected_area"] > 0

    def test_multiple_polygons_consistency(self):
        """Test that same polygon gives consistent results."""
        polygon = Polygon([(0.0, 0.0), (0.0001, 0.0), (0.0001, 0.0001), (0.0, 0.0001)])

        props1 = extract_footprint_properties(polygon)
        props2 = extract_footprint_properties(polygon)

        # Results should be identical
        for key in props1.keys():
            assert props1[key] == props2[key], f"Inconsistent values for {key}"

    def test_different_radius_values(self, building_cluster):
        """Test behavior with different radius values."""
        center = building_cluster[0]

        radii = [100, 300, 600, 1000]
        results = []

        for radius in radii:
            props = extract_footprint_properties(center, all_footprints=building_cluster, neighbor_radius=radius)
            results.append(props["neighbor_count"])

        # Neighbor count should generally increase or stay same with radius
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_crs_consistency(self, simple_square):
        """Test that different CRS values work correctly."""
        props_4326 = extract_footprint_properties(simple_square, crs=4326)
        props_default = extract_footprint_properties(simple_square)

        # Default is 4326, so should be identical
        assert props_4326["unprojected_area"] == props_default["unprojected_area"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
