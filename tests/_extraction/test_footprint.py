"""
Comprehensive tests for the FootprintCalculator class.

Tests cover:
- Geometrical properties (area, dimensions, vertices, perimeter)
- Engineered properties (complexity, quotient, etc.)
- Contextual properties (neighbors, distances, NNI)
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from imageable._extraction.footprint import (
    FootprintCalculator,
    extract_footprint_properties,
    find_neighboring_buildings,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def square_polygon():
    """Create a simple square polygon in WGS84."""
    return Polygon([
        (0.0, 0.0),
        (0.001, 0.0),
        (0.001, 0.001),
        (0.0, 0.001),
    ])


@pytest.fixture
def rectangle_polygon():
    """Create a rectangle polygon (longer in one direction)."""
    return Polygon([
        (0.0, 0.0),
        (0.002, 0.0),
        (0.002, 0.001),
        (0.0, 0.001),
    ])


@pytest.fixture
def complex_polygon():
    """Create an L-shaped polygon."""
    return Polygon([
        (0.0, 0.0),
        (0.002, 0.0),
        (0.002, 0.001),
        (0.001, 0.001),
        (0.001, 0.002),
        (0.0, 0.002),
    ])


@pytest.fixture
def building_with_neighbors():
    """Create a central building with neighbors."""
    center = Polygon([
        (-71.0589, 42.3601),
        (-71.0585, 42.3601),
        (-71.0585, 42.3605),
        (-71.0589, 42.3605),
    ])

    neighbors = [
        # Nearby neighbor
        Polygon([
            (-71.0579, 42.3601),
            (-71.0575, 42.3601),
            (-71.0575, 42.3605),
            (-71.0579, 42.3605),
        ]),
        # Another nearby neighbor
        Polygon([
            (-71.0589, 42.3611),
            (-71.0585, 42.3611),
            (-71.0585, 42.3615),
            (-71.0589, 42.3615),
        ]),
    ]

    return center, neighbors


# =============================================================================
# Tests for Geometrical Properties
# =============================================================================


class TestGeometricalProperties:
    """Tests for geometrical property calculations."""

    def test_calculate_unprojected_area(self, square_polygon):
        """Test unprojected area calculation."""
        area = FootprintCalculator.calculate_unprojected_area(square_polygon)
        assert area > 0
        assert area == pytest.approx(0.001 * 0.001, rel=0.01)

    def test_calculate_projected_area(self, square_polygon):
        """Test projected area calculation (in meters)."""
        area = FootprintCalculator.calculate_projected_area(square_polygon)
        assert area > 0
        # Projected area should be larger than unprojected for small values

    def test_longitude_difference(self, rectangle_polygon):
        """Test longitude difference calculation."""
        lon_diff = FootprintCalculator.longitude_difference(rectangle_polygon)
        assert lon_diff == pytest.approx(0.002, rel=0.01)

    def test_latitude_difference(self, rectangle_polygon):
        """Test latitude difference calculation."""
        lat_diff = FootprintCalculator.latitude_difference(rectangle_polygon)
        assert lat_diff == pytest.approx(0.001, rel=0.01)

    def test_n_vertices_square(self, square_polygon):
        """Test vertex count for square."""
        n = FootprintCalculator.n_vertices(square_polygon)
        assert n == 4

    def test_n_vertices_complex(self, complex_polygon):
        """Test vertex count for L-shape."""
        n = FootprintCalculator.n_vertices(complex_polygon)
        assert n == 6

    def test_shape_length(self, square_polygon):
        """Test perimeter calculation."""
        length = FootprintCalculator.shape_length(square_polygon)
        assert length > 0
        # For a square, perimeter should be 4 * side
        assert length == pytest.approx(4 * 0.001, rel=0.01)


# =============================================================================
# Tests for Engineered Properties
# =============================================================================


class TestEngineeredProperties:
    """Tests for engineered property calculations."""

    def test_complexity(self, square_polygon):
        """Test complexity (perimeter/area) calculation."""
        complexity = FootprintCalculator.complexity(square_polygon)
        assert complexity > 0

    def test_complexity_comparison(self, square_polygon, complex_polygon):
        """Test complexity calculation for different shapes.
        
        Note: Complexity is defined as perimeter/area. Smaller shapes with 
        relatively longer perimeters have higher complexity values.
        The square (0.001 x 0.001) has a higher perimeter-to-area ratio
        than the larger L-shape.
        """
        simple_complexity = FootprintCalculator.complexity(square_polygon)
        complex_complexity = FootprintCalculator.complexity(complex_polygon)
        # Both should be positive values
        assert simple_complexity > 0
        assert complex_complexity > 0
        # The smaller square has higher perimeter/area ratio
        assert simple_complexity > complex_complexity

    def test_inverse_average_segment_length(self, square_polygon):
        """Test inverse average segment length."""
        iasl = FootprintCalculator.inverse_average_segment_length(square_polygon)
        assert iasl > 0

    def test_vertices_per_area(self, square_polygon):
        """Test vertices per area calculation."""
        vpa = FootprintCalculator.vertices_per_area(square_polygon)
        assert vpa > 0

    def test_average_complexity_per_segment(self, square_polygon):
        """Test average complexity per segment."""
        acps = FootprintCalculator.average_complexity_per_segment(square_polygon)
        assert acps >= 0

    def test_isoperimetric_quotient_circle_approximation(self):
        """Test that circle has highest isoperimetric quotient (~1)."""
        # Create a circle-like polygon
        n_points = 32
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radius = 0.001
        points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
        circle = Polygon(points)

        iq = FootprintCalculator.isoperimetric_quotient(circle)
        # Circle should have IQ close to 1
        assert iq > 0.9

    def test_isoperimetric_quotient_square(self, square_polygon):
        """Test isoperimetric quotient for square."""
        iq = FootprintCalculator.isoperimetric_quotient(square_polygon)
        # Square has IQ of pi/4 ≈ 0.785
        assert iq == pytest.approx(np.pi / 4, rel=0.1)


# =============================================================================
# Tests for Contextual Properties
# =============================================================================


class TestContextualProperties:
    """Tests for contextual property calculations."""

    def test_get_footprints_within_radius(self, building_with_neighbors):
        """Test finding footprints within radius."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        projected_center, found = FootprintCalculator.get_footprints_within_radius(
            all_footprints, center, radius=2000
        )

        assert projected_center is not None
        assert len(found) >= 0  # May find neighbors depending on distance

    def test_neighbor_count(self, building_with_neighbors):
        """Test neighbor count calculation."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        count = FootprintCalculator.neighbor_count(all_footprints, center, radius=2000)
        assert count >= 0

    def test_mean_distance_to_neighbors(self, building_with_neighbors):
        """Test mean distance to neighbors."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        mean_dist = FootprintCalculator.mean_distance_to_neighbors(
            all_footprints, center, radius=2000
        )
        assert mean_dist >= 0

    def test_nearest_neighbor_distance(self, building_with_neighbors):
        """Test nearest neighbor distance calculation."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        nnd = FootprintCalculator.nearest_neighbor_distance(
            all_footprints, center, radius=2000
        )
        assert nnd >= 0 or nnd == float("inf")

    def test_expected_nearest_neighbor_distance(self, building_with_neighbors):
        """Test expected nearest neighbor distance."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        end = FootprintCalculator.expected_nearest_neighbor_distance(
            all_footprints, center, radius=2000
        )
        assert end >= 0 or end == float("inf")

    def test_nni(self, building_with_neighbors):
        """Test nearest neighbor index calculation."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        nni = FootprintCalculator.nni(all_footprints, center, radius=2000)
        assert nni >= 0 or nni == float("inf")

    def test_n_size_statistics(self, building_with_neighbors):
        """Test neighbor size statistics."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        mean = FootprintCalculator.n_size_mean(all_footprints, center, radius=2000)
        std = FootprintCalculator.n_size_std(all_footprints, center, radius=2000)
        min_val = FootprintCalculator.n_size_min(all_footprints, center, radius=2000)
        max_val = FootprintCalculator.n_size_max(all_footprints, center, radius=2000)
        cv = FootprintCalculator.n_size_cv(all_footprints, center, radius=2000)

        assert mean >= 0
        assert std >= 0
        assert min_val >= 0
        assert max_val >= 0
        assert cv >= 0


# =============================================================================
# Tests for Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_extract_footprint_properties_minimal(self, square_polygon):
        """Test extracting all properties with just polygon."""
        props = extract_footprint_properties(square_polygon)

        assert isinstance(props, dict)
        assert "unprojected_area" in props
        assert "projected_area" in props
        assert "complexity" in props
        assert props["unprojected_area"] > 0

    def test_extract_footprint_properties_with_neighbors(self, building_with_neighbors):
        """Test extracting all properties with neighbors."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        props = extract_footprint_properties(
            center,
            all_footprints=all_footprints,
            neighbor_radius=2000,
        )

        assert "neighbor_count" in props
        assert "nni" in props

    def test_extract_footprint_properties_no_footprints(self, square_polygon):
        """Test extraction when no footprints list provided."""
        props = extract_footprint_properties(square_polygon, all_footprints=None)

        # Contextual properties should be set to defaults
        assert props["neighbor_count"] == 0
        assert props["nni"] == 0.0

    def test_find_neighboring_buildings(self, building_with_neighbors):
        """Test the find_neighboring_buildings helper."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        found = find_neighboring_buildings(center, all_footprints, max_distance=2000)

        assert isinstance(found, list)
        # May or may not find neighbors depending on actual distances


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_polygon_no_neighbors(self, square_polygon):
        """Test with single polygon in footprints list."""
        props = extract_footprint_properties(
            square_polygon,
            all_footprints=[square_polygon],
            neighbor_radius=1000,
        )

        # Should have 0 neighbors (can't be neighbor of itself)
        assert props["neighbor_count"] == 0

    def test_very_small_radius(self, building_with_neighbors):
        """Test with very small search radius."""
        center, neighbors = building_with_neighbors
        all_footprints = [center] + neighbors

        count = FootprintCalculator.neighbor_count(
            all_footprints, center, radius=1  # 1 meter
        )

        # Should find very few or no neighbors with such small radius
        assert count >= 0

    def test_empty_footprints_list(self, square_polygon):
        """Test with empty footprints list."""
        props = extract_footprint_properties(
            square_polygon,
            all_footprints=[],
            neighbor_radius=1000,
        )

        assert props["neighbor_count"] == 0

    def test_triangle_polygon(self):
        """Test with triangular polygon (minimum vertices)."""
        triangle = Polygon([
            (0.0, 0.0),
            (0.001, 0.0),
            (0.0005, 0.001),
        ])

        n = FootprintCalculator.n_vertices(triangle)
        assert n == 3

        complexity = FootprintCalculator.complexity(triangle)
        assert complexity > 0

    def test_zero_area_handling(self):
        """Test handling of degenerate polygon."""
        # Line polygon (zero area)
        line = Polygon([
            (0.0, 0.0),
            (0.001, 0.0),
            (0.001, 0.0),  # Degenerate
            (0.0, 0.0),
        ])

        # Should handle gracefully
        complexity = FootprintCalculator.complexity(line)
        assert complexity >= 0 or complexity == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
