"""
Footprint property extraction from building polygons.

This module contains the FootprintCalculator class with all geometric, engineered,
and contextual property calculations, plus convenience functions for easy extraction.
"""

from typing import Any

import geopandas as gpd
import numpy as np
import pyproj
from shapely import Polygon
from shapely.ops import transform
from shapely.strtree import STRtree


class FootprintCalculator:
    """
    Calculator for extracting all footprint-based properties from building polygons.

    All methods are static for easy standalone use. Use extract_all_properties()
    for efficient bulk extraction.
    """

    UNPROJECTED_CRS = 4326
    PROJECTED_CRS = 3857  # Mercator

    # ==================== GEOMETRICAL PROPERTIES ====================

    @staticmethod
    def calculate_unprojected_area(footprint: Polygon, original_crs: int = 4326) -> float:
        """
        Calculate the unprojected area of a footprint.

        Parameters
        ----------
        footprint
            The polygon footprint as a shapely Polygon object.
        original_crs
            Original CRS. Default is 4326 (WGS84).

        Returns
        -------
        area
            The unprojected area of the footprint.
        """
        if original_crs == FootprintCalculator.UNPROJECTED_CRS:
            return footprint.area
        start_crs = pyproj.CRS(original_crs)
        end_crs = pyproj.CRS(FootprintCalculator.UNPROJECTED_CRS)
        project = pyproj.Transformer.from_crs(start_crs, end_crs, always_xy=True).transform
        reprojected_footprint = transform(project, footprint)
        return reprojected_footprint.area

    @staticmethod
    def calculate_projected_area(footprint: Polygon, original_crs: int = 4326) -> float:
        """
        Calculate the projected area of a footprint in square meters.

        Parameters
        ----------
        footprint
            The polygon footprint as a shapely Polygon object.
        original_crs
            Original CRS. Default is 4326 (WGS84).

        Returns
        -------
        area
            The projected area in square meters.
        """
        if original_crs == FootprintCalculator.PROJECTED_CRS:
            return footprint.area
        start_crs = pyproj.CRS(original_crs)
        end_crs = pyproj.CRS(FootprintCalculator.PROJECTED_CRS)
        project = pyproj.Transformer.from_crs(start_crs, end_crs, always_xy=True).transform
        reprojected_footprint = transform(project, footprint)
        return reprojected_footprint.area

    @staticmethod
    def longitude_difference(footprint: Polygon) -> float:
        """
        Calculate the difference between the maximum and minimum longitude.
        Assumes the polygon is in EPSG:4326.

        Parameters
        ----------
        footprint
            The polygon footprint as a shapely Polygon object.

        Returns
        -------
        diff_lon
            The difference between max_longitude and min_longitude.
        """
        poly_coords = list(footprint.exterior.coords)
        longitude_values = [x[0] for x in poly_coords]
        max_longitude = max(longitude_values)
        min_longitude = min(longitude_values)
        return max_longitude - min_longitude

    @staticmethod
    def latitude_difference(footprint: Polygon) -> float:
        """
        Calculate the difference between max_latitude and min_latitude.
        Assumes that the polygon is in EPSG:4326.

        Parameters
        ----------
        footprint
            Shapely polygon of the building footprint.

        Returns
        -------
        latitude_diff
            Difference between max_lat and min_lat.
        """
        poly_coords = list(footprint.exterior.coords)
        latitude_values = [x[1] for x in poly_coords]
        max_lat = max(latitude_values)
        min_lat = min(latitude_values)
        return max_lat - min_lat

    @staticmethod
    def n_vertices(footprint: Polygon) -> int:
        """
        Calculate the number of vertices in a polygon footprint.

        Parameters
        ----------
        footprint
            The polygon footprint as a shapely Polygon object.

        Returns
        -------
        n_vertices
            The number of vertices in the polygon footprint.
        """
        poly_coords = list(footprint.exterior.coords)
        return len(poly_coords) - 1  # Exclude the repeated last vertex

    @staticmethod
    def shape_length(footprint: Polygon) -> float:
        """
        Calculate the perimeter length of the polygon footprint.

        Parameters
        ----------
        footprint
            Polygon of the building's footprint.

        Returns
        -------
        shape_length
            Perimeter of the building's polygon.
        """
        poly_coords = list(footprint.exterior.coords)
        shape_length = 0
        for i in range(len(poly_coords) - 1):
            current_vertex = poly_coords[i]
            next_vertex = poly_coords[i + 1]
            dst = np.sqrt((next_vertex[0] - current_vertex[0]) ** 2 + (next_vertex[1] - current_vertex[1]) ** 2)
            shape_length += dst
        return shape_length

    # ==================== ENGINEERED PROPERTIES ====================

    @staticmethod
    def complexity(footprint: Polygon, crs: int = 4326) -> float:
        """
        Calculate the footprint's complexity defined as perimeter/area.

        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint.
        crs
            Optional CRS. Default is 4326 (longitude, latitude).

        Returns
        -------
        complexity
            Complexity of the building's footprint.
        """
        area = FootprintCalculator.calculate_unprojected_area(footprint, original_crs=crs)
        length = FootprintCalculator.shape_length(footprint)
        return length / area if area > 0 else 0.0

    @staticmethod
    def inverse_average_segment_length(footprint: Polygon) -> float:
        """
        Calculate the inverse average segment length.

        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint.

        Returns
        -------
        iasl
            Inverse average segment length.
        """
        coordinates = list(footprint.exterior.coords)
        segment_lengths = [
            np.sqrt((coordinates[i + 1][0] - coordinates[i][0]) ** 2 + (coordinates[i + 1][1] - coordinates[i][1]) ** 2)
            for i in range(len(coordinates) - 1)
        ]
        avg = np.mean(segment_lengths)
        return 1.0 / avg if avg != 0 else 0.0

    @staticmethod
    def vertices_per_area(footprint: Polygon, crs: int = 4326) -> float:
        """
        Calculate vertices per area.

        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint.
        crs
            Original CRS. Default is 4326.

        Returns
        -------
        vpa
            Number of vertices per unprojected area.
        """
        n_vertices = FootprintCalculator.n_vertices(footprint)
        area = FootprintCalculator.calculate_unprojected_area(footprint, original_crs=crs)
        return n_vertices / area if area > 0 else 0.0

    @staticmethod
    def average_complexity_per_segment(footprint: Polygon, crs: int = 4326) -> float:
        """
        Obtain the average complexity per segment.
        Defined as (1/(A*n_vertices))*sum(Li) where Li is the length of the i-th segment.

        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint.
        crs
            Original CRS. Default is 4326.

        Returns
        -------
        acps
            Average complexity per segment.
        """
        coordinates = list(footprint.exterior.coords)
        segment_lengths = [
            np.sqrt((coordinates[i + 1][0] - coordinates[i][0]) ** 2 + (coordinates[i + 1][1] - coordinates[i][1]) ** 2)
            for i in range(len(coordinates) - 1)
        ]
        area = FootprintCalculator.calculate_unprojected_area(footprint, original_crs=crs)
        n_segments = len(segment_lengths)

        if area == 0 or n_segments == 0:
            return 0.0

        return sum(segment_lengths) / (area * n_segments)

    @staticmethod
    def isoperimetric_quotient(footprint: Polygon, crs: int = 4326) -> float:
        """
        Obtain the isoperimetric quotient of a polygon's footprint.
        Defined as 4*pi*Area/(Perimeter^2).

        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint.
        crs
            Original CRS. Default is 4326.

        Returns
        -------
        iq
            Isoperimetric quotient of the polygon footprint.
        """
        area = FootprintCalculator.calculate_unprojected_area(footprint, original_crs=crs)
        perimeter = FootprintCalculator.shape_length(footprint)

        if perimeter == 0:
            return 0.0

        return (4 * np.pi * area) / (perimeter**2)

    # ==================== CONTEXTUAL PROPERTIES ====================

    @staticmethod
    def get_footprints_within_radius(
        footprints: list[Polygon],
        center_footprint: Polygon,
        radius: float = 600,  # meters
    ) -> tuple[Polygon, list[Polygon]]:
        """
        Filter a list of footprints to obtain those within a given radius (in meters).
        Assumes that the footprints are in EPSG:4326.

        Parameters
        ----------
        footprints
            List of shapely polygons to filter.
        center_footprint
            Footprint at the center of the search area.
        radius
            Radius in meters to search for nearby footprints. Default is 600 meters.

        Returns
        -------
        projected_centered_footprint
            Center footprint projected to Mercator projection.
        buffered_footprints
            List of footprints intersecting the search area.
        """
        # Convert to Mercator projection using geopandas
        gdf = gpd.GeoDataFrame(geometry=footprints, crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)

        projected_footprints = gdf["geometry"].tolist()

        # Create spatial tree for efficient querying
        tree = STRtree(projected_footprints)

        # Project the center footprint
        gdf_center = gpd.GeoDataFrame(geometry=[center_footprint], crs="EPSG:4326")
        gdf_center = gdf_center.to_crs(epsg=3857)
        projected_centered_footprint = gdf_center["geometry"].tolist()[0]

        # Buffer and query
        buffer = projected_centered_footprint.buffer(radius)
        candidates_idx = tree.query(buffer)
        buffered_footprints = [
            projected_footprints[i] for i in candidates_idx if projected_footprints[i].intersects(buffer)
        ]

        # Remove the center footprint if present
        if projected_centered_footprint in buffered_footprints:
            buffered_footprints.remove(projected_centered_footprint)

        return projected_centered_footprint, buffered_footprints

    @staticmethod
<<<<<<< HEAD
    def get_footprint_indices_within_radius(
        footprints: list[Polygon],
        center_footprint: Polygon,
        radius: float = 600,  # meters
    ) -> list[int]:
        """
        Filter a list of footprints to obtain those within a given radius (in meters).
        Assumes that the footprints are in EPSG:4326.

        Parameters
        ----------
        footprints
            List of shapely polygons to filter.
        center_footprint
            Footprint at the center of the search area.
        radius
            Radius in meters to search for nearby footprints. Default is 600 meters.

        Returns
        -------
        projected_centered_footprint
            Center footprint projected to Mercator projection.
        buffered_footprints
            List of footprints intersecting the search area.
        """
        # Convert to Mercator projection using geopandas
        gdf = gpd.GeoDataFrame(geometry=footprints, crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)

        projected_footprints = gdf["geometry"].tolist()

        # Create spatial tree for efficient querying
        tree = STRtree(projected_footprints)

        # Project the center footprint
        gdf_center = gpd.GeoDataFrame(geometry=[center_footprint], crs="EPSG:4326")
        gdf_center = gdf_center.to_crs(epsg=3857)
        projected_centered_footprint = gdf_center["geometry"].tolist()[0]

        # Buffer and query
        buffer = projected_centered_footprint.buffer(radius)
        candidates_idx = tree.query(buffer)
        indices = [i for i in candidates_idx if projected_footprints[i].intersects(buffer)]
        footprints = [projected_footprints[i] for i in candidates_idx if projected_footprints[i].intersects(buffer)]

        if projected_centered_footprint in footprints:
            idx = footprints.index(projected_centered_footprint)
            indices.pop(idx)

        return indices

    @staticmethod
    def neighbor_count(footprints: list[Polygon], center_footprint: Polygon, radius: float = 600) -> int:
        """
        Compute the number of neighbors of a footprint within a given radius.

        Parameters
        ----------
        footprints
            Complete list of footprints to search for neighbors.
        center_footprint
            Footprint at the center of the search area.
        radius
            Search radius in meters. Default is 600 meters.

        Returns
        -------
        n_neighbors
            Number of neighboring footprints found within the search area.
        """
        _, filtered_footprints = FootprintCalculator.get_footprints_within_radius(footprints, center_footprint, radius)
        return len(filtered_footprints)

    @staticmethod
    def mean_distance_to_neighbors(footprints: list[Polygon], center_polygon: Polygon, radius: float = 600) -> float:
        """
        Compute the mean distance from a footprint to its neighbors.
        The vicinity is defined by a buffer radius in meters.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        center_polygon
            Footprint at the center of the neighborhood area.
        radius
            Radius that defines the neighborhood area (in meters). Default is 600 meters.

        Returns
        -------
        mdn
            Mean distance to neighbors.
        """
        projected_footprint, neighbors = FootprintCalculator.get_footprints_within_radius(
            footprints, center_polygon, radius
        )

        if len(neighbors) == 0:
            return 0.0

        centroid_center = projected_footprint.centroid
        neighbor_centroids = [n.centroid for n in neighbors]

        distances = [
            np.sqrt((nc.x - centroid_center.x) ** 2 + (nc.y - centroid_center.y) ** 2) for nc in neighbor_centroids
        ]

        return np.mean(distances)

    @staticmethod
    def expected_nearest_neighbor_distance(
        footprints: list[Polygon], center_footprint: Polygon, radius: float = 600
    ) -> float:
        """
        Calculate expected nearest neighbor distance for random distribution.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        center_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.

        Returns
        -------
        expected_distance
            Expected nearest neighbor distance.
        """
        n = FootprintCalculator.neighbor_count(footprints, center_footprint, radius)
        area = np.pi * (radius**2)
        density = n / area

        if density > 0:
            return 1 / (2 * np.sqrt(density))
        return float("inf")

    @staticmethod
    def nearest_neighbor_distance(footprints: list[Polygon], center_footprint: Polygon, radius: float = 600) -> float:
        """
        Compute the distance from a building's polygon to its nearest neighbor.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        center_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.

        Returns
        -------
        nnd
            Distance to nearest neighbor (in meters).
        """
        projected_polygon, neighbors = FootprintCalculator.get_footprints_within_radius(
            footprints, center_footprint, radius
        )

        if len(neighbors) == 0:
            return float("inf")

        polygon_coords = projected_polygon.centroid
        nnd = float("inf")

        for neighbor in neighbors:
            neighbor_coords = neighbor.centroid
            dst = np.sqrt((neighbor_coords.x - polygon_coords.x) ** 2 + (neighbor_coords.y - polygon_coords.y) ** 2)
            if dst > 0 and dst < nnd:
                nnd = dst

        return nnd

    @staticmethod
    def n_size_mean(footprints: list[Polygon], central_footprint: Polygon, radius: float = 600) -> float:
        """
        Obtain the mean footprint area in a neighborhood around the building.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.

        Returns
        -------
        n_size_mean
            Mean area of buildings within the neighborhood.
        """
        projected_footprint, neighbors = FootprintCalculator.get_footprints_within_radius(
            footprints, central_footprint, radius
        )

        # Include the center footprint itself
        all_footprints = neighbors + [projected_footprint]
        areas = [fp.area for fp in all_footprints]
        return np.mean(areas)

    @staticmethod
    def n_size_std(footprints: list[Polygon], central_footprint: Polygon, radius: float = 600) -> float:
        """
        Obtain the standard deviation of footprint areas in a neighborhood.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.

        Returns
        -------
        n_size_std
            Standard deviation of footprint areas within the neighborhood.
        """
        projected_footprint, neighbors = FootprintCalculator.get_footprints_within_radius(
            footprints, central_footprint, radius
        )

        all_footprints = neighbors + [projected_footprint]
        areas = [fp.area for fp in all_footprints]
        return np.std(areas)

    @staticmethod
    def n_size_min(footprints: list[Polygon], central_footprint: Polygon, radius: float = 600) -> float:
        """
        Obtain the minimum footprint area in a neighborhood.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.

        Returns
        -------
        n_size_min
            Minimum footprint area within the neighborhood.
        """
        projected_footprint, neighbors = FootprintCalculator.get_footprints_within_radius(
            footprints, central_footprint, radius
        )

        all_footprints = neighbors + [projected_footprint]
        areas = [fp.area for fp in all_footprints]
        return np.min(areas)

    @staticmethod
    def n_size_max(footprints: list[Polygon], central_footprint: Polygon, radius: float = 600) -> float:
        """
        Obtain the maximum footprint area in a neighborhood.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.

        Returns
        -------
        n_size_max
            Maximum footprint area within the neighborhood.
        """
        projected_footprint, neighbors = FootprintCalculator.get_footprints_within_radius(
            footprints, central_footprint, radius
        )

        all_footprints = neighbors + [projected_footprint]
        areas = [fp.area for fp in all_footprints]
        return np.max(areas)

    @staticmethod
    def n_size_cv(footprints: list[Polygon], central_footprint: Polygon, radius: float = 600) -> float:
        """
        Obtain the coefficient of variation of footprint areas in a neighborhood.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.

        Returns
        -------
        n_size_cv
            Coefficient of variation of footprint areas within the neighborhood.
        """
        projected_footprint, neighbors = FootprintCalculator.get_footprints_within_radius(
            footprints, central_footprint, radius
        )

        all_footprints = neighbors + [projected_footprint]
        areas = [fp.area for fp in all_footprints]
        mean_area = np.mean(areas)

        if mean_area == 0:
            return 0.0

        return np.std(areas) / mean_area

    @staticmethod
    def nni(footprints: list[Polygon], center_building: Polygon, radius: float = 600) -> float:
        """
        Compute the nearest neighbor index (NNI) for a building's footprint.
        The NNI is defined as the ratio of the observed nearest neighbor distance
        to the expected nearest neighbor distance in a random distribution.

        Parameters
        ----------
        footprints
            Complete list of footprints.
        center_building
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood. Default is 600 meters.

        Returns
        -------
        nni
            Nearest neighbor index.
        """
        ond = FootprintCalculator.nearest_neighbor_distance(footprints, center_building, radius)

        end = FootprintCalculator.expected_nearest_neighbor_distance(footprints, center_building, radius)

        if end > 0 and end != float("inf"):
            return ond / end
        return float("inf")


# ==================== CONVENIENCE FUNCTIONS ====================


def extract_footprint_properties(
    polygon: Polygon, all_footprints: list[Polygon] | None = None, crs: int = 4326, neighbor_radius: float = 600
) -> dict[str, Any]:
    """
    Extract ALL footprint properties from a polygon in one efficient call.

    This is the main function you should use. It extracts geometric, engineered,
    and contextual properties efficiently by reusing intermediate calculations.

    Parameters
    ----------
    polygon
        Building footprint polygon.
    all_footprints
        List of ALL building polygons (for contextual features).
        If None, contextual features will be set to defaults.
    crs
        Coordinate reference system. Default is 4326 (WGS84).
    neighbor_radius
        Radius in meters for neighbor detection. Default is 600.

    Returns
    -------
    properties
        Dictionary containing all footprint properties:
        - 6 geometrical properties
        - 5 engineered properties
        - 10 contextual properties (if all_footprints provided)

    Example
    -------
    >>> polygon = Polygon([(-73.98, 40.74), (-73.97, 40.74), ...])
    >>> props = extract_footprint_properties(polygon)
    >>> print(props['projected_area'])  # 2841.32
    >>> print(props['complexity'])      # 0.0432
    """
    calc = FootprintCalculator

    # ========== Geometrical Properties ==========
    properties = {
        "unprojected_area": calc.calculate_unprojected_area(polygon, crs),
        "projected_area": calc.calculate_projected_area(polygon, crs),
        "longitude_difference": calc.longitude_difference(polygon),
        "latitude_difference": calc.latitude_difference(polygon),
        "n_vertices": calc.n_vertices(polygon),
        "shape_length": calc.shape_length(polygon),
    }

    # ========== Engineered Properties ==========
    properties.update(
        {
            "complexity": calc.complexity(polygon, crs),
            "inverse_average_segment_length": calc.inverse_average_segment_length(polygon),
            "vertices_per_area": calc.vertices_per_area(polygon, crs),
            "average_complexity_per_segment": calc.average_complexity_per_segment(polygon, crs),
            "isoperimetric_quotient": calc.isoperimetric_quotient(polygon, crs),
        }
    )

    # ========== Contextual Properties ==========
    if all_footprints is not None:
        # Optimize: get neighbors once, reuse for all calculations
        projected_center, neighbors = calc.get_footprints_within_radius(all_footprints, polygon, neighbor_radius)

        # Calculate all contextual features efficiently
        n_count = len(neighbors)

        if n_count > 0:
            # Get areas once
            neighbor_areas = [n.area for n in neighbors]

            # Get centroids once
            center_centroid = projected_center.centroid
            neighbor_centroids = [n.centroid for n in neighbors]

            # Calculate distances once
            distances = [
                np.sqrt((nc.x - center_centroid.x) ** 2 + (nc.y - center_centroid.y) ** 2) for nc in neighbor_centroids
            ]

            # All contextual properties from cached calculations
            properties.update(
                {
                    "neighbor_count": n_count,
                    "mean_distance_to_neighbors": float(np.mean(distances)),
                    "nearest_neighbor_distance": float(np.min(distances)),
                    "n_size_mean": float(np.mean(neighbor_areas + [projected_center.area])),
                    "n_size_std": float(np.std(neighbor_areas + [projected_center.area])),
                    "n_size_min": float(np.min(neighbor_areas + [projected_center.area])),
                    "n_size_max": float(np.max(neighbor_areas + [projected_center.area])),
                }
            )

            # Coefficient of variation
            mean_area = properties["n_size_mean"]
            properties["n_size_cv"] = properties["n_size_std"] / mean_area if mean_area > 0 else 0.0

            # Expected nearest neighbor distance
            area = np.pi * (neighbor_radius**2)
            density = n_count / area
            expected_nnd = 1 / (2 * np.sqrt(density)) if density > 0 else float("inf")
            properties["expected_nearest_neighbor_distance"] = float(expected_nnd)

            # NNI
            observed_nnd = properties["nearest_neighbor_distance"]
            properties["nni"] = (
                observed_nnd / expected_nnd if expected_nnd > 0 and expected_nnd != float("inf") else float("inf")
            )
        else:
            # No neighbors - set contextual features to defaults
            properties.update(
                {
                    "neighbor_count": 0,
                    "mean_distance_to_neighbors": 0.0,
                    "expected_nearest_neighbor_distance": 0.0,
                    "nearest_neighbor_distance": 0.0,
                    "n_size_mean": 0.0,
                    "n_size_std": 0.0,
                    "n_size_min": 0.0,
                    "n_size_max": 0.0,
                    "n_size_cv": 0.0,
                    "nni": 0.0,
                }
            )
    else:
        # No footprints provided - set contextual features to defaults
        properties.update(
            {
                "neighbor_count": 0,
                "mean_distance_to_neighbors": 0.0,
                "expected_nearest_neighbor_distance": 0.0,
                "nearest_neighbor_distance": 0.0,
                "n_size_mean": 0.0,
                "n_size_std": 0.0,
                "n_size_min": 0.0,
                "n_size_max": 0.0,
                "n_size_cv": 0.0,
                "nni": 0.0,
            }
        )

    return properties


def find_neighboring_buildings(
    target_polygon: Polygon, all_polygons: list[Polygon], max_distance: float = 600.0
) -> list[Polygon]:
    """
    Find neighboring buildings within a certain distance.

    This is a convenience wrapper around get_footprints_within_radius.

    Parameters
    ----------
    target_polygon
        The building to find neighbors for.
    all_polygons
        List of all building polygons in the area.
    max_distance
        Maximum distance to search (in meters). Default is 600.

    Returns
    -------
    neighbors
        List of neighboring polygons (excluding target).
    """
    _, neighbors = FootprintCalculator.get_footprints_within_radius(all_polygons, target_polygon, max_distance)
    return neighbors
<<<<<<< HEAD
=======

>>>>>>> d6a39c2 (test: building + footprint)
