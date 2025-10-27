import logging

import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon

from imageable.utils.geometry.point_geometry import get_euclidean_distance_meters, get_heading_between_points_euclidean
from imageable.utils.geometry.polygons import (
    get_polygon_edge_midpoints,
    get_polygon_outward_vectors,
)
from imageable.utils.geometry.ray_geometry import get_closest_ray_intersection


class ObservationPointEstimator:
    """
    Class for estimating an observation point towards the
    building polygon. This is used to estimate initial camera
    parameters for the building view.
    """

    def __init__(self, polygon: Polygon) -> None:
        """
        Initialize the ObservationPointEstimator.

        Parameters
        ----------
        polygon
            A shapely Polygon object.
        """
        self.polygon: Polygon = polygon

    def get_observation_point(
        self, buffer_constant: float = 2.5e5, true_north: bool = True
    ) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """
        Get the closest point at which a face of the building can be observed
        as orthogonal as possible.

        Parameters
        ----------
        buffer_constant
            A constant to estimate the buffer size based on the polygon area.
            Default is 2.5e5.

        Returns
        -------
        observation_point
            (lon,lat) of the observation point.
        heading
            Heading angle.
        distance_to_building
            Distance to the building in meters.
        """
        # Create a copy of the polygon
        polygon_copy = self.polygon

        self.polygon = polygon_copy

        # Get the midpoints of the polygon faces
        midpoints = get_polygon_edge_midpoints(self.polygon)
        # Get the surrounding street network
        street_network = self._get_surrounding_street_network(buffer_constant=buffer_constant)
        if street_network is None:
            return None, None, None, np.inf

        # get outward vectors
        outward_vectors = get_polygon_outward_vectors(self.polygon)
        intersection_points = []
        min_distances = []
        for i in range(len(midpoints)):
            midpoint = midpoints[i]
            outward_vector = outward_vectors[i]
            # Get the closest intersection point
            intersection, distance = get_closest_ray_intersection(
                start_point=midpoint, ray_direction=outward_vector, boundaries=street_network, max_ray_length=500
            )
            intersection_points.append(intersection)
            min_distances.append(distance)

        min_idx = int(np.argmin(min_distances))
        best_midpoint = midpoints[min_idx]
        best_intersection = intersection_points[min_idx]

        if best_intersection is None:
            return None, None, None, np.inf

        best_distance = get_euclidean_distance_meters(best_midpoint, best_intersection)

        heading = get_heading_between_points_euclidean(best_intersection, best_midpoint, true_north=true_north)

        return best_intersection, best_midpoint, heading, best_distance

    def _get_surrounding_street_network(
        self,
        buffer_constant: float = 2.5e5,
    ) -> gpd.GeoDataFrame:
        """
        Get the surrounding street network of the polygon using a buffer.

        Parameters
        ----------
        buffer_constant
            A constant to estimate the buffer size around the building
            based on the polygon area.

        Returns
        -------
        gdf_surrounding_network
            A GeoDataFrame containing the street network within the buffer.
        """
        # We will estimate a buffer using the polygon area
        area = self.polygon.area
        buffer_distance = area * buffer_constant
        # Get the surrounding street
        buffered_polygon = self.polygon.buffer(buffer_distance)
        if not buffered_polygon.is_valid:
            buffered_polygon = buffered_polygon.buffer(0)
        try:
            g = ox.graph_from_polygon(buffered_polygon, network_type="all", simplify=True)
            # let's get the GDF
            return ox.graph_to_gdfs(g, nodes=False, edges=True)

        except Exception:
            logging.exception("Error obtaining street network")
            return None
