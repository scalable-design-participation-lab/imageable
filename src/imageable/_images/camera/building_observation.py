import logging

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon

from imageable._extraction.extract import extract_building_properties
from imageable._models.distance_to_streets_wrapper import DistanceRegressorWrapper
from imageable._utils.geometry.point_geometry import get_euclidean_distance_meters, get_heading_between_points_euclidean
from imageable._utils.geometry.polygons import (
    get_polygon_edge_midpoints,
    get_polygon_outward_vectors,
)
from imageable._utils.geometry.ray_geometry import get_closest_ray_intersection


class ObservationPointEstimator:
    """
    Class for estimating an observation point towards the
    building polygon. This is used to estimate initial camera
    parameters for the building view.
    """

    def __init__(
        self,
        polygon: Polygon,
        street_network: nx.MultiDiGraph | gpd.GeoDataFrame | None = None,
    ) -> None:
        """
        Initialize the ObservationPointEstimator.

        Parameters
        ----------
        polygon
            A shapely Polygon object.
        street_network
            Optional preloaded city-scale street network. If provided, this
            class will clip it locally using the same bbox geometry that
            OSMnx `graph_from_point(..., dist_type="bbox")` uses.
        """
        self.polygon: Polygon = polygon
        self.street_network: nx.MultiDiGraph | gpd.GeoDataFrame | None = street_network

    def get_observation_point(
        self, buffer_constant: float = 50, true_north: bool = True
    ) -> tuple[tuple[float, float] |None, tuple[float, float]|None, float|None, float]:
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
        buffer_constant: float = 20,
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
        building_id = "0"
        properties = extract_building_properties(
            building_id,
            self.polygon
        )
        feature_vector = [
            properties.projected_area,
            properties.shape_length,
            properties.n_vertices,
            properties.complexity,
            properties.unprojected_area,
            properties.vertices_per_area,
            properties.latitude_difference,
            properties.longitude_difference
        ]

        distance_wrapper = DistanceRegressorWrapper(
            input_dim=8,
            hidden_sizes=(128, 128),
            device=None,
            model_path=None,
            target_mode="log1p"
        )

        x_input = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        pred_meters = float(distance_wrapper.predict(x_input)[0])
        buffer = buffer_constant * pred_meters

        point = self.polygon.centroid
        if buffer > 10000:
            buffer = 600
        try:
            if self.street_network is not None:
                return self._clip_provided_street_network((point.y, point.x), buffer)

            g = ox.graph_from_point((point.y, point.x), dist=buffer, network_type="all", simplify=True)
            return ox.graph_to_gdfs(g, nodes=False, edges=True)

        except Exception:
            logging.exception("Error obtaining street network")
            return None

    def _clip_provided_street_network(
        self,
        center_point: tuple[float, float],
        dist_meters: float,
    ) -> gpd.GeoDataFrame | None:
        """Clip a preloaded street network to match OSMnx bbox-from-point behavior."""
        bbox = ox.utils_geo.bbox_from_point(center_point, dist_meters)
        bbox_polygon = ox.utils_geo.bbox_to_poly(bbox)

        if isinstance(self.street_network, nx.MultiDiGraph):
            clipped = ox.truncate.truncate_graph_polygon(self.street_network, bbox_polygon, truncate_by_edge=False)
            if len(clipped) == 0 or len(clipped.edges) == 0:
                return None

            clipped = ox.truncate.largest_component(clipped, strongly=False)
            if len(clipped) == 0 or len(clipped.edges) == 0:
                return None

            return ox.graph_to_gdfs(clipped, nodes=False, edges=True)

        if isinstance(self.street_network, gpd.GeoDataFrame):
            clip_polygon = bbox_polygon
            if self.street_network.crs is not None:
                clip_polygon = gpd.GeoSeries([bbox_polygon], crs="EPSG:4326").to_crs(self.street_network.crs).iloc[0]

            try:
                candidate_idxs = list(self.street_network.sindex.intersection(clip_polygon.bounds))
                candidates = self.street_network.iloc[candidate_idxs]
            except Exception:
                candidates = self.street_network
            clipped = candidates[candidates.intersects(clip_polygon)]
            if clipped.empty:
                return None
            return clipped.copy()

        msg = (
            "street_network must be a networkx.MultiDiGraph or "
            "a geopandas.GeoDataFrame."
        )
        raise TypeError(msg)


    def _get_surrounding_street_network_with_distance(
            self,
            buffer_distance: float  = 0.055
    ) -> gpd.GeoDataFrame:
        """
        Get the surrounding street network of the polygon using a fixed buffer distance.

        Parameters
        ----------
        buffer_distance
            A fixed distance to create a buffer around the building polygon.

        Returns
        -------
        gdf_surrounding_network
            A GeoDataFrame containing the street network within the buffer.
        """
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
