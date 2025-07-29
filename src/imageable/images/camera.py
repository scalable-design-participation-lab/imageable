from dataclasses import asdict, dataclass
from shapely import Polygon
from typing import Tuple
import osmnx as ox
import numpy as np
from imageable.utils.geometry.polygons import get_polygon_edge_midpoints, get_polygon_outward_vectors
from imageable.utils.geometry.ray_geometry import get_closest_ray_intersection
from imageable.utils.geometry.point_geometry import get_heading_between_points_euclidean
from imageable.utils.geometry.point_geometry import get_euclidean_distance_meters
MAX_DIMENSION = 640
MIN_FOV = 10
MAX_FOV = 120


@dataclass
class CameraParameters:
    """
    Parameters used for building image collection with.

    Parameters
    ----------
    longitude
        Longitude where the image will be taken.
    latitude
        Latitude where the image will be taken.
    fov
        Field of view in degrees. The maximum fov allowed is 120. Default is 90.
    heading
        Heading angle in degrees. Default is 0
    pitch
        Pitch (vertical angle) in degrees. Default is 0.
    width
        Output image width. Must not exceed 640. Default is 640.
    height
        Output image height. Must not exceed 640. Default is 640.
    """

    # Longitude
    longitude: float
    # Latitude
    latitude: float
    # Field of view
    fov: float = 90
    # camera heading
    heading: float = 0
    # pitch
    pitch: float = 0
    # img width
    width: int = 640
    # img height
    height: int = 640

    def __post_init__(self) -> None:
        """Validate parameter ranges after initialization."""
        if self.width > MAX_DIMENSION:
            msg = "width cannot be greater than 640"
            raise ValueError(msg)

        if self.height > MAX_DIMENSION:
            msg = "height cannot be greater than 640"
            raise ValueError(msg)

        if self.fov > MAX_FOV or self.fov < MIN_FOV:
            msg = "FOV should be between 10 and 120"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, float | int | str]:
        """
        Convert camera parameters to a dictionary.

        Returns
        -------
        dict
            Dictionary containing camera parameters.
        """
        return asdict(self)

class CameraParametersEstimator:

    def __init__(self, polygon:Polygon)->None:
        self.polygon = polygon
        self.image_width = 640
        self.image_height = 640
    
    def estimate_first_parameters(
        self, 
        buffer_constant: float = 2.5e5):

        observation_point_estimator = ObservationPointEstimator(self.polygon)
        observation_point, midpoint, heading, distance = observation_point_estimator.get_observation_point(
            buffer_constant=buffer_constant,
            true_north=True
        )

        camera_parameters = CameraParameters(
            longitude=observation_point[0],
            latitude=observation_point[1],
            heading=heading,
            pitch=0,
            fov=90,
            width=640,
            height=640
        )

        return camera_parameters


class ObservationPointEstimator:
    """
    Class for estimating an observation point towards the
    building polygon. This is used to estimate initial camera
    parameters for the building view.
    """

    def __init__(self, polygon:Polygon)->None:
        """
        Teh polygon to analyze. 

        Parameters
        ----------
        polygon
            A shapely Polygon object 
        """
        self.polygon: Polygon = polygon
    
    def get_observation_point(
        self,
        buffer_constant: float = 2.5e5,
        true_north: bool = True)-> Tuple[Tuple[float, float], Tuple[float, float], float, float]:
        """
        gets the closest point at which a face of the building can be observed
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

        #Get the midpoints of the polygon faces
        midpoints = get_polygon_edge_midpoints(self.polygon)
        #Get the surrounding street network
        street_network = self._get_surrounding_street_network()
        
        #get outward vectors
        outward_vectors = get_polygon_outward_vectors(self.polygon)
        intersection_points = []
        min_distances = []
        headings = []
        for i in range(0,len(midpoints)):
            midpoint = midpoints[i]
            outward_vector = outward_vectors[i]
            #Get the closest intersection point
            intersection, distance = get_closest_ray_intersection(
                start_point=midpoint,
                ray_direction=outward_vector,
                boundaries=street_network,
                max_ray_length=500
            )
            intersection_points.append(intersection)
            min_distances.append(distance)
        
    
        min_idx = int(np.argmin(min_distances))
        best_midpoint = midpoints[min_idx]
        best_intersection = intersection_points[min_idx]

        if best_intersection is None:
            return None, None, None, np.inf

        best_distance = get_euclidean_distance_meters(
            best_midpoint, 
            best_intersection
        )

        heading = get_heading_between_points_euclidean(best_intersection, best_midpoint, true_north=true_north)

        return best_intersection, best_midpoint, heading, best_distance
        
    def _get_surrounding_street_network(
        self,
        buffer_constant = 2.5e5,
    ):
        """
        Gets the surrounding street network of the polygon using a buffer.

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
        #We will estimate a buffer using the polygon area
        area = self.polygon.area
        buffer_distance = area * buffer_constant
        #Get the surrounding street 
        buffered_polygon = self.polygon.buffer(buffer_distance)
        if not buffered_polygon.is_valid:
            buffered_polygon = buffered_polygon.buffer(0)
        G = ox.graph_from_polygon(buffered_polygon, network_type='all',simplify = True)
        #let's get the GDF
        gdf_surrounding_network = ox.graph_to_gdfs(G, nodes=False, edges=True)

        return gdf_surrounding_network
    




    
