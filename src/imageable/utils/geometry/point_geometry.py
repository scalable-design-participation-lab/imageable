from typing import Tuple
import math
from pyproj import Geod, Transformer
from typing import Tuple
import numpy as np
geod = Geod(ellps="WGS84")

def get_heading_between_points_euclidean(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
    true_north: bool = True
) -> float:
        """
        Computes the heading angle between two points in degrees.
        The angle is measured clockwise from the north direction.

        Parameters
        ----------
        point_1
            The origin point (x, y).
        point_2
            The target point (x, y).
        true_north 
            If True, heading is measured from north (0°); if False, from east (0°).
        
        Returns
        -------
        heading : float
            Heading angle in degrees, in [0, 360).
        """
        dx = point_2[0] - point_1[0]
        dy = point_2[1] - point_1[1]

        if true_north:
            angle_rad = math.atan2(dx, dy)
        else:
            angle_rad = math.atan2(dy, dx)

        angle_deg = math.degrees(angle_rad)
        return (angle_deg + 360) % 360


def get_euclidean_distance_meters(
    point_1: Tuple[float], 
    point_2: Tuple[float]
) -> float:
    """
    Computes the euclidean distance between two (lon, lat) points in meters. 
    This only works if the scale for the analysis is small. 

    Otherwise the curvature of the earth introduces an error.

    Parameters
    -----------
    point_1
        First point as a tuple (lon, lat).
    point_2
        Second point as a tuple (lon, lat)
    
    Returns
    --------
    distance
        The euclidean distance between the points in meters.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    x1, y1 = transformer.transform(point_1[0], point_1[1])
    x2, y2 = transformer.transform(point_2[0], point_2[1])

    distance = np.sqrt(((x2 - x1)**2 + (y2 - y1)**2))
    return distance

