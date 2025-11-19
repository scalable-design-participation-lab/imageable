import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from imageable._utils.geometry.ray_geometry import get_closest_ray_intersection


def test_closest_ray_intersection():

    center_point = (29.322776540896413, -100.93189549304725)
    reversed_center = (center_point[1], center_point[0])
    points = [
        (29.323414847430044, -100.93249928465585),
        (29.3227797079051, -100.93243006276319),
        (29.322840060700116, -100.93151369675554),
        (29.323509686928475, -100.93157962236755),
        (29.323414847430044, -100.93249928465585),
    ]

    reversed_points = [(x[1], x[0]) for x in points]
    boundaries = gpd.GeoDataFrame(geometry=[LineString(reversed_points)], crs=4326)

    end_point = (np.array(reversed_points[1]) + np.array(reversed_points[2])) / 2
    ray_direction = [end_point[0] - reversed_center[0], end_point[1] - reversed_center[1]]

    ray_magnitude = np.sqrt(ray_direction[0] ** 2 + ray_direction[1] ** 2)
    ray_direction[0] = ray_direction[0] / ray_magnitude
    ray_direction[1] = ray_direction[1] / ray_magnitude

    ray_intersection, distance = get_closest_ray_intersection(
        start_point=reversed_center, ray_direction=ray_direction, boundaries=boundaries, max_ray_length=500
    )
    print(distance)
    threshold = 0.001

    assert distance < threshold
