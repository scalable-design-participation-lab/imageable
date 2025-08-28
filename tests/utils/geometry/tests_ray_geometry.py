import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from imageable.utils.geometry.ray_geometry import get_closest_ray_intersection


def test_closest_ray_intersection():
    center_point = (19.288070936125443, -99.13828508724062)
    reversed_center = (center_point[1], center_point[0])
    points = [
        (19.28264226905163, -99.1385036158896),
        (19.288925458534457, -99.13531202518037),
        (19.290196297996125, -99.13956536018992),
        (19.28378370618827, -99.14070229726086),
        (19.283788527411165, -99.14063248040932)
    ]
    reversed_points = [(x[1], x[0]) for x in points]
    boundaries = gpd.GeoDataFrame(geometry= [LineString(reversed_points)], crs = 4326)


    end_point = (np.array(reversed_points[0])+np.array(reversed_points[1]))/2
    ray_direction = [end_point[0] - reversed_center[0], end_point[1] - reversed_center[1]]

    ray_magnitude = np.sqrt(ray_direction[0] ** 2 + ray_direction[1] ** 2)
    ray_direction[0] = ray_direction[0]/ray_magnitude
    ray_direction[1] = ray_direction[1]/ray_magnitude

    ray_intersection, distance = get_closest_ray_intersection(
        start_point = reversed_center,
        ray_direction = ray_direction,
        boundaries = boundaries,
        max_ray_length = 500)

    threshold = 0.00001

    assert distance < threshold




