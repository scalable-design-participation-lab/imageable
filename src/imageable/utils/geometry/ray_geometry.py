
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon


def get_closest_ray_intersection(
    start_point: tuple[float],
    ray_direction: tuple[float],
    boundaries: gpd.GeoDataFrame,
    max_ray_length: float = 500
) -> tuple[tuple[float, float] | None, float]:
    """
    Obtain the closest intersection of a ray with boundaries defined by a GeoDataFrame.

    Parameters
    ----------
    start_point
        The starting point of the ray as a tuple (lon, lat).
    ray_direction
        A tuple (dx, dy) that defines a unit vector in the direction of the ray.
    boundaries
        A Geodataframe containing the boundaries to check for intersections.
    max_ray_length
        The maximum length of the ray to consider for intersection.

    Returns
    -------
    closest_intersection
        A tuple (lon, lat) of the closest intersection point or None if no intersection is found.
    min_dist
        The distance to the closest intersection point. Returns np.inf if no intersection is found.
    """
    origin = Point(start_point)
    dx, dy = ray_direction
    if not np.isfinite(dx) or not np.isfinite(dy) or (dx == 0 and dy == 0):
        return None, np.inf
    ray_end = (start_point[0] + dx * max_ray_length, start_point[1] + dy * max_ray_length)

    if np.allclose(start_point, ray_end):
        return None, np.inf

    ray = LineString([origin, Point(ray_end)])

    candidate_idxs = list(boundaries.sindex.intersection(ray.bounds))
    candidates = boundaries.iloc[candidate_idxs]

    min_dist = np.inf
    closest_intersection = None

    for geom in candidates.geometry:
        if ray.length == 0 or geom.is_empty:
            continue
        intersection = geom.intersection(ray)
        if intersection.is_empty or isinstance(intersection,Polygon):
            continue

        if isinstance(intersection, Point):
            dist = origin.distance(intersection)
            if dist < min_dist:
                min_dist = dist
                closest_intersection = (intersection.x, intersection.y)

        elif isinstance(intersection, MultiPoint):
            for point in intersection.geoms:
                dist = origin.distance(point)
                if dist < min_dist:
                    min_dist = dist
                    closest_intersection = (point.x, point.y)

        elif isinstance(intersection, LineString):
            for point in intersection.coords:
                dist = origin.distance(Point(point))
                if dist < min_dist:
                    min_dist = dist
                    closest_intersection = point  # already a (x, y) tuple


    if closest_intersection is not None:
        return closest_intersection, min_dist
    return None, np.inf





