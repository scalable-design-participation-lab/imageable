import itertools
import math

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points


def _iter_lines(geom: BaseGeometry):
    gt = geom.geom_type
    if gt == "LineString":
        yield geom
    elif gt == "MultiLineString":
        for g in geom.geoms:
            yield g
    elif gt == "Polygon":
        bd = geom.boundary
        if bd.geom_type == "LineString":
            yield bd
        else:
            for g in bd.geoms:
                yield g
    elif gt == "MultiPolygon":
        for p in geom.geoms:
            bd = p.boundary
            if bd.geom_type == "LineString":
                yield bd
            else:
                for g in bd.geoms:
                    yield g
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            yield from _iter_lines(g)


def _segment_containing_point(
    line: LineString, point: Point, tol: float = 1e-9
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return the two endpoints of the segment in `line` that the ray hit.
    We pick the segment whose distance to `point` is minimal (ideally zero).
    """
    coords = list(line.coords)
    best_pair = None
    best_d = float("inf")
    for a, b in itertools.pairwise(coords):
        seg = LineString([a, b])
        d = seg.distance(point)
        # Prefer exact containment (d ~ 0), otherwise nearest
        if d + tol < best_d or (abs(d - best_d) <= tol and best_pair is None):
            best_d = d
            best_pair = (a, b)
            if best_d <= tol:
                # Early exit if we found the exact segment
                break
    return best_pair if best_pair is not None else (coords[0], coords[-1])


def get_closest_intersected_line(
    start_point: tuple[float, float],
    ray_direction: tuple[float, float],
    boundaries: gpd.GeoDataFrame,
    max_ray_length: float = 500.0,
) -> list[tuple[float, float]]:
    """
    Return the two endpoints (lon, lat) of the closest line *segment* intersected by a ray.
    Empty list if no intersection within max_ray_length.
    """
    ox, oy = start_point
    dx, dy = ray_direction
    mag = math.hypot(dx, dy)
    if mag == 0:
        return []
    ux, uy = dx / mag, dy / mag
    end = (ox + ux * max_ray_length, oy + uy * max_ray_length)

    origin = Point(ox, oy)
    ray = LineString([origin, Point(end)])

    # candidate geometries that intersect the ray's bbox
    try:
        idxs = list(boundaries.sindex.intersection(ray.bounds))
        candidates = boundaries.iloc[idxs]
    except Exception:
        candidates = boundaries

    best_dist = np.inf
    best_line: LineString | None = None
    best_hit_point: Point | None = None

    for geom in candidates.geometry:
        if geom is None or geom.is_empty:
            continue
        for line in _iter_lines(geom):
            if line.is_empty:
                continue
            inter = ray.intersection(line)
            if inter.is_empty:
                continue

            # find the closest point on the intersection to the origin
            _, p_on_inter = nearest_points(origin, inter)
            d = origin.distance(p_on_inter)
            # Ignore intersections at the origin itself unless you want to treat that as a hit
            if d < best_dist and d > 1e-12:
                best_dist = d
                best_line = line
                best_hit_point = p_on_inter

    if best_line is None:
        return []

    # For polygon boundaries (rings), pick the specific edge segment that was hit
    if best_hit_point is not None:
        a, b = _segment_containing_point(best_line, best_hit_point)
        return [a, b]

    # Fallback: return endpoints of the best line
    coords = list(best_line.coords)
    return [coords[0], coords[-1]]


def get_closest_ray_intersection(
    start_point: tuple[float], ray_direction: tuple[float], boundaries: gpd.GeoDataFrame, max_ray_length: float = 500
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
        if intersection.is_empty or isinstance(intersection, Polygon):
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
