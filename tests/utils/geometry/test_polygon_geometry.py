import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from imageable._utils.geometry.polygons import (
    get_convex_hull,
    get_minimum_area_parallelogram,
    get_polygon_edge_midpoints,
    get_polygon_outward_vectors,
    get_signed_area,
)


def test_signed_area():
    polygon_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polygon = Polygon(polygon_points)

    polygon_points_2 = [(0, 0), (0, 1), (1, 1), (1, 0)]
    polygon_2 = Polygon(polygon_points_2)

    area_1 = get_signed_area(polygon)
    area_2 = get_signed_area(polygon_2)

    assert area_1 >= 0
    assert area_2 <= 0


def test_polygon_edge_midpoints():
    polygon_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polygon = Polygon(polygon_points)

    midpoints = get_polygon_edge_midpoints(polygon)
    expected_midpoints = [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]

    assert len(midpoints) == len(expected_midpoints)
    equal_conditions = []

    threshold = 0.0001
    for i in range(len(midpoints)):
        x = midpoints[i]
        y = expected_midpoints[i]

        distance = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

        if distance < threshold:
            equal_conditions.append(True)
        else:
            equal_conditions.append(False)

    assert len(equal_conditions) == len(expected_midpoints)
    assert all(equal_conditions)


def test_polygon_outward_vectors():
    polygon_points:list[tuple[float, float]] = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polygon = Polygon(polygon_points)

    outward_vectors = get_polygon_outward_vectors(polygon)
    expected_vectors:list[tuple[float, float]] = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    assert len(outward_vectors) == len(expected_vectors)
    equal_conditions = []
    threshold = 0.0001

    for i in range(len(outward_vectors)):
        x:tuple[float, float] = outward_vectors[i]
        y:tuple[float, float] = expected_vectors[i]

        distance = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

        if distance < threshold:
            equal_conditions.append(True)
        else:
            equal_conditions.append(False)

    assert len(equal_conditions) == len(expected_vectors)
    assert all(equal_conditions)



def test_convex_hull():
    points: list[tuple[float,float]] = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
    hull = get_convex_hull(points)

    poly = Polygon(hull)
    assert all(poly.covers(Point(p)) for p in points)


def test_min_parallelogram_of_square_is_the_square():
    square:list[tuple[float, float]] = [(0, 0), (1, 0), (1, 1), (0, 1)]
    corners = get_minimum_area_parallelogram(square)

    assert corners is not None
    poly = Polygon([tuple(c) for c in corners[:4]])
    assert poly.area == pytest.approx(1.0)

def test_result_is_a_parallelogram():
    pts: list[tuple[float, float]] = [(0, 0), (3, 0), (4, 2), (1, 2)]
    c = get_minimum_area_parallelogram(pts)
    assert c is not None
    np.testing.assert_allclose(c[0] + c[2], c[1] + c[3])
