import numpy as np
from shapely.geometry import Polygon

from imageable.utils.geometry.polygons import get_polygon_edge_midpoints, get_polygon_outward_vectors, get_signed_area


def test_signed_area():
    polygon_points = [(0,0), (1,0), (1,1), (0,1)]
    polygon = Polygon(polygon_points)

    polygon_points_2 = [(0,0), (0,1), (1,1), (1,0)]
    polygon_2 = Polygon(polygon_points_2)

    area_1 = get_signed_area(polygon)
    area_2 = get_signed_area(polygon_2)

    assert area_1 >= 0
    assert area_2 <= 0


def test_polygon_edge_midpoints():
    polygon_points = [(0,0), (1,0), (1,1), (0,1)]
    polygon = Polygon(polygon_points)

    midpoints = get_polygon_edge_midpoints(polygon)
    expected_midpoints = [(0.5,0), (1,0.5), (0.5,1), (0,0.5)]

    assert len(midpoints) == len(expected_midpoints)
    equal_conditions = []

    threshold = 0.0001
    for i in range(len(midpoints)):
        x = midpoints[i]
        y = expected_midpoints[i]

        distance = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

        if(distance < threshold):
            equal_conditions.append(True)
        else:
            equal_conditions.append(False)


    assert len(equal_conditions) == len(expected_midpoints)
    assert all(equal_conditions)


def test_polygon_outward_vectors():
    polygon_points = [(0,0), (1,0), (1,1), (0,1)]
    polygon = Polygon(polygon_points)

    outward_vectors = get_polygon_outward_vectors(polygon)
    expected_vectors = [(0,-1), (1,0), (0,1), (-1,0)]

    assert len(outward_vectors) == len(expected_vectors)
    equal_conditions = []
    threshold = 0.0001

    for i in range(len(outward_vectors)):
        x = outward_vectors[i]
        y = expected_vectors[i]

        distance = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

        if(distance < threshold):
            equal_conditions.append(True)
        else:
            equal_conditions.append(False)

    assert len(equal_conditions) == len(expected_vectors)
    assert all(equal_conditions)