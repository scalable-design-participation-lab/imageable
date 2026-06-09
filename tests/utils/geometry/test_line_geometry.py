import numpy as np
import pytest

from imageable._utils.geometry.line_geometry import (
    get_angle_between_segments,
    intersection,
    line_coeff,
    order_top_bottom,
    point_line_distance,
    segment_length,
    to_cam_ray,
)


def test_segment_length():
    point1 = np.array([0,0])
    point2 = np.array([1,1])

    seg_length = segment_length(point1, point2)
    epsilon = 1e-9

    assert np.abs(seg_length - np.sqrt(2)) < epsilon


def test_order_top_bottom():

    point1= np.array([1,2])
    point2 = np.array([0,3])


    new_1, _ = order_top_bottom(point1, point2)

    np.testing.assert_allclose(new_1, point1)


    point1 = np.array([1,1])
    point2 = np.array([0,1])

    new_1, _ = order_top_bottom(point1, point2)
    np.testing.assert_allclose(new_1, point2)


def test_point_line_distance():
    line = line_coeff(np.array([0, 0]), np.array([1, 1]))

    distance = point_line_distance(np.array([2, 0]), line)

    np.testing.assert_allclose(distance, np.sqrt(2))


def test_line_coeff():
    # Line through (3,10) and (5,7) -> -3x - 2y + 29 = 0
    coeffs = line_coeff(np.array([3, 10]), np.array([5, 7]))

    assert coeffs == (-3, -2, 29)


def test_intersection():
    # x + y = 1  and  x = y  ->  (1/2, 1/2)
    point = intersection((-1, -1, 1), (1, -1, 0))

    np.testing.assert_allclose(point, [0.5, 0.5])


def test_intersection_parallel_returns_none():
    # Two parallel lines (same A, B) never cross -> None
    assert intersection((1, 1, 0), (1, 1, 5)) is None


def test_to_cam_ray():
    # Identity K_inv leaves the homogeneous point unchanged: [x, y, 1]
    k_inv = 2*np.eye(3)

    ray = to_cam_ray(np.array([3, 5]), k_inv)

    np.testing.assert_allclose(ray, [2*3, 2*5, 2*1])


def test_get_angle_between_segments():
    # Vertical segment (0,1) vs diagonal segment (1,1) -> 45 degrees
    angle = get_angle_between_segments([(0, 0), (0, 1)], [(0, 0), (1, 1)])

    assert angle == pytest.approx(45.0)


def test_get_angle_zero_length_segment_returns_none():
    # Second "segment" is a single point (no direction) -> None
    assert get_angle_between_segments([(0, 0), (0, 1)], [(2, 2), (2, 2)]) is None
