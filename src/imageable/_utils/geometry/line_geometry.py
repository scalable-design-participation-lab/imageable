import numpy as np


def line_coeff(point1: np.ndarray, point2: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate the coefficients of the line
    that passes througn two points.

    Parameters
    ----------
    point1
        First point.
    point2
        Second point.

    Returns
    -------
    line_coeffs
        Coefficients of the line in the form Ax + By + C = 0.
    """
    A = point2[1] - point1[1]
    B = point1[0] - point2[0]
    C = point2[0] * point1[1] - point1[0] * point2[1]
    return A, B, C


def intersection(line1: tuple[float, float, float], line2: tuple[float, float, float]) -> np.ndarray | None:
    """
    Calculate the intersection point of two lines.

    Parameters
    ----------
    line1
        Coefficients of the first line of the form Ax + By + C = 0.
    line2
        Coefficients of the second line of the form Ax + By + C = 0.

    Returns
    -------
    intersection
        The intersection point of the two lines.
    """
    determinant = line1[0] * line2[1] - line2[0] * line1[1]
    if determinant == 0:
        return None
    x = (line1[1] * line2[2] - line2[1] * line1[2]) / determinant
    y = (line2[0] * line1[2] - line1[0] * line2[2]) / determinant
    return np.array([x, y])


def point_line_distance(point: np.ndarray, line: tuple[float, float, float]) -> float:
    """
    Calculate the distance from a point to a line.

    Parameters
    ----------
    point
        The point as a numpy array [x, y].
    line
        Coefficients of the line in the form Ax + By + C = 0.

    Returns
    -------
    distance
        The distance from the point to the line.
    """
    A, B, C = line
    distance = abs(A * point[0] + B * point[1] + C) / np.sqrt(A**2 + B**2)
    return distance


def segment_length(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the length of a line segment defined by two points.

    Parameters
    ----------
    point1
        First endpoint of the segment as a numpy array [x, y].
    point2
        Second endpoint of the segment as a numpy array [x, y].

    Returns
    -------
    length
        The length of the line segment.
    """
    segment_length = float(np.linalg.norm(point2 - point1))
    return segment_length


def order_top_bottom(point1: np.ndarray, point2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Order two points such that the first point is the topmost
    (i.e., has the smallest y-coordinate). If both points have
    the same y-coordinate, the leftmost point (smallest x-coordinate)
    is considered the topmost.

    Parameters
    ----------
    point1
        First point as a numpy array [x, y].
    point2
        Second point as a numpy array [x, y].

    Returns
    -------
    ordered_points
        A tuple containing the topmost and bottommost points.
    """
    if (point1[1] < point2[1]) or (point1[1] == point2[1] and point1[0] < point2[0]):
        return point1, point2
    return point2, point1


def to_cam_ray(point: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """
    Convert an image point in pixel coordinates to a 3D ray in camera coordinates.

    Parameters
    ----------
    point
        The image point [x, y] in pixel coordinates.
    K_inv
        Inverse of the camera intrinsic matrix.

    Returns
    -------
    ray
        3D ray direction in the camera coordinate system.
        The vector is not normalized (normalization is not required
        for sv_measurement formulas).
    """
    x, y = float(point[0]), float(point[1])
    homog = np.array([x, y, 1.0], dtype=float)
    ray = K_inv @ homog
    return ray


def get_angle_between_segments(
    line_1: list[tuple[float, float]],
    line_2: list[tuple[float, float]],
) -> float | None:
    """
    Calculate the angle in degress between two line segments.

    Parameters
    ----------
    __________
    line_1
        List of two tuples corresponding to the endpoints of the first line segment.
    line_2
        List of two tuples corresponding to the endpoints of the second line segment.

    Returns
    -------
    _______
    angle
        The angle in degrees between the two line segments.
    """
    p1_start = np.array(line_1[0])
    p1_end = np.array(line_1[1])
    p2_start = np.array(line_2[0])
    p2_end = np.array(line_2[1])

    line_1_mag = np.linalg.norm(p1_end - p1_start)
    line_2_mag = np.linalg.norm(p2_end - p2_start)

    if line_1_mag != 0 and line_2_mag != 0:
        line_1_unit = (p1_end - p1_start) / line_1_mag
        line_2_unit = (p2_end - p2_start) / line_2_mag

        dot_product = np.clip(np.dot(line_1_unit, line_2_unit), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    return None
