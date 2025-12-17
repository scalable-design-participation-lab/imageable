import math

import numpy as np
from shapely import Polygon


def get_signed_area(polygon: Polygon) -> float:
    """
    Compute the signed area of a polygon using the shoelace formula.
    The signed area is positive if the polygon is oriented counter-clockwise,
    and negative if it is oriented clockwise.

    Parameters
    ----------
    polygon
        A shapely Polygon object representing the polygon.

    Returns
    -------
    A
        The signed area of the polygon.
    """
    area = 0.0
    n = len(polygon.exterior.coords)
    for i in range(n - 1):
        x1, y1 = polygon.exterior.coords[i]
        x2, y2 = polygon.exterior.coords[i + 1]
        area += x1 * y2 - x2 * y1

    return area / 2.0


def get_polygon_edge_midpoints(polygon: Polygon) -> list[tuple[float]]:
    """
    Compute the midpoints of the edges of a polygon.

    Parameters
    ----------
    polygon
        A shapely Polygon object representing the polygon.

    Returns
    -------
    midpoints
        A list of tuples representing the midpoints of the edges of the polygon.
    """
    midpoints = []
    n = len(polygon.exterior.coords)
    for i in range(n - 1):
        x1, y1 = polygon.exterior.coords[i]
        x2, y2 = polygon.exterior.coords[i + 1]
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        midpoints.append((mid_x, mid_y))

    return midpoints


def get_polygon_outward_vectors(polygon: Polygon) -> list[tuple[float]]:
    """
    Compute the orthogonal outward vectors to a polygon's edges.

    Parameters
    ----------
    polygon
        A shapely Polygon object representing the polygon.

    Returns
    -------
    outward_vectors
        A list of tuples representing the outward vectors of the polygon's edges.
        Each vector is represented as a tuple (dx, dy).
    """
    # Get the signed area
    A = get_signed_area(polygon)
    n = len(polygon.exterior.coords)
    outward_vectors = []
    if A > 0:
        # Counter-clockwise orientation
        for i in range(n - 1):
            x1, y1 = polygon.exterior.coords[i]
            x2, y2 = polygon.exterior.coords[i + 1]
            norm = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            outward_vector = ((y2 - y1) / norm, (x1 - x2) / norm)
            outward_vectors.append(outward_vector)
        return outward_vectors

    # Clockwise orientation
    for i in range(n - 1):
        x1, y1 = polygon.exterior.coords[i]
        x2, y2 = polygon.exterior.coords[i + 1]
        norm = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        outward_vector = ((y1 - y2) / norm, (x2 - x1) / norm)
        outward_vectors.append(outward_vector)
    return outward_vectors


def get_convex_hull(points: list[tuple[float, float]], close: bool = False) -> list[tuple[float, float]]:
    """
    Compute the convex hull of a set of 2D points using the Graham scan algorithm.

    Parameters
    ----------
    points
        A list of tuples representing the 2D points.
    close
        A boolean indicating whether to close the polygon by appending the first point at the end.

    Returns
    -------
    stack
        A list of tuples representing the vertices of the convex hull in counter-clockwise order.
    """
    pts = points[:]
    stack = []

    p0 = min(pts, key=lambda p: (p[1], p[0]))
    pts.remove(p0)
    stack.append(p0)

    pts.sort(key=lambda p: (math.atan2(p[1] - p0[1], p[0] - p0[0]), (p[1] - p0[1]) ** 2 + (p[0] - p0[0]) ** 2))
    if not pts:
        return stack

    stack.append(pts[0])
    for p in pts[1:]:
        while len(stack) > 1:
            p1 = stack[-1]
            p2 = stack[-2]
            cross = (p1[0] - p2[0]) * (p[1] - p2[1]) - (p1[1] - p2[1]) * (p[0] - p2[0])
            if cross > 0:
                break
            stack.pop()
        stack.append(p)
    if close:
        stack.append(stack[0])
    return stack


def get_minimum_area_parallelogram(hull_points: list[tuple[float, float]], eps: float = 1e-12):
    """
    Compute the minimum area enclosing parallelogram of a convex polygon. The algorithm
    is based on the rotating calipers method.

    Parameters
    ----------
    hull_points
        A list of tuples representing the vertices of the convex polygon in counter-clockwise order.
    eps
        A small epsilon value to handle numerical precision issues.

    Returns
    -------
    best_corners
        A list of numpy arrays representing the vertices of the minimum area enclosing parallelogram.
    """
    if len(hull_points) >= 2 and hull_points[0] == hull_points[-1]:
        hull_points = hull_points[:-1]

    H = np.asarray(hull_points, dtype=float)
    m = len(H)
    if m == 0:
        return None
    if m == 1:
        p = H[0]
        return [p, p, p, p, p]

    # edge unit directions (canonicalize u and -u)
    dirs = []
    for i in range(m):
        v = H[(i + 1) % m] - H[i]
        L = math.hypot(v[0], v[1])
        if eps > L:
            continue
        u = v / L
        if abs(u[0]) < eps:
            sgn = 1.0 if u[1] >= 0 else -1.0
        else:
            sgn = 1.0 if u[0] >= 0 else -1.0
        u = sgn * u
        dirs.append(tuple(np.round(u, 12)))
    # deduplicate
    uniq = []
    for d in dirs:
        if not any(abs(d[0] - e[0]) < 1e-9 and abs(d[1] - e[1]) < 1e-9 for e in uniq):
            uniq.append(d)
    dirs = [np.array(d) for d in uniq]
    k = len(dirs)

    # degenerate: line/point -> fall back to rotated rectangle
    if k == 0:
        v = H[-1] - H[0]
        L = math.hypot(v[0], v[1])
        if eps > L:
            p = H[0]
            return [p, p, p, p, p]
        u = v / L
        n = np.array([-u[1], u[0]])
        pu = H @ n  # note: rectangle uses normal for span anyway
        pn = H @ u
        a0, a1 = pu.min(), pu.max()
        b0, b1 = pn.min(), pn.max()
        Cll = (-b0) * u + a0 * n
        Clr = (-b1) * u + a0 * n
        Cur = (-b1) * u + a1 * n
        Cul = (-b0) * u + a1 * n
        return [Cll, Clr, Cur, Cul, Cll]

    best_area = float("inf")
    best_corners = None

    for i in range(k):
        u = dirs[i]
        nu = np.array([-u[1], u[0]])
        proj_nu = H @ nu
        a0, a1 = proj_nu.min(), proj_nu.max()

        for j in range(i + 1, k):
            v = dirs[j]
            s = u[0] * v[1] - u[1] * v[0]
            if abs(s) < 1e-9:
                continue

            nv = np.array([-v[1], v[0]])
            proj_nv = H @ nv
            b0, b1 = proj_nv.min(), proj_nv.max()

            area = (a1 - a0) * (b1 - b0) / abs(s)
            if area < best_area:
                # Corner builder: solve nu·x = alpha, nv·x = beta
                # with x = a*u + b*v  -> a = -beta/s, b = alpha/s
                inv_s = 1.0 / s

                def corner(alpha, beta):
                    return (-beta * inv_s) * u + (alpha * inv_s) * v

                Cll = corner(a0, b0)
                Clr = corner(a0, b1)
                Cur = corner(a1, b1)
                Cul = corner(a1, b0)

                best_area = area
                best_corners = [Cll, Clr, Cur, Cul, Cll]

    # very unlikely, but keep fallback
    if best_corners is None:
        u = dirs[0]
        nu = np.array([-u[1], u[0]])
        proj_nu = H @ nu
        a0, a1 = proj_nu.min(), proj_nu.max()
        v = np.array([nu[1], -nu[0]])
        nv = np.array([-v[1], v[0]])
        proj_nv = H @ nv
        b0, b1 = proj_nv.min(), proj_nv.max()
        s = u[0] * v[1] - u[1] * v[0]
        inv_s = 1.0 / s

        def corner(alpha, beta):
            return (-beta * inv_s) * u + (alpha * inv_s) * v

        Cll = corner(a0, b0)
        Clr = corner(a0, b1)
        Cur = corner(a1, b1)
        Cul = corner(a1, b0)
        best_corners = [Cll, Clr, Cur, Cul, Cll]

    return best_corners
