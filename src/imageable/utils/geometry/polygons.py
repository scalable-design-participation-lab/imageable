from shapely import Polygon
from typing import List, Tuple
import numpy as np

def get_signed_area(polygon:Polygon)-> float:
    """
    Computes the signed area of a polygon using the shoelace formula.
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
    A = 0.0
    n = len(polygon.exterior.coords)
    for i in range(n-1):
        x1, y1 = polygon.exterior.coords[i]
        x2, y2 = polygon.exterior.coords[i + 1]
        A += x1*y2 - x2*y1
    
    return A/2.0


def get_polygon_edge_midpoints(polygon:Polygon)-> List[Tuple[float]]:
    """
    Computes the midpoints of the edges of a polygon.

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
    for i in range(n-1):
        x1, y1 = polygon.exterior.coords[i]
        x2, y2 = polygon.exterior.coords[i + 1]
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        midpoints.append((mid_x, mid_y))
    
    return midpoints


def get_polygon_outward_vectors(polygon:Polygon)-> List[Tuple[float]]:
    """
    Computes the orthogonal outward vectors to a polygon's edges.

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

    #Get the signed area
    A = get_signed_area(polygon)
    n = len(polygon.exterior.coords)
    outward_vectors = []
    if(A > 0):
        #Counter-clockwise orientation
        for i in range(n-1):
            x1, y1 = polygon.exterior.coords[i]
            x2, y2 = polygon.exterior.coords[i + 1]
            norm = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            outward_vector = ((y2 - y1)/norm, (x1 - x2)/norm)
            outward_vectors.append(outward_vector)
        return outward_vectors
        
    else:
        #Clockwise orientation
        for i in range(n-1):
            x1, y1 = polygon.exterior.coords[i]
            x2, y2 = polygon.exterior.coords[i + 1]
            norm = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            outward_vector = ((y1 - y2)/norm, (x2 - x1)/norm)
            outward_vectors.append(outward_vector)
        return outward_vectors


