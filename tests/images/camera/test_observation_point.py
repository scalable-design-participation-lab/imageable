from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
from shapely.geometry import Polygon

from imageable._images.camera.building_observation import ObservationPointEstimator

# First test a valid result


def test_observation_point_valid():
    # Define a polygon
    polygon = Polygon(
        [
            (-71.00564872499996, 42.39109031100003),
            (-71.00576326499998, 42.39108336100003),
            (-71.00577879099995, 42.390969031000054),
            (-71.00574336499994, 42.39096638700005),
            (-71.00574602899997, 42.39094677700007),
            (-71.00569376399994, 42.39094287700004),
            (-71.00569110099997, 42.390962487000024),
            (-71.00565440999998, 42.39095974900005),
            (-71.00563896999995, 42.39107345100007),
            (-71.00564872499996, 42.39109031100003),
        ]
    )

    estimator = ObservationPointEstimator(polygon)
    result = estimator.get_observation_point()
    # Check if the result is valid
    assert result is not None

    intersection, midpoint, heading, distance = result
    assert isinstance(intersection, tuple)
    assert len(intersection) == 2
    assert isinstance(midpoint, tuple)
    assert len(midpoint) == 2
    assert isinstance(heading, float)
    assert isinstance(distance, float)
    assert np.isfinite(distance)


def test_observation_point_returns_none_in_desert():
    polygon = Polygon(
        [(-114.883, 31.835), (-114.883, 31.836), (-114.882, 31.836), (-114.882, 31.835), (-114.883, 31.835)]
    )

    estimator = ObservationPointEstimator(polygon)
    result = estimator.get_observation_point()

    assert result == (None, None, None, np.inf)


def test_preloaded_graph_clipped_with_bbox_without_overpass_call():
    polygon = Polygon([(0.0, 0.0), (0.0, 0.001), (0.001, 0.001), (0.001, 0.0), (0.0, 0.0)])

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.001, y=0.0)
    graph.add_node(3, x=0.02, y=0.02)
    graph.add_node(4, x=0.021, y=0.02)
    graph.add_edge(1, 2, key=0, highway="residential")
    graph.add_edge(3, 4, key=0, highway="residential")

    props = SimpleNamespace(
        projected_area=1.0,
        shape_length=1.0,
        n_vertices=4.0,
        complexity=1.0,
        unprojected_area=1.0,
        vertices_per_area=1.0,
        latitude_difference=0.001,
        longitude_difference=0.001,
    )
    estimator = ObservationPointEstimator(polygon, street_network=graph)

    with (
        patch(
            "imageable._images.camera.building_observation.extract_building_properties",
            return_value=props,
        ),
        patch("imageable._images.camera.building_observation.DistanceRegressorWrapper") as mock_distance_wrapper_cls,
        patch("imageable._images.camera.building_observation.ox.graph_from_point") as mock_graph_from_point,
    ):
        mock_distance_wrapper = mock_distance_wrapper_cls.return_value
        mock_distance_wrapper.predict.return_value = np.array([10.0], dtype=np.float32)

        clipped = estimator._get_surrounding_street_network(buffer_constant=20)

    assert clipped is not None
    assert not clipped.empty
    assert clipped.total_bounds[2] < 0.01
    mock_graph_from_point.assert_not_called()
