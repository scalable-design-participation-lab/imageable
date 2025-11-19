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
