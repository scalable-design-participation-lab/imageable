from imageable.utils.geometry.point_geometry import get_euclidean_distance_meters, get_heading_between_points_euclidean


def test_heading_between_points_east_reference():
    true_north = False
    point_1 = (19.299663313632703, -99.10578117536366)
    point_2 = (19.299690538813294, -99.10512540369302)

    # Let's reverse the points
    point_1_reversed = (point_1[1], point_1[0])
    point_2_reversed = (point_2[1], point_2[0])

    get_heading_between_points_euclidean(point_1_reversed, point_2_reversed, true_north=true_north)
    max_error = 15
    heading = get_heading_between_points_euclidean(point_1_reversed, point_2_reversed, true_north=true_north)
    # It must be close to 0 but with an error

    assert (heading < max_error and heading >= 0) or (heading <= 360 and heading > 360 - max_error)


def test_heading_between_points_north_reference():
    true_north = False
    point_1 = (19.299663313632703, -99.10578117536366)
    point_2 = (19.299690538813294, -99.10512540369302)

    point_1_reversed = (point_1[1], point_1[0])
    point_2_reversed = (point_2[1], point_2[0])

    heading = get_heading_between_points_euclidean(point_1_reversed, point_2_reversed, true_north=true_north)

    max_error = 15
    assert heading > 90 - max_error
    assert heading < 90 + max_error


def test_euclidean_distance_meters():
    point_1 = (19.299663313632703, -99.10578117536366)
    point_2 = (19.299690538813294, -99.10512540369302)

    # Let's reverse the points
    point_1_reversed = (point_1[1], point_1[0])
    point_2_reversed = (point_2[1], point_2[0])

    distance = get_euclidean_distance_meters(point_1_reversed, point_2_reversed)

    # The distance should be less than 100 meters
    assert distance > 72
    assert distance < 74
