import pytest

from imageable.images.camera.camera_parameters import CameraParameters


# Camera parameters tests
def test_valid_camera_parameters():
    camera_params = CameraParameters(
        longitude=-73.983138,
        latitude=40.763565,
        fov=90,
        heading=180,
        pitch=0,
        width=320,
        height=320,
    )

    assert camera_params.longitude == -73.983138
    assert camera_params.latitude == 40.763565
    assert camera_params.fov == 90
    assert camera_params.heading == 180
    assert camera_params.pitch == 0
    assert camera_params.width == 320
    assert camera_params.height == 320


# Test case where the user inputs invalid width
def test_invalid_width():
    with pytest.raises(ValueError, match="width cannot be greater than 640"):
        CameraParameters(
            longitude=-73.983138,
            latitude=40.763565,
            fov=90,
            heading=0,
            pitch=0,
            width=1000,
            height=530,
        )


# Test case where the user inputs invalid height
def test_invalid_height():
    with pytest.raises(ValueError, match="height cannot be greater than 640"):
        CameraParameters(
            longitude=-73.983138,
            latitude=40.763565,
            fov=90,
            heading=0,
            pitch=0,
            width=640,
            height=1000,
        )


# Test case where the user inputs invalid fov in camera parameters
def test_invalid_fov():
    with pytest.raises(ValueError, match="FOV should be between 10 and 120"):
        CameraParameters(
            longitude=-73.983138,
            latitude=40.763565,
            fov=200,
            heading=0,
            pitch=0,
        )


# Raises conversion from CameraParameters to dictionary
def test_to_dict_conversion():
    params = CameraParameters(
        longitude=-73.983138,
        latitude=40.763565,
        fov=90,
        heading=0,
        pitch=0,
        width=640,
        height=640,
    )

    d = params.to_dict()
    assert isinstance(d, dict)
    assert d["longitude"] == -73.983138
    assert d["latitude"] == 40.763565
    assert d["fov"] == 90
    assert d["heading"] == 0
    assert d["pitch"] == 0
    assert d["width"] == 640
    assert d["height"] == 640
