
from imageable.images.camera import CameraParameters
import pytest

#Camera parameters tests
def test_valid_camera_parameters():
    camera_params = CameraParameters(
        fov =90,
        heading = 180,
        pitch = 0,
        width = 320,
        height = 320
        )

    assert camera_params.fov == 90
    assert camera_params.heading == 180
    assert camera_params.pitch == 0
    assert camera_params.width == 320
    assert camera_params.height == 320

#Test case where the user inputs invalid width
#in Camera parameters
def test_invalid_width():
    with pytest.raises(ValueError, match = "width .* cannot be greater than 640"):
        CameraParameters(
            fov = 90, 
            heading = 0, 
            pitch = 0, 
            width = 1000,
            height = 530)

#Test case where the user inputs invalid height
# in camera paremters 
def test_invalid_height():
    with pytest.raises(ValueError, match="height .* cannot be greater than 640"):
        CameraParameters(
            fov = 90, 
            heading = 0, 
            pitch = 0, 
            width = 640, 
            height = 1000
        )

#Test case where the user inputs invalid fov in 
#camera parameters
def test_invalid_fov():
    with pytest.raises(ValueError, "FOV .* should be between 10 and 120"):
        CameraParameters(
            fov = 200, 
            heading = 0,
            pitch = 0
        )
#Raises conversion from CameraParameters to dictionary
def test_to_dict_conversion():
    params = CameraParameters(
        fov = 90, 
        heading = 0,
        pitch = 0,
        width = 640, 
        height = 640
    )

    d = params.to_dict()
    assert isinstance(d, dict)
    assert d["fov"] == 90
    assert d["heading"] == 0
    assert d["pitch"] == 0
    assert d["width"] == 640
    assert d["height"] == 640


