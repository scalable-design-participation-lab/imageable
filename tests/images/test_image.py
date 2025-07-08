# Tests for image parameters
from imageable.images.camera import CameraParameters
from imageable.images.image import ImageMetadata


# Test valid metadata
def test_valid_metadata():
    metadata = ImageMetadata(
        status=True,
        latitude=40.7128,
        longitude=-74.0060,
        pano_id="hello_world",
        date="1918-01",
        source="Uriel simulations",
        img_size=(640, 480),
        camera_parameters=CameraParameters(
            latitude=40.7128, longitude=-74.0060, heading=0.0, pitch=0.0, fov=90.0, width=640, height=480
        ),
    )

    assert metadata.status
    assert metadata.latitude == 40.7128
    assert metadata.longitude == -74.0060
    assert metadata.pano_id == "hello_world"
    assert metadata.date == "1918-01"
    assert metadata.source == "Uriel simulations"
    assert metadata.img_size == (640, 480)
    assert isinstance(metadata.camera_parameters, dict)


def test_metadata_to_dict():
    metadata = ImageMetadata(
        status=True,
        latitude=40.7128,
        longitude=-74.0060,
        pano_id="hello_world",
        date="1918-01",
        source="Uriel simulations",
        img_size=(640, 480),
        camera_parameters=CameraParameters(
            latitude=40.7128, longitude=-74.0060, heading=0.0, pitch=0.0, fov=90.0, width=640, height=480
        ),
    )

    # Validate the conversion to dictionary
    metadata_dictionary = metadata.to_dict()
    assert isinstance(metadata_dictionary, dict)
    assert metadata_dictionary["status"]
    assert metadata_dictionary["latitude"] == 40.7128
    assert metadata_dictionary["longitude"] == -74.0060
    assert metadata_dictionary["pano_id"] == "hello_world"
    assert metadata_dictionary["date"] == "1918-01"
    assert metadata_dictionary["source"] == "Uriel simulations"
    assert metadata_dictionary["img_size"] == (640, 480)
    assert "camera_parameters" in metadata_dictionary
    assert isinstance(metadata_dictionary["camera_parameters"], dict)
    assert metadata_dictionary["camera_parameters"]["latitude"] == 40.7128
    assert metadata_dictionary["camera_parameters"]["longitude"] == -74.0060
    assert metadata_dictionary["camera_parameters"]["heading"] == 0.0
    assert metadata_dictionary["camera_parameters"]["pitch"] == 0.0
    assert metadata_dictionary["camera_parameters"]["fov"] == 90.0
    assert metadata_dictionary["camera_parameters"]["width"] == 640
    assert metadata_dictionary["camera_parameters"]["height"] == 480
