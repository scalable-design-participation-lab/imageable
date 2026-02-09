
import numpy as np
import pytest
from shapely import Polygon

from imageable._features.materials.building_materials import (
    BuildingMaterialProperties,
    get_building_materials_segmentation,
)
from imageable._images.camera.camera_parameters import CameraParameters


def _get_mock_building_image():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    start_x = 160
    end_y = 480

    img[start_x:end_y, start_x:end_y, 0] = 255
    img[start_x:end_y, start_x:end_y, 1] = 0
    img[start_x:end_y, start_x:end_y, 2] = 0

    return img




def test_building_material_properties():
    img = _get_mock_building_image()

    building_height = 15.0
    #Define a very simple polygon for the building footprint
    footprint = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    #Mock camera parameters
    camera_parameters = CameraParameters(
        latitude = -10.0,
        longitude = 5.0,
        heading = 0.0,
        fov = 90.0,
        pitch = 0.0,
        width = 640,
        height = 640
    )
    #Define the building material properties
    building_material_props = BuildingMaterialProperties(
        img = img,
        camera_parameters = camera_parameters,
        building_height = building_height,
        footprint = footprint
    )

    #Assert some of the properties that are crucial for building material areas calculation
    assert building_material_props.img is not None
    assert building_material_props.camera_parameters.heading == 0.0
    assert building_material_props.building_height == 15.0
    assert building_material_props.footprint.area == 100.0
    assert isinstance(building_material_props.footprint, Polygon)


@pytest.mark.skip(reason="Requires RMSNet model weights that are not available in CI")
def test_building_material_percentages():
    img = _get_mock_building_image()

    building_height = 15.0
    #Mock camera parameters
    camera_parameters = CameraParameters(
        latitude = -10.0,
        longitude = 5.0,
        heading = 0.0,
        fov = 90.0,
        pitch = 0.0,
        width = 640,
        height = 640
    )
    #Define the building material properties
    building_material_props = BuildingMaterialProperties(
        img = img,
        camera_parameters = camera_parameters,
        building_height = building_height,
        footprint = None,
    )

    material_percentages = get_building_materials_segmentation(building_material_props)

    percentages_list = [material_percentages[k] for k in list(material_percentages.keys())]
    #The sum of the percentages list should equal 1
    #If at some point we decide to multiply the values by 100 this test should be updated.
    total_percentage = sum(percentages_list)
    print("Total material percentage:", total_percentage)
    assert np.isclose(total_percentage, 1.0)

@pytest.mark.skip(reason="Requires RMSNet model weights that are not available in CI")
def test_building_material_areas():
    img = _get_mock_building_image()
    #Define a very simple polygon for the building footprint
    footprint = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])

    building_height = 15.0
    #Mock camera parameters
    camera_parameters = CameraParameters(
        latitude = -10.0,
        longitude = 5.0,
        heading = 0.0,
        fov = 90.0,
        pitch = 0.0,
        width = 640,
        height = 640
    )
    #Define the building material properties
    building_material_props = BuildingMaterialProperties(
        img = img,
        camera_parameters = camera_parameters,
        building_height = building_height,
        footprint = footprint,
    )

    material_areas = get_building_materials_segmentation(building_material_props)

    #The total area should be 10 x 15 if my intuition is correct.
    material_areas_list = [material_areas[k] for k in list(material_areas.keys())]
    total_area = sum(material_areas_list)

    target_area = 1113200 *15.0 # 10 degrees converted to meters times height

    print("Total material area:", total_area)
    print("Difference: ", abs(total_area - target_area))
    assert np.isclose(total_area, target_area, rtol = 100)









