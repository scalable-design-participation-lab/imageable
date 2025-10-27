import json
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
from PIL import Image
from tqdm import tqdm

from imageable.features.height.building_height import HeightEstimationParameters, building_height_from_single_view
from imageable.features.materials.building_materials import (
    BuildingMaterialProperties,
    get_building_materials_segmentation,
)
from imageable.images.camera.camera_parameters import CameraParameters
from imageable.properties.extract import extract_building_properties


@dataclass
class DatasetBuildingParameters:
    """Parameters applied to build a dataset from a GeoDataFrame.

    Parameters
    ----------
    gdf
        A GeoDataFrame containing the footprints of buildings to consider for analysis.
    output_dir_path
        The directory where the final dataset will be saved.
    indices_subset
        An optional list of indices specifying a subset of buildings from the GeoDataFrame to include in the dataset.
    file_extension
        The file format for saving the dataset. Options include "geojson", "shp", "gpkg".
    crs
        The coordinate reference system to use for spatial data. Default is "EPSG:4326".
    height_estimation_parameters
        Parameters used for building height estimation. If None heights won't be estimated and will be set to none_value.
    building_material_properties
        Optional properties for building material calculation. If None, default properties will be used.
    include_footprint_features
        A boolean indicating whether to include footprint features in the dataset. Default is True.
    """

    gdf: gpd.GeoDataFrame
    output_dir_path: str
    indices_subset: list[int] | None = None
    file_extension: Literal["geojson", "shp", "gpkg"] = "geojson"
    crs: str = "EPSG:4326"
    height_estimation_parameters: HeightEstimationParameters | None = None
    building_material_properties: BuildingMaterialProperties | None = None
    include_footprint_features: bool = True
    none_value: float = float("nan")


# ruff: noqa PLR0915
def dataset_from_gdf(
    dataset_params: DatasetBuildingParameters,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """Build a dataset from a GeoDataFrame of building footprints."""
    gdf = dataset_params.gdf.copy()
    n_buildings = len(gdf["geometry"])
    footprint_indices = list(range(n_buildings))
    if dataset_params.indices_subset is not None:
        gdf = gdf.iloc[dataset_params.indices_subset]
        n_buildings = len(gdf["geometry"])
        footprint_indices = dataset_params.indices_subset

    footprints = list(gdf["geometry"])
    building_properties = []

    for i in tqdm(range(len(footprints))):
        footprint = footprints[i]
        footprint_index = footprint_indices[i]

        height = dataset_params.none_value
        height_params = dataset_params.height_estimation_parameters
        if height_params is not None:
            height_params = HeightEstimationParameters(**vars(height_params))
            height_params.building_polygon = footprint
            pictures_dir = f"{height_params.pictures_directory}/{footprint_index}"
            height_params.pictures_directory = pictures_dir
            height = building_height_from_single_view(height_params)
        else:
            pictures_dir = f"{dataset_params.output_dir_path}/{footprint_index}"

        img = None
        metadata = None
        camera_parameters = None
        try:
            img = np.array(Image.open(f"{pictures_dir}/image.jpg"))
            metadata_dir = f"{pictures_dir}/metadata.json"
            with open(metadata_dir) as f:
                metadata = json.load(f)

            camera_parameters_dictionary = metadata["camera_parameters"]
            camera_parameters = CameraParameters(
                longitude=camera_parameters_dictionary["longitude"],
                latitude=camera_parameters_dictionary["latitude"],
            )
            camera_parameters.fov = camera_parameters_dictionary["fov"]
            camera_parameters.heading = camera_parameters_dictionary["heading"]
            camera_parameters.pitch = camera_parameters_dictionary["pitch"]
            camera_parameters.height = camera_parameters_dictionary["height"]
            camera_parameters.width = camera_parameters_dictionary["width"]
        except Exception:
            img = None

        materials = None
        material_segmentation_parameters = dataset_params.building_material_properties
        if (img is not None) and (material_segmentation_parameters is not None):
            msp = BuildingMaterialProperties(**vars(material_segmentation_parameters))
            msp.img = img
            msp.building_height = height
            msp.footprint = footprint
            msp.camera_parameters = camera_parameters
            materials = get_building_materials_segmentation(msp)

        individual_building_props = extract_building_properties(
            building_id=footprint_index,
            polygon=footprint,
            all_buildings=footprints,
            crs=gdf.crs,
            street_view_image=img,
            height_value=height,
            material_percentages=materials,
            verbose=verbose,
        )

        building_properties.append(individual_building_props)

    buildings_labelled_gdf = gpd.GeoDataFrame(building_properties, geometry=footprints, crs=gdf.crs)
    output_data_path = f"{dataset_params.output_dir_path}/dataset.{dataset_params.file_extension}"
    buildings_labelled_gdf.to_file(output_data_path)

    return buildings_labelled_gdf
