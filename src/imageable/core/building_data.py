"""
Public API for building data extraction.

Provides explicit input-specific functions following the pattern:
    get_building_data_from_gdf()
    get_building_data_from_geojson()
    get_building_data_from_file()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from imageable._extraction.building import BuildingProperties
from imageable._extraction.extract import extract_building_properties
from imageable._features.materials.building_materials import BuildingMaterialProperties, get_building_materials_segmentation 
from imageable._images.image import CameraParameters
# Type aliases
OutputFormat = Literal["gdf", "geojson", "dict"]


# =============================================================================
# Public API Functions
# =============================================================================

def get_building_data_from_gdf(
    gdf: gpd.GeoDataFrame,
    image_key: str,
    *,
    id_column: str | None = None,
    neighbor_radius: float = 100.0,
    output_format: OutputFormat = "gdf",
    verbose: bool = False,
    all_city_buildings_gdf: gpd.GeoDataFrame | None = None,
    pictures_directory: str | Path | None = None
) -> gpd.GeoDataFrame | dict[str, Any] | list[dict]:
    """
    Extract building properties from a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Building footprints with geometry column.
    image_key : str
        Google Street View API key for image acquisition.
    id_column : str, optional
        Column name for building IDs. Auto-generates if not provided.
    neighbor_radius : float, default=100.0
        Radius for neighbor analysis in meters.
    output_format : {"gdf", "geojson", "dict"}, default="gdf"
        Output format for results.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    GeoDataFrame | dict | list[dict]
        Building properties in requested format.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.read_file("buildings.geojson")
    >>> result = get_building_data_from_gdf(gdf, api_key)
    >>> print(result.columns)
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Expected GeoDataFrame, got {type(gdf).__name__}")

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty")

    return _extract_building_data_core(
        gdf=gdf,
        image_key=image_key,
        id_column=id_column,
        neighbor_radius=neighbor_radius,
        output_format=output_format,
        verbose=verbose,
        all_city_buildings_gdf=all_city_buildings_gdf,
        pictures_directory=pictures_directory
    )


def get_building_data_from_geojson(
    source: str | Path | dict[str, Any],
    image_key: str,
    *,
    id_property: str | None = None,
    neighbor_radius: float = 100.0,
    output_format: OutputFormat = "gdf",
    verbose: bool = False,
    city_buildings: str | Path | dict[str, Any] | None = None,
    pictures_directory: str | Path | None = None,
) -> gpd.GeoDataFrame | dict[str, Any] | list[dict]:
    """
    Extract building properties from GeoJSON file or dict.

    Parameters
    ----------
    source : str | Path | dict
        Path to GeoJSON file or GeoJSON dict/FeatureCollection.
    image_key : str
        Google Street View API key for image acquisition.
    id_property : str, optional
        Property name for building IDs. Auto-generates if not provided.
    neighbor_radius : float, default=100.0
        Radius for neighbor analysis in meters.
    output_format : {"gdf", "geojson", "dict"}, default="gdf"
        Output format for results.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    GeoDataFrame | dict | list[dict]
        Building properties in requested format.

    Examples
    --------
    >>> result = get_building_data_from_geojson("buildings.geojson", api_key)

    >>> geojson_dict = {"type": "FeatureCollection", "features": [...]}
    >>> result = get_building_data_from_geojson(geojson_dict, api_key)
    """
    gdf = _load_geojson_to_gdf(source)

    all_city_buildings_gdf = None
    if city_buildings is not None:
        all_city_buildings_gdf = _load_geojson_to_gdf(city_buildings)

    return _extract_building_data_core(
        gdf=gdf,
        image_key=image_key,
        id_column=id_property,
        neighbor_radius=neighbor_radius,
        output_format=output_format,
        verbose=verbose,
        all_city_buildings_gdf=all_city_buildings_gdf,
        pictures_directory=pictures_directory
    )


def get_building_data_from_file(
    footprints_path: str | Path,
    images_dir: str | Path,
    *,
    id_column: str | None = None,
    neighbor_radius: float = 100.0,
    output_format: OutputFormat = "gdf",
    verbose: bool = False,
    city_buildings: str | Path | dict[str, Any] | None = None
) -> gpd.GeoDataFrame | dict[str, Any] | list[dict]:
    """
    Extract building properties from local footprints and pre-downloaded images.

    Use this when you have already downloaded street view images locally.
    Note: Height estimation requires API access, so heights won't be included.

    Parameters
    ----------
    footprints_path : str | Path
        Path to footprints file (GeoJSON, Shapefile, etc.).
    images_dir : str | Path
        Directory containing street view images named by building ID.
    id_column : str, optional
        Column name for building IDs. Must match image filenames.
    neighbor_radius : float, default=100.0
        Radius for neighbor analysis in meters.
    output_format : {"gdf", "geojson", "dict"}, default="gdf"
        Output format for results.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    GeoDataFrame | dict | list[dict]
        Building properties in requested format.

    Examples
    --------
    >>> result = get_building_data_from_file(
    ...     "footprints.geojson",
    ...     "images/",
    ...     id_column="building_id"
    ... )
    """
    footprints_path = Path(footprints_path)
    images_dir = Path(images_dir)

    if not footprints_path.exists():
        raise FileNotFoundError(f"Footprints file not found: {footprints_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    gdf = _load_geojson_to_gdf(footprints_path)
    all_city_buildings_gdf = None
    if city_buildings is not None:
        all_city_buildings_gdf = _load_geojson_to_gdf(city_buildings)

    return _extract_building_data_core(
        gdf=gdf,
        images_dir=images_dir,
        id_column=id_column,
        neighbor_radius=neighbor_radius,
        output_format=output_format,
        verbose=verbose,
        all_city_buildings_gdf=all_city_buildings_gdf,
    )


# =============================================================================
# Internal Core Logic
# =============================================================================

def _extract_building_data_core(
    gdf: gpd.GeoDataFrame,
    *,
    image_key: str | None = None,
    images_dir: Path | None = None,
    id_column: str | None = None,
    neighbor_radius: float = 100.0,
    output_format: OutputFormat = "gdf",
    verbose: bool = False,
    all_city_buildings_gdf: gpd.GeoDataFrame | None = None,
    pictures_directory: str | Path | None = None,
) -> gpd.GeoDataFrame | dict[str, Any] | list[dict]:
    """
    Core extraction logic shared by all public functions.

    Either image_key OR images_dir should be provided, not both.
    Height estimation runs automatically when image_key is provided.
    """
    print(f"DEBUG: image_key={image_key!r}, bool={bool(image_key)}")
    # Prepare IDs
    if id_column and id_column in gdf.columns:
        ids = gdf[id_column].astype(str).tolist()
    else:
        ids = [f"building_{i}" for i in range(len(gdf))]

    # Get all polygons for neighbor analysis
    if all_city_buildings_gdf is not None:
        if all_city_buildings_gdf.crs is not None and gdf.crs is not None and all_city_buildings_gdf.crs != gdf.crs:
            all_city_buildings_gdf = all_city_buildings_gdf.to_crs(gdf.crs)
        all_polygons = all_city_buildings_gdf.geometry.tolist()
    else:
        all_polygons = gdf.geometry.tolist()
    crs = gdf.crs.to_epsg() if gdf.crs else 4326

    results: list[BuildingProperties] = []

    for i, (idx, row) in enumerate(gdf.iterrows()):
        building_id = ids[i]
        polygon = row.geometry

        if verbose:
            print(f"[{i+1}/{len(gdf)}] Processing {building_id}...")


        # Get image based on source
        image = None
        metadata = None
        if images_dir:
            image = _load_local_image(images_dir, building_id)
        elif image_key:
            per_building_dir = None
            if pictures_directory is not None:
                per_building_dir = Path(pictures_directory) / building_id
                per_building_dir.mkdir(parents=True, exist_ok=True)

            image, metadata = _fetch_street_view_image_and_metadata(
                polygon,
                image_key,
                pictures_directory=per_building_dir,
            )

        # Estimate height when API key is available
        height = None
        if image_key:
            if verbose:
                print(f"  Estimating height...")
            # This duplicates the image retrieval
            # What we can do is have a way were you can provide it
            if(image is None):
                height = _estimate_height(polygon, image_key, verbose=verbose, all_buildings=all_polygons)
            else:
                #print(f"DEBUG: Estimating height with image.")
                height = _estimate_height(
                    polygon,
                    image_key,
                    verbose = verbose,
                    all_buildings = all_polygons,
                    image = image)

        print(f"DEBUG: building_id={building_id}, height={height}")
        # Estimate the building materialpercentages
        materials_dictionary = None
        if(not image is None):
            camera_parameters_dictionary = metadata.get("camera_parameters",None)
            if(height is None or metadata is None or camera_parameters_dictionary is None):
                building_material_properties = BuildingMaterialProperties(
                    img = image,
                    verbose = verbose
                )
                materials_dictionary = get_building_materials_segmentation(building_material_properties)
            else:
                camera_parameters = CameraParameters(
                    longitude = camera_parameters_dictionary["longitude"],
                    latitude = camera_parameters_dictionary["latitude"],
                    fov = camera_parameters_dictionary["fov"],
                    heading = camera_parameters_dictionary["heading"],
                    pitch = camera_parameters_dictionary["pitch"],
                    width = camera_parameters_dictionary["width"],
                    height = camera_parameters_dictionary["height"]
                )

                building_material_properties = BuildingMaterialProperties(
                    img = image,
                    camera_parameters = camera_parameters,
                    footprint = polygon,
                    verbose = verbose,
                    building_height = height
                )

                materials_dictionary = get_building_materials_segmentation(building_material_properties)
        print(f"DEBUG: materials_dictionary = {materials_dictionary}")
        # Extract properties
        props = extract_building_properties(
            building_id=building_id,
            polygon=polygon,
            all_buildings=all_polygons,
            neighbor_radius=neighbor_radius,
            crs=crs,
            street_view_image=image,
            height_value=height,
            verbose=verbose,
            material_percentages = materials_dictionary
        )

        results.append(props)

    # Format output
    return _format_output(results, gdf, output_format)


def _load_geojson_to_gdf(source: str | Path | dict) -> gpd.GeoDataFrame:
    """Load GeoJSON from path or dict into GeoDataFrame."""
    if isinstance(source, dict):
        # Direct GeoJSON dict
        if source.get("type") == "FeatureCollection":
            return gpd.GeoDataFrame.from_features(source["features"])
        elif source.get("type") == "Feature":
            return gpd.GeoDataFrame.from_features([source])
        elif source.get("type") in ("Polygon", "MultiPolygon"):
            # Single geometry
            return gpd.GeoDataFrame(geometry=[shape(source)])
        else:
            raise ValueError(f"Unsupported GeoJSON type: {source.get('type')}")
    else:
        # File path
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {path}")
        return gpd.read_file(path)


def _load_local_image(images_dir: Path, building_id: str):
    """Load image from local directory by building ID."""
    import numpy as np
    from PIL import Image

    # Try common extensions
    for ext in (".jpg", ".jpeg", ".png"):
        img_path = images_dir / f"{building_id}{ext}"
        if img_path.exists():
            return np.array(Image.open(img_path))

    return None  # No image found


def _fetch_street_view_image(polygon, api_key: str):
    """Fetch street view image for polygon."""
    from imageable._images.download import download_street_view_image
    
    try:
        result = download_street_view_image(
            api_key=api_key,
            building_polygon_or_camera=polygon,
            save_path=None,
        )
        return result.get("image")
    except Exception:
        return None


def _fetch_street_view_image_and_metadata(
    polygon,
    api_key: str,
    pictures_directory: str | Path | None = None,
):
    from imageable._images.camera.camera_adjustment import CameraParametersRefiner
    from PIL import Image
    import json

    refiner = CameraParametersRefiner(polygon)

    camera_params, success, image = refiner.adjust_parameters(
        api_key,
        pictures_directory=pictures_directory,
        save_reel=False,
        overwrite_images=False if pictures_directory is None else True,
        confidence_detection=0.1,
        max_number_of_images=5,
        polygon_buffer_constant=20,
    )

    if not success or image is None:
        return None, None

    metadata = {"camera_parameters": camera_params.to_dict()}

    if pictures_directory is not None:
        pictures_directory = Path(pictures_directory)
        pictures_directory.mkdir(parents=True, exist_ok=True)

        Image.fromarray(image).save(pictures_directory / "image.jpg")
        with (pictures_directory / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

    return image, metadata

def _estimate_height(
        polygon,
        api_key: str,
        verbose: bool = False,
        all_buildings=None,
        image = None
) -> float | None:
    from imageable._features.height.building_height import (
        HeightEstimationParameters,
        building_height_from_single_view,
        corrected_height_from_single_view
    )

    try:
        params = HeightEstimationParameters(
            gsv_api_key=api_key,
            building_polygon=polygon,
            verbose=verbose,
            image = image
        )

        if all_buildings is not None and hasattr(params, "all_buildings"):
            params.all_buildings = all_buildings

        if params.all_buildings is None:
            return building_height_from_single_view(params)
        #print(f"DEBUG: correcting height")
        return corrected_height_from_single_view(
            params,
            params.building_label,
            all_buildings=params.all_buildings,
        )
    except Exception as e:
        print("DEBUG _estimate_height error:", e)
        return None



def _format_output(
    results: list[BuildingProperties],
    original_gdf: gpd.GeoDataFrame,
    output_format: OutputFormat,
) -> gpd.GeoDataFrame | dict[str, Any] | list[dict]:
    """Convert results to requested output format."""
    
    # Build records
    records = [props.to_dict() for props in results]

    if output_format == "dict":
        return records

    elif output_format == "geojson":
        features = []
        for i, record in enumerate(records):
            geom = original_gdf.geometry.iloc[i]
            features.append({
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": record,
            })
        return {
            "type": "FeatureCollection",
            "features": features,
        }

    else:  # gdf (default)
        df = pd.DataFrame(records)
        return gpd.GeoDataFrame(
            df,
            geometry=original_gdf.geometry.values,
            crs=original_gdf.crs,
        )