from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
from PIL import Image

from imageable._models.huggingface.floor_sky_ratio_calculator import FloorSkyRatioCalculator
from imageable._models.huggingface.segformer_segmentation import SegformerSegmentationWrapper
from imageable._models.lcnn.lcnn_wrapper import LCNNWrapper
from imageable._utils.image_quality.image_quality_operations import (
    get_facade_ratio,
    get_lines_and_scores,
    get_sharpness,
    get_sky_and_ground_ratios,
)
from imageable.core.building_data import (
    ProviderType,
    _extract_building_data_core,
    _load_geojson_to_gdf,
)


@dataclass
class FrameworkAttributes:
    building_height: float | None = None
    isoperimetric_quotient: float | None = None
    complexity: float | None = None
    vegetation_pct: float | None = None
    sharpness: float | None = None
    facade_ratio: float | None = None
    sky_ratio: float | None = None
    ground_ratio: float | None = None
    refinement_success: bool | None = None
    n_lines: int | None = None
    line_confidence: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        object_dictionary = {}
        object_dictionary["building_height"] = self.building_height
        object_dictionary["isoperimetric_quotient"] = self.isoperimetric_quotient
        object_dictionary["complexity"] = self.complexity
        object_dictionary["vegetation_pct"] = self.vegetation_pct
        object_dictionary["sharpness"] = self.sharpness
        object_dictionary["facade_ratio"] = self.facade_ratio
        object_dictionary["sky_ratio"] = self.sky_ratio
        object_dictionary["ground_ratio"] = self.ground_ratio
        object_dictionary["refinement_success"] = self.refinement_success
        object_dictionary["n_lines"] = self.n_lines
        object_dictionary["line_confidence"] = self.line_confidence

        return object_dictionary


def get_framework_attributes_from_gdf(
    gdf: gpd.GeoDataFrame,
    image_key: str | None,
    *,
    provider: ProviderType = "google_street_view",
    id_column: str | None = None,
    neighbor_radius: float = 100.0,
    verbose: bool = False,
    all_city_buildings_gdf: gpd.GeoDataFrame | None = None,
    pictures_directory: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> list[Dict[str, Any]]:
    output_format = "dict"
    height_mode = "cluster_then_predict"

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Expected GeoDataFrame, got {type(gdf).__name__}")

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty")

    images_dir = Path(images_dir) if images_dir is not None else None

    results = _extract_building_data_core(
        gdf=gdf,
        image_key=image_key,
        images_dir=images_dir,
        id_column=id_column,
        neighbor_radius=neighbor_radius,
        output_format=output_format,
        height_mode=height_mode,
        provider=provider,
        verbose=verbose,
        all_city_buildings_gdf=all_city_buildings_gdf,
        pictures_directory=pictures_directory,
    )

    framework_results = []
    framework_images_dir = images_dir
    if framework_images_dir is None and pictures_directory is not None:
        framework_images_dir = Path(pictures_directory)

    lcnn_model = None
    sky_ground_model = None
    facade_model = None

    for result in results:
        building_height = result.get("building_height", None)
        isoperimetric_quotient = result.get("isoperimetric_quotient", None)
        complexity = result.get("complexity", None)
        materials = result.get("material_percentages", None) or {}
        materials = materials.get("percentages", materials)
        vegetation_pct = materials.get("leaf", 0.0)

        building_id = result.get("building_id", None)
        refinement_success = None
        sharpness = None
        facade_ratio = None
        sky_ratio = None
        ground_ratio = None
        n_lines = None
        line_confidence = None

        if building_id is not None and framework_images_dir is not None:
            image_path = framework_images_dir / building_id / "image.jpg"
            image_metadata_path = framework_images_dir / building_id / "metadata.json"

            if image_path.exists():
                if image_metadata_path.exists():
                    with image_metadata_path.open() as metadata_file:
                        data = json.load(metadata_file)
                    refinement_success = data.get("adjustment_success", None)

                image = np.array(Image.open(image_path).convert("RGB"))

                if lcnn_model is None:
                    assets_dir = Path(__file__).resolve().parents[1] / "assets"
                    lcnn_model = LCNNWrapper(config_path=str(assets_dir / "wireframe.yaml"))
                    sky_ground_model = FloorSkyRatioCalculator()
                    facade_model = SegformerSegmentationWrapper(
                        model_name="nvidia/segformer-b5-finetuned-ade-640-640",
                        palette_path=str(assets_dir / "ade20k_palette.json"),
                    )

                sharpness = get_sharpness(image)
                facade_ratio = get_facade_ratio(image, facade_model)
                lines, scores = get_lines_and_scores(image, lcnn_model)
                n_lines = len(lines)
                line_confidence = float(np.mean(scores)) if len(scores) > 0 else None
                sky_ratio, ground_ratio = get_sky_and_ground_ratios(image, sky_ground_model)

        framework_results.append(
            FrameworkAttributes(
                building_height=building_height,
                isoperimetric_quotient=isoperimetric_quotient,
                complexity=complexity,
                vegetation_pct=vegetation_pct,
                sharpness=sharpness,
                facade_ratio=facade_ratio,
                sky_ratio=sky_ratio,
                ground_ratio=ground_ratio,
                refinement_success=refinement_success,
                n_lines=n_lines,
                line_confidence=line_confidence,
            ).to_dict()
        )

    return framework_results


def get_framework_attributes_from_geojson(
    source: str | Path | dict[str, Any],
    image_key: str | None,
    *,
    provider: ProviderType = "google_street_view",
    id_column: str | None = None,
    neighbor_radius: float = 100.0,
    verbose: bool = False,
    all_city_buildings_gdf: str | Path | dict[str, Any] | gpd.GeoDataFrame | None = None,
    pictures_directory: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> list[Dict[str, Any]]:
    gdf = _load_geojson_to_gdf(source)

    if all_city_buildings_gdf is not None:
        all_city_buildings_gdf = _load_geojson_to_gdf(all_city_buildings_gdf)

    return get_framework_attributes_from_gdf(
        gdf,
        image_key,
        provider=provider,
        id_column=id_column,
        neighbor_radius=neighbor_radius,
        verbose=verbose,
        all_city_buildings_gdf=all_city_buildings_gdf,
        pictures_directory=pictures_directory,
        images_dir=images_dir,
    )


def get_framework_attributes_from_file(
    footprints_path: str | Path,
    images_dir: str | Path,
    *,
    id_column: str | None = None,
    neighbor_radius: float = 100.0,
    verbose: bool = False,
    all_city_buildings_gdf: str | Path | dict[str, Any] | gpd.GeoDataFrame | None = None,
) -> list[Dict[str, Any]]:
    footprints_path = Path(footprints_path)
    images_dir = Path(images_dir)

    if not footprints_path.exists():
        raise FileNotFoundError(f"Footprints file not found: {footprints_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    gdf = _load_geojson_to_gdf(footprints_path)

    if all_city_buildings_gdf is not None:
        all_city_buildings_gdf = _load_geojson_to_gdf(all_city_buildings_gdf)

    return get_framework_attributes_from_gdf(
        gdf,
        image_key=None,
        id_column=id_column,
        neighbor_radius=neighbor_radius,
        verbose=verbose,
        all_city_buildings_gdf=all_city_buildings_gdf,
        images_dir=images_dir,
    )
