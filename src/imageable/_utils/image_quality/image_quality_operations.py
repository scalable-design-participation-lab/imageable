"""
Image quality metrics for street view imagery.

These functions quantify how usable a street view image is for downstream
tasks such as height estimation: how much structure the line detector finds,
how much of the facade, the sky and the ground are visible, and how sharp or
cluttered the image is.

The model backed metrics receive an already loaded wrapper as an argument, so
importing this module does not instantiate or download any model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from imageable._models.huggingface.floor_sky_ratio_calculator import FloorSkyRatioCalculator
    from imageable._models.huggingface.segformer_segmentation import SegformerSegmentationWrapper
    from imageable._models.lcnn.lcnn_wrapper import LCNNWrapper
    from imageable._models.vpts.vpts_wrapper import VPTSWrapper

# Number of intensity levels of an 8 bit grayscale image.
GRAY_LEVELS = 256

# Weights used to convert an RGB image into grayscale.
RED_LUMA_WEIGHT = 0.299
GREEN_LUMA_WEIGHT = 0.587
BLUE_LUMA_WEIGHT = 0.114

# Labels produced by the remapping below.
FACADE_LABEL = 1
SKY_LABEL = 2
GROUND_LABEL = 11

# ADE20K classes (1 indexed) grouped into facade, sky and ground. Classes that
# are not listed here collapse to 0 when the remapping is applied.
FACADE_REMAPPING: dict[int, int] = {
    2: FACADE_LABEL,  # building
    26: FACADE_LABEL,  # house
    3: SKY_LABEL,  # sky
    4: GROUND_LABEL,  # floor
    7: GROUND_LABEL,  # road
    10: GROUND_LABEL,  # grass
    12: GROUND_LABEL,  # sidewalk
    14: GROUND_LABEL,  # earth
}


def get_grayscale_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert an RGB image to grayscale using the standard luma weights.

    Parameters
    ----------
    image
        An RGB image of shape (H, W, 3).

    Returns
    -------
    gray_image
        The grayscale image of shape (H, W) as unsigned 8 bit integers.
    """
    gray_image: NDArray[np.uint8] = (
        RED_LUMA_WEIGHT * image[:, :, 0] + GREEN_LUMA_WEIGHT * image[:, :, 1] + BLUE_LUMA_WEIGHT * image[:, :, 2]
    ).astype(np.uint8)

    return gray_image


def get_image_entropy(image: NDArray[np.uint8]) -> float:
    """
    Compute the Shannon entropy of the grayscale histogram of an image.

    Low entropy images are flat or featureless, while high entropy images are
    visually cluttered.

    Parameters
    ----------
    image
        An RGB image of shape (H, W, 3).

    Returns
    -------
    entropy
        The entropy of the intensity distribution, in bits.
    """
    gray_image = get_grayscale_image(image)
    counts = np.bincount(gray_image.ravel(), minlength=GRAY_LEVELS)
    probabilities = counts / gray_image.size
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def get_sharpness(image: NDArray[np.uint8]) -> float:
    """
    Compute the sharpness of an image as the variance of its Laplacian.

    Blurry images concentrate their intensity changes over larger regions, and
    therefore yield lower values than sharp ones.

    Parameters
    ----------
    image
        An RGB image of shape (H, W, 3).

    Returns
    -------
    sharpness
        The variance of the Laplacian of the grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_refinement_success(image_metadata: dict[str, Any]) -> bool | None:
    """
    Read whether the camera adjustment of an image succeeded.

    Parameters
    ----------
    image_metadata
        The metadata dictionary saved alongside the image.

    Returns
    -------
    refinement_success
        The value of the "adjustment_success" flag, or None when the metadata
        does not contain it.
    """
    return image_metadata.get("adjustment_success")


def get_lines_and_scores(
    image: NDArray[np.uint8],
    lcnn_model: LCNNWrapper,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Detect the line segments of an image and their confidence scores.

    Few detected lines, or lines with low scores, indicate an image with little
    usable structure.

    Parameters
    ----------
    image
        An RGB image of shape (H, W, 3).
    lcnn_model
        A loaded LCNN wrapper.

    Returns
    -------
    lines_and_scores
        A tuple (lines, scores) with the post-processed line segments and their
        post-processed confidence scores.
    """
    lcnn_results: dict[str, Any] = lcnn_model.predict(image)
    lines: NDArray[np.floating] = lcnn_results["processed_lines"]
    scores: NDArray[np.floating] = lcnn_results["processed_scores"]

    return lines, scores


def get_vanishing_points(
    image: NDArray[np.uint8],
    vpts_model: VPTSWrapper,
    fov: float = 90.0,
    seed: int | None = 42,
    length_threshold: float = 60,
) -> NDArray[np.floating]:
    """
    Obtain the 2d vanishing points of an image.

    Parameters
    ----------
    image
        An RGB image of shape (H, W, 3).
    vpts_model
        A vanishing point detector wrapper.
    fov
        Field of view used to acquire the image, in degrees.
    seed
        Seed of the detector, for reproducible results.
    length_threshold
        Minimum length of the segments used by the detector.

    Returns
    -------
    vanishing_points
        An array of shape (3, 2) with the 2d vanishing points. The detector
        returns an array of zeros when it fails to find any.
    """
    vpts_dictionary: dict[str, Any] = vpts_model.predict(
        image,
        FOV=fov,
        seed=seed,
        length_threshold=length_threshold,
    )
    vanishing_points: NDArray[np.floating] = vpts_dictionary["vpts_2d"]

    return vanishing_points


def get_sky_and_ground_ratios(
    image: NDArray[np.uint8],
    sky_ground_model: FloorSkyRatioCalculator,
    conf: float = 0.5,
) -> tuple[float, float]:
    """
    Compute the fraction of the image occupied by the sky and by the ground.

    Both ratios are restricted to the horizontal extent of the facade closest
    to the center of the image, whenever a facade is detected.

    Parameters
    ----------
    image
        An RGB image of shape (H, W, 3).
    sky_ground_model
        A loaded sky and floor ratio calculator.
    conf
        Confidence threshold of the detector.

    Returns
    -------
    ratios
        A tuple (sky_ratio, ground_ratio).
    """
    ratios = sky_ground_model.predict(image, conf=conf)
    sky_ratio: float = ratios["sky_ratio"]  # type: ignore[assignment]
    ground_ratio: float = ratios["floor_ratio"]  # type: ignore[assignment]

    return float(sky_ratio), float(ground_ratio)


def get_facade_ratio(
    image: NDArray[np.uint8],
    segmentation_model: SegformerSegmentationWrapper,
    remapping_dict: dict[int, int] | None = None,
) -> float:
    """
    Compute the fraction of the image occupied by building facades.

    A low value means the building of interest is barely visible, either
    because it is occluded or because the camera is not pointing at it.

    Parameters
    ----------
    image
        An RGB image of shape (H, W, 3).
    segmentation_model
        A loaded Segformer segmentation wrapper.
    remapping_dict
        Mapping from ADE20K classes (1 indexed) to facade, sky and ground
        labels. Defaults to FACADE_REMAPPING.

    Returns
    -------
    facade_ratio
        The fraction of pixels labelled as facade.
    """
    if remapping_dict is None:
        remapping_dict = FACADE_REMAPPING

    seg_raw: NDArray[np.int_] = segmentation_model.predict(image)
    seg_raw = seg_raw.squeeze().astype("uint8") + 1
    seg: np.ndarray[Any, np.dtype[np.int_]] = segmentation_model.remap_labels(seg_raw, remapping_dict)

    return float((seg == FACADE_LABEL).mean())
