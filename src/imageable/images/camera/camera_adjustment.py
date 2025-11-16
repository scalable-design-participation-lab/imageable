from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

from imageable.images.camera.building_observation import ObservationPointEstimator
from imageable.images.camera.camera_parameters import CameraParameters
from imageable.images.download import _save_metadata, fetch_image
from imageable.models.huggingface.floor_sky_ratio_calculator import FloorSkyRatioCalculator


class CameraParametersEstimator:
    """Apply observation point estimation to obtain initial camera parameters."""

    def __init__(self, polygon: Polygon) -> None:
        self.polygon = polygon
        self.image_width = 640
        self.image_height = 640

    def estimate_first_parameters(self, buffer_constant: float = 2.5e5) -> CameraParameters:
        """
        Estimate initial camera parameters based on the observation point.

        Parameters
        ----------
        buffer_constant
            Buffer to apply around the building for retrieving the street network
            from which the building could be observed.

        Returns
        -------
        camera_parameters
            The estimated initial camera parameters.
        """
        observation_point_estimator = ObservationPointEstimator(self.polygon)
        observation_point, _, heading, _ = observation_point_estimator.get_observation_point(
            buffer_constant=buffer_constant, true_north=True
        )

        # As a fallback use the centroid of the polygon as observation point.
        if observation_point is None or heading is None:
            observation_point = (self.polygon.centroid.x, self.polygon.centroid.y)
            heading = 0

        return CameraParameters(
            longitude=observation_point[0],
            latitude=observation_point[1],
            heading=heading,
            pitch=0,
            fov=90,
            width=640,
            height=640,
        )


class CameraParametersRefiner:
    """
    Refine camera parameters to obtain images where the
    full façade of the building is visible.
    """

    DEFAULT_IMAGE_NAME = "image"
    EXTENSION = ".jpg"
    MAX_PITCH_CHANGE = 90
    MAX_FOV_CHANGE = 90
    MIN_FLOOR_RATIO = 0.00001
    MIN_SKY_RATIO = 0.1

    def __init__(self, polygon: Polygon, model: FloorSkyRatioCalculator = None) -> None:
        self.polygon = polygon
        self.image_width = 640
        self.image_height = 640

        if model is None:
            self.model = FloorSkyRatioCalculator()
            self.model.load_model()

    # ruff : noqa: PLR0913
    def adjust_parameters(
        self,
        api_key: str,
        max_number_of_images: int = 10,
        polygon_buffer_constant: float = 2.5e5,
        pictures_directory: str | None = None,
        save_reel: bool = False,
        overwrite_images: bool = False,
        confidence_detection: float = 0.5,
    ) -> tuple[CameraParameters, bool, np.ndarray | None]:
        """
        Obtain CameraParameters for a view where the
        full façade of the buildings is visible by adjusting the pitch and fov
        of the camera.

        Parameters
        ----------
        api_key
            The GSV Static API Key to use for collecting images.
        max_number_of_images
            The maximum number of images to take as part of the image refinement procedure.
            Default is 10.
        polygon_buffer_constant
            A constant to estimate the buffer size around the building polygon. This is useful for
            estimating the observation point at which images are taken. Default is 2.5e5.
        pictures_directory
            The directory where the images will be stored. If None no images will be stored. Default is
            None
        save_reel
            This parameter interacts with the picture_path parameter.
            If pictures_directory is not None and save_reel is True, all images taken will be saved.
            If pictures_directory is not None and save_reel is False, only the last image will be saved.
            If pictures_directory is None, no images will be saved. Default is False.

        Returns
        -------
        camera_parameters
            The camera parameters that were used to obtain the last image.
        view_obtained
            A boolean indicating whether a full view of the building was obtained.
        image
            The last image taken. If pictures_directory is None, this will be None.
            This case is useful for when the user just wants the parameters to obtain
            the image by themselves.
        """
        # Estimate the initial parameters
        params_estimator = CameraParametersEstimator(self.polygon)
        camera_parameters = params_estimator.estimate_first_parameters(buffer_constant=polygon_buffer_constant)
        # Now we will adjust the parameters to ensure the full façade is visible

        view_obtained = False
        images_taken = 0
        self.pitch_delta = self.MAX_PITCH_CHANGE / max_number_of_images
        self.fov_delta = self.MAX_FOV_CHANGE / max_number_of_images

        image = None
        progress = tqdm(total=max_number_of_images)

        while not view_obtained:
            # Case where the maximum number of images has been taken.
            if images_taken + 1 >= max_number_of_images:
                if pictures_directory is not None:
                    # Fetch the image and save it
                    image, metadata = fetch_image(
                        api_key,
                        camera_parameters,
                        Path(pictures_directory) / (self.DEFAULT_IMAGE_NAME + self.EXTENSION),
                        overwrite_image=overwrite_images,
                    )
                    # We return the camera parameters
                    # and the image
                    return camera_parameters, view_obtained, image

                # We just return the camera_parameters
                # the user will have to fetch the image
                return camera_parameters, view_obtained, None

            # Fetch the image
            # If the user asked to save the reel save the image
            if pictures_directory is not None and save_reel:
                image, metadata = fetch_image(
                    api_key,
                    camera_parameters,
                    Path(pictures_directory) / f"{images_taken}" / (self.DEFAULT_IMAGE_NAME + self.EXTENSION),
                    overwrite_image=overwrite_images,
                )
            else:
                image, metadata = fetch_image(api_key, camera_parameters, None, overwrite_image=overwrite_images)

            if image is not None:
                image_bgr = image[..., ::-1]
                # Obtain the sky and floor ratios
                ratios_dictionary = self.model.predict(image_bgr, conf=confidence_detection)
                sky_ratio = ratios_dictionary["sky_ratio"]
                floor_ratio = ratios_dictionary["floor_ratio"]
                if (sky_ratio >= 0 and sky_ratio <= self.MIN_SKY_RATIO) and floor_ratio > self.MIN_FLOOR_RATIO:
                    # We increase the pitch
                    camera_parameters.pitch += self.pitch_delta
                    camera_parameters.pitch = max(-90, min(90, camera_parameters.pitch))
                    images_taken += 1
                    progress.update(1)

                elif (sky_ratio > self.MIN_SKY_RATIO) and (floor_ratio >= 0 and floor_ratio <= self.MIN_FLOOR_RATIO):
                    # Lower the pitch
                    camera_parameters.pitch -= self.pitch_delta
                    camera_parameters.pitch = max(-90, min(90, camera_parameters.pitch))
                    images_taken += 1
                    progress.update(1)

                elif (sky_ratio >= 0 and sky_ratio <= self.MIN_SKY_RATIO) and (
                    floor_ratio >= 0 and floor_ratio <= self.MIN_FLOOR_RATIO
                ):
                    # Increase the FOV
                    camera_parameters.fov = min(camera_parameters.fov + self.fov_delta, 120)
                    images_taken += 1
                    progress.update(1)

                elif sky_ratio > self.MIN_SKY_RATIO and floor_ratio > self.MIN_FLOOR_RATIO:
                    # Success case
                    view_obtained = True
                    if pictures_directory is not None:
                        # We save the image and metadata
                        pictures_directory = Path(pictures_directory)
                        pictures_directory.mkdir(parents=True, exist_ok=True)
                        path_to_image = pictures_directory / (self.DEFAULT_IMAGE_NAME + self.EXTENSION)
                        path_to_metadata = pictures_directory / "metadata.json"
                        if overwrite_images or not path_to_image.exists():
                            Image.fromarray(image).save(path_to_image)
                            _save_metadata(path_to_metadata, metadata.to_dict())
            else:
                # If the image is None, we return the camera parameters
                # and None as the image
                return camera_parameters, view_obtained, None

        progress.close()
        # You get here if a view was obtained successfully
        return camera_parameters, view_obtained, image
