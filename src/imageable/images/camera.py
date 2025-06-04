from dataclasses import asdict, dataclass

MAX_DIMENSION = 640
MIN_FOV = 10
MAX_FOV = 120


@dataclass
class CameraParameters:
    """
    Parameters used for building image collection with.

    Parameters
    ----------
    fov
        Field of view in degrees. The maximum fov allowed is 120. Default is 90.
    heading
        Heading angle in degrees. Default is 0
    pitch
        Pitch (vertical angle) in degrees. Default is 0.
    width
        Output image width. Must not exceed 640. Default is 640.
    height
        Output image height. Must not exceed 640. Default is 640.
    """

    # Field of view
    fov: float = 90
    # camera heading
    heading: float = 0
    # pitch
    pitch: float = 0
    # img width
    width: int = 640
    # img height
    height: int = 640

    def __post_init__(self) -> None:
        """Validate parameter ranges after initialization."""
        if self.width > MAX_DIMENSION:
            msg = "width cannot be greater than 640"
            raise ValueError(msg)

        if self.height > MAX_DIMENSION:
            msg = "height cannot be greater than 640"
            raise ValueError(msg)

        if self.fov > MAX_FOV or self.fov < MIN_FOV:
            msg = "FOV should be between 10 and 120"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, float | int | str]:
        """
        Convert camera parameters to a dictionary.

        Returns
        -------
        dict
            Dictionary containing camera parameters.
        """
        return asdict(self)
