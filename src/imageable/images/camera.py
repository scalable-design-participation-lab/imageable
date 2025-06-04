from dataclasses import dataclass, asdict


@dataclass
class CameraParameters:

    """
    Encapsulates parameters needed for retrieving images through the
    Google Street View API.

    Parameters
    -----------

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

    #Field of view
    fov:float = 90
    #camera heading
    heading:float = 0
    #pitch
    pitch:float = 0
    #img width
    width:int = 640
    #img height
    height:int = 640


    def __post_init__(self):
        
        if(self.width > 640):
            raise ValueError(f"width ({self.width}) cannot be greater than 640")
        
        if(self.height > 640):
            raise ValueError(f"width ({self.width}) cannot be greater than 640")
        if(self.fov >120 or self.fov < 10):
            raise ValueError(f"FOV values should be in the interval (0, 120)")
    
    def to_dict(self)->dict[str, float|int|str]:
        """
        Convert camera parameters to a dictionary.

        Returns
        -------
        dict
            Dictionary containing camera parameters.
        """
        return asdict(self)
    


    


