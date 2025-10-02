import numpy as np

def get_material_labels() -> list[str]:
    """
    Return the canonical list of material class names.

    Returns
    -------
    list[str]
        List of 20 material classes in fixed order.
    """
    return [
        "asphalt",
        "concrete",
        "metal",
        "road_marking",
        "fabric_leather",
        "glass",
        "plaster",
        "plastic",
        "rubber",
        "sand",
        "gravel",
        "ceramic",
        "cobblestone",
        "brick",
        "grass",
        "wood",
        "leaf",
        "water",
        "human_body",
        "sky",
    ]


def get_material_palette() -> np.ndarray:
    """
    Return the color palette for the 20 material classes.

    Returns
    -------
    np.ndarray
        Array of shape (20,3) with RGB values [0-255].
    """
    return np.array([
        [ 44, 160,  44],  # asphalt
        [ 31, 119, 180],  # concrete
        [255, 127,  14],  # metal
        [214,  39,  40],  # road marking
        [140,  86,  75],  # fabric, leather
        [127, 127, 127],  # glass
        [188, 189,  34],  # plaster
        [255, 152, 150],  # plastic
        [ 23, 190, 207],  # rubber
        [174, 199, 232],  # sand
        [196, 156, 148],  # gravel
        [197, 176, 213],  # ceramic
        [247, 182, 210],  # cobblestone
        [199, 199, 199],  # brick
        [219, 219, 141],  # grass
        [158, 218, 229],  # wood
        [ 57,  59, 121],  # leaf
        [107, 110, 207],  # water
        [156, 158, 222],  # human body
        [ 99, 121,  57],  # sky
    ], dtype=np.uint8)


def get_class_count() -> int:
    """
    Return the number of material classes.

    Returns
    -------
    int
        Number of classes (20).
    """
    return len(get_material_labels())