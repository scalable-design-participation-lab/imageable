
from imageable._models.materials.label_palette import get_class_count, get_material_labels, get_material_palette


def test_label_palette_numbers():
    labels = get_material_labels()
    assert isinstance(labels, list)
    n_labels = len(labels)
    material_colors = get_material_palette()
    n_colors = len(material_colors)

    assert n_labels == n_colors
    assert get_class_count() == n_labels
