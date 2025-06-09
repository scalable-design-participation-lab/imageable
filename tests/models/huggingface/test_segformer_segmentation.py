import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from imageable.models.huggingface.segformer_segmentation import (
    SegformerSegmentationWrapper,
)


@pytest.fixture(scope="module")
def wrapper():
    return SegformerSegmentationWrapper("nvidia/segformer-b5-finetuned-ade-640-640")


def test_load_model(wrapper):
    wrapper.load_model()
    assert wrapper.is_loaded()
    assert wrapper.model is not None
    assert wrapper.processor is not None


def test_predict_output_shape(wrapper):
    wrapper.load_model()

    dummy_image = Image.new("RGB", (640, 640), color=(255, 255, 255))

    prediction = wrapper.predict(dummy_image)

    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (640, 640)
    assert prediction.dtype == np.int64 or prediction.dtype == np.int32


def test_remap_labels(wrapper):
    seg = np.array([[2, 3], [5, 12]])
    mapping = {2: 1, 3: 2, 5: 2, 12: 11}

    expected = np.array([[1, 2], [2, 11]])
    result = wrapper.remap_labels(seg, mapping)

    assert np.array_equal(result, expected)


def test_colorize(wrapper):
    seg = np.array([[1, 2], [3, 4]])
    img = wrapper.colorize(seg)

    assert isinstance(img, Image.Image)
    assert img.mode == "P"
    assert img.size == (2, 2)
