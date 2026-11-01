from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imageable._models.huggingface.yolo_segmentation import (
    YOLOSegmentationWrapper,
)


@pytest.fixture(scope="module")
def wrapper():
    return YOLOSegmentationWrapper()


def test_load_model(wrapper):
    wrapper.load_model()
    assert wrapper.is_loaded()
    assert wrapper.model is not None


def test_predict_output_shape(wrapper):
    wrapper.load_model()

    dummy_image = np.full((480, 640, 3), 255, dtype=np.uint8)

    prediction = wrapper.predict(dummy_image)

    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (480, 640)
    assert prediction.dtype in (np.int64, np.int32)


def test_postprocess_labels(wrapper):
    names = {0: "sky", 1: "facade", 2: "window", 3: "vehicle", 4: "road"}
    masks = torch.tensor(
        [
            [[1, 1, 1], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 1]],
        ],
        dtype=torch.float32,
    )
    outputs = SimpleNamespace(
        orig_shape=(2, 3),
        names=names,
        boxes=SimpleNamespace(cls=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])),
        masks=SimpleNamespace(data=masks),
    )

    expected = np.array([[2, 1, 2], [1, 0, 11]])
    result = wrapper.postprocess(outputs)

    assert np.array_equal(result, expected)
