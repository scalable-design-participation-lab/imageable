from typing import Any

import numpy as np
import pytest
import skimage.io
import torch

from imageable.models.lcnn.lcnn_wrapper import LCNNWrapper


@pytest.fixture
def sample_config():
    return {
        "model": {
            "depth": 4,
            "num_stacks": 2,
            "num_blocks": 1,
            "head_size": [[3], [1], [2]],
            "image": {
                "mean": [109.730, 103.832, 98.681],
                "stddev": [22.275, 22.124, 23.229],
            },
        }
    }


@pytest.fixture
def sample_image():
    # small dummy image, but with a known shape
    return np.zeros((100, 200, 3), dtype=np.uint8)


def make_dummy_model(dummy_lines, dummy_scores) -> Any:
    """
    Return an object that, when called with input_dict, returns
    {"preds": {"lines": dummy_lines, "score": dummy_scores}}
    """

    class Dummy(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()

        def forward(self, _input_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
            return {"preds": {"lines": dummy_lines, "score": dummy_scores}}

    return Dummy()


def test_predict_scales_correctly(monkeypatch, sample_config, sample_image):
    wrapper = LCNNWrapper(config=sample_config)
    original_shape = sample_image.shape[:2]  # (100, 200)

    dummy_lines = torch.tensor([[[0.0, 0.0, 128.0, 128.0]]])
    dummy_scores = torch.tensor([[0.42]])

    def fake_load(self):
        self.model = make_dummy_model(dummy_lines, dummy_scores)

    monkeypatch.setattr(LCNNWrapper, "load_model", fake_load)

    out = wrapper.predict(sample_image)

    assert set(out) == {"lines", "scores", "processed_lines", "processed_scores"}
    assert isinstance(out["lines"], np.ndarray)
    assert isinstance(out["scores"], np.ndarray)

    assert wrapper._original_shape == original_shape

    expected = np.array([[0.0 * 0.78125, 0.0 * 1.5625, 128.0 * 0.78125, 128.0 * 1.5625]])
    np.testing.assert_allclose(out["lines"], expected, atol=1e-5)

    np.testing.assert_allclose(out["scores"], [0.42], atol=1e-6)


def test_predict_with_image_path(tmp_path, monkeypatch, sample_config, sample_image):
    img_path = tmp_path / "img.png"
    skimage.io.imsave(str(img_path), sample_image)

    wrapper = LCNNWrapper(config=sample_config)
    dummy_lines = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
    dummy_scores = torch.tensor([[0.99]])
    monkeypatch.setattr(
        LCNNWrapper,
        "load_model",
        lambda self: setattr(self, "model", make_dummy_model(dummy_lines, dummy_scores)),
    )

    out = wrapper.predict(str(img_path))

    assert out["lines"].shape == (1, 4)
    assert out["scores"].shape == (1,)
    np.testing.assert_allclose(out["scores"], [0.99], atol=1e-6)
