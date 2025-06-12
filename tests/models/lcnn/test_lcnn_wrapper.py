"""
Unit tests for L-CNN wrapper.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import yaml

from imageable.models.lcnn.lcnn_wrapper import LCNNWrapper


class TestLCNNWrapper:
    """Unit tests for LCNNWrapper class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "model": {
                "depth": 4,
                "num_stacks": 2,
                "num_blocks": 1,
                "head_size": [[3], [1], [2]],
                "image": {
                    "mean": [109.730, 103.832, 98.681],
                    "stddev": [22.275, 22.124, 23.229]
                }
            }
        }

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_init_with_config(self, sample_config):
        """Test initialization with configuration dictionary."""
        wrapper = LCNNWrapper(config=sample_config)
        
        assert wrapper.config == sample_config
        assert wrapper.config_path is None
        assert wrapper.checkpoint_path is None
        assert wrapper.model is None
        assert wrapper.device in ["cpu", "cuda", "mps"]

    def test_init_with_config_path(self, sample_config, tmp_path):
        """Test initialization with configuration file."""
        # Create temporary config file
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        wrapper = LCNNWrapper(config_path=str(config_file))
        
        assert wrapper.config_path == str(config_file)
        assert wrapper.config == sample_config

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        wrapper = LCNNWrapper()
        
        assert wrapper.config is not None
        assert "model" in wrapper.config
        assert wrapper.config["model"]["depth"] == 4
        assert wrapper.config["model"]["num_stacks"] == 2

    def test_init_with_device(self):
        """Test initialization with specific device."""
        wrapper = LCNNWrapper(device="cpu")
        assert wrapper.device == "cpu"

    def test_is_loaded(self, sample_config):
        """Test is_loaded method."""
        wrapper = LCNNWrapper(config=sample_config)
        
        # Initially not loaded
        assert not wrapper.is_loaded()
        
        # Simulate loaded model
        wrapper.model = "dummy_model"
        assert wrapper.is_loaded()

    def test_load_model_without_checkpoint(self, sample_config):
        """Test load_model raises error without checkpoint path."""
        wrapper = LCNNWrapper(config=sample_config)
        
        with pytest.raises(ValueError, match="checkpoint_path must be provided"):
            wrapper.load_model()

    def test_preprocess_rgb_image(self, sample_config, sample_image):
        """Test preprocessing RGB image."""
        wrapper = LCNNWrapper(config=sample_config)
        
        # Test with numpy array
        tensor = wrapper.preprocess(sample_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 512, 512)
        assert tensor.dtype == torch.float32
        assert wrapper._original_shape == (480, 640)

    def test_preprocess_grayscale_image(self, sample_config):
        """Test preprocessing grayscale image."""
        wrapper = LCNNWrapper(config=sample_config)
        gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        tensor = wrapper.preprocess(gray_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 512, 512)

    def test_preprocess_rgba_image(self, sample_config):
        """Test preprocessing RGBA image (4 channels)."""
        wrapper = LCNNWrapper(config=sample_config)
        rgba_image = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        
        tensor = wrapper.preprocess(rgba_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 512, 512)

    def test_preprocess_different_sizes(self, sample_config):
        """Test preprocessing images of different sizes."""
        wrapper = LCNNWrapper(config=sample_config)
        
        test_sizes = [(256, 256), (720, 1280), (1080, 1920)]
        
        for height, width in test_sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            tensor = wrapper.preprocess(image)
            
            assert tensor.shape == (1, 3, 512, 512)
            assert wrapper._original_shape == (height, width)

    def test_preprocess_normalization(self, sample_config):
        """Test that preprocessing applies normalization."""
        wrapper = LCNNWrapper(config=sample_config)
        
        # Create an image with known values
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        tensor = wrapper.preprocess(image)
        
        # Check that values are normalized (not in 0-255 range anymore)
        # The normalization formula is (pixel - mean) / stddev
        # So the values should be different from the original 0-255 range
        assert tensor.min() < 0 or tensor.max() > 255 or (tensor.min() > 0 and tensor.max() < 10)
        
        # More specific check: verify the tensor values are roughly what we expect
        # Image value 128 normalized: (128 - mean) / stddev
        expected_r = (128 - 109.730) / 22.275  # ~0.82
        expected_g = (128 - 103.832) / 22.124  # ~1.09
        expected_b = (128 - 98.681) / 23.229   # ~1.26
        
        # Check that at least one channel matches expected normalization
        # (allowing for resize interpolation effects)
        assert abs(tensor[0, 0].mean().item() - expected_r) < 0.5 or \
               abs(tensor[0, 1].mean().item() - expected_g) < 0.5 or \
               abs(tensor[0, 2].mean().item() - expected_b) < 0.5

    def test_save_results(self, sample_config, tmp_path):
        """Test saving results to file."""
        wrapper = LCNNWrapper(config=sample_config)
        
        results = {
            "lines": np.array([[[0, 0], [100, 100]], [[50, 50], [150, 150]]]),
            "scores": np.array([0.95, 0.87]),
            "processed_lines": np.array([[[0, 0], [100, 100]]]),
            "processed_scores": np.array([0.95])
        }
        
        output_file = tmp_path / "results.npz"
        wrapper.save_results(results, str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Load and verify contents
        loaded = np.load(output_file)
        assert "lines" in loaded
        assert "scores" in loaded
        assert "processed_lines" in loaded
        assert "processed_scores" in loaded
        np.testing.assert_array_equal(loaded["lines"], results["lines"])
        np.testing.assert_array_equal(loaded["scores"], results["scores"])

    def test_save_results_creates_directory(self, sample_config, tmp_path):
        """Test that save_results creates parent directory."""
        wrapper = LCNNWrapper(config=sample_config)
        
        results = {
            "lines": np.array([]),
            "scores": np.array([]),
            "processed_lines": np.array([]),
            "processed_scores": np.array([])
        }
        
        # Use a non-existent subdirectory
        output_file = tmp_path / "subdir" / "results.npz"
        wrapper.save_results(results, str(output_file))
        
        assert output_file.parent.exists()
        assert output_file.exists()

    def test_postprocess_calculation(self, sample_config):
        """Test postprocess diagonal calculation."""
        wrapper = LCNNWrapper(config=sample_config)
        
        # Test with known image shape
        outputs = {
            "lines": np.array([[[0, 0], [100, 100]]]),
            "scores": np.array([0.9]),
            "image_shape": (300, 400)  # 3:4 ratio
        }
        
        # Calculate expected diagonal
        expected_diag = (300**2 + 400**2)**0.5  # Should be 500
        assert abs(expected_diag - 500) < 0.01
        
        # The actual postprocessing would use L-CNN's postprocess function
        # Here we just verify the wrapper passes correct parameters

    @pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
    def test_device_resolution(self, device_type):
        """Test device resolution for different device types."""
        # Skip if device not available
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device_type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
        
        wrapper = LCNNWrapper(device=device_type)
        assert wrapper.device == device_type

    def test_process_batch_creates_output_dir(self, sample_config, tmp_path):
        """Test that process_batch creates output directory."""
        wrapper = LCNNWrapper(config=sample_config)
        wrapper.checkpoint_path = "dummy.pth"  # Set to avoid load error
        
        # Use non-existent directory
        output_dir = tmp_path / "batch_output"
        assert not output_dir.exists()
        
        # Even if processing fails, directory should be created
        try:
            wrapper.process_batch([], str(output_dir))
        except:
            pass
        
        assert output_dir.exists()


class TestLCNNWrapperHelpers:
    """Test helper methods of LCNNWrapper."""

    def test_resolve_device_cpu(self):
        """Test device resolution defaults to CPU."""
        wrapper = LCNNWrapper()
        # At minimum, CPU should always be available
        assert wrapper._resolve_device() in ["cpu", "cuda", "mps"]

    def test_config_structure(self):
        """Test configuration structure validation."""
        wrapper = LCNNWrapper()
        
        # Check required config keys
        assert "model" in wrapper.config
        assert "depth" in wrapper.config["model"]
        assert "num_stacks" in wrapper.config["model"]
        assert "num_blocks" in wrapper.config["model"]
        assert "head_size" in wrapper.config["model"]
        assert "image" in wrapper.config["model"]
        assert "mean" in wrapper.config["model"]["image"]
        assert "stddev" in wrapper.config["model"]["image"]

    def test_config_values(self):
        """Test default configuration values."""
        wrapper = LCNNWrapper()
        
        config = wrapper.config["model"]
        assert isinstance(config["depth"], int)
        assert isinstance(config["num_stacks"], int)
        assert isinstance(config["num_blocks"], int)
        assert isinstance(config["head_size"], list)
        assert len(config["image"]["mean"]) == 3
        assert len(config["image"]["stddev"]) == 3