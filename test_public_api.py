"""
Test script to verify the public image acquisition API works correctly.

This script tests the standalone image acquisition functionality with mocked
components to avoid actual API calls.

Run with: python test_public_api.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shapely.geometry import Polygon

from imageable import get_image
from imageable._images.acquisition import (
    ImageAcquisitionConfig,
    ImageAcquisitionResult,
    acquire_building_image,
)
from imageable._images.camera.camera_parameters import CameraParameters
from imageable.core.image import load_image


def create_mock_camera() -> CameraParameters:
    """Create a mock camera parameters object."""
    return CameraParameters(
        longitude=-71.05,
        latitude=42.36,
        fov=90,
        heading=45,
        pitch=10,
        width=640,
        height=640,
    )


def create_mock_image() -> np.ndarray:
    """Create a mock image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


def create_mock_acquisition_result(from_cache: bool = False) -> ImageAcquisitionResult:
    """Create a mock ImageAcquisitionResult."""
    return ImageAcquisitionResult(
        image=create_mock_image(),
        camera_params=create_mock_camera(),
        metadata={
            "refinement_success": True,
            "from_cache": from_cache,
            "refinement_iterations": 5,
            "min_floor_ratio": 0.00001,
            "min_sky_ratio": 0.1,
        },
        success=True,
        from_cache=from_cache,
    )


def test_basic_get_image() -> bool:
    """Test 1: Basic get_image() call with full metadata."""
    print("\n" + "="*70)
    print("TEST 1: Basic get_image() with full metadata")
    print("="*70)

    footprint = Polygon([
        (-71.05, 42.36),
        (-71.05, 42.37),
        (-71.04, 42.37),
        (-71.04, 42.36),
    ])

    with patch("imageable.core.image.acquire_building_image") as mock_acquire:
        mock_acquire.return_value = create_mock_acquisition_result()

        try:
            image, camera_params, metadata = get_image("test_key", footprint)  # type: ignore[misc]

            assert image is not None, "Image should not be None"
            assert image.shape == (640, 640, 3), f"Expected shape (640, 640, 3), got {image.shape}"
            assert isinstance(camera_params, CameraParameters), "Should return CameraParameters"
            assert isinstance(metadata, dict), "Metadata should be a dict"
            assert camera_params.fov == 90, f"Expected FOV 90, got {camera_params.fov}"

            print("✓ Image acquired successfully")
            print(f"✓ Image shape: {image.shape}")
            print(f"✓ Camera FOV: {camera_params.fov}°")
            print(f"✓ Camera Pitch: {camera_params.pitch}°")
            print(f"✓ Metadata keys: {list(metadata.keys())}")
            print("✓ TEST 1 PASSED!")
            return True

        except Exception as e:
            print(f"✗ TEST 1 FAILED: {e}")
            return False


def test_get_image_without_metadata() -> bool:
    """Test 2: get_image() returning only the image."""
    print("\n" + "="*70)
    print("TEST 2: get_image() without metadata return")
    print("="*70)

    footprint = Polygon([
        (-71.05, 42.36),
        (-71.05, 42.37),
        (-71.04, 42.37),
        (-71.04, 42.36),
    ])

    with patch("imageable.core.image.acquire_building_image") as mock_acquire:
        mock_acquire.return_value = create_mock_acquisition_result()

        try:
            image = get_image("test_key", footprint, return_metadata=False)

            assert image is not None, "Image should not be None"
            assert isinstance(image, np.ndarray), "Should return numpy array"
            assert image.shape == (640, 640, 3), f"Expected shape (640, 640, 3), got {image.shape}"

            print("✓ Image acquired successfully")
            print(f"✓ Image shape: {image.shape}")
            print(f"✓ Image type: {type(image)}")
            print("✓ TEST 2 PASSED!")
            return True

        except Exception as e:
            print(f"✗ TEST 2 FAILED: {e}")
            return False


def test_fast_mode() -> bool:
    """Test 3: Fast acquisition mode (no refinement)."""
    print("\n" + "="*70)
    print("TEST 3: Fast mode (refine_camera=False)")
    print("="*70)

    footprint = Polygon([
        (-71.05, 42.36),
        (-71.05, 42.37),
        (-71.04, 42.37),
        (-71.04, 42.36),
    ])

    with patch("imageable.core.image.acquire_building_image") as mock_acquire:
        mock_acquire.return_value = create_mock_acquisition_result()

        try:
            image = get_image(
                "test_key",
                footprint,
                refine_camera=False,
                return_metadata=False
            )

            # Check that config was created with refinement disabled
            call_args = mock_acquire.call_args
            config = call_args[0][1]  # Second argument is config

            assert config.min_floor_ratio == 0.0, "Floor ratio should be 0 in fast mode"
            assert config.min_sky_ratio == 0.0, "Sky ratio should be 0 in fast mode"
            assert config.max_refinement_iterations == 1, "Should have 1 iteration in fast mode"

            print("✓ Fast mode enabled correctly")
            print(f"✓ min_floor_ratio: {config.min_floor_ratio}")
            print(f"✓ min_sky_ratio: {config.min_sky_ratio}")
            print(f"✓ max_refinement_iterations: {config.max_refinement_iterations}")
            print("✓ TEST 3 PASSED!")
            return True

        except Exception as e:
            print(f"✗ TEST 3 FAILED: {e}")
            return False


def test_with_save_path(tmp_path: Path) -> bool:
    """Test 4: Acquisition with save path."""
    print("\n" + "="*70)
    print("TEST 4: Acquisition with save_path")
    print("="*70)

    footprint = Polygon([
        (-71.05, 42.36),
        (-71.05, 42.37),
        (-71.04, 42.37),
        (-71.04, 42.36),
    ])

    save_path = tmp_path / "test_building"

    with patch("imageable.core.image.acquire_building_image") as mock_acquire:
        mock_acquire.return_value = create_mock_acquisition_result()

        try:
            image, camera, metadata = get_image(  # type: ignore[misc]
                "test_key",
                footprint,
                save_path=save_path
            )

            # Check config
            call_args = mock_acquire.call_args
            config = call_args[0][1]

            assert str(config.save_directory) == str(save_path), "Save directory should match"

            print("✓ Save path configured correctly")
            print(f"✓ Save directory: {config.save_directory}")
            print("✓ TEST 4 PASSED!")
            return True

        except Exception as e:
            print(f"✗ TEST 4 FAILED: {e}")
            return False


def test_custom_quality_parameters() -> bool:
    """Test 5: Custom quality thresholds."""
    print("\n" + "="*70)
    print("TEST 5: Custom quality parameters")
    print("="*70)

    footprint = Polygon([
        (-71.05, 42.36),
        (-71.05, 42.37),
        (-71.04, 42.37),
        (-71.04, 42.36),
    ])

    with patch("imageable.core.image.acquire_building_image") as mock_acquire:
        mock_acquire.return_value = create_mock_acquisition_result()

        try:
            image, camera, metadata = get_image(  # type: ignore[misc]
                "test_key",
                footprint,
                min_floor_ratio=0.001,
                min_sky_ratio=0.2,
                max_refinement_iterations=10,
            )

            # Check config
            call_args = mock_acquire.call_args
            config = call_args[0][1]

            assert config.min_floor_ratio == 0.001, f"Expected 0.001, got {config.min_floor_ratio}"
            assert config.min_sky_ratio == 0.2, f"Expected 0.2, got {config.min_sky_ratio}"
            assert config.max_refinement_iterations == 10, f"Expected 10, got {config.max_refinement_iterations}"

            print("✓ Custom parameters configured correctly")
            print(f"✓ min_floor_ratio: {config.min_floor_ratio}")
            print(f"✓ min_sky_ratio: {config.min_sky_ratio}")
            print(f"✓ max_refinement_iterations: {config.max_refinement_iterations}")
            print("✓ TEST 5 PASSED!")
            return True

        except Exception as e:
            print(f"✗ TEST 5 FAILED: {e}")
            return False


def test_load_image(tmp_path: Path) -> bool:
    """Test 6: Loading a previously saved image."""
    print("\n" + "="*70)
    print("TEST 6: load_image() function")
    print("="*70)

    import json

    from PIL import Image

    # Create test image and metadata
    image_path = tmp_path / "image.jpg"
    metadata_path = tmp_path / "metadata.json"

    test_image = create_mock_image()
    Image.fromarray(test_image).save(image_path)

    metadata = {
        "camera_parameters": {
            "longitude": -71.05,
            "latitude": 42.36,
            "fov": 90,
            "heading": 45,
            "pitch": 10,
            "width": 640,
            "height": 640,
        },
        "refinement_success": True,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    try:
        image, camera, meta = load_image(image_path, metadata_path)

        assert image is not None, "Image should not be None"
        assert image.shape == (640, 640, 3), f"Expected shape (640, 640, 3), got {image.shape}"
        assert camera.fov == 90, f"Expected FOV 90, got {camera.fov}"
        assert camera.heading == 45, f"Expected heading 45, got {camera.heading}"
        assert meta["refinement_success"] == True, "Should load metadata"

        print("✓ Image loaded successfully")
        print(f"✓ Image shape: {image.shape}")
        print(f"✓ Camera FOV: {camera.fov}°")
        print(f"✓ Camera heading: {camera.heading}°")
        print("✓ TEST 6 PASSED!")
        return True

    except Exception as e:
        print(f"✗ TEST 6 FAILED: {e}")
        return False


def test_acquisition_config() -> bool:
    """Test 7: ImageAcquisitionConfig creation."""
    print("\n" + "="*70)
    print("TEST 7: ImageAcquisitionConfig")
    print("="*70)

    try:
        # Test default config
        config = ImageAcquisitionConfig(api_key="test_key")

        assert config.api_key == "test_key"
        assert config.save_directory is None
        assert config.save_intermediate is False
        assert config.overwrite is True
        assert config.min_floor_ratio == 0.00001
        assert config.min_sky_ratio == 0.1
        assert config.max_refinement_iterations == 5

        print("✓ Default config created correctly")

        # Test custom config
        custom_config = ImageAcquisitionConfig(
            api_key="custom_key",
            save_directory="/tmp/test",
            save_intermediate=True,
            overwrite=False,
            min_floor_ratio=0.01,
            min_sky_ratio=0.2,
            max_refinement_iterations=10,
        )

        assert custom_config.api_key == "custom_key"
        assert custom_config.save_directory == "/tmp/test"
        assert custom_config.save_intermediate is True
        assert custom_config.overwrite is False
        assert custom_config.min_floor_ratio == 0.01

        print("✓ Custom config created correctly")
        print("✓ TEST 7 PASSED!")
        return True

    except Exception as e:
        print(f"✗ TEST 7 FAILED: {e}")
        return False


def test_low_level_api() -> bool:
    """Test 8: Low-level acquire_building_image() API."""
    print("\n" + "="*70)
    print("TEST 8: Low-level acquire_building_image() API")
    print("="*70)

    footprint = Polygon([
        (-71.05, 42.36),
        (-71.05, 42.37),
        (-71.04, 42.37),
        (-71.04, 42.36),
    ])

    config = ImageAcquisitionConfig(
        api_key="test_key",
        min_floor_ratio=0.001,
        min_sky_ratio=0.2,
    )

    with patch("imageable._images.acquisition.CameraParametersRefiner") as mock_refiner:
        mock_instance = MagicMock()
        mock_refiner.return_value = mock_instance

        mock_camera = create_mock_camera()
        mock_image = create_mock_image()
        mock_instance.adjust_parameters.return_value = (mock_camera, True, mock_image)

        try:
            result = acquire_building_image(footprint, config)

            assert result.is_valid, "Result should be valid"
            assert result.success, "Result should be successful"
            assert result.image is not None, "Image should not be None"
            assert isinstance(result.camera_params, CameraParameters), "Should have camera params"

            print("✓ Low-level API works correctly")
            print(f"✓ Result is valid: {result.is_valid}")
            print(f"✓ Result is successful: {result.success}")
            print(f"✓ Image shape: {result.image.shape}")
            print("✓ TEST 8 PASSED!")
            return True

        except Exception as e:
            print(f"✗ TEST 8 FAILED: {e}")
            return False


def run_all_tests() -> bool:
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("RUNNING PUBLIC API TESTS")
    print("="*70)

    import tempfile
    tmp_path = Path(tempfile.mkdtemp())

    tests = [
        ("Basic get_image()", test_basic_get_image),
        ("get_image() without metadata", test_get_image_without_metadata),
        ("Fast mode", test_fast_mode),
        ("With save_path", lambda: test_with_save_path(tmp_path)),
        ("Custom quality parameters", test_custom_quality_parameters),
        ("load_image()", lambda: test_load_image(tmp_path)),
        ("ImageAcquisitionConfig", test_acquisition_config),
        ("Low-level API", test_low_level_api),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} CRASHED: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    print("\n" + "="*70)
    print(f"RESULTS: {passed_count}/{total_count} tests passed")
    print("="*70)

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe public API is working correctly!")
        return True
    print(f"\n⚠️  {total_count - passed_count} test(s) failed")
    return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
