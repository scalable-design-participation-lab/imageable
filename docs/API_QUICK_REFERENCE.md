# Image Acquisition API Quick Reference

## Public API (`imageable.get_image()`)

### Basic Usage
```python
from imageable import get_image

# Full metadata
image, camera, metadata = get_image(api_key, footprint)

# Just the image
image = get_image(api_key, footprint, return_metadata=False)

# Fast mode (no refinement)
image = get_image(api_key, footprint, refine_camera=False, return_metadata=False)
```

### With Caching
```python
# Save to disk
image, camera, metadata = get_image(
    api_key, 
    footprint,
    save_path="./images/building_001"
)

# Load from cache (skip if exists)
image, camera, metadata = get_image(
    api_key,
    footprint, 
    save_path="./images/building_001",
    overwrite=False  # Won't re-acquire if cached
)
```

### Custom Quality
```python
image, camera, metadata = get_image(
    api_key,
    footprint,
    min_floor_ratio=0.001,      # More lenient
    min_sky_ratio=0.2,          # Require more sky
    max_refinement_iterations=10,  # More attempts
)
```

## Loading Cached Images

```python
from imageable import load_image

# Load from default location (looks for metadata.json in same dir)
image, camera, metadata = load_image("./images/building_001/image.jpg")

# Load with explicit metadata path
image, camera, metadata = load_image(
    "./images/img.jpg",
    metadata_path="./meta/img_meta.json"
)
```

## Advanced API (`acquire_building_image()`)

```python
from imageable._images.acquisition import (
    ImageAcquisitionConfig,
    acquire_building_image,
)

# Create config
config = ImageAcquisitionConfig(
    api_key=api_key,
    save_directory="./images",
    save_intermediate=True,  # Save refinement iterations
    overwrite=True,
    min_floor_ratio=0.00001,
    min_sky_ratio=0.1,
    max_refinement_iterations=5,
    confidence_threshold=0.1,
    polygon_buffer_constant=20,
)

# Acquire
result = acquire_building_image(footprint, config)

# Check result
if result.is_valid:
    image = result.image
    camera = result.camera_params
    meta = result.metadata
    from_cache = result.from_cache
```

## Data Structures

### CameraParameters
```python
camera = CameraParameters(
    longitude=-71.05,
    latitude=42.36,
    fov=90,           # Field of view (degrees)
    heading=45,       # Compass heading (0-360)
    pitch=10,         # Vertical angle (-90 to 90)
    width=640,        # Image width (pixels)
    height=640,       # Image height (pixels)
)
```

### ImageAcquisitionResult
```python
result = ImageAcquisitionResult(
    image=np.array(...),           # RGB numpy array (H, W, 3)
    camera_params=camera,           # CameraParameters
    metadata={...},                 # Additional info
    success=True,                   # Refinement success
    from_cache=False,               # Loaded from cache?
)

# Properties
result.is_valid  # True if image is not None
```

## Common Patterns

### Batch Collection
```python
from pathlib import Path

for i, footprint in enumerate(footprints):
    try:
        image, camera, meta = get_image(
            api_key,
            footprint,
            save_path=Path("./images") / f"building_{i:03d}",
            overwrite=False,
        )
        print(f"✓ Building {i}: {'cached' if meta['from_cache'] else 'acquired'}")
    except Exception as e:
        print(f"✗ Building {i}: {e}")
```

### Pre-Acquisition for Analysis
```python
from imageable import get_image, get_dataset

# Step 1: Acquire all images (cacheable)
for footprint in footprints:
    get_image(api_key, footprint, save_path="./cache")

# Step 2: Analyze (uses cache, no new API calls)
for footprint in footprints:
    props = get_dataset(api_key, footprint)
```

### Dataset Building
```python
import json
from pathlib import Path

dataset_dir = Path("./dataset")
manifest = []

for building_id, footprint in buildings.items():
    save_path = dataset_dir / building_id
    
    image, camera, meta = get_image(
        api_key,
        footprint,
        save_path=save_path
    )
    
    manifest.append({
        'id': building_id,
        'image_path': str(save_path / "image.jpg"),
        'camera': camera.__dict__,
        'metadata': meta,
    })

# Save manifest
with open(dataset_dir / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
```

### Quality Control
```python
# Strict quality requirements
high_quality_images = []

for footprint in footprints:
    try:
        image, camera, meta = get_image(
            api_key,
            footprint,
            min_sky_ratio=0.2,              # 20% sky minimum
            min_floor_ratio=0.01,           # 1% floor minimum
            max_refinement_iterations=10,   # More attempts
        )
        
        if meta['refinement_success']:
            high_quality_images.append(image)
    except Exception:
        continue
```

## Parameters Reference

### get_image() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | str | required | Google Street View API key |
| `building_footprint` | Polygon | required | Building footprint geometry |
| `save_path` | str/Path | None | Directory to save image and metadata |
| `return_metadata` | bool | True | Return camera params and metadata |
| `refine_camera` | bool | True | Refine camera for quality |
| `min_floor_ratio` | float | 0.00001 | Minimum floor pixel ratio |
| `min_sky_ratio` | float | 0.1 | Minimum sky pixel ratio |
| `max_refinement_iterations` | int | 5 | Max refinement attempts |
| `overwrite` | bool | True | Overwrite cached images |

### ImageAcquisitionConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | required | Google Street View API key |
| `save_directory` | str/Path | None | Directory to save results |
| `save_intermediate` | bool | False | Save refinement iterations |
| `overwrite` | bool | True | Overwrite existing cache |
| `min_floor_ratio` | float | 0.00001 | Min floor pixel ratio |
| `min_sky_ratio` | float | 0.1 | Min sky pixel ratio |
| `max_refinement_iterations` | int | 5 | Max refinement attempts |
| `confidence_threshold` | float | 0.1 | Sky/floor detection confidence |
| `polygon_buffer_constant` | float | 20 | Observation point buffer |

## Error Handling

```python
from imageable import get_image

try:
    image, camera, metadata = get_image(api_key, footprint)
    
    if not metadata.get('refinement_success'):
        print("Warning: Refinement didn't meet quality thresholds")
    
except RuntimeError as e:
    print(f"Acquisition failed: {e}")
    # Handle no Street View coverage, network errors, etc.

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Enable caching**: Set `overwrite=False` to skip re-acquiring
2. **Fast mode**: Use `refine_camera=False` when quality isn't critical
3. **Batch processing**: Group buildings by location
4. **Iteration limits**: Lower `max_refinement_iterations` for speed
5. **Pre-acquisition**: Acquire images first, analyze later

## Troubleshooting

### Image Quality Issues
```python
# Increase refinement iterations
image = get_image(api_key, footprint, max_refinement_iterations=10)

# Adjust quality thresholds
image = get_image(
    api_key, 
    footprint,
    min_sky_ratio=0.15,    # More sky
    min_floor_ratio=0.005  # More floor
)
```

### Slow Acquisition
```python
# Fast mode (no refinement)
image = get_image(api_key, footprint, refine_camera=False)

# Fewer iterations
image = get_image(api_key, footprint, max_refinement_iterations=3)

# Use caching
image = get_image(
    api_key,
    footprint,
    save_path="./cache",
    overwrite=False  # Skip if cached
)
```

### Missing Street View Coverage
```python
# The acquisition will fail if no coverage exists
# Check coverage at: https://www.google.com/streetview/

try:
    image = get_image(api_key, footprint)
except RuntimeError:
    print("No Street View coverage for this location")
```

## See Also

- **Complete guide**: `docs/IMAGE_ACQUISITION.md`
- **Working examples**: `examples/standalone_image_acquisition.py`
- **Tests**: `tests/images/test_acquisition.py`
