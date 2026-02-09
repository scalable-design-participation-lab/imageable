# Getting Started with Image Acquisition

A quick guide to acquiring building images with imageable.

## Installation

```bash
pip install imageable
```

## Prerequisites

1. **Google Street View API Key**: Get one from [Google Cloud Console](https://console.cloud.google.com/)
2. **Building Footprint**: A Shapely polygon representing the building

## Basic Usage

### 1. Import and Setup

```python
from imageable import get_image
from shapely.geometry import Polygon

# Your API key
api_key = "YOUR_GOOGLE_STREET_VIEW_API_KEY"

# Building footprint (example coordinates)
footprint = Polygon([
    (-71.0589, 42.3601),  # longitude, latitude
    (-71.0589, 42.3611),
    (-71.0579, 42.3611),
    (-71.0579, 42.3601),
])
```

### 2. Acquire an Image

```python
# Get image with automatic quality optimization
image, camera_params, metadata = get_image(api_key, footprint)

# That's it! You now have:
# - image: numpy array (640, 640, 3) in RGB
# - camera_params: camera location and settings
# - metadata: additional acquisition info

print(f"Image shape: {image.shape}")
print(f"Camera FOV: {camera_params.fov}°")
print(f"Quality check passed: {metadata['refinement_success']}")
```

### 3. Use the Image

```python
# Save to disk
from PIL import Image
Image.fromarray(image).save("building.jpg")

# Or use directly
import matplotlib.pyplot as plt
plt.imshow(image)
plt.title(f"Building at FOV {camera_params.fov}°")
plt.show()
```

## Common Scenarios

### Scenario 1: Collect Multiple Buildings

```python
from pathlib import Path

buildings = [
    Polygon([...]),  # Building 1
    Polygon([...]),  # Building 2
    Polygon([...]),  # Building 3
]

output_dir = Path("./building_images")

for i, footprint in enumerate(buildings):
    image, camera, meta = get_image(
        api_key,
        footprint,
        save_path=output_dir / f"building_{i:03d}"
    )
    print(f"✓ Building {i+1} acquired")
```

### Scenario 2: Fast Collection (No Refinement)

When you need images quickly and don't need perfect quality:

```python
# Single API call, no quality optimization
image = get_image(
    api_key,
    footprint,
    refine_camera=False,
    return_metadata=False
)
```

### Scenario 3: Use Caching to Save API Costs

```python
# First run: acquires from API
get_image(api_key, footprint, save_path="./cache/building_001")

# Second run: loads from disk (no API call!)
get_image(
    api_key, 
    footprint,
    save_path="./cache/building_001",
    overwrite=False  # Skip if already cached
)
```

### Scenario 4: Custom Quality Requirements

```python
# Require more sky and ground visibility
image, camera, meta = get_image(
    api_key,
    footprint,
    min_floor_ratio=0.01,   # 1% floor minimum
    min_sky_ratio=0.2,      # 20% sky minimum
    max_refinement_iterations=10  # More attempts
)
```

### Scenario 5: Load Previously Acquired Images

```python
from imageable import load_image

# Load cached image
image, camera, metadata = load_image("./cache/building_001/image.jpg")

# Camera parameters are automatically loaded
print(f"Camera was at: ({camera.latitude}, {camera.longitude})")
print(f"Heading: {camera.heading}°, Pitch: {camera.pitch}°")
```

## Understanding the Results

### Image
```python
image.shape  # (640, 640, 3) - height, width, channels
image.dtype  # uint8 - values 0-255
# RGB format (not BGR!)
```

### Camera Parameters
```python
camera_params.longitude  # Camera longitude
camera_params.latitude   # Camera latitude
camera_params.fov        # Field of view in degrees (adjusted)
camera_params.heading    # Compass heading 0-360°
camera_params.pitch      # Vertical angle -90 to 90° (adjusted)
camera_params.width      # Image width (640)
camera_params.height     # Image height (640)
```

### Metadata
```python
metadata['refinement_success']     # True if quality met
metadata['refinement_iterations']  # Number of iterations
metadata['from_cache']             # True if loaded from disk
metadata['min_floor_ratio']        # Quality threshold used
metadata['min_sky_ratio']          # Quality threshold used
```

## How It Works

1. **Find Observation Point**: Uses OpenStreetMap to find the best camera position
2. **Download Initial Image**: Fetches image from Google Street View
3. **Check Quality**: Analyzes if sky and ground are visible
4. **Refine If Needed**: Adjusts camera pitch and FOV to improve view
5. **Iterate**: Repeats until quality thresholds met or max iterations reached
6. **Cache**: Saves result to disk if save_path provided

## Tips

### Save API Costs
- Use `save_path` to cache images
- Set `overwrite=False` to reuse cached images
- Pre-acquire images, analyze later

### Faster Acquisition
- Use `refine_camera=False` for quick collection
- Reduce `max_refinement_iterations` to 3 instead of 5
- Process buildings in batches by location

### Better Quality
- Increase `max_refinement_iterations` to 10
- Adjust `min_sky_ratio` and `min_floor_ratio`
- Review failed acquisitions manually

## Troubleshooting

### "No Street View coverage"
**Problem**: The area has no Google Street View imagery  
**Solution**: Check coverage at https://www.google.com/streetview/

### "Quality requirements not met"
**Problem**: Can't find a view meeting sky/ground thresholds  
**Solution**: 
- Lower quality thresholds
- Increase max iterations
- Check if building is very small or obstructed

### "Acquisition is slow"
**Problem**: Taking many iterations per building  
**Solution**:
- Use `refine_camera=False` for fast mode
- Enable caching with `overwrite=False`
- Reduce `max_refinement_iterations`

## Next Steps

### Learn More
- **Complete Guide**: [docs/IMAGE_ACQUISITION.md](../docs/IMAGE_ACQUISITION.md)
- **API Reference**: [docs/API_QUICK_REFERENCE.md](../docs/API_QUICK_REFERENCE.md)
- **Examples**: [examples/standalone_image_acquisition.py](../examples/standalone_image_acquisition.py)

### Build Something
- Create a building image dataset for ML
- Collect images for urban analysis
- Build a visual building catalog
- Pre-acquire images before analysis

### Combine with Analysis
```python
from imageable import get_dataset

# Get image AND analyze it
props = get_dataset(api_key, footprint)

print(f"Height: {props.building_height}m")
print(f"Materials: {props.material_percentages}")
# ... 43+ more properties
```

## Questions?

- **How do I get an API key?** [Google Cloud Console](https://console.cloud.google.com/)
- **What image format is used?** RGB numpy array (H, W, 3), uint8
- **Can I use my own images?** Yes! Use `load_image()` or `estimate_height_from_image()`
- **Is there a batch processing function?** See examples for batch patterns
- **How do I cite this work?** See main README for citation

---

**You're ready to start!** Try the basic example above with your API key and building footprint.
