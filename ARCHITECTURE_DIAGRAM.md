# Image Acquisition Architecture Diagram

## Current Architecture (Already Good!)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PUBLIC API LAYER                            │
│                     (What Users Interact With)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  imageable.get_image(api_key, footprint)                            │
│      ↓                                                               │
│      Simple wrapper with sensible defaults                          │
│      Returns: image, camera_params, metadata                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPONENT ACCESS LAYER                          │
│                   (Advanced Users & Pipelines)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  _images.acquisition.acquire_building_image(polygon, config)        │
│      ↓                                                               │
│      Main acquisition orchestrator                                  │
│      • Checks cache first                                           │
│      • Delegates to camera refiner                                  │
│      • Returns structured result                                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        INTERNAL LAYER                                │
│                    (Implementation Details)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────┐  ┌──────────────────────────┐        │
│  │ CameraParametersRefiner  │  │ ObservationPointEstimator│        │
│  │                          │  │                          │        │
│  │ • Adjust pitch/FOV       │  │ • Find street network    │        │
│  │ • Validate quality       │  │ • Calculate best view    │        │
│  │ • Iterate until good     │  │ • Estimate distance      │        │
│  └──────────────────────────┘  └──────────────────────────┘        │
│              ↓                              ↓                        │
│  ┌──────────────────────────────────────────────────────┐          │
│  │         download_street_view_image()                  │          │
│  │         (Google Street View API calls)                │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Separation from Height Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STANDALONE IMAGE ACQUISITION                      │
│                    (Can be used independently)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  User Code:                                                          │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ from imageable import get_image                     │            │
│  │                                                      │            │
│  │ # Just get images - no analysis                     │            │
│  │ image, camera, meta = get_image(key, footprint)     │            │
│  │                                                      │            │
│  │ # Use for your own purposes:                        │            │
│  │ # - Build datasets                                  │            │
│  │ # - Custom CV tasks                                 │            │
│  │ # - Visual inspection                               │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

                              OR

┌─────────────────────────────────────────────────────────────────────┐
│                    INTEGRATED WITH ANALYSIS                          │
│                    (Height pipeline uses acquisition)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  building_height.py:                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ from imageable._images.acquisition import (         │            │
│  │     acquire_building_image                          │            │
│  │ )                                                    │            │
│  │                                                      │            │
│  │ def building_height_from_single_view(params):       │            │
│  │     # Step 1: Acquire image (reusable component)    │            │
│  │     result = acquire_building_image(                │            │
│  │         polygon, config                             │            │
│  │     )                                                │            │
│  │                                                      │            │
│  │     # Step 2: Estimate height from image            │            │
│  │     return estimate_height_from_image(              │            │
│  │         result.image,                               │            │
│  │         result.camera_params,                       │            │
│  │         polygon                                      │            │
│  │     )                                                │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Standalone Acquisition Flow

```
User Request
    ↓
get_image(api_key, footprint)
    ↓
Check cache?
    ├─ YES → Load from disk → Return cached result
    ↓
    NO
    ↓
Find observation point
    (Use street network + ML model)
    ↓
Download initial image
    (Google Street View API)
    ↓
Check quality (sky + ground ratio)
    ├─ GOOD → Return result
    ↓
    NEEDS REFINEMENT
    ↓
Adjust camera parameters
    (Pitch up/down, FOV in/out)
    ↓
Download refined image
    ↓
Repeat until quality OK or max iterations
    ↓
Save to cache (if save_path provided)
    ↓
Return: image, camera_params, metadata
```

### Height Estimation Flow (Uses Acquisition)

```
User Request
    ↓
get_dataset(api_key, footprint)
    ↓
Acquire image (see above flow)
    ↓
Run segmentation model
    ↓
Detect lines (LCNN)
    ↓
Find vanishing points
    ↓
Calculate height
    ↓
Apply corrections
    ↓
Return: BuildingProperties with height
```

## Key Benefits of This Architecture

### 1. Separation of Concerns ✓
- Image acquisition is its own module
- Analysis pipelines import and use it
- Neither owns the other

### 2. Reusability ✓
- Same acquisition code used by:
  - Height estimation
  - Material segmentation
  - Any future features
  - User custom code

### 3. Testability ✓
- Each layer can be tested independently
- Mock boundaries are clear
- Integration tests are straightforward

### 4. Flexibility ✓
- Users can:
  - Use acquisition alone
  - Use pre-acquired images
  - Combine with analysis
  - Build custom workflows

## Component Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│ Component               │ Responsibility                    │
├─────────────────────────────────────────────────────────────┤
│ get_image()             │ Simple public API                 │
│ acquire_building_image()│ Orchestrate acquisition           │
│ CameraParametersRefiner │ Optimize camera view              │
│ ObservationPointEstimator│ Find best camera position        │
│ download_street_view_image│ GSV API interface               │
│ CameraParameters        │ Data structure for camera         │
│ ImageAcquisitionResult  │ Structured return value           │
└─────────────────────────────────────────────────────────────┘
```

## Usage Patterns

### Pattern 1: Just Get Images
```python
from imageable import get_image

for building in buildings:
    image, camera, meta = get_image(api_key, building.footprint)
    # Use images for whatever you want
```

### Pattern 2: Pre-acquisition + Analysis
```python
from imageable import get_image, get_dataset

# Step 1: Acquire all images with caching
for building in buildings:
    get_image(api_key, building.footprint, save_path=f"./cache/{building.id}")

# Step 2: Run analysis (uses cache, no new API calls)
for building in buildings:
    props = get_dataset(api_key, building.footprint)
```

### Pattern 3: Bring Your Own Images
```python
from imageable import load_image
from imageable._features.height.building_height import estimate_height_from_image

# Load your own image
image, camera, meta = load_image("./my_images/building.jpg")

# Run analysis
height = estimate_height_from_image(image, camera, footprint)
```

### Pattern 4: Combined Workflow (Traditional)
```python
from imageable import get_dataset

# All in one - acquisition + analysis
props = get_dataset(api_key, footprint)
print(f"Height: {props.building_height}m")
```

## Summary

**Your architecture already supports all these patterns!** We just documented it better so users know they can:

1. ✅ Use image acquisition standalone
2. ✅ Pre-acquire images for later analysis
3. ✅ Bring their own images
4. ✅ Combine acquisition + analysis (traditional)

The clean separation of concerns you built makes all of this possible!
