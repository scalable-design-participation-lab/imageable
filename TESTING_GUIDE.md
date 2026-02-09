# Testing the Public API

## Quick Test (Recommended First)

Run this to verify imports and basic structure:

```bash
cd /Users/billngo/Documents/GitHub/imageable
python quick_test_api.py
```

This will test:
- ✓ All imports work
- ✓ Objects can be created
- ✓ API signatures are correct
- ✓ Documentation exists

**Expected output**: All checks should pass with ✓ marks

## Comprehensive Test

Run this for full testing with mocked API calls:

```bash
python test_public_api.py
```

This will test 8 scenarios:
1. Basic get_image() with full metadata
2. get_image() without metadata
3. Fast mode (refine_camera=False)
4. With save_path
5. Custom quality parameters
6. load_image()
7. ImageAcquisitionConfig
8. Low-level acquire_building_image() API

**Expected output**: 8/8 tests passed 🎉

## Using pytest (Alternative)

You can also run the existing test suite:

```bash
# Run all existing tests
pytest tests/ -v

# Run just image acquisition tests
pytest tests/images/test_acquisition.py -v

# Run the new standalone usage tests
pytest tests/images/test_standalone_usage.py -v

# Run with coverage
pytest tests/ --cov=imageable --cov-report=term-missing
```

## Testing with Real API

To test with actual Google Street View API:

1. Set your API key:
```python
API_KEY = "your_actual_api_key_here"
```

2. Run one of the examples:
```bash
# Edit examples/standalone_image_acquisition.py to add your API key
# Then run specific examples:
python -c "
from examples.standalone_image_acquisition import example_basic_acquisition
example_basic_acquisition()
"
```

## What Each Test Validates

### quick_test_api.py
- Imports work correctly
- No circular dependencies
- Objects can be instantiated
- API signatures are correct
- Documentation is present

### test_public_api.py
- get_image() returns correct types
- Fast mode disables refinement
- Custom parameters are passed through
- Save paths are configured correctly
- load_image() works with cached files
- Low-level API functions properly
- All configurations create correct objects

### Existing tests
- Internal functionality (tests/images/test_acquisition.py)
- Usage patterns (tests/images/test_standalone_usage.py)

## Troubleshooting

### Import errors
If you get import errors:
```bash
# Make sure you're in the project root
cd /Users/billngo/Documents/GitHub/imageable

# Check Python can find the src directory
python -c "import sys; sys.path.insert(0, 'src'); import imageable; print('OK')"
```

### Missing dependencies
```bash
# Install in development mode
pip install -e .

# Or install specific dependencies
pip install numpy shapely pillow
```

### Path issues
The test scripts add `src/` to the path automatically, so they should work from the project root.

## Success Criteria

✅ quick_test_api.py shows all ✓ marks
✅ test_public_api.py shows 8/8 tests passed  
✅ pytest tests/ passes all tests
✅ No import errors
✅ API signatures match documentation

## Next Steps After Testing

Once tests pass:
1. Review the enhanced docstrings in `src/imageable/core/image.py`
2. Try the examples in `examples/standalone_image_acquisition.py`
3. Review the documentation in `docs/`
4. Add the README content from the markdown files
5. Commit and push!
