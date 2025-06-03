# imageable

An image processing library built with modern Python tooling.

## Installation

### For Development

1. **Install uv** (if you haven't already):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip
   pip install uv
   ```

2. **Clone and setup the project**:
   ```bash
   git clone https://github.com/yourusername/imageable.git
   cd imageable
   
   # Install all dependencies (including dev dependencies)
   uv sync --group dev
   ```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest --cov=src/imageable --cov-report=html

# Run specific test file
uv run pytest tests/test_specific.py

# Run tests with verbose output
uv run pytest -v

# Run tests and stop at first failure
uv run pytest -x
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add numpy

# Add a development dependency
uv add --dev pytest-mock

# Add optional dependencies
uv add --optional vis matplotlib seaborn
```

### Project Structure

```
imageable/
├── src/
│   └── imageable/
│       ├── __init__.py
│       ├── main.py
│       └── ...
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   └── ...
├── pyproject.toml
├── README.md
└── uv.lock
```

### Available Commands

```bash
# Install dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov-report=html
open htmlcov/index.html  # View coverage report

# Run Python in the project environment
uv run python

# Run any command in the project environment
uv run <command>

# Add/remove dependencies
uv add <package>
uv remove <package>

# Update dependencies
uv sync --upgrade
```

## License

See [LICENSE](LICENSE) file for details.

## Authors

- Khoi Ngo - ngo.kho@northeastern.edu
- Uriel Legaria - fill@gmail.com
- Carlos Sandoval Olascoaga 