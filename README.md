# py-dperf

Device profiler for MLX in Python

## Installation with uv

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Install uv
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew
brew install uv
```

### Project Setup
```bash
# Clone the repository
git clone <repository-url>
cd py-dperf

# Install project dependencies
uv sync
```

### Optional Dependencies
```bash
# For CUDA 12 support
uv pip install -e ".[cuda12]"

# For CUDA 11 support
uv pip install -e ".[cuda11]"

# For ROCm 5.0 support
uv pip install -e ".[rocm50]"
```

## Usage

### Running the Profiler
```bash
# Model profiling
uv run python run.py -p model -m qwen3 -r Qwen/Qwen3-4B-MLX-8bit -b 1 -s 128 -o model_profile.json

# Device profiling
uv run run.py -p device -o device_profile.json   
```
