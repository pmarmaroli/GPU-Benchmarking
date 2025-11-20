# GPU Benchmarking

A deep learning benchmark that trains audio (Wav2Vec2) and video (3D CNN) models to measure GPU performance.

## Quick Start

### With uv (Recommended)
If you have `uv` installed:
```powershell
uv run main.py
```

### Without uv
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## About uv

`uv` is a fast Python package manager that handles dependencies and virtual environments automatically. 

**Install uv:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or visit: https://github.com/astral-sh/uv

**First time setup:**
```powershell
uv sync
```

## What This Script Does

Trains dual deep learning models (Wav2Vec2 for audio and 3D CNN for video) across multiple epochs while measuring:
- Training/inference speed
- GPU memory usage
- CPU utilization

Results are logged and visualized for performance analysis.
