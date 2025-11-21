# GPU Benchmarking

A deep learning benchmark that trains audio (Wav2Vec2) and video (3D CNN) models to measure GPU performance.

## Prerequisites

### GPU Requirements

- NVIDIA GPU with CUDA support
- NVIDIA GPU drivers installed
- CUDA 12.1 or newer (check with `nvidia-smi`)

### Verify GPU

```powershell
nvidia-smi
```

You should see your GPU listed with CUDA version information.

## Installation

### Method 1: With uv (Recommended)

`uv` is a fast Python package manager that handles dependencies and virtual environments automatically.

**Install uv:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or visit: https://github.com/astral-sh/uv

**Setup and run:**

```powershell
# Sync dependencies (installs CUDA-enabled PyTorch automatically)
uv sync

# Run the benchmark
uv run main.py
```

**Note:** The `pyproject.toml` is configured to automatically install PyTorch with CUDA 12.1 support from the PyTorch index.

### Method 2: Without uv (Manual Setup)

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install matplotlib numpy psutil transformers

# Run the benchmark
python main.py
```

**Important:** Do NOT use `pip install -r requirements.txt` as it will install the CPU-only version of PyTorch. Always use the PyTorch index URL for CUDA support.

## Verifying GPU Detection

After installation, verify PyTorch can detect your GPU:

```powershell
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:

```
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 4060
```

## What This Script Does

Trains dual deep learning models (Wav2Vec2 for audio and 3D CNN for video) across multiple epochs while measuring:

- Training/inference speed (samples/sec)
- GPU memory usage (MB)
- Training time per epoch (seconds)
- Loss curves and convergence

## Output

Results are saved to the `outputs/` directory:

- `benchmark_results.json` - Detailed metrics in JSON format
- `benchmark_results.png` - Comprehensive visualization with charts

## Troubleshooting

### GPU Not Detected

If `torch.cuda.is_available()` returns `False`:

1. **Check NVIDIA drivers:**

   ```powershell
   nvidia-smi
   ```

2. **Verify PyTorch version:**

   ```powershell
   uv run python -c "import torch; print(torch.__version__)"
   ```

   Should show `2.5.1+cu121` (not `2.9.1+cpu`)

3. **Reinstall CUDA-enabled PyTorch:**
   ```powershell
   uv pip uninstall torch torchvision torchaudio
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Out of Memory Errors

If you encounter CUDA out of memory errors:

- Reduce `BATCH_SIZE` in `main.py` (default is 4)
- Reduce `NUM_EPOCHS` for faster testing
- Reduce model complexity (frame count, resolution)
