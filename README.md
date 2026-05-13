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

## Quick Start

**Fastest way to run the benchmark:**

```powershell
# Simply double-click start.bat (Windows Explorer)
# OR run from PowerShell:
.\start.bat
```

The `start.bat` script handles everything:
- ✅ Checks for Python, venv, and dependencies
- ✅ Creates virtual environment if needed
- ✅ Installs/verifies dependencies
- ✅ Checks GPU availability (CUDA)
- ✅ Runs the benchmark
- ✅ Shows results and opens outputs folder

**For custom installations or advanced setup, see the Installation section below.**

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

Or use the provided `start.bat` script (see Quick Start above).

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

Or use the provided `start.bat` script (see Quick Start above).

**Important:** Do NOT use `pip install -r requirements.txt` as it will install the CPU-only version of PyTorch. Always use the PyTorch index URL for CUDA support.

### Method 3: Using start.bat (Windows - Easiest)

If you're on Windows, the simplest option is to use the provided batch script:

```powershell
.\start.bat
```

This script automates all setup steps and is recommended for most users. See the Quick Start section above.

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

### Multiple Run Averaging

The benchmark runs **10 complete iterations** by default to provide statistically reliable results. Each metric is reported as:

```
mean ± standard deviation
```

**Example:** `45.23 ± 2.15 samples/sec` means the average throughput across 10 runs was 45.23 samples/sec, with most runs falling within ±2.15 of that value.

You can adjust `NUM_RUNS` in `main.py` to run more iterations (for even higher confidence) or fewer (for faster testing).

## Configuration

This script uses **no command-line arguments**. All benchmark parameters are configured by editing constants in `main.py` (lines 21-42):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_RUNS` | 10 | Number of complete benchmark iterations (results are averaged) |
| `BATCH_SIZE` | 4 | Number of samples processed per training step |
| `NUM_EPOCHS` | 2 | Number of complete passes through the dataset |
| `LEARNING_RATE` | 0.001 | Optimization step size (adjust carefully, typically 0.001-0.01) |
| `NUM_WORKERS` | 2 | CPU threads for data loading (increase for faster I/O) |
| `AUDIO_SAMPLE_RATE` | 16000 | Audio frequency (Hz) |
| `AUDIO_DURATION` | 1 | Audio length per sample (seconds) |
| `VIDEO_NUM_FRAMES` | 16 | Number of frames per video sample |
| `VIDEO_FRAME_HEIGHT` | 64 | Video frame height (pixels) |
| `VIDEO_FRAME_WIDTH` | 64 | Video frame width (pixels) |

### Common Adjustments

**For faster benchmarks (testing):**
```python
BATCH_SIZE = 8        # Larger batches
NUM_EPOCHS = 1        # Fewer epochs
NUM_RUNS = 3          # Fewer runs
```

**For more thorough benchmarks (production):**
```python
BATCH_SIZE = 2        # Smaller batches (more conservative)
NUM_EPOCHS = 5        # More training iterations
NUM_RUNS = 20         # More runs for better averaging
```

**For GPUs with limited memory (RTX 3050, etc.):**
```python
BATCH_SIZE = 2        # Reduce batch size
VIDEO_NUM_FRAMES = 8  # Reduce video complexity
VIDEO_FRAME_HEIGHT = 32
VIDEO_FRAME_WIDTH = 32
```

## Understanding the Metrics

### Throughput (samples/sec)

**What it is:** How many training samples the GPU can process per second.  
**Why it matters:** Higher throughput = faster training. A good GPU will process more samples in less time. This is the most direct measure of your GPU's speed for deep learning tasks.

### Training Time per Epoch (seconds)

**What it is:** How long it takes to train on the entire dataset once.  
**Why it matters:** Lower time = better performance. If an epoch takes 10 seconds instead of 60, you can experiment and iterate 6x faster.

### GPU Memory Usage (MB)

**What it is:** How much of your GPU's RAM is being used during training.  
**Why it matters:** More available memory means you can use larger batch sizes or bigger models. If you hit the limit (e.g., 8GB for RTX 4060), you'll get "out of memory" errors.

### Peak GPU Memory (MB)

**What it is:** The maximum amount of GPU memory used at any point during training.  
**Why it matters:** Shows how close you are to your GPU's limit. Helps you understand if you can increase batch size or need to reduce model complexity.

### Loss (CrossEntropy / MSE)

**What it is:** A number representing how "wrong" the model's predictions are. Lower is better.  
**Why it matters:** Loss should decrease over epochs, showing the model is learning. If it stays flat or increases, something's wrong with training. Different models use different loss functions (CrossEntropy for classification, MSE for regression).

### Speedup (Video/Audio ratio)

**What it is:** How much faster one model trains compared to the other.  
**Why it matters:** Helps compare the computational cost of different model architectures. A 2x speedup means one model trains in half the time.

## Output

All results are saved to the `outputs/` directory with a timestamped filename:

**Filename format:** `benchmark_{GPU_NAME}_{MACHINE_ID}_{TIMESTAMP}.{ext}`

Example: `benchmark_NVIDIA_GeForce_RTX_3050_Laptop_GPU_DESKTOP-F9DN25A_20260505_135505.json`

### Output Files

#### `.json` (Detailed Results)
JSON file containing:
- Device information (GPU name, CUDA version, total memory)
- Configuration used (batch size, epochs, learning rate)
- Per-run results for all `NUM_RUNS` iterations
- Averaged metrics with standard deviations
- Per-epoch breakdown (loss, throughput, memory, time)

Use this for:
- Detailed analysis
- Comparing multiple benchmark runs
- Feeding results into analysis scripts
- Archiving historical results

#### `.png` (Visualization)
Comprehensive 6-panel chart showing:
- Device information banner
- Audio model training loss curve
- Video model training loss curve
- Throughput comparison (samples/sec per epoch)
- Training time per epoch
- Memory usage per epoch
- Summary statistics (time, throughput, memory, speedup)

Use this for:
- Quick visual inspection
- Presentations and reports
- Comparing GPU performance trends

#### `.txt` (Human-Readable Report)
Text report with:
- Device and configuration details
- Tabular breakdown of each epoch (loss, time, throughput, memory)
- Averaged metrics with standard deviations
- Overall speedup comparison

Use this for:
- Quick reference
- Sharing results via email
- Version control / git history

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
