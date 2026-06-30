---
title: Rules — GPU Benchmarking
author: rule-cartographer
date: 2026-06-04
version: 1.0
language: en
---

# Rules — GPU Benchmarking

> **TL;DR** — 49 rules extracted from 1 source file across 7 sections. 0 tagged [TO CONFIRM]. Three documented code/comment contradictions are flagged inline (CFG-006, OUT-010/OUT-011).

## Scope
- Files analyzed: `main.py` (the entire application logic, 740 lines).
- Context-only (no rules extracted, not executable application logic): `README.md`, `start.bat`, `requirements.txt`, `pyproject.toml`.
- Section prefixes:
  - `CFG` = Configuration & device selection
  - `DATA` = Synthetic data generation
  - `MODEL` = Model architecture decisions
  - `TRAIN` = Per-epoch training & metrics
  - `BENCH` = Benchmark orchestration & aggregation
  - `OUT` = Output files, naming & reporting
  - `VIZ` = Visualization
- Not covered: GPU driver / CUDA runtime behavior, PyTorch / transformers internals, and the `start.bat` shell bootstrap (environment setup only, not application rules).

## Rules by section

### Configuration & device selection (`CFG`)

- **`CFG-001`** — The benchmark runs on the first CUDA GPU when one is available; otherwise it runs on the CPU.
  _Source:_ `main.py` · module level:22
- **`CFG-002`** — A full benchmarking session executes 10 complete iterations by default.
  _Source:_ `main.py` · module level:27 (`NUM_RUNS`)
- **`CFG-003`** — Each training step processes 4 samples per batch by default.
  _Source:_ `main.py` · module level:23 (`BATCH_SIZE`), used at 272 and 329
- **`CFG-004`** — Each model trains for 2 epochs per run by default.
  _Source:_ `main.py` · module level:24 (`NUM_EPOCHS`), used at 290 and 348
- **`CFG-005`** — Both models are optimized with a fixed learning rate of 0.001.
  _Source:_ `main.py` · module level:25 (`LEARNING_RATE`), used at 279 and 336
- **`CFG-006`** — The configured `NUM_WORKERS` value has no effect on data loading, because both data loaders are always created with zero worker processes. _(Code/doc contradiction: the README states `NUM_WORKERS` controls data-loading threads; the code wins.)_
  _Source:_ `main.py` · module level:26 vs `benchmark_audio():272` / `benchmark_video():329`
- **`CFG-007`** — Each audio sample represents exactly 1 second of 16 kHz audio (16,000 samples).
  _Source:_ `main.py` · module level:30-32
- **`CFG-008`** — Each video sample is fixed at 16 frames of 64×64 RGB.
  _Source:_ `main.py` · module level:35-38

### Synthetic data generation (`DATA`)

- **`DATA-001`** — The audio benchmark always uses a dataset of exactly 100 synthetic samples.
  _Source:_ `main.py` · `benchmark_audio():271`
- **`DATA-002`** — Each audio sample is random Gaussian noise normalized so its peak absolute amplitude is 1.
  _Source:_ `main.py` · `SyntheticAudioDataset.__getitem__():68-69`
- **`DATA-003`** — Each audio sample is assigned a random binary label (0 or 1).
  _Source:_ `main.py` · `SyntheticAudioDataset.__getitem__():72`
- **`DATA-004`** — The video benchmark always uses a dataset of exactly 200 synthetic samples.
  _Source:_ `main.py` · `benchmark_video():328`
- **`DATA-005`** — Each video sample is uniform random pixel data in the range 0–1.
  _Source:_ `main.py` · `SyntheticVideoDataset.__getitem__():87-88`
- **`DATA-006`** — Each video sample is assigned a random VMAF target uniformly between 0 and 100.
  _Source:_ `main.py` · `SyntheticVideoDataset.__getitem__():91`

### Model architecture decisions (`MODEL`)

- **`MODEL-001`** — The audio model collapses the Wav2Vec2 time dimension into a single vector by averaging over time before classification.
  _Source:_ `main.py` · `AudioClassificationModel.forward():130`
- **`MODEL-002`** — The video network sizes its first fully connected layer by assuming three pooling stages, dividing each spatial/temporal dimension by 8.
  _Source:_ `main.py` · `Video3DCNN.__init__():161`

### Per-epoch training & metrics (`TRAIN`)

- **`TRAIN-001`** — Audio training optimizes cross-entropy loss (treated as a classification task).
  _Source:_ `main.py` · `benchmark_audio():278`
- **`TRAIN-002`** — Video training optimizes mean-squared-error loss (treated as a regression task).
  _Source:_ `main.py` · `benchmark_video():335`
- **`TRAIN-003`** — Both models are trained with the Adam optimizer.
  _Source:_ `main.py` · `benchmark_audio():279` / `benchmark_video():336`
- **`TRAIN-004`** — The reported per-epoch loss is averaged over the number of batches, not the number of samples.
  _Source:_ `main.py` · `train_epoch():245`
- **`TRAIN-005`** — Throughput is computed as samples processed divided by the epoch's wall-clock time.
  _Source:_ `main.py` · `train_epoch():246`
- **`TRAIN-006`** — On a GPU, per-epoch memory used is the GPU allocated memory after the epoch minus the amount before it.
  _Source:_ `main.py` · `train_epoch():242-243`
- **`TRAIN-007`** — When no GPU is available, per-epoch memory is measured from the process's resident memory instead of GPU memory.
  _Source:_ `main.py` · `train_epoch():226,242` via `get_memory_usage():206-209`
- **`TRAIN-008`** — Every batch's inputs and labels are moved to the active device before the forward pass.
  _Source:_ `main.py` · `train_epoch():229-230`
- **`TRAIN-009`** — The model is switched to training mode at the start of every epoch.
  _Source:_ `main.py` · `train_epoch():221`

### Benchmark orchestration & aggregation (`BENCH`)

- **`BENCH-001`** — Within each run, the audio benchmark always executes before the video benchmark.
  _Source:_ `main.py` · `__main__:641-642`
- **`BENCH-002`** — GPU peak-memory statistics are reset before each model's training loop begins.
  _Source:_ `main.py` · `benchmark_audio():288` / `benchmark_video():346`
- **`BENCH-003`** — Peak memory for a model is the maximum GPU memory allocated during its training, reported as 0 when running on CPU.
  _Source:_ `main.py` · `benchmark_audio():302` / `benchmark_video():360`
- **`BENCH-004`** — A model's average throughput within a run is the mean of its per-epoch throughputs.
  _Source:_ `main.py` · `benchmark_audio():314` / `benchmark_video():373`
- **`BENCH-005`** — The GPU cache is emptied after every completed run.
  _Source:_ `main.py` · `__main__:648-649`
- **`BENCH-006`** — Each cross-run metric (total time, average throughput, peak memory) is summarized as a mean and a standard deviation over all runs.
  _Source:_ `main.py` · `__main__:658-663` (audio) / `669-674` (video)
- **`BENCH-007`** — Standard deviations are computed with the population divisor N (NumPy default), not the sample divisor N−1.
  _Source:_ `main.py` · `__main__:659,661,663,670,672,674`
- **`BENCH-008`** — Only the first run's per-epoch data is retained for the loss/throughput/time/memory charts; the per-epoch detail of subsequent runs is not used for plotting.
  _Source:_ `main.py` · `__main__:664,675`

### Output files, naming & reporting (`OUT`)

- **`OUT-001`** — All output files share a base name of the form `benchmark_{GPU}_{machine}_{timestamp}`.
  _Source:_ `main.py` · `__main__:704`
- **`OUT-002`** — The GPU-name component of the filename becomes `CPU` when no GPU is available.
  _Source:_ `main.py` · `__main__:701` / `generate_txt_report():554`
- **`OUT-003`** — In filename components, every character that is not a word character or hyphen is replaced with an underscore, and leading/trailing underscores are stripped.
  _Source:_ `main.py` · `_sanitize_filename():544`
- **`OUT-004`** — The machine identifier used in filenames and reports is the host's network name.
  _Source:_ `main.py` · `_get_machine_id():549`
- **`OUT-005`** — The filename timestamp uses local time formatted as `YYYYMMDD_HHMMSS`.
  _Source:_ `main.py` · `__main__:703`
- **`OUT-006`** — All outputs are written to the `outputs` directory, which is created if it does not already exist.
  _Source:_ `main.py` · `create_benchmark_visualization():534-535` / `generate_txt_report():557-558` / `__main__:726-727`
- **`OUT-007`** — Each benchmarking session produces exactly three result files (PNG, JSON, TXT) sharing one base filename.
  _Source:_ `main.py` · `__main__:709,728,736`
- **`OUT-008`** — The JSON output contains both the averaged metrics and the full per-run results of every iteration.
  _Source:_ `main.py` · `__main__:720-723`
- **`OUT-009`** — Values that cannot be JSON-serialized are written using their string representation.
  _Source:_ `main.py` · `__main__:730`
- **`OUT-010`** — In the text report, the speedup line is labeled "Audio/Video" while computing the ratio of audio total time to video total time.
  _Source:_ `main.py` · `generate_txt_report():612`
- **`OUT-011`** — In the PNG summary panel, the speedup line is labeled "Video/Audio" but computes the same audio-over-video total-time ratio as the text report. _(Labeling contradiction with OUT-010: same formula, opposite label.)_
  _Source:_ `main.py` · `create_benchmark_visualization():505,523`
- **`OUT-012`** — Reported GPU memory is the device's total memory in gigabytes, or 0 when running on CPU.
  _Source:_ `main.py` · `__main__:697`

### Visualization (`VIZ`)

- **`VIZ-001`** — The summary panel shows mean ± standard deviation only when standard-deviation data is present for both models; otherwise it shows single values.
  _Source:_ `main.py` · `create_benchmark_visualization():486-488`
- **`VIZ-002`** — When running on CPU, the device banner reports system RAM instead of GPU memory.
  _Source:_ `main.py` · `create_benchmark_visualization():396-399`
- **`VIZ-003`** — Chart axes use fixed ranges regardless of the data (audio loss 0–1, video loss 0–2000, throughput 0–1000, time 0–50 s, memory −50–500 MB).
  _Source:_ `main.py` · `create_benchmark_visualization():416,427,443,459,475`
- **`VIZ-004`** — The visualization image is saved at 150 DPI with a white background.
  _Source:_ `main.py` · `create_benchmark_visualization():537`

## Traceability index

| Section | ID | Rule | File | Location | Status |
|---|---|---|---|---|---|
| Configuration | `CFG-001` | Runs on first CUDA GPU if available, else CPU | `main.py` | `module:22` | verified |
| Configuration | `CFG-002` | 10 benchmark iterations by default | `main.py` | `module:27` | verified |
| Configuration | `CFG-003` | 4 samples per batch by default | `main.py` | `module:23` | verified |
| Configuration | `CFG-004` | 2 epochs per run by default | `main.py` | `module:24` | verified |
| Configuration | `CFG-005` | Fixed learning rate 0.001 | `main.py` | `module:25` | verified |
| Configuration | `CFG-006` | `NUM_WORKERS` has no effect; loaders use 0 workers | `main.py` | `module:26` vs `272/329` | verified (contradiction) |
| Configuration | `CFG-007` | Audio = 1 s of 16 kHz (16,000 samples) | `main.py` | `module:30-32` | verified |
| Configuration | `CFG-008` | Video = 16 frames of 64×64 RGB | `main.py` | `module:35-38` | verified |
| Data | `DATA-001` | Audio dataset = 100 samples | `main.py` | `benchmark_audio():271` | verified |
| Data | `DATA-002` | Audio = Gaussian noise normalized to peak 1 | `main.py` | `__getitem__():68-69` | verified |
| Data | `DATA-003` | Audio label is random 0/1 | `main.py` | `__getitem__():72` | verified |
| Data | `DATA-004` | Video dataset = 200 samples | `main.py` | `benchmark_video():328` | verified |
| Data | `DATA-005` | Video = uniform random pixels 0–1 | `main.py` | `__getitem__():87-88` | verified |
| Data | `DATA-006` | Video target = random VMAF 0–100 | `main.py` | `__getitem__():91` | verified |
| Model | `MODEL-001` | Audio pools Wav2Vec2 output by time-averaging | `main.py` | `forward():130` | verified |
| Model | `MODEL-002` | Video FC size assumes ÷8 over three pools | `main.py` | `__init__():161` | verified |
| Training | `TRAIN-001` | Audio uses cross-entropy (classification) | `main.py` | `benchmark_audio():278` | verified |
| Training | `TRAIN-002` | Video uses MSE (regression) | `main.py` | `benchmark_video():335` | verified |
| Training | `TRAIN-003` | Both models use Adam | `main.py` | `279/336` | verified |
| Training | `TRAIN-004` | Epoch loss averaged over batches, not samples | `main.py` | `train_epoch():245` | verified |
| Training | `TRAIN-005` | Throughput = samples ÷ epoch time | `main.py` | `train_epoch():246` | verified |
| Training | `TRAIN-006` | GPU memory used = after − before epoch | `main.py` | `train_epoch():242-243` | verified |
| Training | `TRAIN-007` | On CPU, memory measured from process RSS | `main.py` | `train_epoch():226,242` | verified |
| Training | `TRAIN-008` | Inputs/labels moved to device each batch | `main.py` | `train_epoch():229-230` | verified |
| Training | `TRAIN-009` | Model set to train mode each epoch | `main.py` | `train_epoch():221` | verified |
| Benchmark | `BENCH-001` | Audio benchmark runs before video each run | `main.py` | `__main__:641-642` | verified |
| Benchmark | `BENCH-002` | Peak-memory stats reset before each model | `main.py` | `288/346` | verified |
| Benchmark | `BENCH-003` | Peak memory = max GPU allocated, 0 on CPU | `main.py` | `302/360` | verified |
| Benchmark | `BENCH-004` | Avg throughput = mean of per-epoch throughputs | `main.py` | `314/373` | verified |
| Benchmark | `BENCH-005` | GPU cache emptied after every run | `main.py` | `__main__:648-649` | verified |
| Benchmark | `BENCH-006` | Cross-run metrics reported as mean ± std | `main.py` | `__main__:658-674` | verified |
| Benchmark | `BENCH-007` | Std uses population divisor N | `main.py` | `__main__:659-674` | verified |
| Benchmark | `BENCH-008` | Only first run's epoch data used for charts | `main.py` | `__main__:664,675` | verified |
| Output | `OUT-001` | Base name `benchmark_{GPU}_{machine}_{timestamp}` | `main.py` | `__main__:704` | verified |
| Output | `OUT-002` | GPU name = `CPU` when no GPU | `main.py` | `__main__:701` | verified |
| Output | `OUT-003` | Filename: non-word/hyphen → `_`, trim `_` | `main.py` | `_sanitize_filename():544` | verified |
| Output | `OUT-004` | Machine ID = host network name | `main.py` | `_get_machine_id():549` | verified |
| Output | `OUT-005` | Filename timestamp = local `YYYYMMDD_HHMMSS` | `main.py` | `__main__:703` | verified |
| Output | `OUT-006` | Outputs written to `outputs/`, created if missing | `main.py` | `534/557/726` | verified |
| Output | `OUT-007` | Three files (PNG/JSON/TXT) per session | `main.py` | `__main__:709,728,736` | verified |
| Output | `OUT-008` | JSON has averaged + all per-run results | `main.py` | `__main__:720-723` | verified |
| Output | `OUT-009` | Non-serializable JSON values stringified | `main.py` | `__main__:730` | verified |
| Output | `OUT-010` | TXT speedup labeled "Audio/Video", ratio audio÷video | `main.py` | `generate_txt_report():612` | verified |
| Output | `OUT-011` | PNG speedup labeled "Video/Audio", same audio÷video ratio | `main.py` | `create_benchmark_visualization():505` | verified (contradiction) |
| Output | `OUT-012` | Reported GPU memory = total GB, 0 on CPU | `main.py` | `__main__:697` | verified |
| Visualization | `VIZ-001` | Mean ± std shown only when std data present | `main.py` | `create_benchmark_visualization():486-488` | verified |
| Visualization | `VIZ-002` | CPU banner shows system RAM, not GPU memory | `main.py` | `create_benchmark_visualization():396-399` | verified |
| Visualization | `VIZ-003` | Chart axes use fixed, data-independent ranges | `main.py` | `create_benchmark_visualization():416-475` | verified |
| Visualization | `VIZ-004` | Image saved at 150 DPI, white background | `main.py` | `create_benchmark_visualization():537` | verified |
