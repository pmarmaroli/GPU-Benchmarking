import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import psutil
import os
import numpy as np
from datetime import datetime
import json
from transformers import Wav2Vec2Model, Wav2Vec2Config
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
NUM_WORKERS = 2
NUM_RUNS = 10  # Number of times to run the benchmark for averaging

# Audio config
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 1  # seconds (reduced for faster CPU testing)
AUDIO_SEQUENCE_LENGTH = AUDIO_SAMPLE_RATE * AUDIO_DURATION

# Video config
VIDEO_NUM_FRAMES = 16  # increased for more complexity
VIDEO_FRAME_HEIGHT = 64  # increased resolution
VIDEO_FRAME_WIDTH = 64  # increased resolution
VIDEO_CHANNELS = 3

print("=" * 80)
print("DEEP LEARNING BENCHMARK: Audio (Wav2Vec2) + Video (3D CNN) Training")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print("=" * 80)

# ============================================================================
# SYNTHETIC DATASETS
# ============================================================================

class SyntheticAudioDataset(Dataset):
    """Synthetic audio dataset for Wav2Vec2 training (speech classification)"""
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.sequence_length = AUDIO_SEQUENCE_LENGTH
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic audio waveform (random noise normalized)
        audio = np.random.randn(self.sequence_length).astype(np.float32)
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        # Binary classification label (0 or 1)
        label = np.random.randint(0, 2)
        
        return torch.from_numpy(audio), label


class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset for 3D CNN VMAF quality prediction"""
    def __init__(self, num_samples=200):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic video frames (T, C, H, W)
        video = np.random.rand(VIDEO_NUM_FRAMES, VIDEO_CHANNELS, 
                               VIDEO_FRAME_HEIGHT, VIDEO_FRAME_WIDTH).astype(np.float32)
        
        # VMAF score (0-100, continuous regression task)
        vmaf_score = np.float32(np.random.uniform(0, 100))
        
        return torch.from_numpy(video), vmaf_score


# ============================================================================
# AUDIO MODEL: Wav2Vec2-based for Speech Classification
# ============================================================================

class AudioClassificationModel(nn.Module):
    """Wav2Vec2 based audio classifier"""
    def __init__(self, num_labels=2):
        super().__init__()
        
        print("  → Initializing Wav2Vec2 model (this may take a moment)...")
        # Load pretrained Wav2Vec2 config (simplified for CPU)
        config = Wav2Vec2Config(
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=512,
            num_attention_heads=2,
        )
        self.wav2vec2 = Wav2Vec2Model(config)
        print("  ✓ Wav2Vec2 model initialized")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_labels)
        )
    
    def forward(self, input_values):
        # Wav2Vec2 expects (batch, sequence_length)
        outputs = self.wav2vec2(input_values, output_hidden_states=False)
        hidden_states = outputs.last_hidden_state  # (batch, time_steps, hidden_size)
        
        # Global average pooling over time
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)
        
        # Classification
        logits = self.classifier(pooled)
        return logits


# ============================================================================
# VIDEO MODEL: 3D CNN for VMAF Quality Prediction
# ============================================================================

class Video3DCNN(nn.Module):
    """3D CNN for video quality (VMAF) prediction"""
    def __init__(self):
        super().__init__()
        
        # 3D convolutional layers (more complex)
        self.conv3d_1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(64)
        self.pool3d_2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_3 = nn.BatchNorm3d(128)
        self.pool3d_3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Fully connected layers
        # After 3 pooling layers: (frames/8, height/8, width/8)
        self.fc_input_size = 128 * (VIDEO_NUM_FRAMES // 8) * (VIDEO_FRAME_HEIGHT // 8) * (VIDEO_FRAME_WIDTH // 8)
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Regression: VMAF score
        )
    
    def forward(self, x):
        # x: (batch, frames, channels, height, width)
        # Reorder to (batch, channels, frames, height, width) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.conv3d_1(x)
        x = self.bn3d_1(x)
        x = torch.relu(x)
        x = self.pool3d_1(x)
        
        x = self.conv3d_2(x)
        x = self.bn3d_2(x)
        x = torch.relu(x)
        x = self.pool3d_2(x)
        
        x = self.conv3d_3(x)
        x = self.bn3d_3(x)
        x = torch.relu(x)
        x = self.pool3d_3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage():
    """Get GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def train_epoch(model, dataloader, criterion, optimizer, device, model_name):
    """Train for one epoch and collect metrics"""
    model.train()
    total_loss = 0
    samples_processed = 0
    
    epoch_start_time = time.time()
    initial_memory = get_gpu_memory_usage() if torch.cuda.is_available() else get_memory_usage()
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        samples_processed += inputs.size(0)
    
    epoch_time = time.time() - epoch_start_time
    final_memory = get_gpu_memory_usage() if torch.cuda.is_available() else get_memory_usage()
    memory_used = final_memory - initial_memory
    
    avg_loss = total_loss / len(dataloader)
    throughput = samples_processed / epoch_time  # samples per second
    
    return {
        'loss': avg_loss,
        'time': epoch_time,
        'throughput': throughput,
        'samples': samples_processed,
        'memory_used_mb': memory_used
    }


# ============================================================================
# BENCHMARKING PIPELINE
# ============================================================================

def benchmark_audio():
    """Benchmark Wav2Vec2 audio training"""
    print("\n" + "=" * 80)
    print("AUDIO BENCHMARK: Wav2Vec2 Speech Classification")
    print("=" * 80)
    print(f"Dataset: {AUDIO_SEQUENCE_LENGTH} samples, {AUDIO_SAMPLE_RATE} Hz")
    print(f"Model: Wav2Vec2 + Classification Head")
    
    # Create dataset and dataloader
    print("\n[1/4] Creating synthetic audio dataset...")
    dataset = SyntheticAudioDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  ✓ Created {len(dataset)} audio samples")
    
    # Initialize model
    print("\n[2/4] Initializing audio model...")
    model = AudioClassificationModel(num_labels=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    results = []
    total_start = time.time()
    
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS} - Training...", end='', flush=True)
        epoch_metrics = train_epoch(model, dataloader, criterion, optimizer, DEVICE, "Audio")
        results.append(epoch_metrics)
        
        print(" Done!")
        print(f"    Loss: {epoch_metrics['loss']:.4f}")
        print(f"    Time: {epoch_metrics['time']:.2f}s")
        print(f"    Throughput: {epoch_metrics['throughput']:.2f} samples/sec")
        print(f"    Memory used: {epoch_metrics['memory_used_mb']:.2f} MB")
    
    total_time = time.time() - total_start
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    print(f"\nTotal Training Time: {total_time:.2f}s")
    print(f"Average Epoch Time: {total_time / NUM_EPOCHS:.2f}s")
    print(f"Peak GPU Memory: {peak_memory:.2f} MB")
    print(f"Average Throughput: {np.mean([r['throughput'] for r in results]):.2f} samples/sec")
    
    return {
        'model': 'Wav2Vec2',
        'total_time': total_time,
        'epochs': results,
        'peak_memory_mb': peak_memory,
        'avg_throughput': np.mean([r['throughput'] for r in results])
    }


def benchmark_video():
    """Benchmark 3D CNN video quality prediction training"""
    print("\n" + "=" * 80)
    print("VIDEO BENCHMARK: 3D CNN VMAF Quality Prediction")
    print("=" * 80)
    print(f"Dataset: {VIDEO_NUM_FRAMES} frames, {VIDEO_FRAME_HEIGHT}x{VIDEO_FRAME_WIDTH}, {VIDEO_CHANNELS} channels")
    print(f"Model: 3D CNN for VMAF Score Regression")
    
    # Create dataset and dataloader
    print("\n[1/4] Creating synthetic video dataset...")
    dataset = SyntheticVideoDataset(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  ✓ Created {len(dataset)} video samples")
    
    # Initialize model
    print("\n[2/4] Initializing video model...")
    model = Video3DCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"  ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  ✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    print("\n[3/4] Training video model...")
    results = []
    total_start = time.time()
    
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS} - Training...", end='', flush=True)
        epoch_metrics = train_epoch(model, dataloader, criterion, optimizer, DEVICE, "Video")
        results.append(epoch_metrics)
        
        print(" Done!")
        print(f"    Loss (MSE): {epoch_metrics['loss']:.4f}")
        print(f"    Time: {epoch_metrics['time']:.2f}s")
        print(f"    Throughput: {epoch_metrics['throughput']:.2f} samples/sec")
        print(f"    Memory used: {epoch_metrics['memory_used_mb']:.2f} MB")
    
    total_time = time.time() - total_start
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    print("\n[4/4] Video benchmark complete!")
    print(f"  ✓ Total Training Time: {total_time:.2f}s")
    print(f"  ✓ Average Epoch Time: {total_time / NUM_EPOCHS:.2f}s")
    print(f"  ✓ Peak GPU Memory: {peak_memory:.2f} MB")
    print(f"  ✓ Average Throughput: {np.mean([r['throughput'] for r in results]):.2f} samples/sec")
    
    return {
        'model': '3D CNN',
        'total_time': total_time,
        'epochs': results,
        'peak_memory_mb': peak_memory,
        'avg_throughput': np.mean([r['throughput'] for r in results])
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_benchmark_visualization(audio_results, video_results, device_info):
    """Create comprehensive PNG visualization of benchmark results"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Color scheme
    audio_color = '#FF6B6B'
    video_color = '#4ECDC4'
    
    # ========== Device Information Panel ==========
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis('off')
    
    device_text = f"Device: {device_info['device']} | "
    if device_info['cuda_available']:
        device_text += f"GPU: {device_info['gpu_name']} | GPU Memory: {device_info['gpu_memory']:.2f} GB"
    else:
        device_text += "CPU Mode | System RAM: {:.2f} GB".format(psutil.virtual_memory().total / 1e9)
    
    device_text += f" | Timestamp: {device_info['timestamp']}"
    
    ax_info.text(0.5, 0.7, device_text, ha='center', va='center', fontsize=12, 
                 fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax_info.text(0.5, 0.2, f"Batch Size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS} | Learning Rate: {LEARNING_RATE}", 
                 ha='center', va='center', fontsize=10, style='italic')
    
    # ========== Audio Training Loss ==========
    ax1 = fig.add_subplot(gs[1, 0])
    audio_losses = [epoch['loss'] for epoch in audio_results['epochs']]
    epochs = range(1, len(audio_losses) + 1)
    ax1.plot(epochs, audio_losses, marker='o', linewidth=2, markersize=8, color=audio_color, label='Wav2Vec2')
    ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Loss (CrossEntropy)', fontsize=10, fontweight='bold')
    ax1.set_title('Audio Model: Training Loss', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ========== Video Training Loss ==========
    ax2 = fig.add_subplot(gs[1, 1])
    video_losses = [epoch['loss'] for epoch in video_results['epochs']]
    ax2.plot(epochs, video_losses, marker='s', linewidth=2, markersize=8, color=video_color, label='3D CNN')
    ax2.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Loss (MSE)', fontsize=10, fontweight='bold')
    ax2.set_title('Video Model: Training Loss', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 2000)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ========== Throughput Comparison ==========
    ax3 = fig.add_subplot(gs[1, 2])
    audio_throughputs = [epoch['throughput'] for epoch in audio_results['epochs']]
    video_throughputs = [epoch['throughput'] for epoch in video_results['epochs']]
    
    x = np.arange(len(epochs))
    width = 0.35
    ax3.bar(x - width/2, audio_throughputs, width, label='Wav2Vec2', color=audio_color, alpha=0.8)
    ax3.bar(x + width/2, video_throughputs, width, label='3D CNN', color=video_color, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Throughput (samples/sec)', fontsize=10, fontweight='bold')
    ax3.set_title('Throughput per Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1000)
    ax3.set_xticks(x)
    ax3.set_xticklabels(epochs)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== Epoch Time Comparison ==========
    ax4 = fig.add_subplot(gs[2, 0])
    audio_times = [epoch['time'] for epoch in audio_results['epochs']]
    video_times = [epoch['time'] for epoch in video_results['epochs']]
    
    ax4.bar(x - width/2, audio_times, width, label='Wav2Vec2', color=audio_color, alpha=0.8)
    ax4.bar(x + width/2, video_times, width, label='3D CNN', color=video_color, alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax4.set_title('Training Time per Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 50)
    ax4.set_xticks(x)
    ax4.set_xticklabels(epochs)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ========== Memory Usage ==========
    ax5 = fig.add_subplot(gs[2, 1])
    audio_memory = [epoch['memory_used_mb'] for epoch in audio_results['epochs']]
    video_memory = [epoch['memory_used_mb'] for epoch in video_results['epochs']]
    
    ax5.bar(x - width/2, audio_memory, width, label='Wav2Vec2', color=audio_color, alpha=0.8)
    ax5.bar(x + width/2, video_memory, width, label='3D CNN', color=video_color, alpha=0.8)
    ax5.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Memory (MB)', fontsize=10, fontweight='bold')
    ax5.set_title('Memory Usage per Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylim(-50, 500)
    ax5.set_xticks(x)
    ax5.set_xticklabels(epochs)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========== Summary Statistics ==========
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Check if we have standard deviation data (from multiple runs)
    has_std = 'total_time_std' in audio_results and 'total_time_std' in video_results
    
    if has_std:
        summary_text = f"""
    BENCHMARK SUMMARY
    (Average of {NUM_RUNS} runs ± std dev)
    
    Wav2Vec2 (Audio)
    ├─ Avg Total Time: {audio_results['total_time']:.2f} ± {audio_results['total_time_std']:.2f}s
    ├─ Avg Throughput: {audio_results['avg_throughput']:.2f} ± {audio_results['avg_throughput_std']:.2f} samples/sec
    ├─ Avg Peak Memory: {audio_results['peak_memory_mb']:.2f} ± {audio_results['peak_memory_mb_std']:.2f} MB
    └─ Avg Loss: {np.mean(audio_losses):.4f}
    
    3D CNN (Video)
    ├─ Avg Total Time: {video_results['total_time']:.2f} ± {video_results['total_time_std']:.2f}s
    ├─ Avg Throughput: {video_results['avg_throughput']:.2f} ± {video_results['avg_throughput_std']:.2f} samples/sec
    ├─ Avg Peak Memory: {video_results['peak_memory_mb']:.2f} ± {video_results['peak_memory_mb_std']:.2f} MB
    └─ Avg Loss: {np.mean(video_losses):.4f}
    
    Speedup (Video/Audio): {audio_results['total_time']/video_results['total_time']:.2f}x
        """
    else:
        summary_text = f"""
    BENCHMARK SUMMARY
    
    Wav2Vec2 (Audio)
    ├─ Total Time: {audio_results['total_time']:.2f}s
    ├─ Avg Throughput: {audio_results['avg_throughput']:.2f} samples/sec
    ├─ Peak Memory: {audio_results['peak_memory_mb']:.2f} MB
    └─ Avg Loss: {np.mean(audio_losses):.4f}
    
    3D CNN (Video)
    ├─ Total Time: {video_results['total_time']:.2f}s
    ├─ Avg Throughput: {video_results['avg_throughput']:.2f} samples/sec
    ├─ Peak Memory: {video_results['peak_memory_mb']:.2f} MB
    └─ Avg Loss: {np.mean(video_losses):.4f}
    
    Speedup (Video/Audio): {audio_results['total_time']/video_results['total_time']:.2f}x
        """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Deep Learning Benchmark: Audio (Wav2Vec2) vs Video (3D CNN)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'benchmark_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Visualization saved to: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("Starting GPU Benchmarking Suite...")
    print(f"Running {NUM_RUNS} iterations for average metrics")
    print("🚀" * 40)
    
    # Store results from all runs
    all_audio_results = []
    all_video_results = []
    
    for run in range(NUM_RUNS):
        print(f"\n{'='*80}")
        print(f"RUN {run + 1}/{NUM_RUNS}")
        print(f"{'='*80}")
        
        audio_results = benchmark_audio()
        video_results = benchmark_video()
        
        all_audio_results.append(audio_results)
        all_video_results.append(video_results)
        
        # Clear GPU cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate averages
    print("\n" + "=" * 80)
    print("CALCULATING AVERAGE METRICS...")
    print("=" * 80)
    
    avg_audio = {
        'model': 'Wav2Vec2',
        'total_time': np.mean([r['total_time'] for r in all_audio_results]),
        'total_time_std': np.std([r['total_time'] for r in all_audio_results]),
        'avg_throughput': np.mean([r['avg_throughput'] for r in all_audio_results]),
        'avg_throughput_std': np.std([r['avg_throughput'] for r in all_audio_results]),
        'peak_memory_mb': np.mean([r['peak_memory_mb'] for r in all_audio_results]),
        'peak_memory_mb_std': np.std([r['peak_memory_mb'] for r in all_audio_results]),
        'epochs': all_audio_results[0]['epochs']  # Use first run's epoch data for visualization
    }
    
    avg_video = {
        'model': '3D CNN',
        'total_time': np.mean([r['total_time'] for r in all_video_results]),
        'total_time_std': np.std([r['total_time'] for r in all_video_results]),
        'avg_throughput': np.mean([r['avg_throughput'] for r in all_video_results]),
        'avg_throughput_std': np.std([r['avg_throughput'] for r in all_video_results]),
        'peak_memory_mb': np.mean([r['peak_memory_mb'] for r in all_video_results]),
        'peak_memory_mb_std': np.std([r['peak_memory_mb'] for r in all_video_results]),
        'epochs': all_video_results[0]['epochs']  # Use first run's epoch data for visualization
    }
    
    # Summary comparison
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY (Average of {NUM_RUNS} runs)")
    print("=" * 80)
    print(f"\n{'Model':<20} {'Total Time (s)':<25} {'Avg Throughput':<30} {'Peak Memory (MB)':<25}")
    print("-" * 100)
    print(f"{'Wav2Vec2':<20} {avg_audio['total_time']:<8.2f} ± {avg_audio['total_time_std']:<8.2f}   "
          f"{avg_audio['avg_throughput']:<8.2f} ± {avg_audio['avg_throughput_std']:<8.2f} samples/sec   "
          f"{avg_audio['peak_memory_mb']:<8.2f} ± {avg_audio['peak_memory_mb_std']:<8.2f}")
    print(f"{'3D CNN (Video)':<20} {avg_video['total_time']:<8.2f} ± {avg_video['total_time_std']:<8.2f}   "
          f"{avg_video['avg_throughput']:<8.2f} ± {avg_video['avg_throughput_std']:<8.2f} samples/sec   "
          f"{avg_video['peak_memory_mb']:<8.2f} ± {avg_video['peak_memory_mb_std']:<8.2f}")
    
    # Collect device information
    device_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    }
    
    # Create PNG visualization (using averaged results)
    print("\n" + "=" * 80)
    print("Generating visualization...")
    create_benchmark_visualization(avg_audio, avg_video, device_info)
    
    # Save results to JSON
    print("\nSaving results to JSON...")
    results_summary = {
        'timestamp': device_info['timestamp'],
        'device': device_info['device'],
        'gpu_name': device_info['gpu_name'],
        'gpu_memory_gb': device_info['gpu_memory'],
        'cuda_available': device_info['cuda_available'],
        'num_runs': NUM_RUNS,
        'audio_averaged': avg_audio,
        'video_averaged': avg_video,
        'audio_all_runs': all_audio_results,
        'video_all_runs': all_video_results
    }
    
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"  ✓ Results saved to {json_path}")
    print("\n" + "=" * 80)
    print(f"✅ All benchmarks completed successfully! ({NUM_RUNS} runs)")
    print("=" * 80)