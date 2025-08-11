import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter
from pathlib import Path
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
SEGMENTS_DIR = BASE_DIR / "extracted_segments_padded"  # Use padded segments
SPECTROGRAM_DIR = BASE_DIR / "spectrograms_focused"
SPECTROGRAM_DIR.mkdir(exist_ok=True)

# Based on the paper's analysis, Bryde's whale calls are primarily in 0-500 Hz range
# with key components around 42 Hz, 75 Hz, and 105 Hz
SPECTROGRAM_CONFIG = {
    'sample_rate': 8000,      # Target sample rate
    'nfft': 4096,             # FFT size from paper
    'overlap': 0.95,          # 95% overlap from paper
    'freq_min': 0,            # Start from 0 Hz
    'freq_max': 500,          # Up to 500 Hz
    'target_shape': (129, 64), # Fixed output shape (freq_bins, time_bins)
    'window': 'hann',         # Window type
    'min_duration': 0.5,      # Minimum segment duration to process
}

# Enhancement parameters tuned for whale calls
ENHANCEMENT_PARAMS = {
    'noise_floor_percentile': 10,    # Percentile for noise estimation
    'contrast_low_percentile': 10,   # Low percentile for contrast stretch
    'contrast_high_percentile': 95,  # High percentile for contrast stretch
    'median_filter_size': (3, 3),    # Median filter kernel size
    'gaussian_sigma': 0.5,            # Gaussian smoothing sigma
}

def load_and_preprocess_audio(wav_path, target_sr=8000):
    """
    Load and preprocess audio file.
    """
    # Read audio
    sample_rate, audio_data = wavfile.read(wav_path)
    
    # Convert to float32
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)
    
    # Resample if needed
    if sample_rate != target_sr:
        # Calculate new length
        new_length = int(len(audio_data) * target_sr / sample_rate)
        audio_data = signal.resample(audio_data, new_length)
        sample_rate = target_sr
    
    # Normalize
    audio_data = audio_data - np.mean(audio_data)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = 0.95 * audio_data / max_val
    
    return audio_data, sample_rate

def compute_high_quality_spectrogram(audio_data, sample_rate, config):
    """
    Compute high-quality spectrogram similar to Raven Pro.
    """
    nfft = config['nfft']
    overlap_samples = int(nfft * config['overlap'])
    hop_length = nfft - overlap_samples
    
    # Ensure audio is long enough
    if len(audio_data) < nfft:
        # Pad with zeros
        audio_data = np.pad(audio_data, (0, nfft - len(audio_data)), mode='constant')
    
    # Compute spectrogram using Hann window
    window = signal.windows.hann(nfft, sym=False)
    
    freqs, times, Sxx = signal.spectrogram(
        audio_data,
        fs=sample_rate,
        window=window,
        nperseg=nfft,
        noverlap=overlap_samples,
        detrend='constant',
        scaling='density',
        mode='psd'
    )
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Extract frequency range of interest
    freq_mask = (freqs >= config['freq_min']) & (freqs <= config['freq_max'])
    Sxx_db = Sxx_db[freq_mask, :]
    freqs = freqs[freq_mask]
    
    return freqs, times, Sxx_db

def enhance_spectrogram(spec_db, params):
    """
    Enhance spectrogram for better visualization and feature extraction.
    """
    # Step 1: Estimate and remove noise floor
    noise_floor = np.percentile(spec_db, params['noise_floor_percentile'])
    spec_denoised = spec_db - noise_floor
    
    # Step 2: Apply median filter to reduce speckle noise
    spec_filtered = median_filter(spec_denoised, size=params['median_filter_size'])
    
    # Step 3: Apply gentle Gaussian smoothing
    spec_smoothed = gaussian_filter(spec_filtered, sigma=params['gaussian_sigma'])
    
    # Step 4: Contrast stretching
    vmin = np.percentile(spec_smoothed, params['contrast_low_percentile'])
    vmax = np.percentile(spec_smoothed, params['contrast_high_percentile'])
    
    # Clip and normalize
    spec_stretched = np.clip(spec_smoothed, vmin, vmax)
    spec_normalized = (spec_stretched - vmin) / (vmax - vmin + 1e-10)
    
    # Step 5: Apply gamma correction for better contrast
    gamma = 0.7  # Emphasize mid-range values
    spec_enhanced = np.power(spec_normalized, gamma)
    
    # Scale back to dB-like range for consistency
    spec_final = spec_enhanced * 60 - 60  # Map to roughly -60 to 0 dB
    
    return spec_final

def resize_to_fixed_shape(spectrogram, target_shape):
    """
    Resize spectrogram to fixed shape using interpolation.
    """
    from scipy.ndimage import zoom
    
    if spectrogram.shape == target_shape:
        return spectrogram
    
    # Calculate zoom factors
    zoom_factors = (
        target_shape[0] / spectrogram.shape[0],
        target_shape[1] / spectrogram.shape[1]
    )
    
    # Use bilinear interpolation
    resized = zoom(spectrogram, zoom_factors, order=1)
    
    return resized

def visualize_spectrogram(spec, freqs, times, title="Spectrogram"):
    """
    Create a high-quality visualization of the spectrogram.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create extent for proper axis labels
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    
    # Plot spectrogram
    im = ax.imshow(
        spec,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='hot',
        interpolation='bilinear'
    )
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')
    
    return fig

def process_all_segments():
    """
    Process all segments to create high-quality spectrograms.
    """
    # Find all segments
    all_segments = []
    for recording_folder in SEGMENTS_DIR.iterdir():
        if recording_folder.is_dir():
            wav_files = list(recording_folder.glob("*.wav"))
            for wav_file in wav_files:
                all_segments.append({
                    'path': wav_file,
                    'recording': recording_folder.name,
                    'filename': wav_file.name
                })
    
    print(f"Found {len(all_segments)} segments to process")
    
    # Storage for results
    spectrograms_original = []
    spectrograms_enhanced = []
    labels = []
    metadata = []
    
    processed_count = 0
    skipped_count = 0
    
    for idx, segment_info in enumerate(tqdm(all_segments, desc="Processing segments")):
        try:
            # Load audio
            audio_data, sample_rate = load_and_preprocess_audio(
                segment_info['path'], 
                SPECTROGRAM_CONFIG['sample_rate']
            )
            
            # Check duration
            duration = len(audio_data) / sample_rate
            if duration < SPECTROGRAM_CONFIG['min_duration']:
                skipped_count += 1
                continue
            
            # Compute spectrogram
            freqs, times, spec_db = compute_high_quality_spectrogram(
                audio_data, sample_rate, SPECTROGRAM_CONFIG
            )
            
            # Create original version (just normalized)
            spec_original = resize_to_fixed_shape(spec_db, SPECTROGRAM_CONFIG['target_shape'])
            
            # Create enhanced version
            spec_enhanced_full = enhance_spectrogram(spec_db, ENHANCEMENT_PARAMS)
            spec_enhanced = resize_to_fixed_shape(spec_enhanced_full, SPECTROGRAM_CONFIG['target_shape'])
            
            # Store results
            spectrograms_original.append(spec_original)
            spectrograms_enhanced.append(spec_enhanced)
            labels.append(segment_info['recording'])
            
            # Store metadata
            metadata.append({
                'filename': segment_info['filename'],
                'recording': segment_info['recording'],
                'duration': float(duration),
                'index': processed_count,
                'freq_range': [float(freqs[0]), float(freqs[-1])],
                'time_range': [float(times[0]), float(times[-1])]
            })
            
            processed_count += 1
            
            # Save example visualizations for first few
            if processed_count <= 10:
                save_example_comparison(
                    spec_original, spec_enhanced, 
                    freqs, times,
                    segment_info['filename'], 
                    processed_count
                )
            
        except Exception as e:
            print(f"\nError processing {segment_info['filename']}: {e}")
            skipped_count += 1
            continue
    
    print(f"\nProcessed {processed_count} segments, skipped {skipped_count}")
    
    if processed_count > 0:
        # Convert to numpy arrays
        spectrograms_original = np.array(spectrograms_original)
        spectrograms_enhanced = np.array(spectrograms_enhanced)
        labels = np.array(labels)
        
        # Save arrays
        np.save(SPECTROGRAM_DIR / 'spectrograms_original.npy', spectrograms_original)
        np.save(SPECTROGRAM_DIR / 'spectrograms_enhanced.npy', spectrograms_enhanced)
        np.save(SPECTROGRAM_DIR / 'labels.npy', labels)
        
        # Save metadata
        with open(SPECTROGRAM_DIR / 'metadata.json', 'w') as f:
            json.dump({
                'config': SPECTROGRAM_CONFIG,
                'enhancement_params': ENHANCEMENT_PARAMS,
                'shape': SPECTROGRAM_CONFIG['target_shape'],
                'n_samples': processed_count,
                'segments': metadata
            }, f, indent=2, default=str)
        
        print(f"\nSaved spectrograms with shape: {spectrograms_enhanced.shape}")
        print(f"  Original: spectrograms_original.npy")
        print(f"  Enhanced: spectrograms_enhanced.npy (recommended for training)")
        
        # Print statistics
        print(f"\nSpectrogram statistics:")
        print(f"  Original - min: {spectrograms_original.min():.2f}, max: {spectrograms_original.max():.2f}")
        print(f"  Enhanced - min: {spectrograms_enhanced.min():.2f}, max: {spectrograms_enhanced.max():.2f}")
    
    return spectrograms_original, spectrograms_enhanced, labels

def save_example_comparison(spec_original, spec_enhanced, freqs, times, filename, idx):
    """
    Save comparison visualization of original vs enhanced spectrogram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Common extent for both plots
    extent = [0, SPECTROGRAM_CONFIG['target_shape'][1], 
              0, SPECTROGRAM_CONFIG['target_shape'][0]]
    
    # Original
    im1 = axes[0].imshow(spec_original, aspect='auto', origin='lower',
                         cmap='hot', interpolation='bilinear', extent=extent)
    axes[0].set_title('Original Spectrogram')
    axes[0].set_xlabel('Time bins')
    axes[0].set_ylabel('Frequency bins')
    plt.colorbar(im1, ax=axes[0], label='dB')
    
    # Enhanced
    im2 = axes[1].imshow(spec_enhanced, aspect='auto', origin='lower',
                         cmap='hot', interpolation='bilinear', extent=extent)
    axes[1].set_title('Enhanced Spectrogram')
    axes[1].set_xlabel('Time bins')
    axes[1].set_ylabel('Frequency bins')
    plt.colorbar(im2, ax=axes[1], label='Power')
    
    # Add frequency labels on the right
    freq_labels = [0, 100, 200, 300, 400, 500]
    freq_positions = [i * SPECTROGRAM_CONFIG['target_shape'][0] / 500 for i in freq_labels]
    axes[1].set_yticks(freq_positions[:len(freq_labels)])
    axes[1].set_yticklabels(freq_labels[:len(freq_labels)])
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].set_ylabel('Frequency (Hz)', rotation=270, labelpad=20)
    
    fig.suptitle(f'Spectrogram Comparison: {filename[:50]}...')
    plt.tight_layout()
    
    examples_dir = SPECTROGRAM_DIR / "examples"
    examples_dir.mkdir(exist_ok=True)
    plt.savefig(examples_dir / f'comparison_{idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Focused Spectrogram Generation for Bryde's Whale Calls")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    for key, value in SPECTROGRAM_CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\nProcessing segments from: {SEGMENTS_DIR}")
    print(f"Output directory: {SPECTROGRAM_DIR}")
    
    # Process all segments
    specs_orig, specs_enh, labels = process_all_segments()
    
    print("\n✓ Spectrogram generation complete!")
    print(f"Check {SPECTROGRAM_DIR}/examples/ for visualizations")
    
    # Recommendation for autoencoder
    print("\n" + "=" * 60)
    print("RECOMMENDATION FOR AUTOENCODER TRAINING:")
    print("=" * 60)
    print("Use 'spectrograms_enhanced.npy' for training your autoencoder")
    print("These spectrograms have:")
    print("  - Consistent shape (129 freq bins × 64 time bins)")
    print("  - Enhanced harmonic structure")
    print("  - Reduced noise")
    print("  - Normalized contrast")
    print("\nLoad them with:")
    print("  spectrograms = np.load('spectrograms_focused/spectrograms_enhanced.npy')")
    print(f"  Shape: {SPECTROGRAM_CONFIG['target_shape']} per spectrogram")