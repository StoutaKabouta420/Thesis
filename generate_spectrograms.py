import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from pathlib import Path
import json
from tqdm import tqdm
import pickle

# Set up paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
SEGMENTS_DIR = BASE_DIR / "extracted_segments"
SPECTROGRAM_DIR = BASE_DIR / "spectrograms"
SPECTROGRAM_DIR.mkdir(exist_ok=True)

# Spectrogram parameters
PARAMS = {
    'nfft': 512,           # FFT window size
    'hop_length': 128,     # Hop length (75% overlap with nfft=512)
    'window': 'hann',      # Window function
    'target_duration': 0.5, # Target duration in seconds for padding
    'freq_min': 0,         # Min frequency to display
    'freq_max': 1000,      # Max frequency to display
    'log_scale': False,    # Whether to use log scale for frequency
}

def create_spectrogram(audio_data, sample_rate, params):
    """
    Create a spectrogram from audio data.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sample rate
        params: Dictionary of spectrogram parameters
    
    Returns:
        frequencies: Array of frequency values
        times: Array of time values
        spectrogram: 2D array of power values
    """
    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        audio_data,
        fs=sample_rate,
        window=params['window'],
        nperseg=params['nfft'],
        noverlap=params['nfft'] - params['hop_length'],
        scaling='spectrum'
    )
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
    
    return frequencies, times, Sxx_db

def pad_audio(audio_data, sample_rate, target_duration):
    """
    Pad audio to target duration.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sample rate
        target_duration: Target duration in seconds
    
    Returns:
        Padded audio signal
    """
    target_samples = int(target_duration * sample_rate)
    current_samples = len(audio_data)
    
    if current_samples >= target_samples:
        # Truncate if longer
        return audio_data[:target_samples]
    else:
        # Pad with zeros if shorter
        pad_amount = target_samples - current_samples
        # Pad equally on both sides (or as close as possible)
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        return np.pad(audio_data, (pad_left, pad_right), mode='constant')

def process_all_segments():
    """
    Process all segments and generate spectrograms.
    """
    # Collect all segment files
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
    
    # Store metadata
    metadata = {
        'params': PARAMS,
        'segments': []
    }
    
    # Process each segment
    spectrograms = []
    labels = []  # Store recording source as label
    
    for idx, segment_info in enumerate(tqdm(all_segments, desc="Generating spectrograms")):
        try:
            # Read audio
            sample_rate, audio_data = wavfile.read(segment_info['path'])
            
            # Normalize audio to [-1, 1]
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            
            # Pad audio to target duration
            audio_padded = pad_audio(audio_data, sample_rate, PARAMS['target_duration'])
            
            # Create spectrogram
            freqs, times, spec = create_spectrogram(audio_padded, sample_rate, PARAMS)
            
            # Limit frequency range
            freq_mask = (freqs >= PARAMS['freq_min']) & (freqs <= PARAMS['freq_max'])
            spec_filtered = spec[freq_mask, :]
            freqs_filtered = freqs[freq_mask]
            
            # Store spectrogram
            spectrograms.append(spec_filtered)
            labels.append(segment_info['recording'])
            
            # Store metadata
            metadata['segments'].append({
                'index': idx,
                'filename': segment_info['filename'],
                'recording': segment_info['recording'],
                'original_duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'shape': spec_filtered.shape
            })
            
            # Save individual spectrogram image for first 5 segments
            if idx < 5:
                plt.figure(figsize=(10, 6))
                plt.imshow(spec_filtered, aspect='auto', origin='lower',
                          extent=[times[0], times[-1], freqs_filtered[0], freqs_filtered[-1]],
                          cmap='viridis')
                plt.colorbar(label='Power (dB)')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title(f'Spectrogram: {segment_info["filename"]}')
                plt.tight_layout()
                
                # Create examples directory
                examples_dir = SPECTROGRAM_DIR / "examples"
                examples_dir.mkdir(exist_ok=True)
                plt.savefig(examples_dir / f'spectrogram_{idx:04d}.png', dpi=150)
                plt.close()
                
        except Exception as e:
            print(f"\nError processing {segment_info['filename']}: {e}")
            continue
    
    # Convert to numpy arrays
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)
    
    print(f"\nSuccessfully processed {len(spectrograms)} spectrograms")
    print(f"Spectrogram shape: {spectrograms[0].shape}")
    print(f"Dataset shape: {spectrograms.shape}")
    
    # Save processed data
    np.save(SPECTROGRAM_DIR / 'spectrograms.npy', spectrograms)
    np.save(SPECTROGRAM_DIR / 'labels.npy', labels)
    
    # Save metadata
    with open(SPECTROGRAM_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save frequency and time arrays for reference
    np.save(SPECTROGRAM_DIR / 'frequencies.npy', freqs_filtered)
    np.save(SPECTROGRAM_DIR / 'times.npy', times)
    
    print(f"\nData saved to {SPECTROGRAM_DIR}")
    print("Files created:")
    print("  - spectrograms.npy: Main dataset")
    print("  - labels.npy: Recording source labels")
    print("  - metadata.json: Processing parameters and segment info")
    print("  - frequencies.npy: Frequency axis values")
    print("  - times.npy: Time axis values")
    print("  - examples/: Sample spectrogram images")
    
    return spectrograms, labels, metadata

def visualize_dataset_summary(spectrograms, labels):
    """
    Create summary visualizations of the dataset.
    """
    print("\nCreating dataset summary visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Average spectrogram
    avg_spec = np.mean(spectrograms, axis=0)
    im1 = axes[0, 0].imshow(avg_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title('Average Spectrogram')
    axes[0, 0].set_xlabel('Time bins')
    axes[0, 0].set_ylabel('Frequency bins')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Standard deviation
    std_spec = np.std(spectrograms, axis=0)
    im2 = axes[0, 1].imshow(std_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title('Standard Deviation')
    axes[0, 1].set_xlabel('Time bins')
    axes[0, 1].set_ylabel('Frequency bins')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Distribution of recordings
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[1, 0].bar(range(len(unique_labels)), counts)
    axes[1, 0].set_xticks(range(len(unique_labels)))
    axes[1, 0].set_xticklabels([label.split('_')[0] for label in unique_labels], rotation=45)
    axes[1, 0].set_title('Segments per Recording')
    axes[1, 0].set_xlabel('Recording')
    axes[1, 0].set_ylabel('Count')
    
    # 4. Power distribution
    all_powers = spectrograms.flatten()
    axes[1, 1].hist(all_powers, bins=50, edgecolor='black')
    axes[1, 1].set_title('Power Distribution (dB)')
    axes[1, 1].set_xlabel('Power (dB)')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(SPECTROGRAM_DIR / 'dataset_summary.png', dpi=150)
    plt.close()
    
    print(f"Summary visualization saved to {SPECTROGRAM_DIR / 'dataset_summary.png'}")

if __name__ == "__main__":
    print("Starting spectrogram generation...")
    print(f"Input directory: {SEGMENTS_DIR}")
    print(f"Output directory: {SPECTROGRAM_DIR}")
    print(f"\nSpectrogram parameters:")
    for key, value in PARAMS.items():
        print(f"  {key}: {value}")
    
    # Process all segments
    spectrograms, labels, metadata = process_all_segments()
    
    # Create summary visualizations
    visualize_dataset_summary(spectrograms, labels)
    
    print("\n✓ Spectrogram generation complete!")