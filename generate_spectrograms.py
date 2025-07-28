import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
SEGMENTS_DIR = BASE_DIR / "extracted_segments"
SPECTROGRAM_DIR = BASE_DIR / "spectrograms_optimized"
SPECTROGRAM_DIR.mkdir(exist_ok=True)

# Optimized parameters for whale bioacoustics
PARAMS = {
    'nfft': 1024,           # Balance between frequency and time resolution
    'hop_length': 64,       # Much smaller hop for better time resolution (~94% overlap)
    'window': 'hann',       # Window function
    'target_duration': 0.5, # Target duration in seconds
    'freq_min': 10,         # Lower min to capture fundamental frequencies
    'freq_max': 600,        # Slightly higher max to see harmonics
    'pre_emphasis': 0.0,    # No pre-emphasis initially - whale calls are low frequency
    'ref_db': 1.0,          # Reference for dB calculation
    'min_db': -100,         # Wider dynamic range
    'max_db': 0,            # Maximum dB to display
    'normalize_audio': True, # Normalize each segment
    'apply_bandpass': False, # Don't filter initially - see what's there
}

def normalize_audio(audio_data):
    """Normalize audio to [-1, 1] range while preserving dynamics."""
    # Remove DC offset first
    audio_data = audio_data - np.mean(audio_data)
    
    # Find the maximum absolute value
    max_val = np.max(np.abs(audio_data))
    
    if max_val > 0:
        # Normalize to [-0.9, 0.9] to avoid clipping
        return 0.9 * audio_data / max_val
    return audio_data

def compute_spectrogram_scipy(audio_data, sample_rate, params):
    """
    Create spectrogram using scipy with better parameters for whale calls.
    """
    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        audio_data,
        fs=sample_rate,
        window=signal.get_window(params['window'], params['nfft']),
        nperseg=params['nfft'],
        noverlap=params['nfft'] - params['hop_length'],
        scaling='density',
        mode='magnitude'
    )
    
    # Convert to power and then dB
    Sxx_power = Sxx ** 2
    
    # Use a more appropriate reference for underwater acoustics
    # 1 µPa is standard reference for underwater sound
    Sxx_db = 10 * np.log10(Sxx_power / params['ref_db'] + 1e-10)
    
    return frequencies, times, Sxx_db

def adaptive_denoise(spec_db, noise_floor_percentile=10):
    """
    Adaptive denoising based on estimated noise floor.
    """
    # Estimate noise floor from lower percentile of each frequency bin
    noise_floor = np.percentile(spec_db, noise_floor_percentile, axis=1, keepdims=True)
    
    # Subtract noise floor (spectral subtraction)
    spec_denoised = spec_db - noise_floor
    
    # Set negative values to minimum
    spec_denoised[spec_denoised < 0] = 0
    
    return spec_denoised

def enhance_contrast(spec_db, method='percentile'):
    """
    Enhance contrast using various methods.
    """
    if method == 'percentile':
        # Use percentile scaling for better contrast
        vmin = np.percentile(spec_db, 5)
        vmax = np.percentile(spec_db, 95)
        
        # Clip and scale
        spec_scaled = np.clip(spec_db, vmin, vmax)
        spec_scaled = (spec_scaled - vmin) / (vmax - vmin)
        
        # Apply mild gamma correction
        spec_enhanced = np.power(spec_scaled, 0.8)
        
        # Scale back to dB range
        return spec_enhanced * (vmax - vmin) + vmin
    
    return spec_db

def create_multi_resolution_spectrogram(audio_data, sample_rate):
    """
    Create spectrograms at multiple resolutions for comparison.
    """
    resolutions = [
        {'nfft': 512, 'hop': 32, 'name': 'high_time_res'},
        {'nfft': 1024, 'hop': 64, 'name': 'balanced'},
        {'nfft': 2048, 'hop': 128, 'name': 'high_freq_res'}
    ]
    
    results = {}
    
    for res in resolutions:
        f, t, s = signal.spectrogram(
            audio_data,
            fs=sample_rate,
            nperseg=res['nfft'],
            noverlap=res['nfft'] - res['hop'],
            scaling='density'
        )
        
        s_db = 10 * np.log10(s + 1e-10)
        results[res['name']] = {'freqs': f, 'times': t, 'spec': s_db}
    
    return results

def pad_audio_centered(audio_data, sample_rate, target_duration):
    """Pad audio to target duration, centering the signal."""
    target_samples = int(target_duration * sample_rate)
    current_samples = len(audio_data)
    
    if current_samples >= target_samples:
        # Center crop if longer
        excess = current_samples - target_samples
        start = excess // 2
        return audio_data[start:start + target_samples]
    else:
        # Pad with zeros, centering the signal
        pad_amount = target_samples - current_samples
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        return np.pad(audio_data, (pad_left, pad_right), mode='constant')

def process_all_segments():
    """Process all segments with optimized spectrograms."""
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
        'segments': [],
        'processing_info': {
            'normalization': 'Peak normalization to ±0.9',
            'denoising': 'Adaptive spectral subtraction',
            'contrast': 'Percentile-based enhancement'
        }
    }
    
    # Process each segment
    spectrograms = []
    spectrograms_denoised = []
    spectrograms_enhanced = []
    labels = []
    
    # Analyze a few segments first to understand the data
    print("\nAnalyzing first few segments to determine optimal parameters...")
    
    for idx, segment_info in enumerate(tqdm(all_segments, desc="Processing segments")):
        try:
            # Read audio
            sample_rate, audio_data = wavfile.read(segment_info['path'])
            
            # Convert to float
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data / 2147483648.0
            
            # Normalize if requested
            if PARAMS['normalize_audio']:
                audio_data = normalize_audio(audio_data)
            
            # Pad audio
            audio_padded = pad_audio_centered(audio_data, sample_rate, PARAMS['target_duration'])
            
            # Create spectrogram
            freqs, times, spec_db = compute_spectrogram_scipy(audio_padded, sample_rate, PARAMS)
            
            # Apply denoising
            spec_denoised = adaptive_denoise(spec_db)
            
            # Enhance contrast
            spec_enhanced = enhance_contrast(spec_denoised)
            
            # Limit frequency range
            freq_mask = (freqs >= PARAMS['freq_min']) & (freqs <= PARAMS['freq_max'])
            spec_filtered = spec_db[freq_mask, :]
            spec_denoised_filtered = spec_denoised[freq_mask, :]
            spec_enhanced_filtered = spec_enhanced[freq_mask, :]
            freqs_filtered = freqs[freq_mask]
            
            # Store results
            spectrograms.append(spec_filtered)
            spectrograms_denoised.append(spec_denoised_filtered)
            spectrograms_enhanced.append(spec_enhanced_filtered)
            labels.append(segment_info['recording'])
            
            # Store metadata
            metadata['segments'].append({
                'index': idx,
                'filename': segment_info['filename'],
                'recording': segment_info['recording'],
                'original_duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'shape': spec_filtered.shape,
                'time_bins': len(times),
                'freq_bins': len(freqs_filtered),
                'max_power': float(np.max(spec_db)),
                'mean_power': float(np.mean(spec_db))
            })
            
            # Save detailed examples for first few segments
            if idx < 5:
                # Create comprehensive visualization
                fig = plt.figure(figsize=(16, 12))
                
                # Original waveform
                ax1 = plt.subplot(4, 2, 1)
                time_audio = np.arange(len(audio_padded)) / sample_rate
                ax1.plot(time_audio, audio_padded)
                ax1.set_title('Waveform')
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Amplitude')
                
                # Multi-resolution spectrograms
                multi_res = create_multi_resolution_spectrogram(audio_padded, sample_rate)
                
                for i, (res_name, res_data) in enumerate(multi_res.items()):
                    ax = plt.subplot(4, 2, 2 + i)
                    
                    # Limit frequency range for display
                    f_mask = (res_data['freqs'] <= PARAMS['freq_max'])
                    
                    im = ax.imshow(res_data['spec'][f_mask, :], 
                                 aspect='auto', origin='lower',
                                 extent=[res_data['times'][0], res_data['times'][-1], 
                                        res_data['freqs'][f_mask][0], res_data['freqs'][f_mask][-1]],
                                 cmap='viridis', vmin=-100, vmax=-20)
                    ax.set_title(f'Spectrogram - {res_name}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Frequency (Hz)')
                    plt.colorbar(im, ax=ax, label='dB')
                
                # Original spectrogram
                ax5 = plt.subplot(4, 2, 5)
                im5 = ax5.imshow(spec_filtered, aspect='auto', origin='lower',
                               extent=[times[0], times[-1], freqs_filtered[0], freqs_filtered[-1]],
                               cmap='viridis')
                ax5.set_title('Original Spectrogram')
                ax5.set_xlabel('Time (s)')
                ax5.set_ylabel('Frequency (Hz)')
                plt.colorbar(im5, ax=ax5, label='dB')
                
                # Denoised spectrogram
                ax6 = plt.subplot(4, 2, 6)
                im6 = ax6.imshow(spec_denoised_filtered, aspect='auto', origin='lower',
                               extent=[times[0], times[-1], freqs_filtered[0], freqs_filtered[-1]],
                               cmap='viridis')
                ax6.set_title('Denoised Spectrogram')
                ax6.set_xlabel('Time (s)')
                ax6.set_ylabel('Frequency (Hz)')
                plt.colorbar(im6, ax=ax6, label='dB')
                
                # Enhanced spectrogram
                ax7 = plt.subplot(4, 2, 7)
                im7 = ax7.imshow(spec_enhanced_filtered, aspect='auto', origin='lower',
                               extent=[times[0], times[-1], freqs_filtered[0], freqs_filtered[-1]],
                               cmap='viridis')
                ax7.set_title('Enhanced Spectrogram')
                ax7.set_xlabel('Time (s)')
                ax7.set_ylabel('Frequency (Hz)')
                plt.colorbar(im7, ax=ax7, label='dB')
                
                # Power spectrum (average across time)
                ax8 = plt.subplot(4, 2, 8)
                mean_spectrum = np.mean(spec_denoised_filtered, axis=1)
                ax8.plot(freqs_filtered, mean_spectrum)
                ax8.set_title('Average Power Spectrum')
                ax8.set_xlabel('Frequency (Hz)')
                ax8.set_ylabel('Power (dB)')
                ax8.grid(True)
                
                plt.suptitle(f'Comprehensive Analysis: {segment_info["filename"]}')
                plt.tight_layout()
                
                # Save
                examples_dir = SPECTROGRAM_DIR / "detailed_examples"
                examples_dir.mkdir(exist_ok=True)
                plt.savefig(examples_dir / f'detailed_analysis_{idx:04d}.png', dpi=150)
                plt.close()
                
        except Exception as e:
            print(f"\nError processing {segment_info['filename']}: {e}")
            continue
    
    # Convert to arrays
    spectrograms = np.array(spectrograms)
    spectrograms_denoised = np.array(spectrograms_denoised)
    spectrograms_enhanced = np.array(spectrograms_enhanced)
    labels = np.array(labels)
    
    print(f"\nSuccessfully processed {len(spectrograms)} spectrograms")
    print(f"Spectrogram shape: {spectrograms[0].shape if len(spectrograms) > 0 else 'N/A'}")
    print(f"Time resolution: {times[1] - times[0]:.4f} seconds")
    print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")
    print(f"Dataset shape: {spectrograms.shape}")
    
    # Save all variants
    np.save(SPECTROGRAM_DIR / 'spectrograms_original.npy', spectrograms)
    np.save(SPECTROGRAM_DIR / 'spectrograms_denoised.npy', spectrograms_denoised)
    np.save(SPECTROGRAM_DIR / 'spectrograms_enhanced.npy', spectrograms_enhanced)
    np.save(SPECTROGRAM_DIR / 'labels.npy', labels)
    
    # Save metadata
    with open(SPECTROGRAM_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save arrays
    np.save(SPECTROGRAM_DIR / 'frequencies.npy', freqs_filtered)
    np.save(SPECTROGRAM_DIR / 'times.npy', times)
    
    print(f"\nData saved to {SPECTROGRAM_DIR}")
    
    return spectrograms, spectrograms_denoised, spectrograms_enhanced, labels

if __name__ == "__main__":
    print("Starting optimized spectrogram generation...")
    print(f"Input directory: {SEGMENTS_DIR}")
    print(f"Output directory: {SPECTROGRAM_DIR}")
    
    # Process segments
    specs_orig, specs_denoised, specs_enhanced, labels = process_all_segments()
    
    # Print statistics
    print("\n📊 Processing Statistics:")
    if len(specs_orig) > 0:
        print(f"  Original range: [{np.min(specs_orig):.1f}, {np.max(specs_orig):.1f}] dB")
        print(f"  Denoised range: [{np.min(specs_denoised):.1f}, {np.max(specs_denoised):.1f}] dB")
        print(f"  Enhanced range: [{np.min(specs_enhanced):.1f}, {np.max(specs_enhanced):.1f}] dB")
    
    print("\n✓ Optimized spectrogram generation complete!")
    print("\nCheck the 'detailed_examples' folder for comprehensive analysis of sample segments.")