import os
import pandas as pd
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
OUTPUT_DIR = BASE_DIR / "extracted_segments"
ANALYSIS_DIR = BASE_DIR / "data_analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

def analyze_segments():
    """Analyze the extracted segments to understand their characteristics."""
    
    durations = []
    sample_rates = []
    segment_info = []
    
    # Collect information about all segments
    for recording_folder in OUTPUT_DIR.iterdir():
        if recording_folder.is_dir():
            wav_files = list(recording_folder.glob("*.wav"))
            
            for wav_file in wav_files:
                try:
                    sample_rate, audio_data = wavfile.read(wav_file)
                    duration = len(audio_data) / sample_rate
                    
                    durations.append(duration)
                    sample_rates.append(sample_rate)
                    segment_info.append({
                        'file': wav_file.name,
                        'recording': recording_folder.name,
                        'duration': duration,
                        'sample_rate': sample_rate,
                        'samples': len(audio_data)
                    })
                except Exception as e:
                    print(f"Error reading {wav_file.name}: {e}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(segment_info)
    
    # Print statistics
    print("=== Segment Analysis ===")
    print(f"Total segments: {len(df)}")
    print(f"\nDuration statistics (seconds):")
    print(df['duration'].describe())
    print(f"\nUnique sample rates: {df['sample_rate'].unique()}")
    print(f"\nSegments per recording:")
    print(df['recording'].value_counts())
    
    # Plot duration distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.hist(df['duration'], bins=50, edgecolor='black')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.title('Distribution of Segment Durations')
    
    plt.subplot(2, 1, 2)
    plt.hist(df['duration'], bins=50, edgecolor='black', cumulative=True, density=True)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution of Segment Durations')
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'duration_distribution.png')
    plt.close()
    
    # Save detailed analysis
    df.to_csv(ANALYSIS_DIR / 'segment_analysis.csv', index=False)
    
    return df

def create_sample_spectrograms(df, num_samples=10):
    """Create spectrograms for a sample of segments to visualize."""
    
    # Sample random segments
    sample_files = df.sample(min(num_samples, len(df)))
    
    plt.figure(figsize=(15, 10))
    
    for idx, (_, row) in enumerate(sample_files.iterrows()):
        wav_path = OUTPUT_DIR / row['recording'] / row['file']
        
        try:
            sample_rate, audio_data = wavfile.read(wav_path)
            
            # Create spectrogram
            plt.subplot(5, 2, idx + 1)
            plt.specgram(audio_data, Fs=sample_rate, cmap='viridis', 
                        noverlap=128, NFFT=256)
            plt.title(f"Duration: {row['duration']:.3f}s")
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.ylim(0, 1000)  # Focus on low frequencies
            plt.colorbar()
            
        except Exception as e:
            print(f"Error processing {row['file']}: {e}")
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'sample_spectrograms.png', dpi=150)
    plt.close()
    
    print(f"\nSample spectrograms saved to {ANALYSIS_DIR / 'sample_spectrograms.png'}")

if __name__ == "__main__":
    df = analyze_segments()
    create_sample_spectrograms(df)