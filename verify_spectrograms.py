# Create a new file called 'verify_spectrograms.py'
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
SPECTROGRAM_DIR = BASE_DIR / "spectrograms"

# Load the data
spectrograms = np.load(SPECTROGRAM_DIR / 'spectrograms.npy')
labels = np.load(SPECTROGRAM_DIR / 'labels.npy')
frequencies = np.load(SPECTROGRAM_DIR / 'frequencies.npy')
times = np.load(SPECTROGRAM_DIR / 'times.npy')

# Load metadata
with open(SPECTROGRAM_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print("=== Dataset Summary ===")
print(f"Total spectrograms: {len(spectrograms)}")
print(f"Spectrogram shape: {spectrograms.shape}")
print(f"Frequency range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz")
print(f"Time duration: {times[-1]:.3f} seconds")
print(f"Frequency resolution: {frequencies[1] - frequencies[0]:.1f} Hz")
print(f"Time resolution: {(times[1] - times[0])*1000:.1f} ms")
print(f"\nUnique recordings: {np.unique(labels)}")
print(f"Segments per recording:")
for label in np.unique(labels):
    count = np.sum(labels == label)
    print(f"  {label}: {count} segments")

# Show data range
print(f"\nData statistics:")
print(f"  Min value: {spectrograms.min():.2f} dB")
print(f"  Max value: {spectrograms.max():.2f} dB")
print(f"  Mean value: {spectrograms.mean():.2f} dB")
print(f"  Std deviation: {spectrograms.std():.2f} dB")