# Thesis: Spectral Clustering of Underwater Bioacoustics

This repository contains the code, data, and models for the spectral clustering of underwater bioacoustic signals, specifically focusing on Bryde's whale calls.

## Directory Structure

### Key Files and Directories

- **`analyze_segments.py`**: Script for analyzing extracted audio segments.
- **`extract_segments.py`**: Extracts audio segments from raw recordings.
- **`generate_spectrograms.py`**: Generates spectrograms from audio data.
- **`train_autoencoder.py`**: Trains an autoencoder model for feature extraction.
- **`verify_spectrograms.py`**: Verifies the quality of generated spectrograms.
- **`requirements.txt`**: Lists Python dependencies required for the project.
- **`25108016/`**: Contains raw data submissions and metadata.
- **`data_analysis/`**: Includes analysis outputs such as duration distributions, sample spectrograms, and segment analysis.
- **`extracted_segments/`**: Stores extracted audio segments organized by timestamp.
- **`models/`**: Contains trained models and bottleneck features.
- **`spectrograms/`**: Stores generated spectrogram images.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Thesis

2. Install dependencies:
pip install -r requirements.txt