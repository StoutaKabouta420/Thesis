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
    ```bash
    pip install -r requirements.txt

## Usage

1. Extract Audio Segments: Run the extract_segments.py script to extract audio segments from raw recordings:
    ```bash
    python extract_segments.py

2. Generate Spectrograms: Use the generate_spectrograms.py script to create spectrograms:
    ```bash
    python generate_spectrograms.py

3. Train Autoencoder: Train the autoencoder model for feature extraction:
    ```bash
    python train_autoencoder.py

4. Analyze Segments: Perform analysis on the extracted segments:
    ```bash
    python analyze_segments.py

5. Verify Spectrograms: Check the quality of generated spectrograms:
    ```bash
    python verify_spectrograms.py

## Data

Raw Data: Located in 25108016/ and includes metadata and submissions.
Processed Data: Extracted segments are stored in extracted_segments/.
Spectrograms: Generated spectrograms are saved in spectrograms/.

## Outputs
Analysis Results: Found in data_analysis/, including visualizations and CSV files.
Trained Models: Saved in models/, including the autoencoder and bottleneck features.