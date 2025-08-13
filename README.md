# Thesis: Spectral Clustering of Underwater Bioacoustics

This repository contains the code, data, and models for the spectral clustering of underwater bioacoustic signals, specifically focusing on Bryde's whale calls.

## Overview

This project implements an unsupervised learning approach to analyse and categorise Bryde's whale vocalisations using spectral clustering. Two approaches are compared: (1) autoencoder-based feature extraction followed by spectral clustering, and (2) direct spectral clustering on spectrograms. The pipeline processes raw audio recordings, generates optimised spectrograms, and discovers natural acoustic patterns without requiring labelled training data, successfully identifying four distinct call types that match expert manual classification.

## Directory Structure

### Key Files and Directories

- `extract_segments.py` – Extracts audio segments from raw recordings
- `generate_spectrograms.py` – Generates high-quality spectrograms optimised for whale call frequencies
- `synthesize.py` – Augments spectrogram dataset using bioacoustically appropriate transformations
- `train_autoencoder.py` – Trains an autoencoder model for feature extraction
- `spectral_clustering.py` – Performs spectral clustering on autoencoder bottleneck features
- `test.py` – Performs direct spectral clustering on raw spectrograms (alternative approach)
- `evaluate_clusters.py` – Organises clustered results for evaluation and creates comprehensive visualisations
- `parameter_optimization.py` – Automated parameter optimisation for spectral clustering
- `requirements.txt` – Python dependencies for the project
- `README.md` – Project documentation

### Data & Output Folders

- `25108016/` – Raw data submissions and metadata
- `extracted_segments_padded/` – Padded segments with consistent durations
- `models/` – Trained models and bottleneck features
- `spectrograms_focused/` – Optimised spectrograms for whale call frequency range (0–350 Hz)
- `spectrograms_augmented/` – Augmented spectrograms for robust training
- `spectral_clustering_results/` – Clustering outputs from autoencoder approach
- `spectral_clustering_results_direct/` – Clustering outputs from direct approach
- `verify_outputs/` – Organised results for manual evaluation and validation
- `parameter_optimization_results/` – Results from spectral clustering parameter optimisation
- `venv/` – Python virtual environment

## Installation

```bash
git clone <repository-url>
cd Thesis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Method 1: Autoencoder-Based Spectral Clustering (Recommended)

#### 1. Extract Audio Segments
```bash
python extract_segments.py
```
- Pads 300 ms before and after detections
- Merges overlapping detections
- Normalises segment durations (0.5–1.6 s)
- Output: 695 segments

#### 2. Generate Spectrograms
```bash
python generate_spectrograms.py
```
- FFT size: 4096 samples (95% overlap)
- Frequency range: 0–350 Hz (optimised for Bryde's whale calls)
- Final shape: 129 × 64 (freq × time)
- Saves frequency and time arrays for proper axis labeling

#### 3. Augment Dataset (Optional)
```bash
python synthesize.py
```
- Creates ~3000 augmented samples for robust autoencoder training
- Augmentations: time shifting, frequency shifting, amplitude variation, noise addition, time stretching

#### 4. Train Autoencoder
```bash
python train_autoencoder.py
```
- **Critical**: Train for only 10 epochs to preserve discriminative features
- Uses enhanced spectrograms for optimal training
- Bottleneck dimension: 64
- Extracts features from original (non-augmented) data

#### 5. Perform Spectral Clustering
```bash
python spectral_clustering.py \
  --models_dir models \
  --output_dir spectral_clustering_results \
  --n_clusters 4 \
  --similarity_metric knn_self_tuning \
  --n_neighbors 10 \
  --no_mutual \
  --laplacian_type normalized \
  --pca_components 32 \
  --standardize
```

### Method 2: Direct Spectral Clustering (Alternative)

Skip the autoencoder and cluster directly on spectrograms:

```bash
python test.py
```
- Applies PCA directly to flattened spectrograms
- Uses k-NN self-tuning similarity with 30 neighbors
- Often achieves better cluster separation

### Evaluation

```bash
python evaluate_clusters.py
```

Generates:
- `cluster_means.png` – Mean spectrogram per cluster
- `cluster_X_samples.png` – Representative samples per cluster
- `cluster_dominant_freqs.csv` – Frequency analysis per cluster

Note: Currently configured for autoencoder results. For direct clustering results, manually specify the path to labels in `spectral_clustering_results_direct/`.

## Results

### Optimal Parameters (Autoencoder Method)
- **Clusters**: 4
- **Similarity metric**: k-NN self-tuning (n_neighbors = 10)
- **Laplacian type**: normalized
- **PCA components**: 32
- **Standardisation**: enabled
- **Autoencoder epochs**: 10 (critical - more epochs reduce performance)

### Performance Comparison

#### Autoencoder-Based (10 epochs):
- Cluster 0: 40.0% (278 samples)
- Cluster 1: 24.5% (170 samples)
- Cluster 2: 8.3% (58 samples)
- Cluster 3: 27.2% (189 samples)
- Silhouette score: 0.0178

#### Direct Clustering:
- Cluster 0: 39.4% (274 samples)
- Cluster 1: 8.3% (58 samples)
- Cluster 2: 16.4% (114 samples)
- Cluster 3: 35.8% (249 samples)
- Silhouette score: 0.0797

#### Paper's Manual Classification (Reference):
- ST-42: 41% (259 samples)
- MT-42: 32% (197 samples)
- BT-42: 18% (110 samples)
- TD: 9% (57 samples)
- Total: 622 samples

## Key Findings

1. **Both unsupervised methods successfully identify 4 distinct call types** that correspond to expert manual classification
2. **Training duration is critical**: Autoencoder trained for 10 epochs preserves features better than 100 epochs
3. **Frequency focus matters**: Limiting spectrograms to 0-350 Hz (where whale calls occur) improves clustering
4. **Direct clustering can outperform autoencoder approaches** when features are over-compressed

## Methodology

### Spectral Clustering Pipeline
1. **Similarity Matrix Construction** – k-NN self-tuning or Gaussian kernel
2. **Laplacian Matrix Computation** – Normalised graph Laplacian
3. **Eigenvector Computation** – Select k smallest eigenvectors
4. **K-means Clustering** – Assign cluster labels

### Features
- Consistent padded segments
- High-quality spectrograms focused on 0–350 Hz
- Bioacoustically relevant data augmentation
- Flexible similarity metrics and Laplacian types
- PCA for dimensionality reduction
- Automated parameter optimisation
- Comprehensive visual and statistical evaluation

