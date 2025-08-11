Thesis: Spectral Clustering of Underwater Bioacoustics
This repository contains the code, data, and models for the spectral clustering of underwater bioacoustic signals, specifically focusing on Bryde's whale calls.

Overview
This project implements an unsupervised learning approach to analyse and categorise Bryde's whale vocalisations using spectral clustering on autoencoder-derived features. The pipeline processes raw audio recordings, generates optimised spectrograms, extracts meaningful features via deep learning, and discovers natural acoustic patterns without requiring labelled training data.

Directory Structure
Key Files and Directories
extract_segments.py – Extracts audio segments from raw recordings

generate_spectrograms.py – Generates high-quality spectrograms with adaptive FFT sizing for variable-length segments

synthesize.py – Augments spectrogram dataset using bioacoustically appropriate transformations

train_autoencoder.py – Trains an autoencoder model for feature extraction

spectral_clustering.py – Performs spectral clustering on autoencoder bottleneck features

evaluate_clusters.py – Organises clustered results for evaluation and creates comprehensive visualisations

parameter_optimization.py – Automated parameter optimisation for spectral clustering

debug_clustering.py – Debugging utilities for clustering analysis

requirements.txt – Python dependencies for the project

README.md – Project documentation

POA.txt – Project analysis notes

Data & Output Folders

25108016/ – Raw data submissions and metadata

extracted_segments_padded/ – Padded segments with consistent durations

models/ – Trained models and bottleneck features

spectrograms_focused/ – Optimised spectrograms for whale call frequency range (0–500 Hz)

spectrograms_augmented/ – Augmented spectrograms for robust training

spectral_clustering_results/ – Clustering outputs, visualisations, and labels

verify_outputs/ – Organised results for manual evaluation and validation

parameter_optimization_results/ – Results from spectral clustering parameter optimisation

venv/ – Python virtual environment

Installation
bash
Copy
git clone <repository-url>
cd Thesis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Usage
1. Extract Audio Segments
bash
Copy
python extract_segments.py
Pads 300 ms before and after detections

Merges overlapping detections

Normalises segment durations (0.5–1.6 s)

2. Generate Spectrograms
bash
Copy
python generate_spectrograms.py
FFT size: 4096 samples (95% overlap)

Frequency range: 0–500 Hz

Final shape: 129 × 64 (freq × time)

Matches Raven Pro settings

3. Augment Dataset
bash
Copy
python synthesize.py
Time shifting, frequency shifting, amplitude variation, noise addition, time stretching, frequency masking

4. Train Autoencoder
bash
Copy
python train_autoencoder.py
Uses enhanced spectrograms for optimal training

5. Perform Spectral Clustering
Recommended configuration:

bash
Copy
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
Evaluation
bash
Copy
python evaluate_clusters.py
Generates:

cluster_means.png – Mean spectrogram per cluster

cluster_X_samples.png – Representative samples per cluster

cluster_dominant_freqs.csv – Frequency analysis per cluster

Optimal Parameters (Current Best)
Clusters: 4

Similarity metric: k-NN self-tuning (n_neighbors = 10)

Laplacian type: normalized

PCA components: 32

Standardisation: enabled

Performance:

Silhouette score: 0.0154

Distribution:

Cluster 0: 37.3%

Cluster 1: 32.8%

Cluster 2: 8.5%

Cluster 3: 21.4%

Methodology
Similarity Matrix Construction – k-NN self-tuning or Gaussian kernel

Laplacian Matrix Computation – Normalised graph Laplacian

Eigenvector Computation – Select k smallest eigenvectors

K-means Clustering – Assign cluster labels

Features
Consistent padded segments

High-quality spectrograms focused on 0–500 Hz

Bioacoustically relevant data augmentation

Flexible similarity metrics and Laplacian types

PCA for dimensionality reduction

Automated parameter optimisation

Comprehensive visual and statistical evaluation