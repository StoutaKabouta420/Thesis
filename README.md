# Thesis: Spectral Clustering of Underwater Bioacoustics

This repository contains the code, data, and models for the spectral clustering of underwater bioacoustic signals, specifically focusing on Bryde's whale calls.

## Overview

This project implements an unsupervised learning approach to analyze and categorize Bryde's whale vocalizations using spectral clustering on autoencoder-derived features. The pipeline processes raw audio recordings, generates optimized spectrograms, extracts meaningful features via deep learning, and discovers natural acoustic patterns without requiring labeled training data.

## Directory Structure

### Key Files and Directories

- `analyze_segments.py`: Script for analyzing extracted audio segments
- `extract_segments.py`: Extracts audio segments from raw recordings
- `generate_spectrograms.py`: Generates optimized spectrograms from audio data with enhanced parameters for whale bioacoustics
- `train_autoencoder.py`: Trains an autoencoder model for feature extraction
- `verify_spectrograms.py`: Verifies the quality of generated spectrograms
- `spectral_clustering.py`: Performs spectral clustering on autoencoder bottleneck features
- `evaluate_clusters.py`: Organizes clustered results for evaluation and creates comprehensive visualizations
- `parameter_optimization.py`: Automated parameter optimization for spectral clustering
- `requirements.txt`: Lists Python dependencies required for the project
- `25108016/`: Contains raw data submissions and metadata
- `data_analysis/`: Includes analysis outputs such as duration distributions, sample spectrograms, and segment analysis
- `extracted_segments/`: Stores extracted audio segments organized by timestamp
- `models/`: Contains trained models and bottleneck features
- `spectrograms_optimized/`: Stores enhanced spectrogram arrays (numpy format) with improved frequency resolution and contrast
- `spectral_clustering_results/`: Contains spectral clustering outputs, visualizations, and cluster assignments
- `cluster_evaluation/`: Organized cluster results for manual evaluation and validation
- `parameter_optimization_results/`: Results from parameter optimization experiments

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Thesis

Install dependencies:

bashpip install -r requirements.txt
Usage
Data Processing Pipeline

Extract Audio Segments: Run the extract_segments.py script to extract audio segments from raw recordings:

bashpython extract_segments.py

Generate Spectrograms: Use the generate_spectrograms.py script to create optimized spectrograms:

bashpython generate_spectrograms.py
This generates three versions of spectrograms in spectrograms_optimized/:

spectrograms_original.npy: Raw spectrograms
spectrograms_denoised.npy: Noise-reduced spectrograms
spectrograms_enhanced.npy: Contrast-enhanced spectrograms (recommended for clustering)


Train Autoencoder: Train the autoencoder model for feature extraction:

bashpython train_autoencoder.py

Analyze Segments: Perform analysis on the extracted segments:

bashpython analyze_segments.py

Verify Spectrograms: Check the quality of generated spectrograms:

bashpython verify_spectrograms.py
Spectral Clustering

Parameter Optimization (Recommended first step):
Quick test with subset of parameters:
bashpython parameter_optimization.py --quick --standardize
Full grid search (comprehensive but time-consuming):
bashpython parameter_optimization.py --standardize

Perform Spectral Clustering: Apply spectral clustering with optimal parameters:

bashpython spectral_clustering.py --n_clusters 3 --sigma 3.0 --laplacian_type normalized --similarity_metric gaussian --standardize

Evaluate Clusters: Organize and evaluate the clustering results:

bashpython evaluate_clusters.py --cluster_labels spectral_clustering_results/spectral_cluster_labels_TIMESTAMP.npy --features models/bottleneck_features.npy
Spectral Clustering Parameters
The spectral clustering script supports various parameters for customization:
Basic Usage:
bashpython spectral_clustering.py
Custom Parameters:
bashpython spectral_clustering.py --n_clusters 3 --sigma 3.0 --laplacian_type normalized --standardize
Available Parameters:

--models_dir: Directory containing bottleneck features (default: models)
--output_dir: Directory to save results (default: spectral_clustering_results)
--n_clusters: Number of clusters (default: 5)
--sigma: Bandwidth parameter for Gaussian kernel (default: 1.0)
--laplacian_type: Type of Laplacian matrix (normalized, unnormalized, rw)
--similarity_metric: Similarity metric (gaussian, knn, epsilon)
--n_neighbors: Number of neighbors for k-NN graph (default: 10)
--standardize: Standardize features before clustering

Cluster Evaluation Parameters

--cluster_labels: Path to cluster labels .npy file (required)
--extracted_segments: Directory containing audio segments (default: extracted_segments)
--spectrograms: Directory containing spectrograms (default: spectrograms_optimized)
--output_dir: Output directory for organized results (default: cluster_evaluation)
--copy_files: Copy files instead of creating symlinks
--samples_per_cluster: Number of sample spectrograms per cluster (default: 5)
--features: Path to bottleneck features for silhouette analysis (default: models/bottleneck_features.npy)

Data

Raw Data: Located in 25108016/ and includes metadata and submissions
Processed Data: Extracted segments are stored in extracted_segments/
Spectrograms: Enhanced spectrogram arrays saved in spectrograms_optimized/ as numpy arrays
Bottleneck Features: Autoencoder-extracted features stored in models/bottleneck_features.npy

Outputs
Analysis Results
Found in data_analysis/, including visualizations and CSV files.
Trained Models
Saved in models/, including:

Autoencoder model (autoencoder_TIMESTAMP.pth)
Bottleneck features (bottleneck_features.npy)
Spectrogram scaler (spectrogram_scaler.pkl)

Spectral Clustering Results
Saved in spectral_clustering_results/, including:

Cluster assignments (spectral_cluster_labels_TIMESTAMP.npy)
Trained clustering model (spectral_clustering_model_TIMESTAMP.pkl)
Evaluation metrics (clustering_metrics_TIMESTAMP.txt)
Visualizations:

Eigenvalue spectrum of graph Laplacian
Eigenvector plots
Similarity matrix heatmap
Cluster size distribution
2D projection of spectral embedding



Cluster Evaluation Results
Saved in cluster_evaluation/, including:

Organized audio files by cluster (cluster_X/audio/)
Organized spectrograms by cluster (cluster_X/spectrograms/)
Representative sample spectrograms for each cluster
Audio playlists for each cluster (.m3u files)
Cluster comparison visualization (cluster_comparison.png)
Mean spectrograms per cluster (cluster_mean_spectrograms.png)
Silhouette analysis plot (silhouette_analysis.png)
Detailed cluster statistics (cluster_statistics.csv)
Acoustic characteristics per cluster (cluster_acoustic_characteristics.csv)

Parameter Optimization Results
Saved in parameter_optimization_results/, including:

Grid search results (grid_search_results_TIMESTAMP.csv)
Best parameters (best_parameters_TIMESTAMP.json)
Visualization of parameter effects on clustering quality

Spectral Clustering Methodology
The spectral clustering implementation follows a four-step process:

Similarity Matrix Construction: Uses Gaussian kernel to compute pairwise similarities between autoencoder features
Laplacian Matrix Computation: Constructs the normalized graph Laplacian from the similarity matrix
Eigenvector Computation: Finds the k smallest eigenvectors of the Laplacian matrix
K-means Clustering: Applies K-means algorithm to the eigenvector matrix to obtain final cluster assignments

Current Best Results
Latest parameter optimization identified optimal configuration for 710 Bryde's whale call segments:

Optimal Parameters:

Number of clusters: 3
Sigma (bandwidth): 3.0
Laplacian type: normalized
Similarity metric: gaussian
Feature standardization: enabled


Performance Metrics:

Silhouette Score: 0.391 (excellent for bioacoustic data)
Calinski-Harabasz Score: 561.13
Davies-Bouldin Score: 0.943


Cluster Distribution:

Cluster 0: 187 samples (26.3%)
Cluster 1: 268 samples (37.7%)
Cluster 2: 255 samples (35.9%)



The enhanced spectrogram generation (75×47 resolution) provides improved frequency resolution and contrast, better capturing the harmonic structure of Bryde's whale calls compared to initial implementations.
Cluster Evaluation Workflow
After running spectral clustering, use the evaluation tools to validate results:

Visual Comparison: Open cluster_comparison.png to see representative spectrograms from each cluster
Mean Patterns: Check cluster_mean_spectrograms.png for average patterns and variability within clusters
Acoustic Analysis: Review cluster_acoustic_characteristics.csv for frequency and temporal properties
Audio Evaluation: Use the .m3u playlist files to listen to samples from each cluster
Detailed Analysis: Browse individual cluster folders to examine specific samples
Statistical Review: Check cluster_statistics.csv and silhouette_analysis.png for quantitative analysis

Key Features in Current Version

Enhanced Spectrogram Generation: Optimized parameters for whale bioacoustics (1024 FFT size, 64 hop length)
Better Frequency Resolution: Focused on 10-600 Hz range where whale calls occur
Multiple Spectrogram Versions: Original, denoised, and enhanced versions for comparison
Improved Time Resolution: 47 time bins vs 8 in initial version
Adaptive Denoising: Spectral subtraction based on estimated noise floor
Automated Parameter Optimization: Grid search capability to find optimal clustering parameters
Comprehensive Evaluation Tools: Multiple visualizations and metrics for cluster validation
Acoustic Feature Analysis: Automatic extraction of dominant frequency, bandwidth, and duration per cluster

This unsupervised approach successfully discovered distinct acoustic patterns in Bryde's whale calls without requiring labeled training data, providing insights into the natural vocal repertoire of this species.