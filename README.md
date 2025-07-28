Thesis: Spectral Clustering of Underwater Bioacoustics
This repository contains the code, data, and models for the spectral clustering of underwater bioacoustic signals, specifically focusing on Bryde's whale calls.
Directory Structure
Key Files and Directories

analyze_segments.py: Script for analyzing extracted audio segments.
extract_segments.py: Extracts audio segments from raw recordings.
generate_spectrograms.py: Generates optimized spectrograms from audio data with enhanced parameters for whale bioacoustics.
train_autoencoder.py: Trains an autoencoder model for feature extraction.
verify_spectrograms.py: Verifies the quality of generated spectrograms.
spectral_clustering.py: Performs spectral clustering on autoencoder bottleneck features.
evaluate_clusters.py: Organizes clustered results for evaluation and creates visualizations.
requirements.txt: Lists Python dependencies required for the project.
25108016/: Contains raw data submissions and metadata.
data_analysis/: Includes analysis outputs such as duration distributions, sample spectrograms, and segment analysis.
extracted_segments/: Stores extracted audio segments organized by timestamp.
models/: Contains trained models and bottleneck features.
spectrograms_optimized/: Stores enhanced spectrogram arrays (numpy format) with improved frequency resolution and contrast.
spectral_clustering_results/: Contains spectral clustering outputs, visualizations, and cluster assignments.
cluster_evaluation/: Organized cluster results for manual evaluation and validation.

Installation

Clone the repository:

bashgit clone <repository-url>
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

Perform Spectral Clustering: Apply spectral clustering to the autoencoder bottleneck features:

bashpython spectral_clustering.py

Evaluate Clusters: Organize and evaluate the clustering results:

bashpython evaluate_clusters.py --cluster_labels spectral_clustering_results/spectral_cluster_labels_TIMESTAMP.npy
Spectral Clustering Parameters
The spectral clustering script supports various parameters for customization:
Basic Usage:
bashpython spectral_clustering.py
Custom Parameters:
bashpython spectral_clustering.py --n_clusters 8 --sigma 0.5 --laplacian_type normalized
With Feature Standardization:
bashpython spectral_clustering.py --standardize --n_clusters 6
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

Data

Raw Data: Located in 25108016/ and includes metadata and submissions.
Processed Data: Extracted segments are stored in extracted_segments/.
Spectrograms: Enhanced spectrogram arrays saved in spectrograms_optimized/ as numpy arrays.
Bottleneck Features: Autoencoder-extracted features stored in models/bottleneck_features.npy.

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
Detailed cluster statistics (cluster_statistics.csv)

Spectral Clustering Methodology
The spectral clustering implementation follows a four-step process:

Similarity Matrix Construction: Uses Gaussian kernel to compute pairwise similarities between autoencoder features
Laplacian Matrix Computation: Constructs the normalized graph Laplacian from the similarity matrix
Eigenvector Computation: Finds the k smallest eigenvectors of the Laplacian matrix
K-means Clustering: Applies K-means algorithm to the eigenvector matrix to obtain final cluster assignments

Current Results
Latest clustering run identified 5 distinct clusters from 710 Bryde's whale call segments:

Silhouette Score: 0.29 (indicating moderately well-separated clusters)
Cluster Distribution:

Cluster 0: 164 samples (23.1%)
Cluster 1: 38 samples (5.4%) - smallest cluster
Cluster 2: 187 samples (26.3%)
Cluster 3: 97 samples (13.7%)
Cluster 4: 224 samples (31.5%) - largest cluster



The enhanced spectrogram generation (75×47 resolution) provides improved frequency resolution and contrast, better capturing the harmonic structure of Bryde's whale calls compared to the initial implementation.
Cluster Evaluation Workflow
After running spectral clustering, use the evaluation tools to validate results:

Visual Comparison: Open cluster_comparison.png to see representative spectrograms from each cluster
Audio Evaluation: Use the .m3u playlist files to listen to samples from each cluster
Detailed Analysis: Browse individual cluster folders to examine specific samples
Statistical Review: Check cluster_statistics.csv for quantitative analysis

Key Improvements in Current Version

Enhanced Spectrogram Generation: Optimized parameters for whale bioacoustics (1024 FFT size, 64 hop length)
Better Frequency Resolution: Focused on 10-600 Hz range where whale calls occur
Multiple Spectrogram Versions: Original, denoised, and enhanced versions for comparison
Improved Time Resolution: 47 time bins vs 8 in initial version
Adaptive Denoising: Spectral subtraction based on estimated noise floor

This unsupervised approach successfully discovered distinct acoustic patterns in Bryde's whale calls without requiring labeled training data, providing insights into the natural vocal repertoire of this species.