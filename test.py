#!/usr/bin/env python3
"""
Direct spectral clustering on spectrograms without autoencoder.
This bypasses the autoencoder bottleneck that was losing important information.
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.sparse import csgraph
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
SPECTROGRAM_DIR = BASE_DIR / "spectrograms_focused"
OUTPUT_DIR = BASE_DIR / "spectral_clustering_results_direct"
OUTPUT_DIR.mkdir(exist_ok=True)

def construct_knn_self_tuning_similarity(X, n_neighbors=30):
    """Construct k-NN graph with self-tuning sigma."""
    from sklearn.neighbors import NearestNeighbors
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Remove self-connections
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Self-tuning sigma (use distance to k-th neighbor)
    sigmas = distances[:, -1]
    sigmas[sigmas == 0] = 1e-10
    
    # Build similarity matrix
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j, neighbor_idx in enumerate(indices[i]):
            dist = distances[i, j]
            sigma_i = sigmas[i]
            sigma_j = sigmas[neighbor_idx]
            W[i, neighbor_idx] = np.exp(-dist**2 / (sigma_i * sigma_j))
            W[neighbor_idx, i] = W[i, neighbor_idx]  # Symmetric
    
    return W

def spectral_clustering(W, n_clusters=4):
    """Perform spectral clustering given similarity matrix."""
    from sklearn.cluster import KMeans
    
    # Compute degree matrix
    D = np.diag(np.sum(W, axis=1))
    
    # Compute normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-10))
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # Compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Select k smallest eigenvectors
    idx = eigenvalues.argsort()[:n_clusters]
    embedding = eigenvectors[:, idx]
    
    # Normalize rows
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    
    # Apply k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding)
    
    return labels

def main():
    print("=" * 60)
    print("DIRECT SPECTRAL CLUSTERING ON SPECTROGRAMS")
    print("=" * 60)
    
    # Load spectrograms
    print("\nLoading spectrograms...")
    specs = np.load(SPECTROGRAM_DIR / 'spectrograms_enhanced.npy')
    labels_true = np.load(SPECTROGRAM_DIR / 'labels.npy')
    print(f"Loaded {len(specs)} spectrograms with shape {specs[0].shape}")
    
    # Flatten spectrograms
    print("\nFlattening spectrograms...")
    specs_flat = specs.reshape(len(specs), -1)
    print(f"Flattened shape: {specs_flat.shape}")
    
    # Standardize
    print("\nStandardizing features...")
    scaler = StandardScaler()
    specs_scaled = scaler.fit_transform(specs_flat)
    
    # Apply PCA
    print("\nApplying PCA...")
    n_components = 50
    pca = PCA(n_components=n_components)
    specs_pca = pca.fit_transform(specs_scaled)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA: {n_components} components explain {explained_var:.2%} of variance")
    
    # Construct similarity matrix
    print("\nConstructing similarity matrix...")
    n_neighbors = 30
    W = construct_knn_self_tuning_similarity(specs_pca, n_neighbors=n_neighbors)
    print(f"Similarity matrix shape: {W.shape}")
    
    # Perform spectral clustering
    print("\nPerforming spectral clustering...")
    n_clusters = 4
    labels = spectral_clustering(W, n_clusters=n_clusters)
    
    # Evaluate
    score = silhouette_score(specs_pca, labels)
    print(f"\nSilhouette score: {score:.4f}")
    
    # Print distribution
    print("\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / len(labels)
        print(f"  Cluster {u}: {c} samples ({pct:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save labels
    np.save(OUTPUT_DIR / f'cluster_labels_{timestamp}.npy', labels)
    
    # Save parameters
    params = {
        'n_clusters': n_clusters,
        'n_neighbors': n_neighbors,
        'pca_components': n_components,
        'silhouette_score': float(score),
        'cluster_sizes': {int(u): int(c) for u, c in zip(unique, counts)}
    }
    with open(OUTPUT_DIR / f'parameters_{timestamp}.json', 'w') as f:
        import json
        json.dump(params, f, indent=2)
    
    # Create mean spectrograms per cluster
    print("\nCreating cluster visualizations...")
    fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 4))
    
    for i in range(n_clusters):
        cluster_specs = specs[labels == i]
        mean_spec = np.mean(cluster_specs, axis=0)
        
        ax = axes[i] if n_clusters > 1 else axes
        im = ax.imshow(mean_spec, aspect='auto', origin='lower', cmap='hot')
        ax.set_title(f'Cluster {i}\n(n={len(cluster_specs)})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        
        # Add frequency labels
        ax.set_yticks([0, 32, 64, 96, 128])
        ax.set_yticklabels(['0', '88', '175', '262', '350'])
    
    plt.suptitle('Mean Spectrogram per Cluster')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'cluster_means_{timestamp}.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print("\nTo evaluate these clusters, update evaluate_clusters.py to load from:")
    print(f"  {OUTPUT_DIR / f'cluster_labels_{timestamp}.npy'}")

if __name__ == "__main__":
    main()