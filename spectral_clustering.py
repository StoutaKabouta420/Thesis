#!/usr/bin/env python3
"""
Spectral Clustering for Underwater Bioacoustic Signals
Performs spectral clustering on autoencoder bottleneck features of Bryde's whale calls.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
import argparse
import os
from datetime import datetime
import pickle

class SpectralClustering:
    """
    Custom implementation of spectral clustering for bioacoustic analysis.
    """
    
    def __init__(self, n_clusters=5, sigma=1.0, laplacian_type='normalized', 
                 similarity_metric='gaussian', n_neighbors=10):
        """
        Initialize spectral clustering parameters.
        
        Args:
            n_clusters (int): Number of clusters
            sigma (float): Bandwidth parameter for Gaussian kernel
            laplacian_type (str): Type of Laplacian ('normalized', 'unnormalized', 'rw')
            similarity_metric (str): Similarity metric ('gaussian', 'knn', 'epsilon')
            n_neighbors (int): Number of neighbors for k-NN graph
        """
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.laplacian_type = laplacian_type
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.similarity_matrix_ = None
        self.laplacian_ = None
        self.eigenvectors_ = None
        self.eigenvalues_ = None
        self.labels_ = None
        
    def _construct_similarity_matrix(self, X):
        """
        Construct similarity matrix using specified method.
        
        Args:
            X (np.ndarray): Input feature matrix (n_samples, n_features)
            
        Returns:
            np.ndarray: Similarity matrix
        """
        n_samples = X.shape[0]
        
        if self.similarity_metric == 'gaussian':
            # Gaussian (RBF) kernel similarity
            pairwise_dists = pdist(X, metric='euclidean')
            pairwise_dists_sq = pairwise_dists ** 2
            similarity_vector = np.exp(-pairwise_dists_sq / (2 * self.sigma ** 2))
            similarity_matrix = squareform(similarity_vector)
            np.fill_diagonal(similarity_matrix, 1.0)
            
        elif self.similarity_metric == 'knn':
            # k-NN similarity graph
            knn_graph = kneighbors_graph(X, n_neighbors=self.n_neighbors, 
                                       mode='connectivity', include_self=False)
            # Make symmetric
            similarity_matrix = (knn_graph + knn_graph.T).toarray()
            similarity_matrix = np.clip(similarity_matrix, 0, 1)
            
        elif self.similarity_metric == 'epsilon':
            # Epsilon-neighborhood graph
            pairwise_dists = squareform(pdist(X, metric='euclidean'))
            epsilon = np.percentile(pairwise_dists, 10)  # Use 10th percentile as epsilon
            similarity_matrix = (pairwise_dists <= epsilon).astype(float)
            np.fill_diagonal(similarity_matrix, 1.0)
            
        return similarity_matrix
    
    def _compute_laplacian(self, W):
        """
        Compute graph Laplacian matrix.
        
        Args:
            W (np.ndarray): Similarity/adjacency matrix
            
        Returns:
            np.ndarray: Laplacian matrix
        """
        # Degree matrix
        D = np.diag(np.sum(W, axis=1))
        
        if self.laplacian_type == 'unnormalized':
            # L = D - W
            L = D - W
            
        elif self.laplacian_type == 'normalized':
            # L_sym = D^(-1/2) * (D - W) * D^(-1/2)
            D_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-12))
            L = D_sqrt_inv @ (D - W) @ D_sqrt_inv
            
        elif self.laplacian_type == 'rw':
            # L_rw = D^(-1) * (D - W)
            D_inv = np.diag(1.0 / (np.sum(W, axis=1) + 1e-12))
            L = D_inv @ (D - W)
            
        return L
    
    def _find_eigenvectors(self, L):
        """
        Find the k smallest eigenvectors of the Laplacian matrix.
        
        Args:
            L (np.ndarray): Laplacian matrix
            
        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        # Find smallest eigenvalues and corresponding eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort by eigenvalue (smallest first)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select k smallest eigenvectors
        selected_eigenvectors = eigenvectors[:, :self.n_clusters]
        selected_eigenvalues = eigenvalues[:self.n_clusters]
        
        return selected_eigenvalues, selected_eigenvectors
    
    def fit_predict(self, X):
        """
        Perform spectral clustering on input data.
        
        Args:
            X (np.ndarray): Input feature matrix (n_samples, n_features)
            
        Returns:
            np.ndarray: Cluster labels
        """
        print(f"Performing spectral clustering with {self.n_clusters} clusters...")
        print(f"Input shape: {X.shape}")
        
        # Step 1: Construct similarity matrix
        print("Step 1: Constructing similarity matrix...")
        self.similarity_matrix_ = self._construct_similarity_matrix(X)
        print(f"Similarity matrix shape: {self.similarity_matrix_.shape}")
        
        # Step 2: Compute Laplacian matrix
        print("Step 2: Computing Laplacian matrix...")
        self.laplacian_ = self._compute_laplacian(self.similarity_matrix_)
        
        # Step 3: Find eigenvectors
        print("Step 3: Computing eigenvectors...")
        self.eigenvalues_, self.eigenvectors_ = self._find_eigenvectors(self.laplacian_)
        print(f"Selected eigenvalues: {self.eigenvalues_}")
        
        # Step 4: Normalize eigenvectors (for normalized Laplacian)
        if self.laplacian_type == 'normalized':
            # Normalize rows to unit length
            row_norms = np.linalg.norm(self.eigenvectors_, axis=1, keepdims=True)
            self.eigenvectors_ = self.eigenvectors_ / (row_norms + 1e-12)
        
        # Step 5: Apply K-means to eigenvectors
        print("Step 4: Applying K-means to eigenvectors...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        self.labels_ = kmeans.fit_predict(self.eigenvectors_)
        
        print(f"Clustering completed. Found {len(np.unique(self.labels_))} clusters.")
        return self.labels_

def load_data(models_dir):
    """
    Load bottleneck features and labels from the models directory.
    
    Args:
        models_dir (str): Path to models directory
        
    Returns:
        tuple: (features, labels)
    """
    features_path = os.path.join(models_dir, 'bottleneck_features.npy')
    labels_path = os.path.join(models_dir, 'labels.npy')
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Bottleneck features not found at {features_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels not found at {labels_path}")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    
    return features, labels

def evaluate_clustering(true_labels, pred_labels, features):
    """
    Evaluate clustering performance using multiple metrics.
    
    Args:
        true_labels (np.ndarray): Ground truth labels
        pred_labels (np.ndarray): Predicted cluster labels
        features (np.ndarray): Original features for silhouette score
        
    Returns:
        dict: Evaluation metrics
    """
    metrics = {}
    
    # External evaluation metrics (if ground truth is available)
    if true_labels is not None and len(np.unique(true_labels)) > 1:
        metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, pred_labels)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, pred_labels)
    
    # Internal evaluation metrics
    if len(np.unique(pred_labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(features, pred_labels)
    
    metrics['n_clusters_found'] = len(np.unique(pred_labels))
    
    return metrics

def plot_clustering_results(features, labels, eigenvalues, eigenvectors, 
                          similarity_matrix, output_dir):
    """
    Create visualizations of clustering results.
    
    Args:
        features (np.ndarray): Original features
        labels (np.ndarray): Cluster labels
        eigenvalues (np.ndarray): Eigenvalues from spectral decomposition
        eigenvectors (np.ndarray): Eigenvectors used for clustering
        similarity_matrix (np.ndarray): Similarity matrix
        output_dir (str): Directory to save plots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Eigenvalue spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Spectrum of Graph Laplacian')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'eigenvalue_spectrum_{timestamp}.png'), dpi=300)
    plt.show()
    
    # Plot 2: First few eigenvectors
    plt.figure(figsize=(15, 10))
    n_eigenvectors_to_plot = min(6, eigenvectors.shape[1])
    for i in range(n_eigenvectors_to_plot):
        plt.subplot(2, 3, i + 1)
        plt.plot(eigenvectors[:, i], 'b-', alpha=0.7)
        plt.title(f'Eigenvector {i + 1} (λ = {eigenvalues[i]:.4f})')
        plt.xlabel('Sample Index')
        plt.ylabel('Eigenvector Value')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'eigenvectors_{timestamp}.png'), dpi=300)
    plt.show()
    
    # Plot 3: Similarity matrix heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Similarity')
    plt.title('Similarity Matrix')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'similarity_matrix_{timestamp}.png'), dpi=300)
    plt.show()
    
    # Plot 4: Cluster distribution
    plt.figure(figsize=(10, 6))
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.bar(unique_labels, counts, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    plt.grid(True, alpha=0.3, axis='y')
    for i, count in enumerate(counts):
        plt.text(unique_labels[i], count + 0.5, str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cluster_distribution_{timestamp}.png'), dpi=300)
    plt.show()
    
    # Plot 5: 2D projection of eigenvectors (first two)
    if eigenvectors.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(eigenvectors[:, 0], eigenvectors[:, 1], 
                            c=labels, cmap='tab10', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('First Eigenvector')
        plt.ylabel('Second Eigenvector')
        plt.title('2D Projection of Spectral Embedding')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spectral_embedding_2d_{timestamp}.png'), dpi=300)
        plt.show()

def save_results(clustering_model, labels, metrics, output_dir):
    """
    Save clustering results and model.
    
    Args:
        clustering_model: Trained spectral clustering model
        labels (np.ndarray): Cluster labels
        metrics (dict): Evaluation metrics
        output_dir (str): Directory to save results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save cluster labels
    labels_path = os.path.join(output_dir, f'spectral_cluster_labels_{timestamp}.npy')
    np.save(labels_path, labels)
    print(f"Cluster labels saved to: {labels_path}")
    
    # Save clustering model
    model_path = os.path.join(output_dir, f'spectral_clustering_model_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clustering_model, f)
    print(f"Clustering model saved to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'clustering_metrics_{timestamp}.txt')
    with open(metrics_path, 'w') as f:
        f.write("Spectral Clustering Results\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of clusters: {clustering_model.n_clusters}\n")
        f.write(f"Similarity metric: {clustering_model.similarity_metric}\n")
        f.write(f"Laplacian type: {clustering_model.laplacian_type}\n")
        f.write(f"Sigma (bandwidth): {clustering_model.sigma}\n\n")
        
        f.write("Evaluation Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"Metrics saved to: {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description='Spectral Clustering for Underwater Bioacoustics')
    parser.add_argument('--models_dir', type=str, default='models', 
                       help='Directory containing bottleneck features and labels')
    parser.add_argument('--output_dir', type=str, default='spectral_clustering_results',
                       help='Directory to save results')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Bandwidth parameter for Gaussian kernel')
    parser.add_argument('--laplacian_type', type=str, default='normalized',
                       choices=['normalized', 'unnormalized', 'rw'],
                       help='Type of Laplacian matrix')
    parser.add_argument('--similarity_metric', type=str, default='gaussian',
                       choices=['gaussian', 'knn', 'epsilon'],
                       help='Similarity metric for constructing adjacency matrix')
    parser.add_argument('--n_neighbors', type=int, default=10,
                       help='Number of neighbors for k-NN similarity graph')
    parser.add_argument('--standardize', action='store_true',
                       help='Standardize features before clustering')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SPECTRAL CLUSTERING FOR UNDERWATER BIOACOUSTICS")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    features, true_labels = load_data(args.models_dir)
    
    # Standardize features if requested
    if args.standardize:
        print("Standardizing features...")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run spectral clustering
    spectral_clustering = SpectralClustering(
        n_clusters=args.n_clusters,
        sigma=args.sigma,
        laplacian_type=args.laplacian_type,
        similarity_metric=args.similarity_metric,
        n_neighbors=args.n_neighbors
    )
    
    # Perform clustering
    cluster_labels = spectral_clustering.fit_predict(features)
    
    # Evaluate clustering
    print("\nEvaluating clustering performance...")
    metrics = evaluate_clustering(true_labels, cluster_labels, features)
    
    print("\nClustering Results:")
    print("-" * 20)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_clustering_results(
        features, cluster_labels, 
        spectral_clustering.eigenvalues_, 
        spectral_clustering.eigenvectors_,
        spectral_clustering.similarity_matrix_,
        args.output_dir
    )
    
    # Save results
    print("\nSaving results...")
    save_results(spectral_clustering, cluster_labels, metrics, args.output_dir)
    
    print(f"\nSpectral clustering completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Found {len(np.unique(cluster_labels))} clusters")

if __name__ == "__main__":
    main()