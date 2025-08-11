#!/usr/bin/env python3
"""
Spectral Clustering for Underwater Bioacoustic Signals
Performs spectral clustering on autoencoder bottleneck features of Bryde's whale calls.

Changes:
- Optional PCA whitening before graph construction (--pca_components)
- Mutual kNN graph with self-tuning sigmas (similarity_metric='knn_self_tuning')
- kNN heat-kernel graph (euclidean) and cosine-kNN variants
- Keeps classic gaussian RBF and epsilon graphs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # only used for color palettes in plots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csgraph
import argparse
import os
from datetime import datetime
import pickle


# -------------------------------
# Utility: build mutual kNN edges
# -------------------------------
def _mutual_knn_indices(idxs):
    """
    Given idxs (n x (k+1)) where idxs[i, 0] = i (self),
    return a boolean mutual-edge matrix mask (n x n).
    """
    n = idxs.shape[0]
    k = idxs.shape[1] - 1
    neigh_sets = [set(row[1:]) for row in idxs]  # drop self
    mutual = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in neigh_sets[i]:
            if i in neigh_sets[j]:
                mutual[i][j] = True
                mutual[j][i] = True
    return np.array(mutual, dtype=bool)


class SpectralClustering:
    """
    Custom implementation of spectral clustering for bioacoustic analysis.
    """
    def __init__(
        self,
        n_clusters=5,
        sigma=1.0,
        laplacian_type='normalized',
        similarity_metric='gaussian',
        n_neighbors=15,
        use_mutual=True,
        self_tuning=False,
        distance_metric='euclidean'
    ):
        """
        Args:
            n_clusters: number of clusters
            sigma: bandwidth for global Gaussian kernel
            laplacian_type: 'normalized', 'unnormalized', or 'rw'
            similarity_metric:
                'gaussian'         -> full RBF
                'knn'              -> kNN heat kernel (global sigma)
                'knn_self_tuning'  -> kNN with self-tuning sigma_i*sigma_j
                'cosine_knn'       -> cosine-distance kNN with heat kernel
                'epsilon'          -> epsilon-neighborhood (binary)
            n_neighbors: k for the kNN graph
            use_mutual: keep only mutual edges for kNN variants
            self_tuning: enable self-tuning on kNN if True (overridden by 'knn_self_tuning')
            distance_metric: for kNN ('euclidean' or 'cosine')
        """
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.laplacian_type = laplacian_type
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.use_mutual = use_mutual
        self.self_tuning = self_tuning
        self.distance_metric = distance_metric

        self.similarity_matrix_ = None
        self.laplacian_ = None
        self.eigenvectors_ = None
        self.eigenvalues_ = None
        self.labels_ = None

    # -----------------------------
    # Similarity / adjacency matrix
    # -----------------------------
    def _construct_similarity_matrix(self, X):
        n = X.shape[0]

        if self.similarity_metric == 'gaussian':
            # Full dense RBF with global sigma
            d2 = squareform(pdist(X, metric='euclidean')) ** 2
            W = np.exp(-d2 / (2 * (self.sigma ** 2)))
            np.fill_diagonal(W, 1.0)
            return W

        # kNN variants
        if self.similarity_metric in ('knn', 'knn_self_tuning', 'cosine_knn'):
            metric = 'cosine' if self.similarity_metric == 'cosine_knn' else self.distance_metric
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric=metric).fit(X)
            dists, idxs = nbrs.kneighbors(X)  # dists[:,0]=0 to self; idxs includes self

            # Mutual kNN mask (optional)
            if self.use_mutual:
                mutual = _mutual_knn_indices(idxs)
            else:
                mutual = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in idxs[i, 1:]:
                        mutual[i, j] = True
                        mutual[j, i] = True

            # Self-tuning sigmas (sigma_i = distance to k-th neighbor of i)
            use_self_tuning = self.self_tuning or (self.similarity_metric == 'knn_self_tuning')
            if use_self_tuning:
                # distance to k-th neighbor (last column, excluding self)
                sigma_i = dists[:, -1] + 1e-12
            else:
                sigma_i = None  # use global sigma

            # Build weighted adjacency
            W = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in np.where(mutual[i])[0]:
                    if metric == 'cosine':
                        # cosine distance in [0,2], convert to similarity weight via heat kernel
                        dij2 = (1.0 - (1.0 - dists[i, np.where(idxs[i] == j)[0][0]])) ** 2  # not exact; fallback
                        # better: compute directly from vectors to be safe
                        # but we'll use Euclidean-like schema for kernel scaling
                    # compute squared Euclidean distance (safe)
                    diff2 = np.sum((X[i] - X[j]) ** 2)
                    if use_self_tuning:
                        Wij = np.exp(-diff2 / (sigma_i[i] * sigma_i[j]))
                    else:
                        Wij = np.exp(-diff2 / (2 * (self.sigma ** 2)))
                    W[i, j] = Wij
                    W[j, i] = Wij
            # ensure diagonal
            np.fill_diagonal(W, 1.0)
            return W

        if self.similarity_metric == 'epsilon':
            D = squareform(pdist(X, metric='euclidean'))
            eps = np.percentile(D, 10)  # heuristic
            W = (D <= eps).astype(float)
            np.fill_diagonal(W, 1.0)
            return W

        raise ValueError(f"Unknown similarity_metric: {self.similarity_metric}")

    # ----------------
    # Graph Laplacian
    # ----------------
    def _compute_laplacian(self, W):
        D = np.diag(W.sum(axis=1))
        if self.laplacian_type == 'unnormalized':
            return D - W
        if self.laplacian_type == 'normalized':
            D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-12))
            return D_sqrt_inv @ (D - W) @ D_sqrt_inv
        if self.laplacian_type == 'rw':
            D_inv = np.diag(1.0 / (np.diag(D) + 1e-12))
            return D_inv @ (D - W)
        raise ValueError(f"Unknown laplacian_type: {self.laplacian_type}")

    # ----------------
    # Eigen-decompose
    # ----------------
    def _find_eigenvectors(self, L):
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        U = eigvecs[:, :self.n_clusters]
        lam = eigvals[:self.n_clusters]
        return lam, U

    # -------------
    # Fit-predict
    # -------------
    def fit_predict(self, X):
        print(f"Performing spectral clustering with {self.n_clusters} clusters...")
        print(f"Input shape: {X.shape}")

        print("Step 1: Constructing similarity matrix...")
        self.similarity_matrix_ = self._construct_similarity_matrix(X)
        print(f"Similarity matrix shape: {self.similarity_matrix_.shape}")

        print("Step 2: Computing Laplacian matrix...")
        self.laplacian_ = self._compute_laplacian(self.similarity_matrix_)

        print("Step 3: Computing eigenvectors...")
        self.eigenvalues_, self.eigenvectors_ = self._find_eigenvectors(self.laplacian_)
        print(f"Selected eigenvalues: {self.eigenvalues_}")

        if self.laplacian_type == 'normalized':
            # Row-normalize embedding (Ng, Jordan & Weiss 2002)
            norms = np.linalg.norm(self.eigenvectors_, axis=1, keepdims=True)
            self.eigenvectors_ = self.eigenvectors_ / (norms + 1e-12)

        print("Step 4: Applying K-means to eigenvectors...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=50)
        self.labels_ = kmeans.fit_predict(self.eigenvectors_)

        print(f"Clustering completed. Found {len(np.unique(self.labels_))} clusters.")
        return self.labels_


# ------------
# IO helpers
# ------------
def load_data(models_dir):
    features_path = os.path.join(models_dir, 'bottleneck_features.npy')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Bottleneck features not found at {features_path}")
    feats = np.load(features_path)
    print(f"Loaded features shape: {feats.shape}")
    return feats


def evaluate_clustering(pred_labels, features):
    metrics = {}
    if len(np.unique(pred_labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(features, pred_labels)
    metrics['n_clusters_found'] = len(np.unique(pred_labels))
    unique_labels, counts = np.unique(pred_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        metrics[f'cluster_{label}_size'] = count
        metrics[f'cluster_{label}_percentage'] = count / len(pred_labels) * 100
    return metrics


def plot_clustering_results(features, labels, eigenvalues, eigenvectors, similarity_matrix, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Eigenvalue spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    plt.xlabel('Eigenvalue Index'); plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Spectrum of Graph Laplacian'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'eigenvalue_spectrum_{timestamp}.png'), dpi=300); plt.close()

    # First eigenvectors
    plt.figure(figsize=(15, 10))
    m = min(6, eigenvectors.shape[1])
    for i in range(m):
        plt.subplot(2, 3, i + 1)
        plt.plot(eigenvectors[:, i], 'b-', alpha=0.7)
        plt.title(f'Eigenvector {i + 1} (λ = {eigenvalues[i]:.4f})')
        plt.xlabel('Sample Index'); plt.ylabel('Value'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'eigenvectors_{timestamp}.png'), dpi=300); plt.close()

    # Similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Similarity'); plt.title('Similarity Matrix')
    plt.xlabel('Sample Index'); plt.ylabel('Sample Index')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'similarity_matrix_{timestamp}.png'), dpi=300); plt.close()

    # Cluster distribution
    plt.figure(figsize=(10, 6))
    uniq, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(uniq)))
    plt.bar(uniq, counts, alpha=0.8, color=colors, edgecolor='black')
    for l, c in zip(uniq, counts):
        plt.text(l, c + 0.5, f'{c}\n({c/len(labels)*100:.1f}%)', ha='center', va='bottom')
    plt.xlabel('Cluster ID'); plt.ylabel('Number of Samples'); plt.title('Cluster Size Distribution')
    plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cluster_distribution_{timestamp}.png'), dpi=300); plt.close()

    # 2D projection of spectral embedding
    if eigenvectors.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(eigenvectors[:, 0], eigenvectors[:, 1], c=labels, cmap='tab10', s=50, alpha=0.75)
        plt.colorbar(sc, label='Cluster'); plt.xlabel('First Eigenvector'); plt.ylabel('Second Eigenvector')
        plt.title('2D Projection of Spectral Embedding'); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spectral_embedding_2d_{timestamp}.png'), dpi=300); plt.close()


def save_results(clustering_model, labels, metrics, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels_path = os.path.join(output_dir, f'spectral_cluster_labels_{timestamp}.npy')
    np.save(labels_path, labels); print(f"Cluster labels saved to: {labels_path}")

    model_path = os.path.join(output_dir, f'spectral_clustering_model_{timestamp}.pkl')
    with open(model_path, 'wb') as f: pickle.dump(clustering_model, f)
    print(f"Clustering model saved to: {model_path}")

    metrics_path = os.path.join(output_dir, f'clustering_metrics_{timestamp}.txt')
    with open(metrics_path, 'w') as f:
        f.write("Spectral Clustering Results\n" + "="*30 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of clusters: {clustering_model.n_clusters}\n")
        f.write(f"Similarity metric: {clustering_model.similarity_metric}\n")
        f.write(f"Laplacian type: {clustering_model.laplacian_type}\n")
        f.write(f"Sigma (bandwidth): {clustering_model.sigma}\n")
        f.write(f"Neighbors (k): {clustering_model.n_neighbors}\n")
        f.write(f"Mutual edges: {clustering_model.use_mutual}\n")
        f.write(f"Self-tuning: {clustering_model.self_tuning}\n\n")
        f.write("Evaluation Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
    print(f"Metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Spectral Clustering for Underwater Bioacoustics')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing bottleneck features')
    parser.add_argument('--output_dir', type=str, default='spectral_clustering_results', help='Directory to save results')
    parser.add_argument('--n_clusters', type=int, default=4, help='Number of clusters')
    parser.add_argument('--sigma', type=float, default=1.0, help='Bandwidth parameter for Gaussian kernel')
    parser.add_argument('--laplacian_type', type=str, default='normalized', choices=['normalized', 'unnormalized', 'rw'])
    parser.add_argument('--similarity_metric', type=str, default='knn_self_tuning',
                        choices=['gaussian', 'knn', 'knn_self_tuning', 'cosine_knn', 'epsilon'])
    parser.add_argument('--n_neighbors', type=int, default=15, help='Number of neighbors for k-NN similarity graph')
    parser.add_argument('--no_mutual', action='store_true', help='Do not enforce mutual kNN edges')
    parser.add_argument('--self_tuning', action='store_true', help='Enable self-tuning sigmas for kNN')
    parser.add_argument('--standardize', action='store_true', help='Standardize features before clustering')
    parser.add_argument('--pca_components', type=int, default=32, help='PCA components (<=0 to disable)')
    parser.add_argument('--distance_metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='Distance metric for kNN (ignored for gaussian/epsilon)')
    args = parser.parse_args()

    print("=" * 50)
    print("SPECTRAL CLUSTERING FOR UNDERWATER BIOACOUSTICS")
    print("=" * 50)

    # Load
    print("\nLoading data...")
    X = load_data(args.models_dir)

    # Standardize
    if args.standardize:
        print("Standardizing features...")
        X = StandardScaler().fit_transform(X)

    # PCA whiten
    if args.pca_components and args.pca_components > 0:
        print(f"Applying PCA whitening to {args.pca_components} components...")
        X = PCA(n_components=args.pca_components, whiten=True, random_state=0).fit_transform(X)
        print(f"PCA output shape: {X.shape}")

    # Init clustering
    spectral = SpectralClustering(
        n_clusters=args.n_clusters,
        sigma=args.sigma,
        laplacian_type=args.laplacian_type,
        similarity_metric=args.similarity_metric,
        n_neighbors=args.n_neighbors,
        use_mutual=not args.no_mutual,
        self_tuning=args.self_tuning,
        distance_metric=args.distance_metric
    )

    # Run
    labels = spectral.fit_predict(X)

    # Evaluate on the same feature space used for the graph (X)
    print("\nEvaluating clustering performance...")
    metrics = evaluate_clustering(labels, X)

    print("\nClustering Results:")
    print("-" * 20)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Visualizations
    print("\nGenerating visualizations...")
    plot_clustering_results(
        X, labels,
        spectral.eigenvalues_, spectral.eigenvectors_,
        spectral.similarity_matrix_, args.output_dir
    )

    # Save
    print("\nSaving results...")
    save_results(spectral, labels, metrics, args.output_dir)

    print("\nSpectral clustering completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Found {len(np.unique(labels))} clusters")


if __name__ == "__main__":
    main()
