import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns

# Load the bottleneck features
features = np.load('models/bottleneck_features.npy')
print(f"Features shape: {features.shape}")
print(f"Features dtype: {features.dtype}")

# Check for any NaN or infinite values
print(f"\nContains NaN: {np.any(np.isnan(features))}")
print(f"Contains Inf: {np.any(np.isinf(features))}")

# Basic statistics
print("\nFeature statistics:")
print(f"Mean: {np.mean(features):.4f}")
print(f"Std: {np.std(features):.4f}")
print(f"Min: {np.min(features):.4f}")
print(f"Max: {np.max(features):.4f}")

# Check variance per feature
feature_vars = np.var(features, axis=0)
print(f"\nFeature variances - Min: {np.min(feature_vars):.6f}, Max: {np.max(feature_vars):.6f}")
print(f"Number of low-variance features (<0.01): {np.sum(feature_vars < 0.01)}")

# Visualize feature distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Feature variance plot
axes[0, 0].bar(range(len(feature_vars)), feature_vars)
axes[0, 0].set_title('Variance per Feature')
axes[0, 0].set_xlabel('Feature Index')
axes[0, 0].set_ylabel('Variance')

# 2. PCA visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)
axes[0, 1].scatter(features_pca[:, 0], features_pca[:, 1], alpha=0.5)
axes[0, 1].set_title(f'PCA Projection (explained var: {pca.explained_variance_ratio_.sum():.2%})')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')

# 3. Standardized features PCA
scaler = StandardScaler()
features_std = scaler.fit_transform(features)
features_pca_std = pca.fit_transform(features_std)
axes[1, 0].scatter(features_pca_std[:, 0], features_pca_std[:, 1], alpha=0.5)
axes[1, 0].set_title(f'PCA on Standardized Features')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')

# 4. t-SNE visualization (on standardized features)
print("\nComputing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_tsne = tsne.fit_transform(features_std)
axes[1, 1].scatter(features_tsne[:, 0], features_tsne[:, 1], alpha=0.5)
axes[1, 1].set_title('t-SNE Projection')
axes[1, 1].set_xlabel('t-SNE 1')
axes[1, 1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig('data_analysis/bottleneck_features_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Check pairwise distances to understand sigma selection
from scipy.spatial.distance import pdist
distances = pdist(features_std, metric='euclidean')
print(f"\nPairwise distances (standardized features):")
print(f"Mean: {np.mean(distances):.4f}")
print(f"Std: {np.std(distances):.4f}")
print(f"Percentiles: 10th={np.percentile(distances, 10):.4f}, 50th={np.percentile(distances, 50):.4f}, 90th={np.percentile(distances, 90):.4f}")

# Suggest sigma values based on distance distribution
suggested_sigmas = [
    np.percentile(distances, 10),
    np.percentile(distances, 25),
    np.percentile(distances, 50),
]
print(f"\nSuggested sigma values to try: {suggested_sigmas}")

# Quick test: Can we see any natural clusters?
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("\nTesting different numbers of clusters with K-means:")
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_std)
    score = silhouette_score(features_std, labels)
    sizes = [np.sum(labels == i) for i in range(k)]
    print(f"k={k}: silhouette={score:.3f}, cluster sizes={sizes}")