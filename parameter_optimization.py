#!/usr/bin/env python3
"""
Parameter optimization for spectral clustering of underwater bioacoustic signals.
Tests different parameter combinations and finds the best configuration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import argparse
import os
from datetime import datetime
import json
from itertools import product
import sys
sys.path.append('.')  # Add current directory to path
from spectral_clustering import SpectralClustering

class ParameterOptimizer:
    def __init__(self, features, output_dir='parameter_optimization_results'):
        """
        Initialize parameter optimizer.
        
        Args:
            features: Input features for clustering
            output_dir: Directory to save results
        """
        self.features = features
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define parameter grid
        self.param_grid = {
            'n_clusters': [3, 4, 5, 6, 7, 8],
            'sigma': [0.5, 1.0, 1.5, 2.0, 3.0],
            'laplacian_type': ['normalized', 'unnormalized', 'rw'],
            'similarity_metric': ['gaussian', 'knn'],
            'n_neighbors': [5, 10, 15, 20]  # Only used for knn
        }
        
        self.results = []
        
    def evaluate_clustering(self, labels, features):
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            labels: Cluster labels
            features: Input features
            
        Returns:
            dict: Evaluation metrics
        """
        n_clusters = len(np.unique(labels))
        
        # Skip if only one cluster found
        if n_clusters == 1:
            return {
                'silhouette_score': -1,
                'calinski_harabasz_score': 0,
                'davies_bouldin_score': float('inf'),
                'n_clusters_found': 1
            }
        
        metrics = {
            'silhouette_score': silhouette_score(features, labels),
            'calinski_harabasz_score': calinski_harabasz_score(features, labels),
            'davies_bouldin_score': davies_bouldin_score(features, labels),
            'n_clusters_found': n_clusters
        }
        
        # Calculate cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['min_cluster_size'] = np.min(counts)
        metrics['max_cluster_size'] = np.max(counts)
        metrics['cluster_size_std'] = np.std(counts)
        
        return metrics
    
    def test_parameters(self, params):
        """
        Test a single parameter combination.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            dict: Results including parameters and metrics
        """
        # Skip invalid combinations
        if params['similarity_metric'] != 'knn' and 'n_neighbors' in params:
            # For non-knn methods, n_neighbors is not used
            params = params.copy()
            params['n_neighbors'] = None
        
        try:
            # Create clustering model
            model_params = {k: v for k, v in params.items() if v is not None}
            spectral = SpectralClustering(**model_params)
            
            # Perform clustering
            labels = spectral.fit_predict(self.features)
            
            # Evaluate
            metrics = self.evaluate_clustering(labels, self.features)
            
            # Combine parameters and metrics
            result = params.copy()
            result.update(metrics)
            result['success'] = True
            result['error'] = None
            
            return result
            
        except Exception as e:
            result = params.copy()
            result['success'] = False
            result['error'] = str(e)
            return result
    
    def run_grid_search(self):
        """
        Run grid search over all parameter combinations.
        """
        print("Starting parameter grid search...")
        
        # Generate all parameter combinations
        param_combinations = []
        
        for params in product(*[self.param_grid[k] for k in self.param_grid.keys()]):
            param_dict = dict(zip(self.param_grid.keys(), params))
            
            # Skip invalid combinations
            if param_dict['similarity_metric'] != 'knn':
                # For non-knn, only use one n_neighbors value
                if param_dict['n_neighbors'] != self.param_grid['n_neighbors'][0]:
                    continue
            
            param_combinations.append(param_dict)
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Test each combination
        for i, params in enumerate(param_combinations):
            print(f"\nTesting combination {i+1}/{len(param_combinations)}:")
            print(f"  Parameters: {params}")
            
            result = self.test_parameters(params)
            self.results.append(result)
            
            if result['success']:
                print(f"  Silhouette score: {result['silhouette_score']:.3f}")
            else:
                print(f"  Failed: {result['error']}")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f'grid_search_results_{timestamp}.csv')
        self.results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
    def analyze_results(self):
        """
        Analyze grid search results and find best parameters.
        """
        # Filter successful runs
        successful_results = self.results_df[self.results_df['success']]
        
        if len(successful_results) == 0:
            print("No successful clustering runs!")
            return
        
        print(f"\nAnalyzing {len(successful_results)} successful runs...")
        
        # Find best parameters by different metrics
        best_silhouette = successful_results.loc[successful_results['silhouette_score'].idxmax()]
        best_calinski = successful_results.loc[successful_results['calinski_harabasz_score'].idxmax()]
        best_davies_bouldin = successful_results.loc[successful_results['davies_bouldin_score'].idxmin()]
        
        print("\nBest parameters by metric:")
        print("\n1. Best Silhouette Score:")
        self._print_best_params(best_silhouette)
        
        print("\n2. Best Calinski-Harabasz Score:")
        self._print_best_params(best_calinski)
        
        print("\n3. Best Davies-Bouldin Score:")
        self._print_best_params(best_davies_bouldin)
        
        # Save best parameters
        best_params = {
            'best_silhouette': best_silhouette.to_dict(),
            'best_calinski': best_calinski.to_dict(),
            'best_davies_bouldin': best_davies_bouldin.to_dict()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_params_path = os.path.join(self.output_dir, f'best_parameters_{timestamp}.json')
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\nBest parameters saved to: {best_params_path}")
        
    def _print_best_params(self, params_row):
        """Print parameters and metrics for a result row."""
        print(f"  n_clusters: {int(params_row['n_clusters'])}")
        print(f"  sigma: {params_row['sigma']}")
        print(f"  laplacian_type: {params_row['laplacian_type']}")
        print(f"  similarity_metric: {params_row['similarity_metric']}")
        if params_row['similarity_metric'] == 'knn':
            print(f"  n_neighbors: {int(params_row['n_neighbors'])}")
        print(f"  Metrics:")
        print(f"    Silhouette: {params_row['silhouette_score']:.3f}")
        print(f"    Calinski-Harabasz: {params_row['calinski_harabasz_score']:.1f}")
        print(f"    Davies-Bouldin: {params_row['davies_bouldin_score']:.3f}")
        print(f"    Clusters found: {int(params_row['n_clusters_found'])}")
        print(f"    Min cluster size: {int(params_row['min_cluster_size'])}")
    
    def plot_results(self):
        """Create visualizations of parameter optimization results."""
        successful_results = self.results_df[self.results_df['success']]
        
        if len(successful_results) == 0:
            return
        
        # Plot 1: Silhouette score vs number of clusters
        plt.figure(figsize=(12, 8))
        
        # Group by similarity metric
        for metric in successful_results['similarity_metric'].unique():
            metric_data = successful_results[successful_results['similarity_metric'] == metric]
            
            # Average across other parameters
            avg_scores = metric_data.groupby('n_clusters')['silhouette_score'].agg(['mean', 'std'])
            
            plt.errorbar(avg_scores.index, avg_scores['mean'], 
                        yerr=avg_scores['std'], 
                        label=f'{metric}', marker='o', capsize=5)
        
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.output_dir, f'silhouette_vs_clusters_{timestamp}.png'), dpi=300)
        plt.close()
        
        # Plot 2: Heatmap of silhouette scores for Gaussian similarity
        gaussian_results = successful_results[successful_results['similarity_metric'] == 'gaussian']
        
        if len(gaussian_results) > 0:
            # Create pivot table
            pivot_data = gaussian_results.pivot_table(
                values='silhouette_score',
                index='n_clusters',
                columns='sigma',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                       cbar_kws={'label': 'Silhouette Score'})
            plt.title('Silhouette Score Heatmap (Gaussian Similarity)')
            plt.xlabel('Sigma')
            plt.ylabel('Number of Clusters')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'gaussian_heatmap_{timestamp}.png'), dpi=300)
            plt.close()
        
        # Plot 3: Comparison of metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Silhouette score distribution
        axes[0, 0].hist(successful_results['silhouette_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Silhouette Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Silhouette Scores')
        axes[0, 0].axvline(successful_results['silhouette_score'].max(), color='red', 
                          linestyle='--', label=f'Best: {successful_results["silhouette_score"].max():.3f}')
        axes[0, 0].legend()
        
        # Davies-Bouldin score distribution
        axes[0, 1].hist(successful_results['davies_bouldin_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Davies-Bouldin Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Davies-Bouldin Scores')
        axes[0, 1].axvline(successful_results['davies_bouldin_score'].min(), color='red', 
                          linestyle='--', label=f'Best: {successful_results["davies_bouldin_score"].min():.3f}')
        axes[0, 1].legend()
        
        # Cluster size variation
        axes[1, 0].scatter(successful_results['silhouette_score'], 
                          successful_results['cluster_size_std'], 
                          alpha=0.6)
        axes[1, 0].set_xlabel('Silhouette Score')
        axes[1, 0].set_ylabel('Cluster Size Std Dev')
        axes[1, 0].set_title('Silhouette Score vs Cluster Size Variation')
        
        # Laplacian type comparison
        laplacian_scores = successful_results.groupby('laplacian_type')['silhouette_score'].agg(['mean', 'std'])
        axes[1, 1].bar(laplacian_scores.index, laplacian_scores['mean'], 
                      yerr=laplacian_scores['std'], capsize=5, alpha=0.7)
        axes[1, 1].set_xlabel('Laplacian Type')
        axes[1, 1].set_ylabel('Mean Silhouette Score')
        axes[1, 1].set_title('Performance by Laplacian Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'metrics_comparison_{timestamp}.png'), dpi=300)
        plt.close()
        
        print(f"\nVisualization saved to: {self.output_dir}")

def run_quick_test(features, output_dir):
    """
    Run a quick test with a subset of parameters for faster results.
    """
    print("\nRunning quick parameter test...")
    
    quick_params = [
        {'n_clusters': 4, 'sigma': 1.0, 'laplacian_type': 'normalized', 'similarity_metric': 'gaussian'},
        {'n_clusters': 5, 'sigma': 1.0, 'laplacian_type': 'normalized', 'similarity_metric': 'gaussian'},
        {'n_clusters': 6, 'sigma': 1.0, 'laplacian_type': 'normalized', 'similarity_metric': 'gaussian'},
        {'n_clusters': 5, 'sigma': 0.5, 'laplacian_type': 'normalized', 'similarity_metric': 'gaussian'},
        {'n_clusters': 5, 'sigma': 1.5, 'laplacian_type': 'normalized', 'similarity_metric': 'gaussian'},
        {'n_clusters': 5, 'sigma': 1.0, 'laplacian_type': 'unnormalized', 'similarity_metric': 'gaussian'},
        {'n_clusters': 5, 'sigma': 1.0, 'laplacian_type': 'normalized', 'similarity_metric': 'knn', 'n_neighbors': 10},
    ]
    
    results = []
    
    for params in quick_params:
        print(f"\nTesting: {params}")
        
        # Create model
        spectral = SpectralClustering(**params)
        labels = spectral.fit_predict(features)
        
        # Evaluate
        silhouette = silhouette_score(features, labels)
        
        result = params.copy()
        result['silhouette_score'] = silhouette
        result['n_clusters_found'] = len(np.unique(labels))
        results.append(result)
        
        print(f"  Silhouette: {silhouette:.3f}, Clusters found: {result['n_clusters_found']}")
    
    # Find best
    results_df = pd.DataFrame(results)
    best_idx = results_df['silhouette_score'].idxmax()
    best_params = results_df.loc[best_idx]
    
    print("\nQuick test - Best parameters:")
    print(best_params.to_dict())
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f'quick_test_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    
    return best_params.to_dict()

def main():
    parser = argparse.ArgumentParser(description='Parameter optimization for spectral clustering')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing bottleneck features')
    parser.add_argument('--output_dir', type=str, default='parameter_optimization_results',
                       help='Directory to save results')
    parser.add_argument('--standardize', action='store_true',
                       help='Standardize features before clustering')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with subset of parameters')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPECTRAL CLUSTERING PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Load features
    features_path = os.path.join(args.models_dir, 'bottleneck_features.npy')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found at {features_path}")
    
    features = np.load(features_path)
    print(f"Loaded features shape: {features.shape}")
    
    # Standardize if requested
    if args.standardize:
        print("Standardizing features...")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.quick:
        # Run quick test
        best_params = run_quick_test(features, args.output_dir)
        
        print("\n" + "=" * 60)
        print("QUICK TEST COMPLETE!")
        print("Recommended parameters for full clustering:")
        for k, v in best_params.items():
            if not k.endswith('_score') and not k.endswith('_found'):
                print(f"  --{k} {v}")
    else:
        # Run full grid search
        optimizer = ParameterOptimizer(features, args.output_dir)
        optimizer.run_grid_search()
        optimizer.analyze_results()
        optimizer.plot_results()
        
        print("\n" + "=" * 60)
        print("PARAMETER OPTIMIZATION COMPLETE!")
        print(f"Results saved to: {args.output_dir}")
        print("\nNext steps:")
        print("1. Review best parameters in best_parameters_*.json")
        print("2. Check visualizations for parameter trends")
        print("3. Run spectral_clustering.py with optimal parameters")
    
    print("=" * 60)

if __name__ == "__main__":
    main()