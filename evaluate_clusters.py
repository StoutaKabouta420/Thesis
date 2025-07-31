import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from collections import defaultdict
import pandas as pd
from scipy.io import wavfile
import argparse
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import silhouette_samples

class ClusterEvaluator:
    def __init__(self, cluster_labels_path, extracted_segments_dir, spectrograms_dir, output_dir):
        """
        Initialize the cluster evaluator.
        
        Args:
            cluster_labels_path: Path to the cluster labels .npy file
            extracted_segments_dir: Directory containing extracted audio segments
            spectrograms_dir: Directory containing spectrogram arrays
            output_dir: Directory to save evaluation results
        """
        self.cluster_labels = np.load(cluster_labels_path)
        self.extracted_segments_dir = Path(extracted_segments_dir)
        self.spectrograms_dir = Path(spectrograms_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load spectrogram data
        self.spectrograms = np.load(self.spectrograms_dir / 'spectrograms_enhanced.npy')
        self.frequencies = np.load(self.spectrograms_dir / 'frequencies.npy')
        self.times = np.load(self.spectrograms_dir / 'times.npy')
        
        # Calculate global min/max for consistent visualization
        self.global_vmin = np.percentile(self.spectrograms, 5)
        self.global_vmax = np.percentile(self.spectrograms, 95)
        
        print(f"Loaded {len(self.cluster_labels)} cluster labels")
        print(f"Loaded {len(self.spectrograms)} spectrograms")
        print(f"Spectrogram value range: {self.global_vmin:.2f} to {self.global_vmax:.2f}")
        
        # Collect all audio files
        self.audio_files = self._collect_audio_files()
        print(f"Found {len(self.audio_files)} audio files")
        
        # Get unique clusters
        self.unique_clusters = np.unique(self.cluster_labels)
        self.n_clusters = len(self.unique_clusters)
        print(f"Number of clusters: {self.n_clusters}")
        
    def _collect_audio_files(self):
        """Collect all audio files from the extracted segments directory."""
        audio_files = []
        
        for recording_dir in self.extracted_segments_dir.iterdir():
            if recording_dir.is_dir():
                wav_files = list(recording_dir.glob("*.wav"))
                audio_files.extend(wav_files)
        
        # Sort to ensure consistent ordering
        audio_files.sort()
        return audio_files
    
    def calculate_cluster_characteristics(self):
        """Calculate acoustic characteristics for each cluster."""
        print("\nCalculating cluster characteristics...")
        
        cluster_chars = []
        
        for cluster_id in self.unique_clusters:
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            cluster_specs = self.spectrograms[cluster_indices]
            
            # Calculate mean spectrogram
            mean_spec = np.mean(cluster_specs, axis=0)
            
            # Calculate dominant frequency (weighted average)
            freq_power = np.mean(mean_spec, axis=1)
            dominant_freq_idx = np.argmax(freq_power)
            dominant_freq = self.frequencies[dominant_freq_idx]
            
            # Calculate bandwidth (frequency spread)
            threshold = np.max(freq_power) * 0.5  # -3dB bandwidth
            freq_mask = freq_power > threshold
            if np.any(freq_mask):
                freq_indices = np.where(freq_mask)[0]
                bandwidth = self.frequencies[freq_indices[-1]] - self.frequencies[freq_indices[0]]
            else:
                bandwidth = 0
            
            # Calculate temporal characteristics
            time_power = np.mean(mean_spec, axis=0)
            mean_duration = np.sum(time_power > np.mean(time_power)) * (self.times[1] - self.times[0])
            
            cluster_chars.append({
                'cluster_id': cluster_id,
                'sample_count': len(cluster_indices),
                'dominant_frequency_hz': dominant_freq,
                'bandwidth_hz': bandwidth,
                'mean_duration_s': mean_duration,
                'mean_power_db': np.mean(cluster_specs)
            })
        
        chars_df = pd.DataFrame(cluster_chars)
        chars_df.to_csv(self.output_dir / 'cluster_acoustic_characteristics.csv', index=False)
        
        print("\nCluster Acoustic Characteristics:")
        print(chars_df.to_string(index=False))
        
        return chars_df
    
    def organize_by_cluster(self, copy_files=False):
        """
        Organize files by cluster, creating directories and optionally copying files.
        
        Args:
            copy_files: If True, copy files. If False, create symbolic links.
        """
        print("\nOrganizing files by cluster...")
        
        cluster_stats = []
        
        for cluster_id in self.unique_clusters:
            # Get indices for this cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            # Create cluster directory
            cluster_dir = self.output_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            audio_dir = cluster_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            
            spectrograms_dir = cluster_dir / "spectrograms"
            spectrograms_dir.mkdir(exist_ok=True)
            
            print(f"\nProcessing Cluster {cluster_id}: {len(cluster_indices)} samples")
            
            # Process files for this cluster
            audio_count = 0
            
            for idx in cluster_indices:
                if idx < len(self.audio_files):
                    # Handle audio file
                    src_audio = self.audio_files[idx]
                    dst_audio = audio_dir / src_audio.name
                    
                    if copy_files:
                        shutil.copy2(src_audio, dst_audio)
                    else:
                        if dst_audio.exists():
                            dst_audio.unlink()
                        dst_audio.symlink_to(src_audio.resolve())
                    
                    audio_count += 1
                    
                    # Save spectrogram as image
                    if idx < len(self.spectrograms):
                        spec_filename = spectrograms_dir / f"{src_audio.stem}_spectrogram.png"
                        self._save_spectrogram_image(self.spectrograms[idx], spec_filename)
            
            # Store statistics
            cluster_stats.append({
                'cluster_id': cluster_id,
                'sample_count': len(cluster_indices),
                'percentage': len(cluster_indices) / len(self.cluster_labels) * 100,
                'audio_files_count': audio_count
            })
        
        # Create summary statistics
        stats_df = pd.DataFrame(cluster_stats)
        stats_df.to_csv(self.output_dir / 'cluster_statistics.csv', index=False)
        
        print("\nCluster Statistics:")
        print(stats_df.to_string(index=False))
        
        return stats_df
    
    def _save_spectrogram_image(self, spectrogram, filename):
        """Save a single spectrogram as an image."""
        plt.figure(figsize=(10, 6))
        
        # Use consistent color scaling
        plt.imshow(spectrogram, aspect='auto', origin='lower',
                  extent=[self.times[0], self.times[-1], 
                         self.frequencies[0], self.frequencies[-1]],
                  cmap='viridis', vmin=self.global_vmin, vmax=self.global_vmax)
        
        plt.colorbar(label='Power (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Spectrogram: {filename.stem}')
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    
    def create_cluster_summary_spectrograms(self, samples_per_cluster=5):
        """
        Create summary spectrograms showing representative samples from each cluster.
        
        Args:
            samples_per_cluster: Number of samples to show per cluster
        """
        print("\nCreating cluster summary spectrograms...")
        
        for cluster_id in self.unique_clusters:
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            # Select representative samples
            if len(cluster_indices) > samples_per_cluster:
                # Select evenly spaced samples
                selected_indices = cluster_indices[::len(cluster_indices)//samples_per_cluster][:samples_per_cluster]
            else:
                selected_indices = cluster_indices
            
            # Create figure with subplots
            n_samples = len(selected_indices)
            fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
            
            if n_samples == 1:
                axes = [axes]
            
            for i, idx in enumerate(selected_indices):
                if idx < len(self.spectrograms):
                    spec = self.spectrograms[idx]
                    
                    im = axes[i].imshow(spec, aspect='auto', origin='lower',
                                       extent=[self.times[0], self.times[-1],
                                              self.frequencies[0], self.frequencies[-1]],
                                       cmap='viridis', vmin=self.global_vmin, vmax=self.global_vmax)
                    
                    if idx < len(self.audio_files):
                        axes[i].set_title(f'{self.audio_files[idx].stem[:20]}...')
                    
                    axes[i].set_xlabel('Time (s)')
                    if i == 0:
                        axes[i].set_ylabel('Frequency (Hz)')
            
            plt.suptitle(f'Cluster {cluster_id} - Representative Samples ({len(cluster_indices)} total)')
            plt.tight_layout()
            
            # Save
            summary_path = self.output_dir / f'cluster_{cluster_id}_summary.png'
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved summary for Cluster {cluster_id}")
    
    def create_cluster_comparison_plot(self):
        """Create a single plot comparing one representative sample from each cluster."""
        print("\nCreating cluster comparison plot...")
        
        n_clusters = len(self.unique_clusters)
        fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 4))
        
        if n_clusters == 1:
            axes = [axes]
        
        for i, cluster_id in enumerate(self.unique_clusters):
            # Get middle sample from cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Pick middle sample
                middle_idx = cluster_indices[len(cluster_indices)//2]
                
                if middle_idx < len(self.spectrograms):
                    spec = self.spectrograms[middle_idx]
                    
                    # Use consistent color scaling based on actual data range
                    im = axes[i].imshow(spec, aspect='auto', origin='lower',
                                       extent=[self.times[0], self.times[-1],
                                              self.frequencies[0], self.frequencies[-1]],
                                       cmap='viridis', vmin=self.global_vmin, vmax=self.global_vmax)
                    
                    axes[i].set_title(f'Cluster {cluster_id}\n({len(cluster_indices)} samples)')
                    axes[i].set_xlabel('Time (s)')
                    
                    if i == 0:
                        axes[i].set_ylabel('Frequency (Hz)')
        
        # Add colorbar to the last subplot
        cbar = plt.colorbar(im, ax=axes[-1])
        cbar.set_label('Power (dB)')
        
        plt.suptitle('Cluster Comparison - One Representative Sample per Cluster', fontsize=14)
        plt.tight_layout()
        
        comparison_path = self.output_dir / 'cluster_comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster comparison plot saved to: {comparison_path}")
    
    def create_mean_spectrograms_plot(self):
        """Create a plot showing the mean spectrogram for each cluster."""
        print("\nCreating mean spectrograms plot...")
        
        n_clusters = len(self.unique_clusters)
        fig, axes = plt.subplots(2, n_clusters, figsize=(4*n_clusters, 8))
        
        if n_clusters == 1:
            axes = axes.reshape(2, 1)
        
        for i, cluster_id in enumerate(self.unique_clusters):
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Calculate mean and std spectrogram
                cluster_specs = self.spectrograms[cluster_indices]
                mean_spec = np.mean(cluster_specs, axis=0)
                std_spec = np.std(cluster_specs, axis=0)
                
                # Plot mean spectrogram
                im1 = axes[0, i].imshow(mean_spec, aspect='auto', origin='lower',
                                       extent=[self.times[0], self.times[-1],
                                              self.frequencies[0], self.frequencies[-1]],
                                       cmap='viridis', vmin=self.global_vmin, vmax=self.global_vmax)
                
                axes[0, i].set_title(f'Cluster {cluster_id} Mean\n({len(cluster_indices)} samples)')
                axes[0, i].set_xlabel('Time (s)')
                
                # Plot std spectrogram
                im2 = axes[1, i].imshow(std_spec, aspect='auto', origin='lower',
                                       extent=[self.times[0], self.times[-1],
                                              self.frequencies[0], self.frequencies[-1]],
                                       cmap='plasma')
                
                axes[1, i].set_title(f'Cluster {cluster_id} Std Dev')
                axes[1, i].set_xlabel('Time (s)')
                
                if i == 0:
                    axes[0, i].set_ylabel('Frequency (Hz)')
                    axes[1, i].set_ylabel('Frequency (Hz)')
        
        plt.suptitle('Mean and Standard Deviation Spectrograms per Cluster', fontsize=14)
        plt.tight_layout()
        
        mean_path = self.output_dir / 'cluster_mean_spectrograms.png'
        plt.savefig(mean_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Mean spectrograms plot saved to: {mean_path}")
    
    def plot_silhouette_analysis(self, features_path=None):
        """
        Create silhouette plot to visualize cluster quality.
        
        Args:
            features_path: Path to bottleneck features (if available)
        """
        if features_path and Path(features_path).exists():
            print("\nCreating silhouette analysis plot...")
            
            features = np.load(features_path)
            silhouette_vals = silhouette_samples(features, self.cluster_labels)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            y_lower = 10
            for cluster_id in self.unique_clusters:
                cluster_silhouette_vals = silhouette_vals[self.cluster_labels == cluster_id]
                cluster_silhouette_vals.sort()
                
                size_cluster = cluster_silhouette_vals.shape[0]
                y_upper = y_lower + size_cluster
                
                color = plt.cm.nipy_spectral(float(cluster_id) / self.n_clusters)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                0, cluster_silhouette_vals,
                                facecolor=color, edgecolor=color, alpha=0.7)
                
                ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))
                y_lower = y_upper + 10
            
            ax.set_xlabel("Silhouette Coefficient Values")
            ax.set_ylabel("Cluster Label")
            ax.set_title("Silhouette Plot for Each Cluster")
            
            # Add average silhouette score line
            avg_score = np.mean(silhouette_vals)
            ax.axvline(x=avg_score, color="red", linestyle="--", label=f'Average: {avg_score:.3f}')
            ax.legend()
            
            plt.tight_layout()
            silhouette_path = self.output_dir / 'silhouette_analysis.png'
            plt.savefig(silhouette_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Silhouette analysis saved to: {silhouette_path}")
    
    def create_playlists(self, max_files_per_playlist=50):
        """
        Create .m3u playlist files for each cluster.
        
        Args:
            max_files_per_playlist: Maximum number of files to include in each playlist
        """
        print("\nCreating playlist files...")
        
        for cluster_id in self.unique_clusters:
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            # Create playlist
            playlist_path = self.output_dir / f'cluster_{cluster_id}_samples.m3u'
            
            with open(playlist_path, 'w') as f:
                f.write('#EXTM3U\n')
                
                count = 0
                for idx in cluster_indices:
                    if idx < len(self.audio_files) and count < max_files_per_playlist:
                        audio_file = self.audio_files[idx]
                        f.write(f'#EXTINF:-1,{audio_file.stem}\n')
                        f.write(f'{audio_file.resolve()}\n')
                        count += 1
            
            print(f"Created playlist for Cluster {cluster_id}: {count} files")

def main():
    parser = argparse.ArgumentParser(description='Evaluate clustering results for underwater bioacoustics')
    
    parser.add_argument('--cluster_labels', type=str, required=True,
                       help='Path to cluster labels .npy file')
    parser.add_argument('--extracted_segments', type=str, 
                       default='extracted_segments',
                       help='Directory containing extracted audio segments')
    parser.add_argument('--spectrograms', type=str, 
                       default='spectrograms_optimized',
                       help='Directory containing spectrogram arrays')
    parser.add_argument('--output_dir', type=str, 
                       default='cluster_evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--copy_files', action='store_true',
                       help='Copy files instead of creating symbolic links')
    parser.add_argument('--samples_per_cluster', type=int, default=5,
                       help='Number of sample spectrograms to show per cluster')
    parser.add_argument('--features', type=str, default='models/bottleneck_features.npy',
                       help='Path to bottleneck features for silhouette analysis')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CLUSTER EVALUATION FOR UNDERWATER BIOACOUSTICS")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ClusterEvaluator(
        cluster_labels_path=args.cluster_labels,
        extracted_segments_dir=args.extracted_segments,
        spectrograms_dir=args.spectrograms,
        output_dir=args.output_dir
    )
    
    # Organize files by cluster
    stats_df = evaluator.organize_by_cluster(copy_files=args.copy_files)
    print(f"\nStatistics saved to: {evaluator.output_dir / 'cluster_statistics.csv'}")
    
    # Calculate acoustic characteristics
    evaluator.calculate_cluster_characteristics()
    
    # Create visualizations
    evaluator.create_cluster_summary_spectrograms(samples_per_cluster=args.samples_per_cluster)
    evaluator.create_cluster_comparison_plot()
    evaluator.create_mean_spectrograms_plot()
    
    # Create silhouette analysis if features available
    evaluator.plot_silhouette_analysis(args.features)
    
    # Create playlists
    evaluator.create_playlists()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print(f"Results organized in: {evaluator.output_dir}")
    print(f"Found {evaluator.n_clusters} clusters")
    print("\nNext steps:")
    print("1. Open cluster_comparison.png to see overview of all clusters")
    print("2. Check cluster_mean_spectrograms.png for average patterns")
    print("3. Review cluster_acoustic_characteristics.csv for acoustic properties")
    print("4. Navigate to each cluster_X folder to examine samples")
    print("5. Use the .m3u playlist files to listen to audio samples")
    print("6. Check silhouette_analysis.png for cluster quality visualization")
    print("=" * 60)

if __name__ == "__main__":
    main()