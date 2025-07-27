#!/usr/bin/env python3
"""
Cluster Evaluation Tool for Underwater Bioacoustics
Organizes audio segments and spectrograms by cluster for manual evaluation.
"""

import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import pandas as pd
import argparse
from datetime import datetime
import random

class ClusterEvaluator:
    """
    Tool for evaluating and organizing clustered bioacoustic data.
    """
    
    def __init__(self, cluster_labels_path, extracted_segments_dir, 
                 spectrograms_dir, output_dir):
        """
        Initialize the cluster evaluator.
        
        Args:
            cluster_labels_path (str): Path to cluster labels .npy file
            extracted_segments_dir (str): Directory containing audio segments
            spectrograms_dir (str): Directory containing spectrograms
            output_dir (str): Output directory for organized results
        """
        self.cluster_labels = np.load(cluster_labels_path)
        self.extracted_segments_dir = Path(extracted_segments_dir)
        self.spectrograms_dir = Path(spectrograms_dir)
        self.output_dir = Path(output_dir)
        self.n_clusters = len(np.unique(self.cluster_labels))
        
        # Create output directory structure
        self.setup_output_directories()
        
        # Get file lists
        self.audio_files = self.get_audio_files()
        self.spectrogram_files = self.get_spectrogram_files()
        
        print(f"Loaded {len(self.cluster_labels)} cluster labels")
        print(f"Found {len(self.audio_files)} audio files")
        print(f"Found {len(self.spectrogram_files)} spectrogram files")
        print(f"Number of clusters: {self.n_clusters}")
    
    def setup_output_directories(self):
        """Create organized directory structure for clusters."""
        self.output_dir.mkdir(exist_ok=True)
        
        # Create cluster directories
        for cluster_id in range(self.n_clusters):
            cluster_dir = self.output_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            (cluster_dir / "audio").mkdir(exist_ok=True)
            (cluster_dir / "spectrograms").mkdir(exist_ok=True)
            (cluster_dir / "sample_spectrograms").mkdir(exist_ok=True)
    
    def get_audio_files(self):
        """Get list of audio files in extracted_segments directory."""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(self.extracted_segments_dir.glob(f"**/*{ext}")))
        
        return sorted(audio_files)
    
    def get_spectrogram_files(self):
        """Get list of spectrogram files."""
        spectrogram_files = list(self.spectrograms_dir.glob("**/*.png"))
        return sorted(spectrogram_files)
    
    def organize_files_by_cluster(self, copy_files=True):
        """
        Organize audio files and spectrograms by cluster.
        
        Args:
            copy_files (bool): Whether to copy files (True) or create symlinks (False)
        """
        print("Organizing files by cluster...")
        
        cluster_stats = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            cluster_stats[cluster_id] = {
                'count': len(cluster_indices),
                'audio_files': [],
                'spectrogram_files': []
            }
            
            cluster_dir = self.output_dir / f"cluster_{cluster_id}"
            
            print(f"Processing Cluster {cluster_id}: {len(cluster_indices)} samples")
            
            # Organize audio files
            for idx in cluster_indices:
                if idx < len(self.audio_files):
                    src_audio = self.audio_files[idx]
                    dst_audio = cluster_dir / "audio" / src_audio.name
                    
                    if copy_files:
                        shutil.copy2(src_audio, dst_audio)
                    else:
                        if not dst_audio.exists():
                            dst_audio.symlink_to(src_audio.absolute())
                    
                    cluster_stats[cluster_id]['audio_files'].append(src_audio.name)
                
                # Organize spectrogram files
                if idx < len(self.spectrogram_files):
                    src_spec = self.spectrogram_files[idx]
                    dst_spec = cluster_dir / "spectrograms" / src_spec.name
                    
                    if copy_files:
                        shutil.copy2(src_spec, dst_spec)
                    else:
                        if not dst_spec.exists():
                            dst_spec.symlink_to(src_spec.absolute())
                    
                    cluster_stats[cluster_id]['spectrogram_files'].append(src_spec.name)
        
        # Save cluster statistics
        self.save_cluster_statistics(cluster_stats)
        return cluster_stats
    
    def create_cluster_summary_spectrograms(self, n_samples_per_cluster=5):
        """
        Create summary spectrograms showing representative samples from each cluster.
        
        Args:
            n_samples_per_cluster (int): Number of sample spectrograms per cluster
        """
        print("Creating cluster summary spectrograms...")
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            # Randomly sample indices for this cluster
            n_samples = min(n_samples_per_cluster, len(cluster_indices))
            sampled_indices = random.sample(list(cluster_indices), n_samples)
            
            # Create subplot grid
            fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
            if n_samples == 1:
                axes = [axes]
            
            fig.suptitle(f'Cluster {cluster_id} - Representative Samples ({len(cluster_indices)} total)', 
                        fontsize=16, fontweight='bold')
            
            for i, idx in enumerate(sampled_indices):
                if idx < len(self.audio_files):
                    audio_file = self.audio_files[idx]
                    
                    try:
                        # Load audio and create spectrogram
                        y, sr = librosa.load(audio_file, sr=None)
                        
                        # Create mel spectrogram
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                        S_dB = librosa.power_to_db(S, ref=np.max)
                        
                        # Plot spectrogram
                        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                                               ax=axes[i], cmap='viridis')
                        axes[i].set_title(f'Sample {i+1}\n{audio_file.name}', fontsize=10)
                        axes[i].set_xlabel('Time (s)')
                        if i == 0:
                            axes[i].set_ylabel('Mel Frequency')
                        
                    except Exception as e:
                        axes[i].text(0.5, 0.5, f'Error loading\n{audio_file.name}\n{str(e)}', 
                                   ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f'Sample {i+1} - Error')
            
            plt.tight_layout()
            
            # Save summary plot
            summary_path = self.output_dir / f"cluster_{cluster_id}" / "sample_spectrograms" / "cluster_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_cluster_comparison_plot(self):
        """Create a comparison plot showing one sample from each cluster."""
        print("Creating cluster comparison plot...")
        
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(4*self.n_clusters, 4))
        if self.n_clusters == 1:
            axes = [axes]
        
        fig.suptitle('Cluster Comparison - One Representative Sample per Cluster', 
                    fontsize=16, fontweight='bold')
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Take first sample from cluster
                sample_idx = cluster_indices[0]
                
                if sample_idx < len(self.audio_files):
                    audio_file = self.audio_files[sample_idx]
                    
                    try:
                        # Load audio and create spectrogram
                        y, sr = librosa.load(audio_file, sr=None)
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                        S_dB = librosa.power_to_db(S, ref=np.max)
                        
                        # Plot spectrogram
                        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                                               ax=axes[cluster_id], cmap='viridis')
                        axes[cluster_id].set_title(f'Cluster {cluster_id}\n({len(cluster_indices)} samples)', 
                                                  fontsize=12, fontweight='bold')
                        axes[cluster_id].set_xlabel('Time (s)')
                        if cluster_id == 0:
                            axes[cluster_id].set_ylabel('Mel Frequency')
                        
                    except Exception as e:
                        axes[cluster_id].text(0.5, 0.5, f'Error loading\nCluster {cluster_id}', 
                                            ha='center', va='center', transform=axes[cluster_id].transAxes)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = self.output_dir / "cluster_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster comparison plot saved to: {comparison_path}")
    
    def save_cluster_statistics(self, cluster_stats):
        """Save cluster statistics to CSV file."""
        stats_data = []
        
        for cluster_id, stats in cluster_stats.items():
            stats_data.append({
                'cluster_id': cluster_id,
                'sample_count': stats['count'],
                'percentage': (stats['count'] / len(self.cluster_labels)) * 100,
                'audio_files_count': len(stats['audio_files']),
                'spectrogram_files_count': len(stats['spectrogram_files'])
            })
        
        df = pd.DataFrame(stats_data)
        stats_path = self.output_dir / "cluster_statistics.csv"
        df.to_csv(stats_path, index=False)
        
        print(f"\nCluster Statistics:")
        print(df.to_string(index=False))
        print(f"\nStatistics saved to: {stats_path}")
    
    def create_playlist_files(self):
        """Create M3U playlist files for each cluster for easy audio playback."""
        print("Creating playlist files...")
        
        for cluster_id in range(self.n_clusters):
            cluster_dir = self.output_dir / f"cluster_{cluster_id}"
            audio_dir = cluster_dir / "audio"
            
            # Get all audio files in this cluster
            audio_files = list(audio_dir.glob("*.*"))
            
            if audio_files:
                playlist_path = cluster_dir / f"cluster_{cluster_id}_playlist.m3u"
                
                with open(playlist_path, 'w') as f:
                    f.write("#EXTM3U\n")
                    for audio_file in sorted(audio_files):
                        f.write(f"audio/{audio_file.name}\n")
                
                print(f"Created playlist for Cluster {cluster_id}: {len(audio_files)} files")

def main():
    parser = argparse.ArgumentParser(description='Evaluate and organize clustered bioacoustic data')
    parser.add_argument('--cluster_labels', type=str, required=True,
                       help='Path to cluster labels .npy file')
    parser.add_argument('--extracted_segments', type=str, default='extracted_segments',
                       help='Directory containing extracted audio segments')
    parser.add_argument('--spectrograms', type=str, default='spectrograms',
                       help='Directory containing spectrograms')
    parser.add_argument('--output_dir', type=str, default='cluster_evaluation',
                       help='Output directory for organized results')
    parser.add_argument('--copy_files', action='store_true',
                       help='Copy files instead of creating symlinks')
    parser.add_argument('--samples_per_cluster', type=int, default=5,
                       help='Number of sample spectrograms per cluster')
    
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
    cluster_stats = evaluator.organize_files_by_cluster(copy_files=args.copy_files)
    
    # Create summary visualizations
    evaluator.create_cluster_summary_spectrograms(n_samples_per_cluster=args.samples_per_cluster)
    evaluator.create_cluster_comparison_plot()
    
    # Create playlist files for easy listening
    evaluator.create_playlist_files()
    
    print(f"\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print(f"Results organized in: {args.output_dir}")
    print(f"Found {evaluator.n_clusters} clusters")
    print("\nNext steps:")
    print("1. Open cluster_comparison.png to see overview of all clusters")
    print("2. Navigate to each cluster_X folder to examine samples")
    print("3. Use the .m3u playlist files to listen to audio samples")
    print("4. Check cluster_statistics.csv for detailed statistics")
    print("="*60)

if __name__ == "__main__":
    main()