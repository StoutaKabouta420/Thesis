import numpy as np
import os
from pathlib import Path
from scipy.ndimage import shift, zoom
import json
from tqdm import tqdm

# Set up paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
SPECTROGRAM_DIR = BASE_DIR / "spectrograms_focused"
AUGMENTED_DIR = BASE_DIR / "spectrograms_augmented"
AUGMENTED_DIR.mkdir(exist_ok=True)

def load_spectrograms():
    """Load original spectrograms and metadata."""
    print("Loading original spectrograms from spectrograms_focused...")
    
    spec_file = SPECTROGRAM_DIR / 'spectrograms_enhanced.npy'
    if not spec_file.exists():
        spec_file = SPECTROGRAM_DIR / 'spectrograms_original.npy'
        print("Note: Using original spectrograms (enhanced not found)")
    
    spectrograms = np.load(spec_file)
    labels = np.load(SPECTROGRAM_DIR / 'labels.npy')
    
    with open(SPECTROGRAM_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(spectrograms)} original spectrograms")
    print(f"Shape: {spectrograms[0].shape}")
    print(f"Value range: [{spectrograms.min():.2f}, {spectrograms.max():.2f}]")
    
    return spectrograms, labels, metadata

# -----------------
# Augmentation funcs
# -----------------

def time_shift(spec, max_shift=2):
    """Small time shift with reflect padding to avoid hard bars."""
    shift_amt = np.random.randint(-max_shift, max_shift + 1)
    return shift(spec, shift=(0, shift_amt), mode="reflect", order=1)

def frequency_shift(spec, max_shift=1):
    """Tiny frequency shift with reflect padding."""
    shift_amt = np.random.randint(-max_shift, max_shift + 1)
    return shift(spec, shift=(shift_amt, 0), mode="reflect", order=1)

def add_noise(spec, snr_db=25):
    """Add Gaussian noise at a target SNR in dB."""
    p = np.mean(spec**2)
    sigma = np.sqrt(p / (10**(snr_db/10)))
    return spec + np.random.normal(0, sigma, spec.shape)

def amplitude_variation(spec, min_db=-3, max_db=3):
    """Scale amplitude in dB (safe for log-mel or dB spectrograms)."""
    gain_db = np.random.uniform(min_db, max_db)
    return spec + gain_db

def time_stretch(spec, stretch_range=(0.97, 1.03)):
    """Slight time stretching with reflect padding."""
    factor = np.random.uniform(*stretch_range)
    stretched = zoom(spec, (1.0, factor), order=1)
    T = spec.shape[1]
    if stretched.shape[1] > T:
        start = (stretched.shape[1] - T) // 2
        return stretched[:, start:start+T]
    elif stretched.shape[1] < T:
        pad_total = T - stretched.shape[1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(stretched, ((0, 0), (pad_left, pad_right)), mode="reflect")
    return stretched

def renorm(spec):
    """Per-spectrogram z-score normalization with clipping."""
    m, s = spec.mean(), spec.std() + 1e-6
    x = (spec - m) / s
    return np.clip(x, -3, 3)

# -----------------
# Augmentation loop
# -----------------

def augment_spectrograms(spectrograms, labels, target_count=3000):
    n_original = len(spectrograms)
    n_aug_needed = target_count - n_original
    augs_per_sample = int(np.ceil(n_aug_needed / n_original))
    
    print(f"\nAugmentation plan:")
    print(f"  Original samples: {n_original}")
    print(f"  Target count: {target_count}")
    print(f"  Augmentations per sample: ~{augs_per_sample}")
    
    augmented_specs = []
    augmented_labels = []
    augmented_info = []
    
    augmented_specs.extend(spectrograms)
    augmented_labels.extend(labels)
    augmented_info.extend([{'type': 'original', 'source_idx': i} for i in range(n_original)])
    
    augmentation_funcs = [
        ('time_shift', lambda x: time_shift(x, 2), 0.5),
        ('time_stretch', lambda x: time_stretch(x, (0.97, 1.03)), 0.4),
        ('noise', lambda x: add_noise(x, snr_db=25), 0.5),
        ('frequency_shift', lambda x: frequency_shift(x, 1), 0.2),
        ('amplitude', lambda x: amplitude_variation(x, -3, 3), 0.3),
    ]
    
    print("\nGenerating augmented spectrograms...")
    
    with tqdm(total=n_aug_needed) as pbar:
        for aug_round in range(augs_per_sample + 1):
            for idx, (spec, label) in enumerate(zip(spectrograms, labels)):
                if len(augmented_specs) >= target_count:
                    break
                if aug_round > 0 and np.random.random() > 0.9:
                    continue
                aug_spec = spec.copy()
                applied_augs = []
                for aug_name, aug_func, prob in augmentation_funcs:
                    if np.random.random() < prob:
                        aug_spec = aug_func(aug_spec)
                        applied_augs.append(aug_name)
                if applied_augs and aug_round > 0:
                    aug_spec = renorm(aug_spec)
                    augmented_specs.append(aug_spec)
                    augmented_labels.append(label)
                    augmented_info.append({
                        'type': 'augmented',
                        'source_idx': idx,
                        'augmentations': applied_augs,
                        'round': aug_round
                    })
                    pbar.update(1)
                if len(augmented_specs) >= target_count:
                    break
    
    augmented_specs = np.array(augmented_specs[:target_count])
    augmented_labels = np.array(augmented_labels[:target_count])
    
    print(f"\nGenerated {len(augmented_specs)} total spectrograms")
    print(f"Final shape: {augmented_specs.shape}")
    
    return augmented_specs, augmented_labels, augmented_info

# -----------------
# Save & visualize
# -----------------

def save_augmented_data(specs, labels, info, metadata):
    print("\nSaving augmented dataset...")
    np.save(AUGMENTED_DIR / 'spectrograms_augmented.npy', specs)
    np.save(AUGMENTED_DIR / 'labels_augmented.npy', labels)
    augmented_metadata = metadata.copy()
    augmented_metadata['augmentation_info'] = {
        'total_samples': len(specs),
        'original_samples': sum(1 for i in info if i['type'] == 'original'),
        'augmented_samples': sum(1 for i in info if i['type'] == 'augmented'),
        'shape': specs[0].shape,
        'augmentation_methods': [f[0] for f in [
            ('time_shift', None), ('time_stretch', None),
            ('noise', None), ('frequency_shift', None),
            ('amplitude', None)
        ]]
    }
    with open(AUGMENTED_DIR / 'metadata_augmented.json', 'w') as f:
        json.dump(augmented_metadata, f, indent=2, default=str)
    with open(AUGMENTED_DIR / 'augmentation_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved to {AUGMENTED_DIR}")

def visualize_augmentations(original_specs, augmented_specs, augmented_info):
    import matplotlib.pyplot as plt
    print("\nCreating augmentation examples...")
    examples_dir = AUGMENTED_DIR / "augmentation_examples"
    examples_dir.mkdir(exist_ok=True)
    CMAP = 'hot'
    for orig_idx in range(min(3, len(original_specs))):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes[1, 1].imshow(original_specs[orig_idx], aspect='auto', origin='lower',
                          cmap=CMAP, interpolation='bilinear')
        axes[1, 1].set_title('Original', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time bins (64)')
        axes[1, 1].set_ylabel('Frequency bins (129)')
        aug_versions = [(i, info) for i, info in enumerate(augmented_info) 
                        if info.get('source_idx') == orig_idx and info['type'] == 'augmented']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
        for (aug_idx, aug_info), (row, col) in zip(aug_versions[:8], positions):
            axes[row, col].imshow(augmented_specs[aug_idx], aspect='auto', origin='lower',
                                  cmap=CMAP, interpolation='bilinear')
            aug_types = ', '.join(aug_info['augmentations'][:2])
            if len(aug_info['augmentations']) > 2:
                aug_types += f" (+{len(aug_info['augmentations'])-2})"
            axes[row, col].set_title(aug_types, fontsize=10)
        for idx in range(len(aug_versions), 8):
            if idx < len(positions):
                row, col = positions[idx]
                axes[row, col].axis('off')
        plt.suptitle(f'Augmentation Examples - Spectrogram {orig_idx} (129×64 shape)', fontsize=14)
        plt.tight_layout()
        plt.savefig(examples_dir / f'augmentation_example_{orig_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f"Saved examples to {examples_dir}")

# -----------------
# Main
# -----------------

def main():
    print("=" * 60)
    print("Spectrogram Data Augmentation (clean version)")
    print("=" * 60)
    specs, labels, metadata = load_spectrograms()
    if specs[0].shape != (129, 64):
        print(f"WARNING: Expected shape (129, 64), got {specs[0].shape}")
    augmented_specs, augmented_labels, augmented_info = augment_spectrograms(
        specs, labels, target_count=3000
    )
    save_augmented_data(augmented_specs, augmented_labels, augmented_info, metadata)
    visualize_augmentations(specs, augmented_specs, augmented_info)
    print("\n✓ Augmentation complete!")
    print(f"Ready for AE training with {len(augmented_specs)} samples")

if __name__ == "__main__":
    main()
