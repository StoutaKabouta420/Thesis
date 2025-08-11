import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import pickle

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
AUGMENTED_DIR = BASE_DIR / "spectrograms_augmented"
ORIGINAL_DIR = BASE_DIR / "spectrograms_focused"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for 129x64 spectrograms.
    Designed for Bryde's whale call spectrograms.
    """
    def __init__(self, bottleneck_dim=32):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (1, 129, 64)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 64, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 32, 16)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (128, 16, 8)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (256, 8, 4)
        )
        
        # Calculate flattened size
        self.flattened_size = 256 * 8 * 4  # 8192
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, bottleneck_dim),  # Compress to bottleneck
            nn.ReLU(),
            nn.Linear(bottleneck_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.flattened_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # (256, 8, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # (128, 16, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # (64, 32, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (32, 64, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # (1, 128, 64)
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),  # Adjust to (1, 129, 64)
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self.bottleneck_dim = bottleneck_dim
    
    def encode(self, x):
        """Encode input to bottleneck representation."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # Return the middle layer (bottleneck features)
        for i, layer in enumerate(self.bottleneck):
            x = layer(x)
            if i == 3:  # After the bottleneck linear layer
                bottleneck_features = x.clone()
        return bottleneck_features
    
    def forward(self, x):
        """Full forward pass through autoencoder."""
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        # Bottleneck
        bottleneck_input = x.clone()
        x = self.bottleneck(x)
        
        # Reshape for decoder
        x = x.view(x.size(0), 256, 8, 4)
        
        # Decode
        x = self.decoder(x)
        
        # Ensure output matches input size exactly
        if x.shape[-2:] != (129, 64):
            x = nn.functional.interpolate(x, size=(129, 64), mode='bilinear', align_corners=False)
        
        return x

def load_data():
    """Load augmented training data and original data for evaluation."""
    print("\nLoading data...")
    
    # Load augmented data for training
    augmented_specs = np.load(AUGMENTED_DIR / 'spectrograms_augmented.npy')
    augmented_labels = np.load(AUGMENTED_DIR / 'labels_augmented.npy')
    
    # Load original data for feature extraction
    original_specs = np.load(ORIGINAL_DIR / 'spectrograms_enhanced.npy')
    original_labels = np.load(ORIGINAL_DIR / 'labels.npy')
    
    print(f"Augmented data shape: {augmented_specs.shape}")
    print(f"Original data shape: {original_specs.shape}")
    
    # Normalize to [0, 1] if not already
    if augmented_specs.min() < 0:
        augmented_specs = (augmented_specs - augmented_specs.min()) / (augmented_specs.max() - augmented_specs.min())
    if original_specs.min() < 0:
        original_specs = (original_specs - original_specs.min()) / (original_specs.max() - original_specs.min())
    
    return augmented_specs, augmented_labels, original_specs, original_labels

def create_dataloaders(augmented_specs, batch_size=32, val_split=0.2):
    """Create training and validation dataloaders."""
    n_samples = len(augmented_specs)
    n_val = int(n_samples * val_split)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Split data
    train_specs = augmented_specs[train_indices]
    val_specs = augmented_specs[val_indices]
    
    # Add channel dimension and convert to tensors
    train_specs = torch.FloatTensor(train_specs).unsqueeze(1)  # (N, 1, 129, 64)
    val_specs = torch.FloatTensor(val_specs).unsqueeze(1)
    
    # Create datasets
    train_dataset = TensorDataset(train_specs, train_specs)  # Input and target are the same
    val_dataset = TensorDataset(val_specs, val_specs)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_specs)}")
    print(f"Validation samples: {len(val_specs)}")
    
    return train_loader, val_loader

def train_autoencoder(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """Train the autoencoder."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / 'best_autoencoder.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / 'best_autoencoder.pth'))
    
    return model, train_losses, val_losses

def extract_bottleneck_features(model, original_specs):
    """Extract bottleneck features from original (non-augmented) data."""
    print("\nExtracting bottleneck features from original data...")
    
    model.eval()
    features = []
    
    # Add channel dimension
    specs_tensor = torch.FloatTensor(original_specs).unsqueeze(1)
    
    # Create dataloader for batch processing
    dataset = TensorDataset(specs_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            data = batch[0].to(device)
            bottleneck = model.encode(data)
            features.append(bottleneck.cpu().numpy())
    
    features = np.vstack(features)
    print(f"Extracted features shape: {features.shape}")
    
    return features

def visualize_reconstruction(model, original_specs, n_examples=5):
    """Visualize original vs reconstructed spectrograms."""
    print("\nCreating reconstruction visualizations...")
    
    model.eval()
    
    # Select random examples
    indices = np.random.choice(len(original_specs), n_examples, replace=False)
    
    fig, axes = plt.subplots(n_examples, 2, figsize=(10, n_examples * 3))
    
    for i, idx in enumerate(indices):
        # Get original
        original = original_specs[idx]
        
        # Get reconstruction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(original).unsqueeze(0).unsqueeze(0).to(device)
            reconstruction = model(input_tensor).cpu().numpy()[0, 0]
        
        # Plot original
        axes[i, 0].imshow(original, aspect='auto', origin='lower', cmap='hot')
        axes[i, 0].set_title(f'Original {idx}')
        axes[i, 0].set_ylabel('Frequency bins')
        
        # Plot reconstruction
        axes[i, 1].imshow(reconstruction, aspect='auto', origin='lower', cmap='hot')
        axes[i, 1].set_title(f'Reconstructed {idx}')
        
        if i == n_examples - 1:
            axes[i, 0].set_xlabel('Time bins')
            axes[i, 1].set_xlabel('Time bins')
    
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'reconstruction_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reconstruction examples to {MODELS_DIR}/reconstruction_examples.png")

def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss history."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(MODELS_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history to {MODELS_DIR}/training_history.png")

def save_features_and_metadata(features, original_labels, model, train_losses, val_losses):
    """Save bottleneck features and training metadata."""
    # Save features
    np.save(MODELS_DIR / 'bottleneck_features.npy', features)
    np.save(MODELS_DIR / 'bottleneck_labels.npy', original_labels)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), MODELS_DIR / f'autoencoder_{timestamp}.pth')
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'bottleneck_dim': model.bottleneck_dim,
        'input_shape': [129, 64],
        'n_training_samples': 3000,
        'n_original_samples': len(features),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'n_epochs': len(train_losses),
        'architecture': 'ConvAutoencoder',
        'device': str(device)
    }
    
    with open(MODELS_DIR / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved model and features to {MODELS_DIR}/")
    print(f"  - bottleneck_features.npy: {features.shape}")
    print(f"  - autoencoder_{timestamp}.pth")
    print(f"  - training_metadata.json")

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Training Autoencoder for Bryde's Whale Call Spectrograms")
    print("=" * 60)
    
    # Load data
    augmented_specs, augmented_labels, original_specs, original_labels = load_data()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(augmented_specs, batch_size=32)
    
    # Initialize model
    bottleneck_dim = 64  # Compressed representation size
    model = ConvAutoencoder(bottleneck_dim=bottleneck_dim).to(device)
    print(f"\nModel initialized with bottleneck dimension: {bottleneck_dim}")
    
    # Train model
    print("\nTraining autoencoder...")
    model, train_losses, val_losses = train_autoencoder(
        model, train_loader, val_loader, 
        num_epochs=10, lr=0.001
    )
    
    # Extract features from ORIGINAL data only
    features = extract_bottleneck_features(model, original_specs)
    
    # Visualize results
    visualize_reconstruction(model, original_specs)
    plot_training_history(train_losses, val_losses)
    
    # Save everything
    save_features_and_metadata(features, original_labels, model, train_losses, val_losses)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Bottleneck features extracted: {features.shape}")
    print(f"These features are ready for spectral clustering")
    print("\nNext step:")
    print("  python spectral_clustering.py --standardize")

if __name__ == "__main__":
    main()