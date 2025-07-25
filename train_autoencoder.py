import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
SPECTROGRAM_DIR = BASE_DIR / "spectrograms"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BOTTLENECK_DIM = 32  # Dimension of compressed representation
VALIDATION_SPLIT = 0.2

class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape=(65, 28), bottleneck_dim=32):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (1, 65, 28)
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (16, 32, 14)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 16, 7)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 8, 3)
        )
        
        # Calculate flattened size after convolutions
        self.flattened_size = 64 * 8 * 3  # 1536
        
        # Bottleneck
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim)
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.flattened_size),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            # (64, 8, 3)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # (32, 16, 6) -> need to adjust to (32, 16, 7)
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # (16, 32, 12) -> need to adjust to (16, 32, 14)
            
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Output: (1, 64, 24) -> need to adjust to (1, 65, 28)
        )
        
        # Adjustment layers to match exact dimensions
        self.final_adjust = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder_fc(x)
        return x
    
    def decode(self, x):
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 64, 8, 3)  # Reshape
        x = self.decoder(x)
        # Crop or pad to match original size (65, 28)
        if x.size(2) < 65:
            # Pad
            pad_h = 65 - x.size(2)
            x = nn.functional.pad(x, (0, 0, 0, pad_h))
        elif x.size(2) > 65:
            # Crop
            x = x[:, :, :65, :]
            
        if x.size(3) < 28:
            # Pad
            pad_w = 28 - x.size(3)
            x = nn.functional.pad(x, (0, pad_w))
        elif x.size(3) > 28:
            # Crop
            x = x[:, :, :, :28]
            
        return x
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded

def load_data():
    """Load and prepare the spectrogram data."""
    print("Loading data...")
    
    # Load spectrograms and labels
    spectrograms = np.load(SPECTROGRAM_DIR / 'spectrograms.npy')
    labels = np.load(SPECTROGRAM_DIR / 'labels.npy')
    
    # Normalize to [0, 1]
    spec_min = spectrograms.min()
    spec_max = spectrograms.max()
    spectrograms_norm = (spectrograms - spec_min) / (spec_max - spec_min)
    
    # Convert to PyTorch tensors
    # Add channel dimension: (N, H, W) -> (N, 1, H, W)
    spectrograms_tensor = torch.FloatTensor(spectrograms_norm).unsqueeze(1)
    
    # Create dataset
    dataset = TensorDataset(spectrograms_tensor, spectrograms_tensor)
    
    # Split into train and validation
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    return train_loader, val_loader, (spec_min, spec_max)

def train_autoencoder(model, train_loader, val_loader, num_epochs=100):
    """Train the autoencoder."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            output, _ = model(data)
            loss = criterion(output, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                output, _ = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def visualize_reconstruction(model, data_loader, spec_range, num_examples=5):
    """Visualize original vs reconstructed spectrograms."""
    model.eval()
    
    # Get a batch of data
    data, _ = next(iter(data_loader))
    data = data[:num_examples].to(device)
    
    with torch.no_grad():
        reconstructed, _ = model(data)
    
    # Move to CPU and remove channel dimension
    data = data.cpu().squeeze(1)
    reconstructed = reconstructed.cpu().squeeze(1)
    
    # Denormalize
    spec_min, spec_max = spec_range
    data = data * (spec_max - spec_min) + spec_min
    reconstructed = reconstructed * (spec_max - spec_min) + spec_min
    
    # Plot
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
    
    for i in range(num_examples):
        # Original
        axes[0, i].imshow(data[i], aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        if i == 0:
            axes[0, i].set_ylabel('Frequency bins')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i], aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        if i == 0:
            axes[1, i].set_ylabel('Frequency bins')
        axes[1, i].set_xlabel('Time bins')
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'reconstruction_examples.png', dpi=150)
    plt.close()
    
    print(f"Reconstruction examples saved to {MODEL_DIR / 'reconstruction_examples.png'}")

def extract_bottleneck_features(model, data_loader):
    """Extract bottleneck features for all data."""
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Extracting features"):
            data = data.to(device)
            _, encoded = model(data)
            all_features.append(encoded.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    return features

if __name__ == "__main__":
    # Load data
    train_loader, val_loader, spec_range = load_data()
    
    # Create model
    model = ConvAutoencoder(bottleneck_dim=BOTTLENECK_DIM).to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # Train model
    print(f"\nTraining autoencoder for {NUM_EPOCHS} epochs...")
    train_losses, val_losses = train_autoencoder(model, train_loader, val_loader, NUM_EPOCHS)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR / 'training_history.png', dpi=150)
    plt.close()
    
    # Visualize reconstructions
    visualize_reconstruction(model, val_loader, spec_range)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f'autoencoder_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'bottleneck_dim': BOTTLENECK_DIM,
        'spec_range': spec_range,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Extract bottleneck features for all data
    print("\nExtracting bottleneck features...")
    
    # Load all data for feature extraction
    spectrograms = np.load(SPECTROGRAM_DIR / 'spectrograms.npy')
    labels = np.load(SPECTROGRAM_DIR / 'labels.npy')
    
    # Normalize
    spectrograms_norm = (spectrograms - spec_range[0]) / (spec_range[1] - spec_range[0])
    spectrograms_tensor = torch.FloatTensor(spectrograms_norm).unsqueeze(1)
    
    # Create dataset for all data
    full_dataset = TensorDataset(spectrograms_tensor, spectrograms_tensor)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Extract features
    bottleneck_features = extract_bottleneck_features(model, full_loader)
    
    # Save features
    np.save(MODEL_DIR / 'bottleneck_features.npy', bottleneck_features)
    np.save(MODEL_DIR / 'labels.npy', labels)  # Save labels too for reference
    
    print(f"Bottleneck features shape: {bottleneck_features.shape}")
    print(f"Features saved to {MODEL_DIR / 'bottleneck_features.npy'}")
    
    print("\n✓ Autoencoder training complete!")
    print(f"\nNext step: Use the {BOTTLENECK_DIM}-dimensional features for spectral clustering")