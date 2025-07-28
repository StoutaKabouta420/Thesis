import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
SPECTROGRAM_DIR = BASE_DIR / "spectrograms_optimized"  # Updated path
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BOTTLENECK_DIM = 32  # Dimension of the bottleneck layer

class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape, bottleneck_dim=32):
        super(ConvAutoencoder, self).__init__()
        
        self.input_shape = input_shape  # (75, 47) for new spectrograms
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (1, 75, 47)
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (16, 37, 23)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 18, 11)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 9, 5)
        )
        
        # Calculate the flattened size after encoder
        # After 3 pooling layers: 75->37->18->9, 47->23->11->5
        self.encoder_output_size = 64 * 9 * 5  # 2880
        
        # Bottleneck
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.encoder_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim)
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.encoder_output_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Reshape will be done in forward pass
            # Input: (64, 9, 5)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # (32, 18, 10) -> need to adjust
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),  # (16, 37, 20) -> need to adjust
            
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            # Output: (1, 75, 40) -> need final adjustment
        )
        
        # Final adjustment layer to match exact input dimensions
        self.final_adjust = nn.Conv2d(1, 1, kernel_size=(1, 8), stride=1, padding=(0, 0))
        
    def encode(self, x):
        # Pass through convolutional layers
        x = self.encoder(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Pass through bottleneck
        x = self.encoder_fc(x)
        return x
    
    def decode(self, x):
        # Pass through decoder FC layers
        x = self.decoder_fc(x)
        # Reshape to match encoder output
        x = x.view(x.size(0), 64, 9, 5)
        # Pass through deconvolutional layers
        x = self.decoder(x)
        # Final adjustment to match input size exactly
        if x.shape[2:] != (self.input_shape[0], self.input_shape[1]):
            # Crop or pad to match exact dimensions
            target_h, target_w = self.input_shape
            current_h, current_w = x.shape[2], x.shape[3]
            
            # Height adjustment
            if current_h > target_h:
                x = x[:, :, :target_h, :]
            elif current_h < target_h:
                pad_h = target_h - current_h
                x = F.pad(x, (0, 0, 0, pad_h))
            
            # Width adjustment
            if current_w > target_w:
                x = x[:, :, :, :target_w]
            elif current_w < target_w:
                pad_w = target_w - current_w
                x = F.pad(x, (0, pad_w, 0, 0))
        
        return x
    
    def forward(self, x):
        bottleneck = self.encode(x)
        reconstructed = self.decode(bottleneck)
        return reconstructed, bottleneck

def load_data():
    """Load and preprocess the spectrogram data."""
    print("Loading data...")
    
    # Load the enhanced spectrograms
    spectrograms = np.load(SPECTROGRAM_DIR / 'spectrograms_enhanced.npy')
    labels = np.load(SPECTROGRAM_DIR / 'labels.npy')
    
    # Print data info
    print(f"Loaded {len(spectrograms)} spectrograms")
    print(f"Spectrogram shape: {spectrograms[0].shape}")
    
    # Normalize spectrograms to [0, 1]
    scaler = MinMaxScaler()
    n_samples, n_freq, n_time = spectrograms.shape
    
    # Reshape for scaling
    spectrograms_flat = spectrograms.reshape(n_samples, -1)
    spectrograms_normalized = scaler.fit_transform(spectrograms_flat)
    spectrograms_normalized = spectrograms_normalized.reshape(n_samples, n_freq, n_time)
    
    # Add channel dimension for Conv2D
    spectrograms_normalized = spectrograms_normalized[:, np.newaxis, :, :]
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        spectrograms_normalized, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, MODEL_DIR / 'spectrogram_scaler.pkl')
    
    return train_loader, val_loader, spectrograms[0].shape

def train_autoencoder(model, train_loader, val_loader, num_epochs):
    """Train the autoencoder."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    print(f"\nTraining autoencoder for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, data)
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

def visualize_reconstruction(model, val_loader):
    """Visualize original vs reconstructed spectrograms."""
    model.eval()
    
    # Get a batch of validation data
    data_iter = iter(val_loader)
    data, _ = next(data_iter)
    data = data.to(device)
    
    with torch.no_grad():
        reconstructed, _ = model(data)
    
    # Move to CPU for plotting
    data = data.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Plot first 4 examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Original
        axes[i, 0].imshow(data[i, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Frequency')
        
        # Reconstructed
        axes[i, 1].imshow(reconstructed[i, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed {i+1}')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'reconstruction_comparison.png', dpi=150)
    plt.close()
    
    print(f"Reconstruction comparison saved to {MODEL_DIR / 'reconstruction_comparison.png'}")

def extract_bottleneck_features(model, spectrograms, batch_size=32):
    """Extract bottleneck features for all spectrograms."""
    model.eval()
    
    # Ensure spectrograms have the right shape
    if len(spectrograms.shape) == 3:
        spectrograms = spectrograms[:, np.newaxis, :, :]
    
    n_samples = len(spectrograms)
    bottleneck_features = []
    
    print("\nExtracting bottleneck features...")
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = spectrograms[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            _, bottleneck = model(batch_tensor)
            bottleneck_features.append(bottleneck.cpu().numpy())
    
    bottleneck_features = np.vstack(bottleneck_features)
    
    print(f"Bottleneck features shape: {bottleneck_features.shape}")
    
    return bottleneck_features

if __name__ == "__main__":
    # Load data
    train_loader, val_loader, input_shape = load_data()
    
    # Create model
    model = ConvAutoencoder(input_shape=input_shape, bottleneck_dim=BOTTLENECK_DIM).to(device)
    print(f"\nModel architecture:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Train model
    train_losses, val_losses = train_autoencoder(model, train_loader, val_loader, NUM_EPOCHS)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR / 'training_history.png', dpi=150)
    plt.close()
    
    print(f"\nTraining history plot saved to {MODEL_DIR / 'training_history.png'}")
    
    # Visualize reconstructions
    visualize_reconstruction(model, val_loader)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f'autoencoder_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_shape': input_shape,
        'bottleneck_dim': BOTTLENECK_DIM,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    
    # Extract and save bottleneck features for all data
    # Load all spectrograms (normalized)
    spectrograms = np.load(SPECTROGRAM_DIR / 'spectrograms_enhanced.npy')
    import joblib
    scaler = joblib.load(MODEL_DIR / 'spectrogram_scaler.pkl')
    
    # Normalize
    n_samples, n_freq, n_time = spectrograms.shape
    spectrograms_flat = spectrograms.reshape(n_samples, -1)
    spectrograms_normalized = scaler.transform(spectrograms_flat)
    spectrograms_normalized = spectrograms_normalized.reshape(n_samples, n_freq, n_time)
    
    # Extract features
    bottleneck_features = extract_bottleneck_features(model, spectrograms_normalized)
    
    # Save bottleneck features
    np.save(MODEL_DIR / 'bottleneck_features.npy', bottleneck_features)
    
    # Save metadata
    metadata = {
        'model_path': str(model_path),
        'input_shape': input_shape,
        'bottleneck_dim': BOTTLENECK_DIM,
        'n_samples': len(bottleneck_features),
        'timestamp': timestamp,
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1])
    }
    
    with open(MODEL_DIR / 'autoencoder_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nBottleneck features saved to {MODEL_DIR / 'bottleneck_features.npy'}")
    print(f"Metadata saved to {MODEL_DIR / 'autoencoder_metadata.json'}")
    
    print("\n✓ Autoencoder training complete!")