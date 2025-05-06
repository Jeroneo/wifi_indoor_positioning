from torch import nn

import numpy as np

import torch
import os


class MissingValueClassifier(nn.Module):
    """Neural network classifier that handles missing values using indicator features"""
    
    def __init__(self, input_dim, output_dim=15, hidden_dims=[256, 128, 64]):
        super(MissingValueClassifier, self).__init__()
        augmented_input_dim = input_dim * 2
        layers = []
        prev_dim = augmented_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def load_model_and_preprocessors(model_path):
    """Load the trained model, imputer, and scaler from the specified path."""
    checkpoint = torch.load(model_path, weights_only=True)  # Set weights_only=True for security
    model = MissingValueClassifier(input_dim=173, output_dim=15)  # Define the model architecture
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    imputer = checkpoint.get('imputer')  # Load imputer if available
    scaler = checkpoint.get('scaler')    # Load scaler if available
    return model, imputer, scaler

def preprocess_live_data(data, imputer, scaler):
    """Preprocess the live data using the imputer and scaler."""
    data_imputed = imputer.transform(data)
    data_scaled = scaler.transform(data_imputed)
    if np.isnan(data).all():
        return None
    data_missing = np.isnan(data).astype(np.float32)
    data_combined = np.concatenate([data_scaled, data_missing], axis=1)
    return torch.tensor(data_combined, dtype=torch.float32)

def load_mac_address_order(order_path):
    """Load the order of MAC addresses from the specified path."""
    return np.load(order_path, allow_pickle=True).tolist()

def load_zone_labels(labels_path):
    """Load the zone labels from the specified path."""
    return np.load(labels_path, allow_pickle=True)

def load_mac_address_order(path):
    """Load the MAC address order used during training."""
    return np.load(path, allow_pickle=True).tolist()

def load_zone_labels(path):
    """Load the zone labels used during training."""
    return np.load(path, allow_pickle=True).tolist()

def preprocess_live_data(data):
    """Preprocess live data for inference."""
    # Replace NaNs with a constant (-95)
    return np.nan_to_num(data, nan=-95)

def get_pth_file_name(directory):
    """Retrieve the .pth file name from the specified directory."""
    try:
        files = os.listdir(directory)
        for file in files:
            if file.endswith('.pth'):
                print(f"Found .pth file: {file} in {directory}")  # Log the found .pth file
                return file
        print(f"No .pth file found in {directory}")  # Log if no .pth file is found
        return None
    except FileNotFoundError:
        print(f"Directory not found: {directory}")  # Log if the directory does not exist
        return None

def find_pth_file(directory):
    """Recursively search for a .pth file in the specified directory."""
    try:
        print(f"Searching for .pth file in directory: {directory}")  # Debug log
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.pth'):
                    print(f"Found .pth file: {file} in {root}")  # Debug log
                    return os.path.join(root, file)
        print(f"No .pth file found in {directory}")  # Debug log
        return None
    except FileNotFoundError:
        print(f"Directory not found: {directory}")  # Debug log
        return None

