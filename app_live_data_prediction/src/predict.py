from utils import load_mac_address_order, load_zone_labels, preprocess_live_data
import pandas as pd
import numpy as np
import pywifi
import torch
import time
import os
import re


def map_live_data_to_training_columns(live_data, mac_address_order):
    """Map the live data to the correct columns based on the MAC address order used during training."""
    # Create a DataFrame with all columns set to NaN
    mapped_data = pd.DataFrame(np.nan, index=range(len(live_data)), columns=mac_address_order)
    
    # Fill the mapped DataFrame with the values from the live data
    for mac in live_data.columns:
        if mac in mac_address_order:
            mapped_data[mac] = live_data[mac]
    
    return mapped_data


def predict_live_data(model, data, mac_address_order, zone_labels, model_name):
    """Predict the top 3 zones for live data using the trained model."""
    # Map live data to the correct columns
    data = map_live_data_to_training_columns(data, mac_address_order)
    
    # Preprocess the data
    data_preprocessed = preprocess_live_data(data.values)

    if data_preprocessed is None:
        return [[('Unknown', 0.0), ('Unknown', 0.0), ('Unknown', 0.0)]]  # Return unknown if all values are missing
    
    if model_name == 'SVM':
        # Handle SVM predictions
        probabilities = model.decision_function(data_preprocessed)
        top3_pred = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
        top3_prob = np.sort(probabilities, axis=1)[:, -3:][:, ::-1]
    else:
        # Handle PyTorch model predictions
        data_tensor = torch.tensor(data_preprocessed, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(data_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top3_prob, top3_pred = probabilities.topk(3, dim=1)
    
    # Convert predictions to zone labels and format the output
    top3_predictions = []
    for i in range(top3_pred.shape[0]):
        predictions = [(zone_labels[top3_pred[i, j]], top3_prob[i, j]) for j in range(3)]
        top3_predictions.append(predictions)
    
    return top3_predictions


def scan_wifi_realtime(iface):
    """Scan WiFi networks and return live data as a DataFrame."""
    rssi_dict = {}
    
    # Trigger scan
    iface.scan()
    
    # Get scan results
    results = iface.scan_results()
    time.sleep(2)  # Wait for a while before the next scan

    # Create dictionary with MAC addresses as keys and RSSI values as values
    rssi_dict = {re.sub(r'[:-]$', '', network.bssid.upper().replace(':', '-')): network.signal for network in results}

    # Create single-row DataFrame with MAC addresses as columns
    return pd.DataFrame([rssi_dict])


def load_model(model_name, model_path):
    """
    Load the specified PyTorch model from the given path.
    """
    if model_name == 'NN':
        class MissingValueClassifier(torch.nn.Module):
            """Neural network classifier that handles missing values using indicator features."""
            def __init__(self, input_dim, output_dim=15, hidden_dims=[256, 128, 64]):
                super(MissingValueClassifier, self).__init__()
                augmented_input_dim = input_dim * 2
                layers = []
                prev_dim = augmented_input_dim
                for hidden_dim in hidden_dims:
                    layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.BatchNorm1d(hidden_dim))
                    layers.append(torch.nn.Dropout(0.3))
                    prev_dim = hidden_dim
                layers.append(torch.nn.Linear(prev_dim, output_dim))
                self.model = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        print("1")

        model = MissingValueClassifier(input_dim=30)  # Adjust input_dim as needed

        print("2")

        model_path = os.path.abspath(model_path)
        checkpoint = torch.load(model_path, weights_only=False)

        print("3")

        model.load_state_dict(checkpoint['model_state_dict'])

        print("4")

        model.eval()

        print("5")

        return model
    elif model_name in ['KNN', 'SVM', 'RF']:
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def preprocess_input(data, model_name):
    """
    Preprocess input data for the specified model.
    """
    if model_name in ['NN', 'KNN', 'SVM', 'RF']:
        return torch.tensor(np.nan_to_num(data, nan=-95), dtype=torch.float32)  # Replace NaNs with a constant
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def predict(iface, model_path, data_path):
    """Perform prediction using the specified model."""
    # Extract the model name from the model path
    model_name = os.path.basename(model_path).split('_')[0]

    # Correctly pass model_name and model_path to load_model
    model = load_model(model_name, model_path)

    # Scan WiFi networks and get live data
    live_data = scan_wifi_realtime(iface)

    # Load model, MAC address order, and zone labels
    mac_address_order_path = f'../models/{data_path}/mac_address_order.npy'
    zone_labels_path = f'../models/{data_path}/zone_labels.npy'

    mac_address_order = load_mac_address_order(mac_address_order_path)
    zone_labels = load_zone_labels(zone_labels_path)

    # Predict zones for live data
    top3_predictions = predict_live_data(model, live_data, mac_address_order, zone_labels, model_name)

    return top3_predictions


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    iface = pywifi.PyWiFi().interfaces()[0]  # Assuming the first interface is the one to use
    model_name = 'NN'  # Replace with the desired model name
    split_name = 'split1'  # Replace with the desired split name
    top3_predictions = predict(iface, model_name, split_name)
    
    # Print predictions
    for i, predictions in enumerate(top3_predictions):
        print(f"Sample {i+1}:")
        for zone, confidence in predictions:
            print(f"  Predicted Zone - {zone} with confidence {confidence:.6f}")
