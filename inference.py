import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def load_model(model_path):
    """Load a saved model from the specified path."""
    return torch.load(model_path, map_location=torch.device('cpu'))

def preprocess_input(input_data, mac_address_order_path):
    """
    Preprocess input data by aligning it with the saved MAC address order.
    """
    mac_address_order = np.load(mac_address_order_path, allow_pickle=True)
    input_df = pd.DataFrame([input_data], columns=mac_address_order)
    input_array = input_df.values
    input_array = np.nan_to_num(input_array, nan=-95)  # Replace NaN with -95
    return input_array

def infer_nn(model_path, input_data, mac_address_order_path, zone_labels_path):
    """Perform inference using the Neural Network model."""
    model = load_model(model_path)
    model.eval()
    input_array = preprocess_input(input_data, mac_address_order_path)
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    input_tensor = torch.cat([input_tensor, torch.tensor(np.isnan(input_array).astype(np.float32))], dim=1)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    zone_labels = np.load(zone_labels_path, allow_pickle=True)
    return zone_labels[predicted_class]

def infer_knn(model_path, input_data, mac_address_order_path, zone_labels_path):
    """Perform inference using the K-Nearest Neighbors model."""
    model = load_model(model_path)
    input_array = preprocess_input(input_data, mac_address_order_path)
    predicted_class = model.predict(input_array)[0]
    zone_labels = np.load(zone_labels_path, allow_pickle=True)
    return zone_labels[predicted_class]

def infer_svm(model_path, input_data, mac_address_order_path, zone_labels_path):
    """Perform inference using the Support Vector Machine model."""
    model = load_model(model_path)
    input_array = preprocess_input(input_data, mac_address_order_path)
    predicted_class = model.predict(input_array)[0]
    zone_labels = np.load(zone_labels_path, allow_pickle=True)
    return zone_labels[predicted_class]

def infer_rf(model_path, input_data, mac_address_order_path, zone_labels_path):
    """Perform inference using the Random Forest model."""
    model = load_model(model_path)
    input_array = preprocess_input(input_data, mac_address_order_path)
    predicted_class = model.predict(input_array)[0]
    zone_labels = np.load(zone_labels_path, allow_pickle=True)
    return zone_labels[predicted_class]

if __name__ == "__main__":
    # Example usage
    input_data = {
        "MAC1": -80,
        "MAC2": -75,
        "MAC3": np.nan,  # Missing value
        # Add more MAC addresses as needed
    }

    mac_address_order_path = "./models/mac_address_order.npy"
    zone_labels_path = "./models/zone_labels.npy"

    nn_model_path = "./models/split_70_30/NN/wifi_rssi_data_100_clean/NN_missing_value.pth"
    knn_model_path = "./models/split_70_30/KNN/wifi_rssi_data_100_clean/KNN_missing_value.pth"
    svm_model_path = "./models/split_70_30/SVM/wifi_rssi_data_100_clean/SVM_missing_value.pth"
    rf_model_path = "./models/split_70_30/RandomForest/wifi_rssi_data_100_clean/RF_n3_missing_value.pth"

    print("NN Prediction:", infer_nn(nn_model_path, input_data, mac_address_order_path, zone_labels_path))
    print("KNN Prediction:", infer_knn(knn_model_path, input_data, mac_address_order_path, zone_labels_path))
    print("SVM Prediction:", infer_svm(svm_model_path, input_data, mac_address_order_path, zone_labels_path))
    print("RF Prediction:", infer_rf(rf_model_path, input_data, mac_address_order_path, zone_labels_path))
