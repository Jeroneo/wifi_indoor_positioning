import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def load_model(model_path):
    """Load a saved model from the specified path."""
    return torch.load(model_path, weights_only=False) # , map_location=torch.device('cpu')

def preprocess_input(input_data, mac_address_order_path, replace_missing=True):
    """
    Preprocess input data by aligning it with the saved MAC address order.
    """
    mac_address_order = np.load(mac_address_order_path, allow_pickle=True)
    input_df = pd.DataFrame([input_data], columns=mac_address_order)
    input_array = input_df.values
    if replace_missing:
        input_array = np.nan_to_num(input_array, nan=-95)  # Replace NaN with -95 if enabled
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
        probabilities = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
        probabilities = np.nan_to_num(probabilities, nan=0.0)  # Replace NaN with 0.0
        top3_indices = np.argsort(probabilities)[-3:][::-1]  # Get indices of top 3 probabilities
        top3_probs = probabilities[top3_indices]  # Get top 3 probabilities
    zone_labels = np.load(zone_labels_path, allow_pickle=True)
    top3_predictions = [(zone_labels[idx], prob) for idx, prob in zip(top3_indices, top3_probs)]
    return top3_predictions

def infer_knn(model_path, input_data, mac_address_order_path, zone_labels_path):
    """Perform inference using the K-Nearest Neighbors model."""
    model = load_model(model_path)
    input_array = preprocess_input(input_data, mac_address_order_path)
    predicted_probs = model.predict_proba(input_array)[0]
    predicted_probs = np.nan_to_num(predicted_probs, nan=0.0)  # Replace NaN with 0.0
    top3_indices = predicted_probs.argsort()[-3:][::-1]
    zone_labels = np.load(zone_labels_path, allow_pickle=True)
    top3_predictions = [(zone_labels[idx], predicted_probs[idx]) for idx in top3_indices]
    return top3_predictions

def infer_svm(model_path, input_data, mac_address_order_path, zone_labels_path):
    """Perform inference using the Support Vector Machine model."""
    model = load_model(model_path)
    input_array = preprocess_input(input_data, mac_address_order_path)
    decision_function = model.decision_function(input_array)
    probabilities = np.exp(decision_function) / np.sum(np.exp(decision_function), axis=1, keepdims=True)
    probabilities = np.nan_to_num(probabilities, nan=0.0)  # Replace NaN with 0.0
    top3_indices = probabilities[0].argsort()[-3:][::-1]
    zone_labels = np.load(zone_labels_path, allow_pickle=True)
    top3_predictions = [(zone_labels[idx], probabilities[0][idx]) for idx in top3_indices]
    return top3_predictions

def infer_rf(model_path, input_data, mac_address_order_path, zone_labels_path):
    """Perform inference using the Random Forest model."""
    model = load_model(model_path)
    input_array = preprocess_input(input_data, mac_address_order_path)
    predicted_probs = model.predict_proba(input_array)[0]
    predicted_probs = np.nan_to_num(predicted_probs, nan=0.0)  # Replace NaN with 0.0
    top3_indices = predicted_probs.argsort()[-3:][::-1]
    zone_labels = np.load(zone_labels_path, allow_pickle=True)
    top3_predictions = [(zone_labels[idx], predicted_probs[idx]) for idx in top3_indices]
    return top3_predictions

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

    nn_model_path = "./models/NN/dataset/NN_missing_value.pth"
    knn_model_path = "./models/KNN/dataset/KNN_missing_value.pth"
    svm_model_path = "./models/SVM/dataset/SVM_missing_value.pth"
    rf_model_path = "./models/RandomForest/dataset/RF_n50_missing_value.pth"

    print("NN Prediction:", infer_nn(nn_model_path, input_data, mac_address_order_path, zone_labels_path))
    print("KNN Prediction:", infer_knn(knn_model_path, input_data, mac_address_order_path, zone_labels_path))
    print("SVM Prediction:", infer_svm(svm_model_path, input_data, mac_address_order_path, zone_labels_path))
    print("RF Prediction:", infer_rf(rf_model_path, input_data, mac_address_order_path, zone_labels_path))
