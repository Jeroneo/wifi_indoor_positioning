from flask import Flask, render_template, jsonify, request
from utils import find_pth_file
from inference import infer_nn, infer_knn, infer_svm, infer_rf

import pandas as pd
import numpy as np

import pywifi
import time
import os
import re


app = Flask(__name__)

# Initialize WiFi interface
wifi = pywifi.PyWiFi()
iface = wifi.interfaces()[0]

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
    return rssi_dict

@app.route('/')
def index():
    return render_template('wifi-locator-webapp.html')

@app.route('/get_splits', methods=['GET'])
def get_splits():
    """Return the directory names inside the models directory, formatted for the dropdown."""
    splits_dir = '../../models'
    directory_tree = {}

    for dirs in os.listdir(splits_dir):
        dir_path = os.path.join(splits_dir, dirs)
        if os.path.isdir(dir_path):
            formatted_name = dirs
            directory_tree[formatted_name] = []

            # Add all folders and include their names as is
            for model_dir in os.listdir(dir_path):
                model_path = os.path.join(dir_path, model_dir)
                if os.path.isdir(model_path):
                    percentages = [
                        folder.replace("wifi_rssi_data_", "")
                        for folder in os.listdir(model_path)
                        if folder.startswith("wifi_rssi_data_")
                    ]
                    directory_tree[formatted_name].append({
                        "model": model_dir,
                        "percentages": sorted(percentages, key=lambda x: int(re.sub(r'\D', '', x)) if re.search(r'\d', x) else float('inf'))
                    })

    return jsonify({'splits': directory_tree})

@app.route('/get_location', methods=['GET'])
def get_location():
    """Perform inference using the selected model, split, and access point percentage."""
    try:
        model_name = request.args.get('model')
        split_name = request.args.get('split')
        percentage = request.args.get('percentage')
        replace_missing = request.args.get('replace_missing', 'true').lower() == 'true'

        # Perform inference using the selected model
        mac_address_order_path = os.path.abspath('../models/mac_address_order.npy')
        zone_labels_path = os.path.abspath('../models/zone_labels.npy')

        # Map short model names to their folder names
        model_folder_map = {
            "NN": "NN",
            "KNN": "KNN",
            "SVM": "SVM",
            "RF": "RandomForest"
        }

        if not model_name or not split_name or not percentage:
            return jsonify({'error': 'Model, split, and percentage must be specified'}), 400

        # Get the correct folder name for the model
        model_folder = model_folder_map.get(model_name)
        if not model_folder:
            return jsonify({'error': f"Invalid model name: {model_name}"}), 400

        print(f"Selected model: {model_name} (folder: {model_folder}), split: {split_name}, percentage: {percentage}")

        # Construct the data path
        data_path = f'../../models/{split_name}/{model_folder}/wifi_rssi_data_{percentage}'
        print(f"Data path: {data_path}")

        # Check if the data directory exists
        if not os.path.exists(data_path):
            return jsonify({'error': f"Data path does not exist: {data_path}"}), 404

        # Search for the .pth file
        model_file_path = find_pth_file(data_path)
        if not model_file_path:
            print(f"No .pth file found in directory: {data_path}")  # Debug log
            return jsonify({'error': f"No .pth file found in directory: {data_path}. Scanning stopped."}), 404
        
        model_file_path = os.path.abspath(model_file_path)  # Get absolute path

        print(f"Model file path: {model_file_path}")  # Debug log

        # Scan WiFi networks and get live data
        live_data = scan_wifi_realtime(iface)

        if replace_missing:
            live_data = {key: (value if not np.isnan(value) else -95) for key, value in live_data.items()}

        if model_name == "NN":
            top3_predictions = infer_nn(model_file_path, live_data, mac_address_order_path, zone_labels_path)
        elif model_name == "KNN":
            top3_predictions = infer_knn(model_file_path, live_data, mac_address_order_path, zone_labels_path)
        elif model_name == "SVM":
            top3_predictions = infer_svm(model_file_path, live_data, mac_address_order_path, zone_labels_path)
        elif model_name == "RF":
            top3_predictions = infer_rf(model_file_path, live_data, mac_address_order_path, zone_labels_path)
        else:
            return jsonify({'error': f"Unsupported model: {model_name}"}), 400

        # Handle NaN confidence values
        def format_confidence(confidence):
            return f"{confidence:.2%}" if not np.isnan(confidence) else "0.00%"

        return jsonify({
            'location': top3_predictions[0][0],
            'confidence': format_confidence(top3_predictions[0][1]),
            'second_location': top3_predictions[1][0],
            'second_confidence': format_confidence(top3_predictions[1][1]),
            'third_location': top3_predictions[2][0],
            'third_confidence': format_confidence(top3_predictions[2][1])
        })
    except Exception as e:
        print(f"Error in /get_location: {e}")  # Log the error
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app.run(debug=True)

