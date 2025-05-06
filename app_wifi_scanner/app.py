from flask import Flask, render_template, request, redirect, url_for, jsonify
from contextlib import contextmanager
import subprocess
import time
import os
import json
from tqdm import tqdm
import threading

app = Flask(__name__)

# Store scan status information
scan_status = {
    "is_scanning": False,
    "current_acquisition": 0,
    "total_acquisitions": 0,
    "zone": "",
    "position": ""
}

# Change the current directory to the one where the script is located
@contextmanager
def change_directory(destination):
    current_directory = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(current_directory)

def scan_wifi(acquisition_file_name):
    with change_directory('./wifiinfoview-x64'):
        # Execute the command and redirect output to subprocess.DEVNULL
        subprocess.run('WifiInfoView.exe /stab "" | GetNir "SSID,MACAddress,RSSI" "Connected=Yes"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.system(f'WifiInfoView.exe /scomma {acquisition_file_name}.csv')

def move_files(zone, position, acquisition_file_name):
    os.makedirs(f'./data/{zone}/{position}', exist_ok=True)
    os.rename(f'./wifiinfoview-x64/{acquisition_file_name}.csv', f'./data/{zone}/{position}/{acquisition_file_name}.csv')

def perform_scans(zone, position, nb_acquisitions=10):
    """Perform multiple WiFi scans with status updates"""
    global scan_status
    
    scan_status["is_scanning"] = True
    scan_status["zone"] = zone
    scan_status["position"] = position
    scan_status["total_acquisitions"] = nb_acquisitions
    scan_status["current_acquisition"] = 0
    
    try:
        for i in range(nb_acquisitions):
            acquisition_file_name = f"acquisition_{i}"
            scan_wifi(acquisition_file_name)
            move_files(zone, position, acquisition_file_name)
            
            # Update status
            scan_status["current_acquisition"] = i + 1
            
            time.sleep(2)  # Wait 2 seconds between acquisitions
    finally:
        # Reset scan status when done
        scan_status["is_scanning"] = False

def get_directory_structure():
    """Get the directory structure for the data folder"""
    structure = {}
    data_dir = './data'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return structure
    
    for zone in os.listdir(data_dir):
        zone_path = os.path.join(data_dir, zone)
        if os.path.isdir(zone_path):
            structure[zone] = {}
            for position in os.listdir(zone_path):
                position_path = os.path.join(zone_path, position)
                if os.path.isdir(position_path):
                    structure[zone][position] = []
                    for file in os.listdir(position_path):
                        if file.endswith('.csv'):
                            structure[zone][position].append(file)
    
    return structure

@app.route('/')
def index():
    structure = get_directory_structure()
    zones = list(structure.keys())
    
    # Get all positions across all zones
    positions = []
    for zone_data in structure.values():
        positions.extend(list(zone_data.keys()))
    positions = list(set(positions))  # Remove duplicates
    
    return render_template('index.html', structure=structure, zones=zones, positions=positions)

@app.route('/scan', methods=['POST'])
def scan():
    # If already scanning, don't start a new scan
    if scan_status["is_scanning"]:
        return jsonify({"status": "error", "message": "A scan is already in progress"})
    
    zone = request.form.get('zone')
    position = request.form.get('position')
    nb_acquisitions = int(request.form.get('nb_acquisitions', 10))
    
    # Create a new thread for scanning to prevent blocking the web server
    scan_thread = threading.Thread(
        target=perform_scans, 
        args=(zone, position, nb_acquisitions)
    )
    scan_thread.daemon = True
    scan_thread.start()
    
    return jsonify({"status": "success", "message": "Scan started"})

@app.route('/get_structure')
def get_structure():
    structure = get_directory_structure()
    return jsonify(structure)

@app.route('/scan_status')
def get_scan_status():
    return jsonify(scan_status)

@app.route('/get_positions/<zone>')
def get_positions(zone):
    """Get available positions for a specific zone"""
    structure = get_directory_structure()
    
    if zone in structure:
        positions = list(structure[zone].keys())
        return jsonify({"status": "success", "positions": positions})
    else:
        return jsonify({"status": "error", "message": "Zone not found", "positions": []})

if __name__ == '__main__':
    # Ensure data directory exists
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    app.run(debug=True)

