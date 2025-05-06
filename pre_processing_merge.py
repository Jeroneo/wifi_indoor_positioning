import os
import pandas as pd
import csv
import re

def process_wifi_data(directory):
    """
    Process WiFi data files from a directory structure organized as:
    Zone/Position/files
    
    Returns:
    1. A DataFrame with RSSI values where:
       - Each row represents a file
       - Each column represents a unique MAC address with its RSSI value
       - Last two columns are 'position' and 'zone'
    
    2. A DataFrame mapping MAC addresses to SSIDs
    """
    data = []
    all_mac_addresses = set()
    mac_to_ssid = {}  # Dictionary to store MAC address to SSID mapping
    
    # Walk through the directory structure
    for zone in os.listdir(directory):
        zone_path = os.path.join(directory, zone)
        if not os.path.isdir(zone_path):
            continue
            
        for position in os.listdir(zone_path):
            position_path = os.path.join(zone_path, position)
            if not os.path.isdir(position_path):
                continue
                
            for filename in os.listdir(position_path):
                file_path = os.path.join(position_path, filename)
                if not os.path.isfile(file_path):
                    continue
                
                # Process the file
                mac_rssi_dict = {}
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader)  # Read header row
                        
                        # Find the indices of required columns
                        column_indices = {}
                        for i, col_name in enumerate(header):
                            col_name = col_name.strip()
                            if col_name in ["MAC Address", "RSSI", "Channel Utilization", "SSID"]:
                                column_indices[col_name] = i
                        
                        required_columns = ["MAC Address", "RSSI", "Channel Utilization"]
                        missing_columns = [col for col in required_columns if col not in column_indices]
                        if missing_columns:
                            print(f"Warning: Missing required columns {missing_columns} in {file_path}")
                            continue
                        
                        # Process each row
                        for row in reader:
                            if len(row) <= max(column_indices.values()):
                                continue  # Skip rows that don't have enough columns
                            
                            # Check if Channel Utilization exists
                            channel_util = row[column_indices["Channel Utilization"]].strip()
                            if not channel_util:
                                continue  # Skip hotspots (no Channel Utilization)
                            
                            # Extract MAC address and RSSI
                            mac_address = row[column_indices["MAC Address"]].strip()
                            rssi = row[column_indices["RSSI"]].strip()
                            
                            # Validate MAC address format
                            if re.match(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$', mac_address):
                                try:
                                    rssi_value = int(rssi)
                                    mac_rssi_dict[mac_address] = rssi_value
                                    all_mac_addresses.add(mac_address)
                                    
                                    # Store SSID if available
                                    if "SSID" in column_indices:
                                        ssid = row[column_indices["SSID"]].strip()
                                        if ssid:  # Only store non-empty SSIDs
                                            mac_to_ssid[mac_address] = ssid
                                except ValueError:
                                    # Skip if RSSI can't be converted to integer
                                    pass
                                    
                    # Add the data for this file
                    if mac_rssi_dict:  # Only add if we have data
                        mac_rssi_dict['position'] = position
                        mac_rssi_dict['zone'] = zone
                        data.append(mac_rssi_dict)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    # Create main DataFrame with a simple numeric index
    df = pd.DataFrame(data)
    
    # Ensure all MAC addresses are present as columns
    for mac in all_mac_addresses:
        if mac not in df.columns:
            df[mac] = None
    
    # Reorder columns to put position and zone at the end
    cols = [col for col in df.columns if col not in ['position', 'zone']]
    cols = cols + ['position', 'zone']
    df = df[cols]
    
    # Create MAC to SSID mapping DataFrame
    mac_ssid_data = []
    for mac, ssid in mac_to_ssid.items():
        mac_ssid_data.append({'MAC Address': mac, 'SSID': ssid})
    
    mac_ssid_df = pd.DataFrame(mac_ssid_data)
    
    return df, mac_ssid_df

def main():
    directory = "../Acquisition des données/wifi_scanner_app/data"
    
    # Process the files
    rssi_df, mac_ssid_df = process_wifi_data(directory)
    
    # Save to CSV files
    rssi_df.to_csv("wifi_rssi_data.csv", index=False)
    mac_ssid_df.to_csv("mac_ssid_mapping.csv", index=False)
    
    print(f"Processed RSSI data saved to wifi_rssi_data.csv")
    print(f"DataFrame shape: {rssi_df.shape}")
    print(f"Number of unique MAC addresses: {len(rssi_df.columns) - 2}")  # Subtract 2 for position and zone
    print(f"Number of files processed: {len(rssi_df.index)}")
    
    print(f"\nMAC to SSID mapping saved to mac_ssid_mapping.csv")
    print(f"Number of MAC addresses with SSID information: {len(mac_ssid_df)}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

