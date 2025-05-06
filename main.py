import pre_processing_merge
import post_processing
import models

import pandas as pd

import os


def main():
    # -------------------------- Pre-processing and merge --------------------------

    # Set the directory containing the raw data files
    raw_data_directory = "./app_wifi_scanner/data"

    # Process the files
    rssi_df, mac_ssid_df = pre_processing_merge.process_wifi_data(raw_data_directory)
    
    # Save to CSV files
    rssi_df.to_csv("./raw_data/wifi_rssi_data.csv", index=False)
    mac_ssid_df.to_csv("./raw_data/mac_ssid_mapping.csv", index=False)

    print(f"Processed RSSI data saved to wifi_rssi_data.csv")
    print(f"DataFrame shape: {rssi_df.shape}")
    print(f"Number of unique MAC addresses: {len(rssi_df.columns) - 2}")  # Subtract 2 for position and zone
    print(f"Number of files processed: {len(rssi_df.index)}")

    print(f"\nMAC to SSID mapping saved to mac_ssid_mapping.csv")
    print(f"Number of MAC addresses with SSID information: {len(mac_ssid_df)}")

    # -------------------------- Post-processing --------------------------

    post_processing.process_wifi_data("./raw_data/wifi_rssi_data.csv")

    print("Post-processing completed.")

    # -------------------------- Models --------------------------

    models.main()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()


