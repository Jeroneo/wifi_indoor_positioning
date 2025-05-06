import pandas as pd
import numpy as np
import os

def process_wifi_data(directory):
    """
    Process WiFi data from the specified directory.

    Parameters:
    - directory: The directory containing the WiFi data files.

    Returns:
    - None. Saves processed files to the appropriate directories.
    """

    df = pd.read_csv(directory).drop('position', axis=1)

    def df_percentage_per_zone():
        df_nb_not_null_value = df[df.columns[:-1]].notnull().astype(float).groupby(df[df.columns[-1]]).sum()
        df_nb_not_null_value['zone'] = df_nb_not_null_value.index
        df_nb_not_null_value.reset_index(drop=True, inplace=True)

        df_nb_value = pd.DataFrame(0, index=range(df_nb_not_null_value[df_nb_not_null_value.columns[:-1]].shape[0]),
                                   columns=df_nb_not_null_value.columns[:-1])
        df_nb_value['zone'] = df_nb_not_null_value[df_nb_not_null_value.columns[-1]].values

        for idx, zone in enumerate(df_nb_not_null_value[df_nb_not_null_value.columns[-1]].values):
            if zone in df_nb_value[df_nb_value.columns[-1]].values:
                value = float(df[df.columns[-1]].value_counts()[zone])
                df_nb_value.iloc[idx, :-1] = [value] * (df_nb_value.shape[1] - 1)

        df_mask = df_nb_not_null_value[df_nb_not_null_value.columns[:-1]] / df_nb_value[df_nb_value.columns[:-1]]
        df_mask['zone'] = df_nb_not_null_value['zone']

        df_mask.to_csv('percentage_zone.csv', index=False)
        return df_mask

    def only_masked_values(df, df_mask):
        df_20 = df.copy()
        for col in df.columns[:-1]:
            df_20[col] = df.apply(
                lambda row: row[col] if df_mask.loc[df_mask['zone'] == row[df.columns[-1]], col].values[0] == 1 else np.nan,
                axis=1
            )
        return df_20

    df_mask = df_percentage_per_zone()

    if not os.path.exists('./datasets/raw/mask/'):
        os.makedirs('./datasets/raw/mask/')

    thresholds = [0.2, 0.5, 0.8, 0.95, 1.0]
    for threshold in thresholds:
        df_mask_threshold = df_mask.copy()
        df_mask_threshold.iloc[:, :-1] = df_mask_threshold.iloc[:, :-1].ge(threshold).astype(int)

        mask_file = f'./datasets/raw/mask/mask_zone_{int(threshold * 100)}.csv'
        df_mask_threshold.to_csv(mask_file, index=False)

        df_threshold = only_masked_values(df, df_mask_threshold)
        data_file = f'./datasets/raw/wifi_rssi_data_{int(threshold * 100)}.csv'
        df_threshold.to_csv(data_file, index=False)

        df_mean = df.groupby(df.columns[-1]).mean()
        df_std = df.groupby(df.columns[-1]).std()
        df_2_std = df_std * 2

        df_clipped = clip_outliers(df_threshold, df_mean, df_2_std)
        df_ap_completed = replace_missing_with_mean(df_clipped, df_mean, df_mask_threshold)

        ap_completed_file = f'./datasets/ap_completed/wifi_rssi_data_{int(threshold * 100)}_ap_completed.csv'
        df_ap_completed.to_csv(ap_completed_file, index=False)

        if not os.path.exists('./datasets/clean/'):
            os.makedirs('./datasets/clean/')

        df_ap_completed.fillna(-95, inplace=True)
        clean_file = f'./datasets/clean/wifi_rssi_data_{int(threshold * 100)}_clean.csv'
        df_ap_completed.to_csv(clean_file, index=False)

def clip_outliers(df, df_mean, df_2_std):
    df_clipped = df.copy()
    for col in df_clipped.columns[:-1]:
        df_clipped[col] = df_clipped.apply(
            lambda row: np.clip(
                row[col],
                df_mean.loc[row[df_clipped.columns[-1]], col] - df_2_std.loc[row[df_clipped.columns[-1]], col],
                df_mean.loc[row[df_clipped.columns[-1]], col] + df_2_std.loc[row[df_clipped.columns[-1]], col]
            ) if not pd.isna(row[col]) else np.nan,
            axis=1
        )
    return df_clipped

def replace_missing_with_mean(df_clipped, df_mean, df_mask):
    df_filled = df_clipped.copy()
    for col in df_filled.columns[:-1]:
        df_filled[col] = df_filled.apply(
            lambda row: df_mean.loc[row[df_filled.columns[-1]], col]
            if pd.isna(row[col]) and df_mask.loc[df_mask['zone'] == row[df_filled.columns[-1]], col].values[0] == 1
            else row[col],
            axis=1
        )
    return df_filled

# Example usage:
# process_wifi_data("path/to/your/directory")


