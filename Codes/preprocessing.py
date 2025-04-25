import pandas as pd
import numpy as np
from sklearn.utils import shuffle

#Loading the dataset
df = pd.read_csv(r"C:\Users\Cherry\OneDrive\Documents\CHS_project\RFID - Activity, Location and PID Labels.csv")

#Converting the timestamp 
df['Timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='ms')
df.drop(['unix_timestamp'], axis=1, inplace=True)

#Defined RSSI columns
meta_cols = ['location_label', 'activity_label', 'PID', 'Timestamp']
rssi_columns = df.columns.difference(meta_cols)
df[rssi_columns] = df[rssi_columns].replace(0, np.nan)

#Bin timestamps into 30sec windows
df['Time_Bin'] = df['Timestamp'].dt.floor('30s')  # rounding to nearest 30 seconds

#Container for features
feature_list = []

#Group by PID and Time_Bin to extract rows
for (pid, time_bin), group in df.groupby(['PID', 'Time_Bin']):
    rssi_data = group[rssi_columns]

    #Statistical RSSI features
    mean_rssi = rssi_data.mean().mean()
    median_rssi = rssi_data.median().median()
    std_rssi = rssi_data.std().mean()
    var_rssi = rssi_data.var().mean()
    max_rssi = rssi_data.max().max()
    min_rssi = rssi_data.min().min()

    #Temporal features
    time_diffs = group['Timestamp'].diff().dt.total_seconds().dropna()
    mean_interval = time_diffs.mean()
    std_interval = time_diffs.std()

    #Spatial and signal behavior
    unique_locations = group['location_label'].nunique()
    rssi_count = rssi_data.count().sum()

    feature_list.append({
        'PID': pid,
        'Time_Bin': time_bin,
        'Mean_RSSI': mean_rssi,
        'Median_RSSI': median_rssi,
        'Std_RSSI': std_rssi,
        'Var_RSSI': var_rssi,
        'Max_RSSI': max_rssi,
        'Min_RSSI': min_rssi,
        'Mean_Time_Interval': mean_interval,
        'Std_Time_Interval': std_interval,
        'Unique_Locations': unique_locations,
        'RSSI_Count': rssi_count,
        'Counterfeit': 0
    })

#Converting them to dataframe
features_df = pd.DataFrame(feature_list)

#Fill in the missing values for numeric features
numeric_cols = features_df.select_dtypes(include=[np.number]).columns
features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())

#Now, Simulating the counterfeit tags

#Injected 10% counterfeit rows
num_fakes = max(5, int(0.1 * len(features_df)))
fake_tags = features_df.sample(num_fakes, random_state=42).copy()

np.random.seed(42)
fake_tags['Mean_RSSI'] += np.random.normal(0, 1.0, num_fakes)
fake_tags['Std_RSSI'] *= np.random.uniform(1.05, 1.2, num_fakes)
fake_tags['Mean_Time_Interval'] *= np.random.uniform(0.8, 1.2, num_fakes)
fake_tags['Unique_Locations'] += np.random.randint(1, 2, num_fakes)
fake_tags['RSSI_Count'] *= np.random.uniform(0.5, 1.5, num_fakes)
fake_tags['Counterfeit'] = 1

#Combined and shuffled them a bit
final_df = pd.concat([features_df, fake_tags], ignore_index=True)
final_df = shuffle(final_df, random_state=42).reset_index(drop=True)

#Saved the final processed dataset as shown here
final_df.to_csv("Processed_RFID_Features_Windowed.csv", index=False)

print("Updated preprocessing is done.")
print(f"Total no. of records: {len(final_df)}")
print(f"Counterfeits injected: {num_fakes}")
print("\n Class Distribution:\n", final_df['Counterfeit'].value_counts())
