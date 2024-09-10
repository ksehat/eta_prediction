import os
from pathlib import Path
import pandas as pd
import numpy as np

project_dir = Path(__file__).parent.parent.resolve()
os.chdir(project_dir)
data_path = 'data/raw/rides_data.pq'
df = pd.read_parquet(data_path)

df['accept_event_timestamp'] = pd.to_datetime(df['accept_event_timestamp'])

# Extract hour of the day, day of the week, and whether it's a weekend
df['accept_hour'] = df['accept_event_timestamp'].dt.hour
df['accept_day_of_week'] = df['accept_event_timestamp'].dt.dayofweek
df['is_weekend'] = df['accept_event_timestamp'].dt.dayofweek

# In Iran, 3rd (Wednesday) and 4th (Thursday) days are weekends
df['is_weekend'] = df['accept_day_of_week'].isin([3, 4])

# Calculate the absolute error between each provider's ETA and the actual ATA
df['error_provider_A'] = np.abs(df['provider_A'] - df['ata'])
df['error_provider_B'] = np.abs(df['provider_B'] - df['ata'])
df['error_provider_C'] = np.abs(df['provider_C'] - df['ata'])
df['error_provider_D'] = np.abs(df['provider_D'] - df['ata'])

# Create the 'accurate_provider' column, finding the provider with the minimum error
df['accurate_provider'] = df[['error_provider_A', 'error_provider_B', 'error_provider_C', 'error_provider_D']].idxmin(
    axis=1)

df['accurate_provider'] = df['accurate_provider'].map({
    'error_provider_A': 0,
    'error_provider_B': 1,
    'error_provider_C': 2,
    'error_provider_D': 3
})

df.drop(columns=['ata', 'accept_event_timestamp', 'error_provider_A', 'error_provider_B', 'error_provider_C',
                 'error_provider_D'], inplace=True)

# Saving the processed dataset
processed_data_path = 'data/processed/processed_rides_data_solution1.pq'
df.to_parquet(processed_data_path)

print("Feature engineering completed and saved to processed data folder.")
