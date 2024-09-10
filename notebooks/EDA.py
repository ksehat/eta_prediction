import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Get the current script's directory
project_dir = Path(__file__).parent.parent.resolve()

# Change the working directory to the script's directory
os.chdir(project_dir)

# Load the data
data_path = 'data/raw/rides_data.pq'
df = pd.read_parquet(data_path)

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Data types and general information
print("\nData types and general info:")
df.info()

# Visualize distributions of each ETA provider's error
def calculate_error(provider_col):
    return np.abs(df[provider_col] - df['ata'])

df['error_provider_A'] = calculate_error('provider_A')
df['error_provider_B'] = calculate_error('provider_B')
df['error_provider_C'] = calculate_error('provider_C')
df['error_provider_D'] = calculate_error('provider_D')

# Plot the distribution of ETA errors for each provider
plt.figure(figsize=(10, 6))
sns.kdeplot(df['error_provider_A'], label='Provider A', fill=False)
sns.kdeplot(df['error_provider_B'], label='Provider B', fill=False)
sns.kdeplot(df['error_provider_C'], label='Provider C', fill=False)
sns.kdeplot(df['error_provider_D'], label='Provider D', fill=False)
plt.title('Distribution of ETA Errors by Provider')
plt.xlabel('Error (Seconds)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Correlation matrix to understand feature relationships
plt.figure(figsize=(10, 8))
corr_matrix = df.drop(['city_id'], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Scatter plot of driven distance (edd) vs. errors
plt.figure(figsize=(10, 6))
sns.scatterplot(x='edd', y='error_provider_A', data=df, label='Provider A', alpha=0.5)
sns.scatterplot(x='edd', y='error_provider_B', data=df, label='Provider B', alpha=0.5)
sns.scatterplot(x='edd', y='error_provider_C', data=df, label='Provider C', alpha=0.5)
sns.scatterplot(x='edd', y='error_provider_D', data=df, label='Provider D', alpha=0.5)
plt.title('EDD vs. ETA Errors')
plt.xlabel('Estimated Driven Distance (EDD)')
plt.ylabel('ETA Error (Seconds)')
plt.legend()
plt.show()

# Analyzing ride's origin and destination
plt.figure(figsize=(10, 6))
sns.scatterplot(x='origin_lon', y='origin_lat', data=df, hue='city_id', alpha=0.5)
plt.title('Ride Origins by City')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Time of day analysis: extract hour from the timestamp
df['accept_hour'] = pd.to_datetime(df['accept_event_timestamp']).dt.hour

plt.figure(figsize=(10, 6))
sns.boxplot(x='accept_hour', y='error_provider_A', data=df, label='Provider A', color='lightblue')
plt.title('ETA Errors by Hour of Day for Provider A')
plt.xlabel('Hour of Day')
plt.ylabel('ETA Error (Seconds)')
plt.show()
