import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_pred_error = None
        self.scaler = StandardScaler()

    def load_data(self):
        self.df = pd.read_parquet(self.file_path)

    def feature_engineering(self):
        provider_cols = ['provider_A', 'provider_B', 'provider_C', 'provider_D']
        high_value = 0
        self.df[provider_cols] = self.df[provider_cols].fillna(high_value)

        self.df['accept_event_timestamp'] = pd.to_datetime(self.df['accept_event_timestamp'])

        # Extract hour of the day, day of the week, and whether it's a weekend (3rd and 4th day of the week are the weekend in Iran)
        self.df['accept_hour'] = self.df['accept_event_timestamp'].dt.hour
        self.df['accept_day_of_week'] = self.df['accept_event_timestamp'].dt.dayofweek
        self.df['accept_month'] = self.df['accept_event_timestamp'].dt.month
        self.df['is_weekend'] = self.df['accept_day_of_week'].isin([3, 4]).astype(int)

        self.df.drop(columns=['accept_event_timestamp'], inplace=True)

        # Calculate errors for each provider
        self.df['error_provider_A'] = np.abs(self.df['provider_A'] - self.df['ata'])
        self.df['error_provider_B'] = np.abs(self.df['provider_B'] - self.df['ata'])
        self.df['error_provider_C'] = np.abs(self.df['provider_C'] - self.df['ata'])
        self.df['error_provider_D'] = np.abs(self.df['provider_D'] - self.df['ata'])

        self.df['best_prediction_error'] = self.df[
            ['error_provider_A', 'error_provider_B', 'error_provider_C', 'error_provider_D']].min(axis=1)

        # Determine the most accurate provider for classification
        self.df['accurate_provider'] = self.df[
            ['error_provider_A', 'error_provider_B', 'error_provider_C', 'error_provider_D']].idxmin(axis=1)
        self.df['accurate_provider'] = self.df['accurate_provider'].map({
            'error_provider_A': 0,
            'error_provider_B': 1,
            'error_provider_C': 2,
            'error_provider_D': 3
        })

    def split_data(self, feature_cols, target_col):
        X = self.df[feature_cols]
        y = self.df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42, stratify=self.df[
                ['city_id', 'accept_day_of_week',
                 'accept_hour']])

    def scale_data(self, feature_cols):
        self.X_train[feature_cols] = self.scaler.fit_transform(self.X_train)
        self.X_test[feature_cols] = self.scaler.transform(self.X_test)

        with open('C://Users\kanan\Desktop\ML-Map\eta_prediction//artifact/standard_scaler.pkl', 'wb') as file:
            pickle.dump(self.scaler, file)

    def cluster_data(self):
        kmeans = KMeans(n_clusters=20, random_state=42)
        self.X_train['cluster'] = kmeans.fit_predict(self.X_train)

        # Predict clusters for X_test and add them as a new column
        self.X_test['cluster'] = kmeans.predict(self.X_test)

        with open('C://Users\kanan\Desktop\ML-Map\eta_prediction//artifact/kmeans_model.pkl', 'wb') as file:
            pickle.dump(kmeans, file)

    def knn_regressor(self):
        knn_regressor = KNeighborsRegressor(n_neighbors=5)

        knn_regressor.fit(self.X_train, self.df['ata'][self.X_train.index])

        self.X_train['knn_pred'] = knn_regressor.predict(self.X_train)
        self.X_test['knn_pred'] = knn_regressor.predict(self.X_test)