import os
import copy
import pickle
import pandas as pd


class Preprocessing:
    def __init__(self):
        base_path = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base_path, 'artifact', 'standard_scaler.pkl'), 'rb') as file:
            self.standard_scaler = pickle.load(file)
        with open(os.path.join(base_path, 'artifact', 'kmeans_model.pkl'), 'rb') as file:
            self.kmeans = pickle.load(file)
        self.df = None

    def preprocess(self, request_data):
        feature_cols = ['accept_hour', 'accept_day_of_week', 'accept_month', 'is_weekend',
                'origin_lat', 'origin_lon', 'destination_lat', 'destination_lon',
                'edd', 'provider_A', 'provider_B', 'provider_C', 'provider_D']
        self.df = pd.DataFrame([request_data])

        provider_cols = ['provider_A', 'provider_B', 'provider_C', 'provider_D']
        high_value = 0
        self.df[provider_cols] = self.df[provider_cols].fillna(high_value)

        self.df['accept_event_timestamp'] = pd.to_datetime(self.df['accept_event_timestamp'])

        self.df['accept_hour'] = self.df['accept_event_timestamp'].dt.hour
        self.df['accept_day_of_week'] = self.df['accept_event_timestamp'].dt.dayofweek
        self.df['accept_month'] = self.df['accept_event_timestamp'].dt.month
        self.df['is_weekend'] = self.df['accept_day_of_week'].isin([3, 4]).astype(int)

        self.df.drop(columns=['accept_event_timestamp'], inplace=True)

        self.df[feature_cols] = self.standard_scaler.transform(self.df[feature_cols])

        self.X_pred = copy.deepcopy(self.df[feature_cols])

        self.X_pred['cluster'] = self.kmeans.predict(self.X_pred)

        return self.X_pred
