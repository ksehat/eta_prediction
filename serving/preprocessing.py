import pickle

import pandas as pd


class Preprocessing:
    def __init__(self):
        with open('C://Users\kanan\Desktop\ML-Map\eta_prediction//artifact\label_encoder.pkl', 'rb') as file:
            self.label_encoder = pickle.load(file)
        self.df = None

    def preprocess(self, request_data):
        feature_cols = ['city_id', 'accept_hour', 'accept_day_of_week', 'is_weekend', 'origin_lat', 'origin_lon',
                        'destination_lat', 'destination_lon',
                        'edd', 'provider_A', 'provider_B',
                        'provider_C', 'provider_D']
        self.df = pd.DataFrame([request_data])
        self.df['city_id'] = self.label_encoder.transform(self.df['city_id'])

        provider_cols = ['provider_A', 'provider_B', 'provider_C', 'provider_D']
        high_value = 10000
        self.df[provider_cols] = self.df[provider_cols].fillna(high_value)

        self.df['accept_event_timestamp'] = pd.to_datetime(self.df['accept_event_timestamp'])

        self.df['accept_hour'] = self.df['accept_event_timestamp'].dt.hour
        self.df['accept_day_of_week'] = self.df['accept_event_timestamp'].dt.dayofweek
        self.df['is_weekend'] = self.df['accept_day_of_week'].isin([3, 4]).astype(int)

        self.df.drop(columns=['accept_event_timestamp'], inplace=True)

        return self.df[feature_cols]
