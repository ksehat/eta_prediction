import streamlit as st
import os
from preprocessing import Preprocessing
from model_service import ModelService

base_path = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_path, 'logs', 'model_weights', 'model1.h5')
weights_path = os.path.join(base_path, 'logs', 'model_weights', 'model1', 'weights_epoch320.h5')
rf_model_path = os.path.join(base_path, 'artifact', 'random_forest_model.pkl')

preprocessor = Preprocessing()
model_service = ModelService(model_path, weights_path, rf_model_path=rf_model_path)

st.title("ETA Prediction")

# Input fields
city_id = st.text_input("City ID", "C")
accept_event_timestamp = st.text_input("Accept Event Timestamp", "2024-09-10T12:34:56Z")
origin_lat = st.number_input("Origin Latitude", value=35.6892)
origin_lon = st.number_input("Origin Longitude", value=51.3890)
destination_lat = st.number_input("Destination Latitude", value=36.292)
destination_lon = st.number_input("Destination Longitude", value=52.3890)
edd = st.number_input("EDD", value=12000)
provider_A = st.number_input("Provider A", value=3600)
provider_B = st.number_input("Provider B", value=3700)
provider_C = st.number_input("Provider C", value=3400)
provider_D = st.number_input("Provider D", value=3300)

# Collect input data
features = {
    "city_id": city_id,
    "accept_event_timestamp": accept_event_timestamp,
    "origin_lat": origin_lat,
    "origin_lon": origin_lon,
    "destination_lat": destination_lat,
    "destination_lon": destination_lon,
    "edd": edd,
    "provider_A": provider_A,
    "provider_B": provider_B,
    "provider_C": provider_C,
    "provider_D": provider_D
}

# Make prediction
if st.button("Predict"):
    # Preprocess input data
    processed_data = preprocessor.preprocess(features)
    # Get prediction from model
    prediction = model_service.predict(processed_data)
    # Display prediction
    st.write(f"Prediction: {prediction}")

