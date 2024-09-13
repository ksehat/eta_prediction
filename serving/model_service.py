import pickle
import tensorflow as tf
import numpy as np


class ModelService:
    def __init__(self, nn_model_path, nn_model_weights_path, rf_model_path):
        # Load Neural Network model (TensorFlow)
        self.nn_model = tf.keras.models.load_model(nn_model_path)
        self.nn_model.load_weights(nn_model_weights_path)

        # Load Random Forest (Scikit-learn)
        with open(rf_model_path, 'rb') as rf_file:
            self.rf_model = pickle.load(rf_file)

    def predict(self, preprocessed_data):
        # Get predictions from all three models
        nn_prediction = self.nn_model.predict(preprocessed_data)
        rf_prediction = self.rf_model.predict(preprocessed_data)

        # You can aggregate predictions or return them separately
        return {
            'neural_network': np.argmax(nn_prediction, axis=1).tolist(),  # Get the predicted class
            'random_forest': rf_prediction.tolist()
        }