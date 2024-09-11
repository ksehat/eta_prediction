from keras.models import load_model
import numpy as np

class ModelService:
    def __init__(self, model_path, weights_path):
        self.model = load_model(model_path)
        self.model.load_weights(weights_path)

    def predict(self, data):
        prediction = self.model.predict(np.array([data]))
        return prediction