import os
import waitress
from fastapi import FastAPI, Request
from serving.preprocessing import Preprocessing
from serving.model_service import ModelService

base_path = os.path.dirname(os.path.dirname(__file__))

app = FastAPI()
preprocessor = Preprocessing()
model_service = ModelService(os.path.join(base_path, 'logs', 'model_weights', 'model1.h5'),
                             os.path.join(base_path, 'logs', 'model_weights', 'model1', 'weights_epoch320.h5'),
                             rf_model_path=os.path.join(base_path, 'artifact', 'random_forest_model.pkl'))


@app.post("/predict")
async def predict(request: Request):
    # Extract request data
    request_data = await request.json()
    # Preprocess input data
    processed_data = preprocessor.preprocess(request_data['features'])
    # Get prediction from model
    prediction = model_service.predict(processed_data)
    return {"prediction": prediction}


if __name__ == "__main__":
    waitress.serve(app, host="127.0.0.1", port=8080)
