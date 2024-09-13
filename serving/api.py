import json
import waitress
from fastapi import FastAPI, Request
from serving.preprocessing import Preprocessing
from serving.model_service import ModelService

app = FastAPI()
preprocessor = Preprocessing()
model_service = ModelService("C://Users\kanan\Desktop\ML-Map\eta_prediction\logs\model_weights\model1.h5",
                             "C://Users\kanan\Desktop\ML-Map\eta_prediction\logs\model_weights\model1\weights_epoch320.h5",
                             rf_model_path="C://Users\kanan\Desktop\ML-Map\eta_prediction//artifact//random_forest_model.pkl")


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
    waitress.serve(app, host="127.0.0.1", port=5000)
