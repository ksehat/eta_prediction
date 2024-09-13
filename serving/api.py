import os
import uvicorn
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
    print('get the request.')
    request_data = await request.json()
    processed_data = preprocessor.preprocess(request_data['features'])
    prediction = model_service.predict(processed_data)
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("serving.api:app", host="0.0.0.0", port=5000)
