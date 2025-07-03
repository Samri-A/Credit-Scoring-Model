from fastapi import FastAPI
from pydantic_models import CreditRiskInput
import numpy as np
import joblib
import os

app = FastAPI()

# Load the model
MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_model/model.pkl")
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(input_data: CreditRiskInput):
    features = np.array([[
        input_data.Total_Amount,
        input_data.Average_Amount,
        input_data.Transaction_Count,
        input_data.Std_Amount
    ]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
