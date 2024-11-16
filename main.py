from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model import predict

# FastAPI app
app = FastAPI()

# Pydantic model for request body validation
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# POST request endpoint to get predictions
@app.post("/predict/")
async def get_prediction(data: InputData):
    # Convert input data to pandas DataFrame (assuming 4 features for DT model)
    input_df = pd.DataFrame([{
        "feature1": data.feature1,
        "feature2": data.feature2,
        "feature3": data.feature3,
        "feature4": data.feature4,
    }])
    
    # Get model prediction
    prediction = predict(input_df)
    
    return {"prediction": prediction}
