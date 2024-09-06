from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Define the path for the model
MODEL_PATH = 'model.pkl'

# Example dataset and training
house_sizes = np.array([[600], [800], [1000], [1200], [1400], [1600], [1800], [2000], [2200], [2400]])
house_prices = np.array([150000, 180000, 210000, 240000, 270000, 300000, 330000, 360000, 390000, 420000])

# Load or train model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    model = LinearRegression()
    model.fit(house_sizes, house_prices)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

class HouseSize(BaseModel):
    size: float

@app.post("/predict_price/")
async def predict_price(house_size: HouseSize):
    try:
        size = np.array([[house_size.size]])
        predicted_price = model.predict(size)[0]
        return {"predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Mount the static files directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")
