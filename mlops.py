from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

# FastAPI app instance
app = FastAPI()

# Define fixed inflow rate (liters per minute)
INFLOW_RATE = 10  # liters per minute

# Example dataset: Tank capacities (liters) and time to fill (minutes)
# Time to fill = capacity / inflow rate
tank_capacities = np.array([[500], [1000], [1500], [2000], [2500], [3000], [3500], [4000], [4500], [5000]])
time_to_fill = tank_capacities / INFLOW_RATE

# Train a simple linear regression model
model = LinearRegression()
model.fit(tank_capacities, time_to_fill)

# Request model for input validation
class TankRequest(BaseModel):
    capacity: float  # Capacity in liters

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# API endpoint to predict time to fill
@app.post("/predict-time/")
def predict_fill_time(request: TankRequest):
    capacity = request.capacity
    if capacity <= 0:
        raise HTTPException(status_code=400, detail="Capacity must be a positive value.")
    
    # Reshape input for prediction
    predicted_time = model.predict([[capacity]])[0][0]
    return {"capacity": capacity, "predicted_time_minutes": predicted_time}

# Start the server (you can run this using `uvicorn <filename>:app --reload`)
