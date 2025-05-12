from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept requests from any domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class data_sent(BaseModel):
    battery_power :  int
    fc            :  int
    four_g        :  int
    int_memory    :  int
    mobile_wt     :  int
    n_cores       :  int
    pc            :  int
    px_height     :  int
    px_width      :  int
    ram           :  int
    touch_screen  :  int
    wifi          :  int

with open("xgbClassifier.pkl","rb") as model:
    model = pickle.load(model)

@app.get("/predict_model/monitor")
def monitor_prediction_route():
    return {"status": "up", "message": "Monitor check successful"}


@app.get("/")
def read_root():
    return "Success ✅: Your app is functioning properly"
    
@app.post("/predict_model")
async def predicted_category(user_values : data_sent):
    input_array =  np.array([list(user_values.dict().values())])

    prediction = model.predict(input_array)

    if prediction[0] == 0:
        result = "low"
    elif prediction[0] == 1:
        result = "Medium"
    else:
        result = "High"
        
    return {"prediction": result}
