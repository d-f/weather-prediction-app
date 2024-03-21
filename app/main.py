from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import predict
from typing import List


app = FastAPI()


class InputList(BaseModel):
    """data type for input"""
    values: list[list[float]]


class Prediction(BaseModel):
    """data type for prediction"""
    forecast: list


@app.get("/")
def home():
    """returns health check"""
    return {"health_check": "OK"}


@app.post("/predict", response_model=Prediction)
def get_model_pred(payload: InputList):
    """predicts forecast from input"""
    weather_forecast = predict(payload.values)
    return {"forecast": weather_forecast}
