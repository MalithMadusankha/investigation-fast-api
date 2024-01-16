from fastapi import APIRouter
from typing import List, Dict
from models.dataModel import DataModel
from schemas.serialize import serializeDict, serializeList
from controller.resultPredictor import PredictorForList, PredictorForCity
predict = APIRouter()


@predict.post('/predict-max')
async def predict_max(dataObj: DataModel):
    res = PredictorForList(dataObj)
    return res

@predict.post('/predict-city')
async def predict_city(dataObj: DataModel):
    res = PredictorForCity(dataObj)
    return res
