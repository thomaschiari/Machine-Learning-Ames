from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sc_preprocessing import sc_preprocessing
import pandas as pd
import numpy as np
import importlib
import pathlib
import joblib
import json 
import warnings

warnings.filterwarnings('ignore')

try:
    importlib.import_module('catboost')
    print("CatBoost está instalado.")
except ImportError:
    print("CatBoost não está instalado. Instalando agora...")
    
    # Instalar o CatBoost usando o pip
    import subprocess
    subprocess.check_call(['pip', 'install', 'catboost'])
    
    print("CatBoost foi instalado com sucesso.")

from catboost import CatBoostRegressor

app = FastAPI()

@app.post("/predict/{predict_type}")
async def create_item(predict_type: str, item: dict):
    if predict_type == "catboost":

        model = CatBoostRegressor()
        model.load_model(f"{pathlib.Path.cwd().parent}/models/catboost_model.cbm")
        item = sc_preprocessing(item)
        try:
            prediction = model.predict(item)
            return  {"prediction": prediction}.toJSON()
        except:
            return {"prediction": "Erro na predição"}
        
    elif predict_type == "linear":

        model = joblib.load(f"{pathlib.Path.cwd().parent}/models/linear_model.pkl")
        item = sc_preprocessing(item)
        try:
            prediction = model.predict(item)
            return  {"prediction": prediction}.toJSON()
        except:
            return {"prediction": "Erro na predição"}
    
    elif predict_type == "both":

        model_catboost = CatBoostRegressor()
        model_catboost.load_model(f"{pathlib.Path.cwd().parent}/models/catboost_model.cbm")
        model_linear = joblib.load(f"{pathlib.Path.cwd().parent}/models/linear_model.pkl")
        item = sc_preprocessing(item)
        try:
            prediction_catboost = model_catboost.predict(item)
            prediction_linear = model_linear.predict(item)
            return {"catboost": prediction_catboost, "linear": prediction_linear}.toJSON()
        except:
            return {"prediction": "Erro na predição"}
    
    else:
        raise HTTPException(status_code=404, detail="Invalid predict type")