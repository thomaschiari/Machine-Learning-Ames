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

@app.post("/predict/{model_name}")
async def create_item(model_name: str, item: dict):

    if model_name == "catboost":
        model = CatBoostRegressor()
        model.load_model(f"{pathlib.Path.cwd().parent}/models/catboost_model.cbm")
        item = sc_preprocessing(item)
        try:
            prediction = model.predict(item)
            return  {"prediction log10": prediction[0],
                        "final prediction": 10**prediction[0],
                        "feature_importance": dict(zip(model.feature_names_, model.get_feature_importance()))}
        except Exception as e:
            return {"prediction": "Erro na predição: " + str(e)}
    
    elif model_name == "ridge":
        model = joblib.load(f"{pathlib.Path.cwd().parent}/models/linear_model.pkl")
        item = sc_preprocessing(item)
        try:
            prediction = model.predict(item)
            return  {"prediction log10": prediction[0],
                        "final prediction": 10**prediction[0]}
        except Exception as e:
            return {"prediction": "Erro na predição: " + str(e)}
    
    elif model_name == "both":
        cat_model = CatBoostRegressor()
        cat_model.load_model(f"{pathlib.Path.cwd().parent}/models/catboost_model.cbm")

        ridge_model = joblib.load(f"{pathlib.Path.cwd().parent}/models/linear_model.pkl")

        item = sc_preprocessing(item)

        try:
            prediction_cat = cat_model.predict(item)
            prediction_ridge = ridge_model.predict(item)
            return  {"prediction catboost log10": prediction_cat[0],
                     "prediction ridge log10": prediction_ridge[0],
                     "prediction catboost": 10**prediction_cat[0],
                     "prediction ridge": 10**prediction_ridge[0],
                     "prediction log10": (prediction_cat[0] + prediction_ridge[0])/2,
                     "final prediction": 10**((prediction_cat[0] + prediction_ridge[0])/2)}
        except Exception as e:
            return {"prediction": "Erro na predição: " + str(e)}
    
    else:
        raise HTTPException(status_code=404, detail="Modelo não encontrado.")
