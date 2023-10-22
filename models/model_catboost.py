import pandas as pd
import numpy as np
import importlib
import pathlib

# Verificar se o módulo CatBoost está instalado
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

df = pd.read_parquet(f'{pathlib.Path.cwd().parent}/fetch_data/data_final.parquet')
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

cat = CatBoostRegressor(random_state=42, verbose=0, iterations=20000, learning_rate=0.025)
cat.fit(X, y)

cat.save_model('catboost_model.cbm')