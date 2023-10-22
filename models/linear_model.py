import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import importlib
import pathlib
import joblib

df = pd.read_parquet(f'{pathlib.Path.cwd().parent}/fetch_data/data_final.parquet')
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

linear = LinearRegression()
linear.fit(X, y)

joblib.dump(linear, 'linear_model.pkl')