from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def get_models():
    return {
        "Ridge": Ridge(),
        "LightGBM": LGBMRegressor(n_estimators=500),
        "CatBoost": CatBoostRegressor(verbose=0),
        "XGBoost": XGBRegressor(n_estimators=500, verbosity=0)
    }

def evaluate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )
    return -scores.mean(), scores.std()

def train_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    return -scores.mean(), scores.std()

# def get_models():
#     return {
#         "Ridge": Ridge(),
#         "LightGBM": LGBMRegressor(n_estimators=500),
#         "CatBoost": CatBoostRegressor(verbose=0)
#     }