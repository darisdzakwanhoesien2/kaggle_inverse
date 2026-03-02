from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def build_stacking_model():

    base_models = [
        ("lgbm", LGBMRegressor(n_estimators=500)),
        ("cat", CatBoostRegressor(verbose=0)),
        ("xgb", XGBRegressor(n_estimators=500, verbosity=0))
    ]

    meta_model = Ridge()

    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    return stack


def evaluate_stacking(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )
    return -scores.mean(), scores.std()