import os
import pandas as pd
import numpy as np
import polars as pl
import joblib

from jane_kaggle.constant import FEAT_COLS
from kaggle_evaluation.jane_street_inference_server import JSInferenceServer


lags_ : pl.DataFrame | None = None

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    global lags_
    if lags is not None:
        lags_ = lags

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6')
    )
    
    feat = test[FEAT_COLS].to_pandas()

    import xgboost as xgb
    dmatrix_feat = xgb.DMatrix(feat)

    model = models[0]
    pred = model.predict(dmatrix_feat)
    
    # Remove the mean to retain per-row predictions
    pred = pred.ravel()

    # Optional: Validate the prediction length
    assert len(pred) == len(test), "Mismatch between predictions and test set."

    predictions = predictions.with_columns(pl.Series('responder_6', pred))

    print(predictions)
    
    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    
    assert list(predictions.columns) == ['row_id', 'responder_6']
    assert len(predictions) == len(test)
    
    return predictions

model_path = 'models/'
model_name = 'xgb_model_fold4'
models = []

# Load the XGBoost model from a JSON file
import xgboost as xgb
model = xgb.Booster()
model.load_model(f'{model_path}/{model_name}.json')
models.append(model)

print(f"Loaded XGBoost model from the saved JSON file.")

inference_server = JSInferenceServer(predict) # type: ignore

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-realtime-marketdata-forecasting/test.parquet',
            '/kaggle/input/jane-street-realtime-marketdata-forecasting/lags.parquet',
        )
    )
