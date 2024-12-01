import datetime
import gc
import os
import warnings
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import dill
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import polars as pl
import torch
import xgboost as xgb

from config import ModelConfig

warnings.filterwarnings('ignore')

class BaseModel:
    """Base model class for easy extension"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._register_custom_metrics()
    
    def _register_custom_metrics(self):
        """Register custom metrics if needed"""
        pass
    
    def fit(self, train_data: Tuple[np.ndarray, np.ndarray], 
           val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            dill.dump(self.model, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = dill.load(f)

class LightGBMModel(BaseModel):
    def _register_custom_metrics(self):
        """Register custom metrics for LightGBM"""
        # Instead of registering metrics directly, we'll add them to params
        if self.config.custom_metrics:
            self.config.params['metric'] = list(self.config.custom_metrics.keys())
    
    def fit(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray], 
           val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None):
        train_X, train_y, train_w = train_data
        print(f"\nMemory usage before training:")
        print(f"train_X: {train_X.nbytes / 1024 / 1024 / 1024:.2f} GB")
        print(f"train_y: {train_y.nbytes / 1024 / 1024 / 1024:.2f} GB")
        if train_w is not None:
            print(f"train_w: {train_w.nbytes / 1024 / 1024 / 1024:.2f} GB")
        
        train_set = lgb.Dataset(train_X, train_y, weight=train_w, free_raw_data=False, params={'feature_pre_filter': False})
        
        del train_X, train_y, train_w, train_data
        gc.collect()
        
        val_set = None
        if val_data is not None:
            val_X, val_y, val_w = val_data
            print(f"\nValidation data memory usage:")
            print(f"val_X: {val_X.nbytes / 1024 / 1024 / 1024:.2f} GB")
            print(f"val_y: {val_y.nbytes / 1024 / 1024 / 1024:.2f} GB")
            if val_w is not None:
                print(f"val_w: {val_w.nbytes / 1024 / 1024 / 1024:.2f} GB")
            val_set = lgb.Dataset(val_X, val_y, weight=val_w, free_raw_data=False, params={'feature_pre_filter': False})
            
            del val_X, val_y, val_w, val_data
            gc.collect()
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            self.config.params,
            train_set,
            num_boost_round=1000,
            valid_sets=[val_set] if val_set else None,
            valid_names=['valid'] if val_set else None,
            callbacks=callbacks
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class XGBoostModel(BaseModel):
    def _register_custom_metrics(self):
        """Register custom metrics for XGBoost"""
        if self.config.custom_metrics:
            self.config.params['custom_metric'] = list(self.config.custom_metrics.values())
    
    def fit(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray], 
           val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None):
        """Train XGBoost model"""
        train_X, train_y, train_w = train_data
        train_set = xgb.DMatrix(train_X, train_y, weight=train_w)
        
        del train_X, train_y, train_w
        gc.collect()
        
        watchlist = [(train_set, 'train')]
        if val_data is not None:
            val_X, val_y, val_w = val_data
            val_set = xgb.DMatrix(val_X, val_y, weight=val_w)
            watchlist.append((val_set, 'valid'))
            
            del val_X, val_y, val_w
            gc.collect()
        
        self.model = xgb.train(
            self.config.params,
            train_set,
            num_boost_round=1000,
            evals=watchlist,
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        del train_set
        if val_data is not None:
            del val_set
        gc.collect()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(xgb.DMatrix(X))

class NeuralNetworkModel(BaseModel):
    def _register_custom_metrics(self):
        # PyTorch/TensorFlow에서는 metrics를 모델 컴파일 시 등록
        self.metrics = list(self.config.custom_metrics.values())
    
    def fit(self, train_data: Tuple[np.ndarray, np.ndarray], 
           val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        train_X, train_y = train_data
        
        # PyTorch example
        self.model = torch.nn.Sequential(
            torch.nn.Linear(train_X.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=self.config.params.get('learning_rate', 0.001))
        
        # Training loop with custom metrics
        for epoch in range(self.config.params.get('epochs', 10)):
            self.model.train()
            # ... training implementation ...
            
            if val_data is not None:
                self.model.eval()
                # ... validation implementation ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            return self.model(X_tensor).numpy()

class EnsembleModel(BaseModel):
    def __init__(self, config: "ModelConfig", models: List[BaseModel]):
        super().__init__(config)
        self.models = models
    
    def _register_custom_metrics(self):
        # 각 모델의 custom metrics는 이미 등록되어 있음
        pass
    
    def fit(self, train_data: Tuple[np.ndarray, np.ndarray], 
           val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        for model in self.models:
            model.fit(train_data, val_data)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = [model.predict(X) for model in self.models]
        # Weighted average if weights are specified in config
        weights = self.config.params.get('weights', None)
        if weights is not None:
            return np.average(predictions, axis=0, weights=weights)
        return np.mean(predictions, axis=0)
