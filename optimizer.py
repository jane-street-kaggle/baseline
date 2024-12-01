from typing import Any, Dict, Tuple

import numpy as np
import optuna

from config import Config


class OptimizationHandler:
    def __init__(self, config: Config, model_class: type):
        self.config = config
        self.model_class = model_class
    
    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define search space for each model type"""
        if self.config.model.name == 'lightgbm':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 96),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
        """
        elif self.config.model.name == 'xgboost':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            }
        """
        return {}
    
    def objective(self, trial: optuna.Trial, train_data: Tuple[np.ndarray, np.ndarray], 
                 val_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Optimization objective"""
        params = self.get_search_space(trial)
        self.config.model.params.update(params)
        
        model = self.model_class(self.config.model)
        model.fit(train_data, val_data)
        
        val_X, val_y = val_data
        predictions = model.predict(val_X)
        
        return np.mean((predictions - val_y) ** 2) ** 0.5
    
    def optimize(self, train_data: Tuple[np.ndarray, np.ndarray], 
                val_data: Tuple[np.ndarray, np.ndarray], n_trials: int = 100) -> Dict[str, Any]:
        """Run optimization"""
        study = optuna.create_study(direction='minimize')
        objective = lambda trial: self.objective(trial, train_data, val_data)
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best score: {study.best_value:.4f}")
        print("Best params:", study.best_params)
        
        return study.best_params