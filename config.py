from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from constant import MODEL_PATH
from spliter import SplitStrategy, TimeBasedSplit


@dataclass
class ModelConfig:
    name: str = 'lightgbm'
    params: Optional[Dict[str, Any]] = None
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.params is None:
            self.params = self.get_default_params()
    
    def get_default_params(self) -> Dict[str, Any]:
        params = {
            'lightgbm': {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
            },
            'xgboost': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'verbosity': 0,
            },
            'neural_network': {
                'learning_rate': 0.001,
                'batch_size': 512,
                'epochs': 10,
            }
        }
        return params.get(self.name, {}) # type: ignore

@dataclass
class Config:
    # Model
    model: ModelConfig = ModelConfig()
    # Paths
    model_path: str = f"{MODEL_PATH}/pipeline.pkl"
    dataset_name: str = "jane-street-model"
    # Data loading
    partition_range: Optional[List[int]] = None
    # Training
    split_strategy: SplitStrategy = field(default_factory=lambda: TimeBasedSplit(train_ratio=0.75, test_ratio=0.2))
    seed: int = 42
    
    def __post_init__(self):
        np.random.seed(self.seed)