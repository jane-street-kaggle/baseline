import datetime
import gc
import os
from typing import Callable, Optional

import dill # type: ignore
import numpy as np
import polars as pl

from config import Config
from constant import IS_KAGGLE
from data import DataHandler
from kaggle_handler import KaggleHandler
from metrics import r2_metric
from model import BaseModel, LightGBMModel, XGBoostModel
from optimizer import OptimizationHandler
from spliter import PurgedGroupTimeSeriesSplit, TimeBasedSplit, TimeSeriesKFold


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_handler = DataHandler(config)
        self.model = self._get_model()
        self.kaggle_handler = KaggleHandler(config) if not IS_KAGGLE else None
    
    def _get_model(self) -> BaseModel:
        """Get model instance based on config"""
        model_map = {
            'lightgbm': LightGBMModel,
            'xgboost': XGBoostModel,
            # 'neural_network': NeuralNetworkModel,
        }
        model_class = model_map.get(self.config.model.name)
        if model_class is None:
            raise ValueError(f"Unknown model: {self.config.model.name}")
        return model_class(self.config.model)
    
    def train(self, preprocessor: Optional[Callable] = None, 
            feature_generator: Optional[Callable] = None,
            optimize: bool = False,
            n_trials: int = 100) -> pl.DataFrame:
        """Train pipeline and return holdout test set"""
        # Load and prepare data
        print("Loading and preparing data...")
        _, test_df = self.data_handler.load_data()
        
        # Split data using configured strategy
        print("Splitting data using configured strategy...")
        train_range, holdout_test_range = self.config.split_strategy.get_holdout_test(self.data_handler.date_ranges)
        splits = self.config.split_strategy.split(train_range)
        self.config.split_strategy.visualize_splits(train_range)
        
        best_model = None
        best_score = float('-inf')
        
        # Train and validate on each split
        for i, (train_dates, val_dates) in enumerate(splits):
            print(f"\nTraining fold {i+1}/{len(splits)}")
           
            # Prepare data for training and validation
            # do we have to hand over preprocessor and feature_generator here? just init it in the class would be better
            train_df = self.data_handler.prepare_data(train_dates, preprocessor, feature_generator)
            val_df = self.data_handler.prepare_data(val_dates, preprocessor, feature_generator)
            
            # X, y, w
            train_data = self.data_handler.get_feature_data(train_df)
            val_data = self.data_handler.get_feature_data(val_df)

            # Clear memory
            del train_df, val_df
            gc.collect()

            # Create new model instance for each fold
            fold_model = self._get_model()

            # Optionally run optimization (only on first fold)
            if optimize and i == 0:
                print("Running hyperparameter optimization...")
                optimizer = OptimizationHandler(self.config, type(fold_model))
                best_params = optimizer.optimize(
                    (train_data[0], train_data[1]), # type: ignore
                    (val_data[0], val_data[1]), # type: ignore
                    n_trials
                )
                if self.config.model.params is None:
                    raise ValueError("Model parameters are not initialized")
                self.config.model.params.update(best_params)
                # Recreate model with optimized parameters
                del fold_model
                gc.collect()
                fold_model = self._get_model()
            
            # Train model
            print("Training model...")
            if train_data is None:
                raise ValueError("Training data is not provided")
            if val_data is None:
                raise ValueError("Validation data is not provided")
            fold_model.fit(
                (train_data[0], train_data[1]), # type: ignore
                (val_data[0], val_data[1]) # type: ignore
            )
            
            # Evaluate on validation set using R2
            val_X, val_y, val_w = val_data
            val_pred = fold_model.predict(val_X)
            _, val_score, _ = r2_metric(val_y, val_pred, val_w)
            print(f"Validation R2 score for fold {i+1}: {val_score:.4f}")
            
            # Keep track of best model
            # if need to transfer learning, we need to keep the best model and continue training
            # or if we want to ensemble fold models, we need to keep all models
            if val_score > best_score:
                best_score = val_score
                if best_model is not None:
                    del best_model
                    gc.collect()
                best_model = fold_model
            else:
                del fold_model
                gc.collect()

            # Clear fold data
            del train_data, val_data, val_X, val_y, val_w, val_pred
            gc.collect()
        # Use best model for final predictions
        if best_model is None:
            raise ValueError("No valid model was found during training")
        self.model = best_model
        print(f"\nBest validation score: {best_score:.4f}")
        
        # Save pipeline
        print("\nSaving pipeline...")
        self.save()

        # After finding best model, generate features for holdout test
        holdout_test_df = self.data_handler.prepare_data(holdout_test_range, preprocessor, feature_generator) 
        
        # Clear any remaining memory
        gc.collect()

        return holdout_test_df

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        print("Starting prediction...")
        print(f"Available features: {self.data_handler.features}")
        print(f"X shape: {X.shape}")

        predictions = self.model.predict(X)

        print(f"Predictions shape: {predictions.shape}")

        print("Feature names:", self.data_handler.features)
        print("First row of features:", X[0])  # 또는 holdout_test_X[0]
        print("Feature matrix shape:", X.shape)  # 또는 holdout_test_X.shape
        
        return predictions

    def save(self):
        """Save pipeline with detailed logging and version tracking"""
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        
        # Verify model state
        if not hasattr(self.model, 'config') or not hasattr(self.model, 'model'):
            raise ValueError("Model appears to be uninitialized or invalid")

        # Create detailed config dictionary
        config_dict = {
            'model': {
                'name': self.config.model.name,
                'params': dict(self.config.model.params),
                'custom_metrics': {k: v.__name__ for k, v in self.config.model.custom_metrics.items()}
            },
            'paths': {
                'model_path': self.config.model_path,
                'dataset_name': self.config.dataset_name
            },
            'seed': self.config.seed
        }

        # Update split strategy section in config_dict
        split_strategy_params = {
            'test_ratio': self.config.split_strategy.test_ratio
        }
        
        # Add strategy-specific parameters
        if isinstance(self.config.split_strategy, TimeBasedSplit):
            split_strategy_params['train_ratio'] = self.config.split_strategy.train_ratio
        elif isinstance(self.config.split_strategy, TimeSeriesKFold):
            split_strategy_params['n_splits'] = self.config.split_strategy.n_splits
        elif isinstance(self.config.split_strategy, PurgedGroupTimeSeriesSplit):
            split_strategy_params.update({
                'n_splits': self.config.split_strategy.n_splits,
                'max_train_group_size': self.config.split_strategy.max_train_group_size,
                'max_test_group_size': self.config.split_strategy.max_test_group_size,
                'group_gap': self.config.split_strategy.group_gap
            })
        
        config_dict['split_strategy'] = {
            'type': type(self.config.split_strategy).__name__,
            'params': split_strategy_params
        }

        # Create pipeline metadata with version information
        pipeline_data = {
            'model': self.model,
            'config': config_dict,
            'data_handler': {
                'preprocessor': self.data_handler.preprocessor,
                'preprocessor_version': getattr(self.data_handler.preprocessor, 'version', 'unknown'),
                'preprocessor_description': getattr(self.data_handler.preprocessor, 'description', ''),
                'feature_generator': self.data_handler.feature_generator,
                'feature_generator_version': getattr(self.data_handler.feature_generator, 'version', 'unknown'),
                'feature_generator_description': getattr(self.data_handler.feature_generator, 'description', ''),
                'features': self.data_handler.features,
                'n_features': len(self.data_handler.features)
            },
            'version': '1.0',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Print pipeline information
        print("\n" + "="*50)
        print("Pipeline Information:")
        print("="*50)
        
        print(f"\n1. Model Configuration:")
        print(f"   - Model Type: {config_dict['model']['name']}")
        print(f"   - Number of Features: {len(self.data_handler.features)}")
        print(f"   - Model Parameters:")
        for k, v in config_dict['model']['params'].items():
            print(f"     * {k}: {v}")
        
        print(f"\n2. Data Processing:")
        print(f"   - Preprocessor: {self.data_handler.preprocessor.__name__ if self.data_handler.preprocessor else 'None'}")
        print(f"     * Version: {getattr(self.data_handler.preprocessor, 'version', 'unknown')}")
        print(f"     * Description: {getattr(self.data_handler.preprocessor, 'description', '')}")
        print(f"   - Feature Generator: {self.data_handler.feature_generator.__name__ if self.data_handler.feature_generator else 'None'}")
        print(f"     * Version: {getattr(self.data_handler.feature_generator, 'version', 'unknown')}")
        print(f"     * Description: {getattr(self.data_handler.feature_generator, 'description', '')}")
        print(f"   - First 5 Features: {self.data_handler.features[:5]}")
        
        print(f"\n3. Split Strategy:")
        print(f"   - Type: {config_dict['split_strategy']['type']}")
        for k, v in config_dict['split_strategy']['params'].items():
            print(f"   - {k}: {v}")
        
        print(f"\n4. Save Location:")
        print(f"   - Path: {self.config.model_path}")
        print(f"   - Dataset Name: {self.config.dataset_name}")
        print(f"   - Timestamp: {pipeline_data['timestamp']}")
        
        try:
            with open(self.config.model_path, 'wb') as f:
                dill.dump(pipeline_data, f)
            print("\nPipeline saved successfully! ✓")
        except Exception as e:
            print(f"\nError saving pipeline: {str(e)}")
            raise

    def load(self, path: Optional[str] = None):
        """Load pipeline with extensive validation and version tracking"""
        load_path = path if path is not None else self.config.model_path
        
        print("\n" + "="*50)
        print("Loading Pipeline:")
        print("="*50)
        
        try:
            with open(load_path, 'rb') as f:
                pipeline_data = dill.load(f)
            
            # Phase 1: Structure Validation
            print("\nPhase 1: Structure Validation")
            print("-" * 30)
            
            required_components = ['model', 'config', 'data_handler', 'version']
            missing = [comp for comp in required_components if comp not in pipeline_data]
            if missing:
                raise ValueError(f"Missing required components in pipeline: {missing}")
            print("✓ Basic structure validation passed")
            
            required_data_handler = ['preprocessor', 'feature_generator', 'features', 'n_features',
                                'preprocessor_version', 'feature_generator_version']
            missing_dh = [comp for comp in required_data_handler if comp not in pipeline_data['data_handler']]
            if missing_dh:
                raise ValueError(f"Missing data handler components: {missing_dh}")
            print("✓ Data handler structure validation passed")
            
            # Phase 2: Model and Config Reconstruction
            print("\nPhase 2: Model and Config Reconstruction")
            print("-" * 30)
            
            self.model = pipeline_data['model']
            config_dict = pipeline_data['config']
            
            # Reconstruct split strategy
            split_strategy_type = config_dict['split_strategy']['type']
            split_params = config_dict['split_strategy']['params']
            
            if split_strategy_type == 'TimeBasedSplit':
                self.config.split_strategy = TimeBasedSplit(
                    train_ratio=split_params.get('train_ratio', 0.75),
                    test_ratio=split_params.get('test_ratio', 0.2)
                )
            elif split_strategy_type == 'TimeSeriesKFold':
                self.config.split_strategy = TimeSeriesKFold(
                    n_splits=split_params.get('n_splits', 5),
                    test_ratio=split_params.get('test_ratio', 0.2)
                )
            elif split_strategy_type == 'PurgedGroupTimeSeriesSplit':
                self.config.split_strategy = PurgedGroupTimeSeriesSplit(
                    n_splits=split_params.get('n_splits', 5),
                    max_train_group_size=split_params.get('max_train_group_size', np.inf),
                    max_test_group_size=split_params.get('max_test_group_size', np.inf),
                    group_gap=split_params.get('group_gap', None),
                    test_ratio=split_params.get('test_ratio', 0.2)
                )
            else:
                raise ValueError(f"Unknown split strategy type: {split_strategy_type}")
            
            print("✓ Split strategy reconstruction passed")
            
            # Model validation
            required_model_attrs = ['predict', 'model', 'config']
            missing_attrs = [attr for attr in required_model_attrs if not hasattr(self.model, attr)]
            if missing_attrs:
                raise AttributeError(f"Model missing required attributes: {missing_attrs}")
            print("✓ Model attributes validation passed")
            
            # Test model with dummy data
            try:
                n_features = len(pipeline_data['data_handler']['features'])
                dummy_input = np.random.random((5, n_features))
                dummy_pred = self.model.predict(dummy_input)
                if not isinstance(dummy_pred, np.ndarray):
                    raise TypeError(f"Model prediction returned {type(dummy_pred)}, expected numpy.ndarray")
                if len(dummy_pred.shape) != 1 or len(dummy_pred) != 5:
                    raise ValueError(f"Unexpected prediction shape: {dummy_pred.shape}, expected (5,)")
                print("✓ Model prediction test passed")
            except Exception as e:
                raise RuntimeError(f"Model prediction test failed: {str(e)}")
            
            # Phase 3: Function Version Validation
            print("\nPhase 3: Function Version Validation")
            print("-" * 30)
            
            self.data_handler.preprocessor = pipeline_data['data_handler']['preprocessor']
            self.data_handler.feature_generator = pipeline_data['data_handler']['feature_generator']
            self.data_handler.features = pipeline_data['data_handler']['features']
            
            # Version validation
            preprocessor_version = getattr(self.data_handler.preprocessor, 'version', 'unknown')
            saved_preprocessor_version = pipeline_data['data_handler']['preprocessor_version']
            if preprocessor_version != saved_preprocessor_version:
                print(f"⚠️ Warning: Current preprocessor version ({preprocessor_version}) "
                    f"differs from saved version ({saved_preprocessor_version})")
            
            feature_gen_version = getattr(self.data_handler.feature_generator, 'version', 'unknown')
            saved_feature_gen_version = pipeline_data['data_handler']['feature_generator_version']
            if feature_gen_version != saved_feature_gen_version:
                print(f"⚠️ Warning: Current feature generator version ({feature_gen_version}) "
                    f"differs from saved version ({saved_feature_gen_version})")
            
            # Phase 4: Data Handler Function Validation
            print("\nPhase 4: Data Handler Function Validation")
            print("-" * 30)
            
            # Validate preprocessor
            if self.data_handler.preprocessor:
                try:
                    dummy_df = pl.DataFrame({
                        'time_id': np.arange(5),
                        'symbol_id': np.ones(5),
                        'weight': np.ones(5),
                        **{f'feature_{i:02d}': np.random.random(5) for i in range(79)}
                    })
                    processed_df = self.data_handler.preprocessor(dummy_df)
                    if not isinstance(processed_df, pl.DataFrame):
                        raise TypeError(f"Preprocessor returned {type(processed_df)}, expected polars.DataFrame")
                    print("✓ Preprocessor function test passed")
                except Exception as e:
                    raise RuntimeError(f"Preprocessor function test failed: {str(e)}")
            
            # Validate feature generator
            if self.data_handler.feature_generator:
                try:
                    dummy_df = pl.DataFrame({
                        'time_id': np.arange(5),
                        'symbol_id': np.ones(5),
                        'weight': np.ones(5),
                        **{f'feature_{i:02d}': np.random.random(5) for i in range(79)}
                    })
                    generated_df = self.data_handler.feature_generator(dummy_df)
                    if not isinstance(generated_df, pl.DataFrame):
                        raise TypeError(f"Feature generator returned {type(generated_df)}, expected polars.DataFrame")
                    print("✓ Feature generator function test passed")
                except Exception as e:
                    raise RuntimeError(f"Feature generator function test failed: {str(e)}")
            
            # Validate features list
            if not self.data_handler.features:
                raise ValueError("Features list is empty")
            if not all(isinstance(f, str) for f in self.data_handler.features):
                raise TypeError("All feature names must be strings")
            if len(self.data_handler.features) != len(set(self.data_handler.features)):
                raise ValueError("Duplicate feature names found")
            print("✓ Features list validation passed")
            
            # Phase 5: Pipeline Information
            print("\n" + "="*50)
            print("Pipeline Information:")
            print("="*50)
            
            print(f"\n1. Model Configuration:")
            print(f"   - Model Type: {config_dict['model']['name']}")
            print(f"   - Number of Features: {pipeline_data['data_handler']['n_features']}")
            print(f"   - Model Parameters:")
            for k, v in config_dict['model']['params'].items():
                print(f"     * {k}: {v}")
            
            if self.data_handler.preprocessor is None:
                raise ValueError("Preprocessor function is not defined")
            if self.data_handler.feature_generator is None:
                raise ValueError("Feature generator function is not defined")
            
            print(f"\n2. Data Processing:")
            print(f"   - Preprocessor: {self.data_handler.preprocessor.__name__ if self.data_handler.preprocessor else 'None'}")
            print(f"     * Version: {saved_preprocessor_version}")
            print(f"     * Description: {pipeline_data['data_handler']['preprocessor_description']}")
            print(f"   - Feature Generator: {self.data_handler.feature_generator.__name__ if self.data_handler.feature_generator else 'None'}")
            print(f"     * Version: {saved_feature_gen_version}")
            print(f"     * Description: {pipeline_data['data_handler']['feature_generator_description']}")
            print(f"   - First 5 Features: {self.data_handler.features[:5]}")
            
            print(f"\n3. Split Strategy:")
            print(f"   - Type: {config_dict['split_strategy']['type']}")
            for k, v in config_dict['split_strategy']['params'].items():
                print(f"   - {k}: {v}")
            
            print(f"\n4. Load Location:")
            print(f"   - Path: {load_path}")
            print(f"   - Dataset Name: {config_dict['paths']['dataset_name']}")
            print(f"   - Original Save Timestamp: {pipeline_data['timestamp']}")
            
            # Final validation: Complete pipeline test
            try:
                dummy_features = {f: np.random.random(5) for f in self.data_handler.features}
                dummy_df = pl.DataFrame({
                    **dummy_features,
                    'weight': np.ones(5)
                })
                X, _, _ = self.data_handler.get_feature_data(dummy_df)
                final_pred = self.model.predict(X)
                print("\n✓ Complete pipeline test passed")
            except Exception as e:
                raise RuntimeError(f"Complete pipeline test failed: {str(e)}")
            
            print("\nPipeline loaded and validated successfully! ✓")
            
        except Exception as e:
            print(f"\nError loading pipeline: {str(e)}")
            raise

    def upload_to_kaggle(self, dataset_title: Optional[str] = None):
        """Upload this pipeline to Kaggle dataset"""
        if self.kaggle_handler is not None:
            self.kaggle_handler.upload_pipeline(self, dataset_title)
        else:
            print("KaggleHandler is not initialized. Cannot upload to Kaggle.")