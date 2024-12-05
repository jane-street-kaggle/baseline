import os

import polars as pl

from config import Config, ModelConfig
from constant import BASE_PATH, IS_KAGGLE
from function_wrappers import versioned_function
from metrics import r2_metric
from pipeline import Pipeline
from spliter import PurgedGroupTimeSeriesSplit
from utility import default_preprocessor, predict, run_inference_only
import numpy as np

INFERENCE_ONLY = False  # True: inference만 실행, False: 학습 포함
OPTIMIZE_HYPERPARAMS = False  # True: 하이퍼파라미터 최적화 실행
NICKNAME = "alvinlee9"  # Kaggle nickname
BASE_DATASET_NAME = "jane-street-model-v3"  # Base dataset name
DATASET_NAME = f"{BASE_DATASET_NAME}" if IS_KAGGLE else f"{NICKNAME}/{BASE_DATASET_NAME}"



@versioned_function("1.1.0", "Added time-based features and symbol_id processing")
def default_feature_generator(df: pl.DataFrame) -> pl.DataFrame:
    """Feature generation with time-based features"""
    # Add time-based features using polars expressions
    result = df.with_columns([
        (2 * np.pi * pl.col('time_id') / 967).sin().alias('feature_sin_time_id'),
        (2 * np.pi * pl.col('time_id') / 967).cos().alias('feature_cos_time_id'),
        (2 * np.pi * pl.col('time_id') / 483).sin().alias('feature_sin_time_id_halfday'),
        (2 * np.pi * pl.col('time_id') / 483).cos().alias('feature_cos_time_id_halfday')
    ])

    # Fill NA values and rename columns
    result = (result
    .fill_null(-1)
    .rename({
        'symbol_id': 'feature_symbol_id',
        'weight': 'feature_weight'
    }))

    # Select and reorder columns
    feature_cols = ['feature_symbol_id', 'feature_sin_time_id', 'feature_cos_time_id',
                    'feature_sin_time_id_halfday', 'feature_cos_time_id_halfday', 'feature_weight']
    feature_cols.extend([f'feature_0{i}' if i < 10 else f'feature_{i}'
                         for i in range(79)])

    # Add target column if it exists
    if 'responder_6' in result.columns:
        feature_cols.insert(0, 'responder_6')

    return result.select(feature_cols) 

if INFERENCE_ONLY and IS_KAGGLE:
    # Inference only mode
    print("Running in inference-only mode...")
    pipeline = run_inference_only(DATASET_NAME)
else:
    # Training mode
    # config = Config(
    #     # partition_range=[6,7,8,9],
    #     model=ModelConfig(
    #         name='lightgbm',
    #         params={
    #             'objective': 'regression_l2',
    #             'metric': 'rmse',
    #             'boosting_type': 'gbdt',
    #             'learning_rate': 0.1,
    #             'random_state': 42,
    #             'verbose': 1,
    #             'device': 'cpu',
    #         },
    #         custom_metrics={},
    #     ),
    #     dataset_name=DATASET_NAME,
    #     split_strategy=PurgedGroupTimeSeriesSplit(
    #         # n_splits=5,
    #         # group_gap=15,
    #         # max_train_group_size=200,
    #         # max_test_group_size=50,
    #         # test_ratio=0.2,
    #         n_splits=5,
    #         group_gap=50,
    #         max_train_group_size=400,
    #         max_test_group_size=200,
    #         test_ratio=0.2,
    #     ),
    #     seed=42
    # )
    config = Config(
        # partition_range=[6,7,8,9],
        model=ModelConfig(
            name='xgboost',
            params={
                'objective': 'reg:squarederror',  # regression task
                'eval_metric': 'rmse',            # evaluation metric
                'booster': 'gbtree',             # use tree booster
                'learning_rate': 0.1,            # eta
                'max_depth': 6,                  # maximum tree depth
                'min_child_weight': 1,           # minimum sum of instance weight in a child
                'subsample': 0.8,                # sampling ratio of training instances
                'colsample_bytree': 0.8,         # sampling ratio of columns when constructing each tree
                'colsample_bylevel': 0.8,        # sampling ratio of columns for each level
                'lambda': 1,                     # L2 regularization
                'alpha': 0,                      # L1 regularization
                'tree_method': 'hist',           # use histogram-based algorithm
                'random_state': 42,
                'n_jobs': -1,                    # use all CPU cores
                'verbosity': 1,
            },
            custom_metrics={},
        ),
        dataset_name=DATASET_NAME,
        split_strategy=PurgedGroupTimeSeriesSplit(
            n_splits=5,
            group_gap=50,
            max_train_group_size=400,
            max_test_group_size=200,
            test_ratio=0.2,
        ),
        seed=42
    )
    
    pipeline = Pipeline(config)
    
    if not IS_KAGGLE:
        # Local training
        print("Training model locally...")
        holdout_test = pipeline.train(
            preprocessor=default_preprocessor,
            feature_generator=default_feature_generator,
            optimize=OPTIMIZE_HYPERPARAMS,
            n_trials=100 if OPTIMIZE_HYPERPARAMS else None # type: ignore
        )

        print("\nUploading pipeline to Kaggle...")
        pipeline.upload_to_kaggle()

        # Evaluate on holdout test set using R2
        print("\nEvaluating on holdout test set...")
        holdout_test_X, holdout_test_y, holdout_test_w = pipeline.data_handler.get_feature_data(holdout_test)
        holdout_test_pred = pipeline.predict(holdout_test_X)
        
        # Calculate R2 score
        _, r2_score, _ = r2_metric(holdout_test_y, holdout_test_pred, holdout_test_w)
        print(f"Holdout test R2 score: {r2_score:.4f}")
        
        # Predict on competition test set if available
        if pipeline.data_handler.test_data is not None:
            print("\nPredicting on competition test set...")
            raw_test_data = pipeline.data_handler.test_data
            test_data = pipeline.data_handler._process_and_generate_features(raw_test_data)
            print(f"Test data shape: {test_data.shape}")
            _test_data = test_data.to_numpy()
            test_pred = pipeline.predict(_test_data)  
            print(f"Test predictions shape: {test_pred.shape}")
            print(test_pred)
    else:
        # Kaggle training
        print("Training model in Kaggle environment...")
        pipeline.train(
            preprocessor=default_preprocessor,
            feature_generator=default_feature_generator,
            optimize=False
        )

if IS_KAGGLE:
    import kaggle_evaluation.jane_street_inference_server # type: ignore

    if not 'pipeline' in globals():  # pipeline이 아직 정의되지 않은 경우  # noqa: E713
        # Inference only mode로 가정하고 모델 로드
        pipeline = run_inference_only(DATASET_NAME)
    
    print("Setting up for competition submission...")
    inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(
        predict
    )
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("Starting inference server...")
        inference_server.serve()
    else:
        print("Running local gateway...")
        inference_server.run_local_gateway(
            (f'{BASE_PATH}/test.parquet', f'{BASE_PATH}/lags.parquet')
        )


# Custom preprocessing and feature generation example
def my_preprocessor(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col('weight').fill_null(pl.col('weight').mean()),
        pl.col('feature_00').clip(-3, 3),
        pl.col('feature_01').clip(-3, 3),
    ])

def my_feature_generator(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # Moving statistics
        pl.col('feature_00').rolling_mean(window_size=10).alias('feature_00_ma10'),
        pl.col('feature_00').rolling_std(window_size=10).alias('feature_00_std10'),
        
        # Feature interactions
        (pl.col('feature_02') / (pl.col('feature_03') + 1e-7)).alias('feature_ratio_02_03'),
        
        # Group statistics
        pl.col('feature_00').mean().over('symbol_id').alias('feature_00_symbol_mean'),
    ])

# Run custom experiment
config = Config()
pipeline = Pipeline(config)
pipeline.train(
    preprocessor=my_preprocessor,
    feature_generator=my_feature_generator,
    optimize=True
)