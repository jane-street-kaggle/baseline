import numpy as np
import pandas as pd
import polars as pl

from config import Config
from constant import IS_KAGGLE
from function_wrappers import versioned_function
from pipeline import Pipeline


def reduce_memory(df: pl.DataFrame) -> pl.DataFrame:
    """Optimize data types for memory usage in Polars"""
    start_mem = df.estimated_size() / (1024**2)
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            c_min = df[col].drop_nulls().min()
            c_max = df[col].drop_nulls().max()
            
            if c_min is not None and c_max is not None:  # null check 추가
                if col_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: # noqa: E501 
                        df = df.with_columns(pl.col(col).cast(pl.Int8))
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:  # noqa: E501
                        df = df.with_columns(pl.col(col).cast(pl.Int16))
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:  # noqa: E501
                        df = df.with_columns(pl.col(col).cast(pl.Int32))
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:  # noqa: E501
                        df = df.with_columns(pl.col(col).cast(pl.Int64))
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df = df.with_columns(pl.col(col).cast(pl.Float32))
                    else:
                        df = df.with_columns(pl.col(col).cast(pl.Float64))
        
        elif col_type == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
    
    end_mem = df.estimated_size() / (1024**2)
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

@versioned_function("1.0.0", "Initial preprocessor with basic cleaning")
def default_preprocessor(df: pl.DataFrame) -> pl.DataFrame:
    """Default preprocessing function"""
    df = reduce_memory(df)
    return df


def predict(pipeline: Pipeline, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Competition prediction function"""

    row_ids = test['row_id'].to_numpy()

    if pipeline.data_handler.preprocessor:
        test = pipeline.data_handler.preprocessor(test)
    if pipeline.data_handler.feature_generator:
        test = pipeline.data_handler.feature_generator(test)

    test_X, _, _ = pipeline.data_handler.get_feature_data(test)
    predictions = pipeline.predict(test_X)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    result = pl.DataFrame({
        'row_id': row_ids,
        'responder_6': predictions
    })
    
    # Validation checks
    assert isinstance(result, (pl.DataFrame, pd.DataFrame))
    assert result.columns == ['row_id', 'responder_6']
    assert len(result) == len(test)
    
    return result

def run_inference_only(dataset_name: str, model_filename: str = 'pipeline.pkl') -> Pipeline:
    """Kaggle dataset에서 모델 로드하고 inference 준비"""
    if not IS_KAGGLE:
        raise ValueError("This function is for Kaggle environment only")
    
    # Kaggle dataset에서 모델 파일 경로
    model_path = f'/kaggle/input/{dataset_name}/{dataset_name}.pkl'
    
    # 파이프라인 초기화 및 모델 로드
    pipeline = Pipeline(Config())
    print(f"Loading model from {model_path}")
    pipeline.load(model_path)
    
    return pipeline