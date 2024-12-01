import gc
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from config import Config
from constant import BASE_PATH


class DataHandler:
    def __init__(self, config: Config):
        self.config = config
        self.date_ranges: Dict[int, Tuple[int, int]] = {} 
        self.test_data: Optional[pl.DataFrame] = None
        self.features: List[str] = [] 
        self.preprocessor = None
        self.feature_generator = None
    
    def load_data(self) -> Tuple[Dict[int, Tuple[int, int]], Optional[pl.DataFrame]]:
        """Load test data and get date ranges for train partitions"""
        try:
            # Load test data
            self.test_data = pl.read_parquet(f"{BASE_PATH}/test.parquet")
            
            # Initialize date_ranges as an empty dictionary with correct typing
            self.date_ranges: Dict[int, Tuple[int, int]] = {} # noqa 
            # Determine the partition range, defaulting to range(10) if not specified
            partition_range = self.config.partition_range or range(10)
            
            # Iterate over each partition in the determined range
            for i in partition_range:
                date_range_df = pl.scan_parquet(
                    f"{BASE_PATH}/train.parquet/partition_id={i}/part-0.parquet"
                ).select('date_id')
                
                # 수정된 부분: min과 max에 각각 다른 alias 지정
                date_stats = date_range_df.select([
                    pl.col('date_id').min().alias('min_date'),
                    pl.col('date_id').max().alias('max_date')
                ]).collect()
                
                self.date_ranges[i] = ( # noqa
                    date_stats[0, 'min_date'],  # alias로 접근
                    date_stats[0, 'max_date']   # alias로 접근
                )
            
            print("\nTrain Data Date Ranges per Partition:")
            for partition, (min_date, max_date) in self.date_ranges.items():
                print(f"Partition {partition}: date_id {min_date} to {max_date}")
            
            return self.date_ranges, self.test_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def _load_partition_data_by_datarange(self, date_range: Optional[Tuple[int, int]] = None) -> Optional[pl.DataFrame]:
        """Load and return train data for specified date range"""
        try:
            relevant_partitions = self._get_relevant_partitions(date_range)
            print(f"\nLoading train data from partitions: {relevant_partitions}")
            
            # Load relevant partitions
            train_parts = []
            for i in relevant_partitions:
                part_df = pl.scan_parquet(  # read_parquet 대신 scan_parquet 사용
                    f"{BASE_PATH}/train.parquet/partition_id={i}/part-0.parquet"
                )
                
                # Filter by date range if specified
                start_date, end_date = date_range
                part_df = part_df.filter(
                    (pl.col('date_id') >= start_date) & 
                    (pl.col('date_id') <= end_date)
                )
                train_parts.append(part_df)
 
            # Concatenate parts
            train_data = pl.concat(train_parts, how='vertical').collect()
 
            # Clear memory
            del train_parts
            gc.collect()
            
            print(f"Loaded train data shape: {train_data.shape}")
            # Decode date_id if it's a byte string
            min_date = train_data['date_id'].min()
            max_date = train_data['date_id'].max()
            if isinstance(min_date, bytes):
                min_date = min_date.decode('utf-8')
            if isinstance(max_date, bytes):
                max_date = max_date.decode('utf-8')
            print(f"Date range in loaded data: {min_date} to {max_date}")
            return train_data
        except Exception as e:
            print(f"Error loading train data: {e}")
            return None

    def _process_and_generate_features(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        """Apply preprocessing and generate features from raw DataFrame"""
        try:
            # Preprocessing
            if self.preprocessor:
                processed_df = self.preprocessor(raw_df)
                del raw_df
                gc.collect()
            else:
                processed_df = raw_df

            # Feature generation
            if self.feature_generator:
                featured_df = self.feature_generator(processed_df)
                del processed_df
                gc.collect()
                self.features = [col for col in featured_df.columns
                                if col.startswith('feature_')]
                return featured_df
            
            return processed_df

        except Exception as e:
            print(f"Error in data processing and feature generation: {e}")
            return raw_df
    
    def prepare_data(self, date_range: Optional[Tuple[int, int]] = None, 
                    preprocessor: Optional[Callable] = None,
                    feature_generator: Optional[Callable] = None) -> pl.DataFrame:
        """Prepare data with custom preprocessing and feature generation"""
        if date_range is None:
            raise ValueError("Date range is required")
        if not preprocessor and not feature_generator:
            raise ValueError("Preprocessor or feature generator is required")
        try:
            self.preprocessor = preprocessor
            self.feature_generator = feature_generator
            
            # Load data
            raw_df = self._load_partition_data_by_datarange(date_range)
            if raw_df is None:
                raise ValueError("Failed to load partition data")

            # Process and generate features
            return self._process_and_generate_features(raw_df)
                
        except Exception as e:
            print(f"Error in prepare_data: {e}")
            return None
            
        finally:
            gc.collect()
    
    def _get_relevant_partitions(self, date_range: Tuple[int, int]) -> List[int]:
        """Get relevant partitions for a given date range"""
        start_date, end_date = date_range
        relevant_partitions = []
        for partition, (min_date, max_date) in self.date_ranges.items():
            if (start_date <= max_date) and (end_date >= min_date):
                relevant_partitions.append(partition)
        return relevant_partitions

    def generate_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate features for a specific DataFrame"""
        if self.feature_generator:
            df = self.feature_generator(df)
            self.features = [col for col in df.columns
                            if col.startswith('feature_')]
            return df
        return df
    
    def get_feature_data(self, df: pl.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract features, target, and weights"""
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")

        X = df.select(self.features).to_numpy()
        y = df.select('responder_6').to_numpy() if 'responder_6' in df.columns else None
        w = df.select('weight').to_numpy() if 'weight' in df.columns else None

        return X, y, w