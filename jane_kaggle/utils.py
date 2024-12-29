import polars as pl

from jane_kaggle.constant import DEFAULT_DATA_DIR


def load_data(
    data_dir: str = DEFAULT_DATA_DIR,
    date_id_range=None, 
    time_id_range=None, 
    columns=None, 
    return_type='pl'
):
    data = pl.scan_parquet(f'{data_dir}/train.parquet')

    if date_id_range is not None:
        start_date, end_date = date_id_range
        data = data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") <= end_date))
    
    if time_id_range is not None:
        start_time, end_time = time_id_range
        data = data.filter((pl.col("time_id") >= start_time) & (pl.col("time_id") <= end_time))
    
    if columns is not None:
        data = data.select(columns)

    if return_type == 'pd':
        return data.collect().to_pandas()
    else:
        return data.collect()
