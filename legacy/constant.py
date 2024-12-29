import os

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if not IS_KAGGLE: 
    os.chdir('./kaggle')
BASE_PATH = '/kaggle/input/jane-street-real-time-market-data-forecasting' if IS_KAGGLE else './data'  # noqa: E501
MODEL_PATH = '/kaggle/working' if IS_KAGGLE else './models'