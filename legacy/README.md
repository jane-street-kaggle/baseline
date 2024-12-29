# Setup
```bash
micromamba create --name kaggle python=3.11 -c conda-forge
micromamba activate kaggle 
micromamba config append channels conda-forge
micromamba config append channels torch 
# macos
micromamba install pytorch::pytorch torchvision torchaudio -c pytorch
# linux-cuda 12.4
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
micromamba install polars numpy xgboost lightgbm dill matplotlib optuna kaggle pandas pyarrow fastparquet -c conda-forge
kaggle competitions download -c jane-street-real-time-market-data-forecasting
```

```
1. kaggle api token: [here](https://teddylee777.github.io/kaggle/Kaggle-API-%EC%82%AC%EC%9A%A9%EB%B2%95/)
```