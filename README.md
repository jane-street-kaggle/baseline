# Setup
```bash
micromamba create --name myenv python=3.12 -c conda-forge
micromamba activate myenv
micromamba config append channels conda-forge
micromamba config append channels torch 
micromamba install pytorch::pytorch torchvision torchaudio -c pytorch
micromamba install polars numpy xgboost lightgbm dill matplotlib optuna -c conda-forge


```
1. kaggle api token: [here](https://teddylee777.github.io/kaggle/Kaggle-API-%EC%82%AC%EC%9A%A9%EB%B2%95/)