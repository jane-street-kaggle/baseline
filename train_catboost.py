import os
import joblib # type: ignore
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor # type: ignore

from jane_kaggle.constant import FEAT_COLS, TARGET
from jane_kaggle.metrics import calculate_r2
from jane_kaggle.utils import load_data

def create_catboost_features_kfold(total_days=1699, n_splits=5, cat_features=None):
    max_valid_days = 1200
    valid_days = min(total_days, max_valid_days)
    valid_start = 1699 - valid_days
    
    fold_size = valid_days // n_splits
    folds = [(valid_start + i * fold_size, valid_start + (i + 1) * fold_size - 1) for i in range(n_splits)]
    
    all_data_with_preds = None
    catboost_models = []
    
    for fold_idx in range(n_splits):
        valid_range = folds[fold_idx]
        train_ranges = [folds[i] for i in range(n_splits) if i != fold_idx]
        print(f'Fold {fold_idx}: Creating CatBoost predictions')
        
        # 검증 데이터 로드
        valid_data = load_data(
            date_id_range=valid_range,
            columns=["date_id", "weight"] + FEAT_COLS + [TARGET],
            return_type='pl')
        
        # 학습 데이터 로드
        train_data = None
        for train_range in train_ranges:
            partial_train_data = load_data(date_id_range=train_range,
                                         columns=["date_id", "weight"] + FEAT_COLS + [TARGET],
                                         return_type='pl')
            if train_data is None:
                train_data = partial_train_data
            else:
                train_data = train_data.vstack(partial_train_data)
        
        # CatBoost 모델 학습
        catboost_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            early_stopping_rounds=50,
            verbose=100,
            cat_features=list(range(len(cat_features))),  # 인덱스로 변경
            task_type='GPU'
        )
        
        # Polars to pandas conversion for CatBoost
        train_df = train_data.to_pandas()
        valid_df = valid_data.to_pandas()

        print(f"Using only categorical features: {cat_features}")
        print(f"Train shape with only cat features: {train_df[cat_features].shape}")
        print(f"Valid shape with only cat features: {valid_df[cat_features].shape}")
        
        catboost_model.fit(
            train_df[cat_features],  # FEAT_COLS 대신 cat_features만 사용
            train_df[TARGET],
            eval_set=(valid_df[cat_features], valid_df[TARGET]),
            sample_weight=train_df['weight']
        )
        
        # 예측값 생성
        valid_df['catboost_pred'] = catboost_model.predict(valid_df[cat_features])
        
        # 결과 저장
        if all_data_with_preds is None:
            all_data_with_preds = valid_df
        else:
            all_data_with_preds = pd.concat([all_data_with_preds, valid_df])
        
        catboost_models.append(catboost_model)
    
    return all_data_with_preds, catboost_models
   

def train_xgb_kfold_with_catboost(total_days=1699, n_splits=5, save_model=True, save_path='models/', cat_features=None):
    if save_model and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # CatBoost 예측값 생성
    print("Creating CatBoost features...")
    data_with_catboost, catboost_models = create_catboost_features_kfold(
        total_days=total_days,
        n_splits=n_splits,
        cat_features=cat_features
    )
    
    # CatBoost 예측값을 포함한 새로운 피처 리스트
    FEAT_COLS_WITH_CATBOOST = FEAT_COLS + ['catboost_pred']
    
    max_valid_days = 1200
    valid_days = min(total_days, max_valid_days)
    valid_start = 1699 - valid_days
    
    fold_size = valid_days // n_splits
    folds = [(valid_start + i * fold_size, valid_start + (i + 1) * fold_size -1) for i in range(n_splits)]
    
    cv_scores = []
    xgb_models = []
    
    for fold_idx in range(n_splits):
        valid_range = folds[fold_idx]
        train_ranges = [folds[i] for i in range(n_splits) if i != fold_idx]
        print(f'Fold {fold_idx}: Training XGBoost')
        
        # 폴드별 데이터 분할
        valid_mask = (data_with_catboost['date_id'] >= valid_range[0]) & (data_with_catboost['date_id'] <= valid_range[1])
        valid_data = data_with_catboost[valid_mask]
        train_data = data_with_catboost[~valid_mask]
        
        # XGBoost 데이터 준비
        import xgboost as xgb

        train_dmatrix = xgb.DMatrix(train_data[FEAT_COLS_WITH_CATBOOST], label=train_data[TARGET], weight=train_data['weight'])
        valid_dmatrix = xgb.DMatrix(valid_data[FEAT_COLS_WITH_CATBOOST], label=valid_data[TARGET], weight=valid_data['weight'])
        
        # XGBoost 파라미터
        XGB_PARAMS = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'gpu',
            'gpu_id': 0  # 사용하려는 GPU ID 설정
        }
        
        evals = [(train_dmatrix, 'train'), (valid_dmatrix, 'valid')]
        
        # 모델 학습
        xgb_model = xgb.train(
            params=XGB_PARAMS,
            dtrain=train_dmatrix,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=50
        )
        
        xgb_models.append(xgb_model)
        
        # R2 점수 계산
        y_valid_pred = xgb_model.predict(valid_dmatrix)
        r2_score = calculate_r2(valid_data[TARGET], y_valid_pred, valid_data['weight'])
        print(f"Fold {fold_idx} validation R2 score: {r2_score}")
        
        cv_scores.append(r2_score)
    
    # 모델 저장
    if save_model:
        joblib.dump({
            'xgb_models': xgb_models,
            'catboost_models': catboost_models
        }, os.path.join(save_path, "stacking_models.pkl"))
        print("Saved all models to stacking_models.pkl")
    
    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean R2 score: {np.mean(cv_scores)}, Std: {np.std(cv_scores)}")
    
    return xgb_models, catboost_models, np.mean(cv_scores), np.std(cv_scores)

cat_features = ['feature_09', 'feature_10', 'feature_11']

xgb_models, catboost_models, mean_r2, std_r2 = train_xgb_kfold_with_catboost(
    total_days=500,
    cat_features=cat_features
)