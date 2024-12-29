import os
import joblib # type: ignore
import numpy as np
import lightgbm as lgb
import xgboost as xgb

from jane_kaggle.constant import FEAT_COLS, TARGET
from jane_kaggle.metrics import calculate_r2
from jane_kaggle.utils import load_data

def train_xgb_holdout_single(total_days=1699, valid_days=120, save_model=True, save_path='models/'):
    if save_model and not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_date = 0
    valid_start = total_days - valid_days  # 검증 데이터의 시작 날짜

    print(f'Validation range: {valid_start} to {total_days - 1}')

    # 검증 데이터 로드
    valid_data = load_data(
        date_id_range=(valid_start, total_days - 1),
        columns=["date_id", "weight"] + FEAT_COLS + [TARGET],
        return_type='pl'
    )
    valid_weight = valid_data.select("weight").to_pandas()["weight"]
    valid_X = valid_data.select(FEAT_COLS).to_pandas()
    valid_y = valid_data.select(TARGET).to_pandas()[TARGET]

    # 학습 데이터 로드
    train_data = load_data(
        date_id_range=(skip_date, valid_start - 1),
        columns=["date_id", "weight"] + FEAT_COLS + [TARGET],
        return_type='pl'
    )
    train_weight = train_data.select("weight").to_pandas()["weight"]
    train_X = train_data.select(FEAT_COLS).to_pandas()
    train_y = train_data.select(TARGET).to_pandas()[TARGET]

    # XGBoost DMatrix 생성
    train_dmatrix = xgb.DMatrix(data=train_X, label=train_y, weight=train_weight)
    valid_dmatrix = xgb.DMatrix(data=valid_X, label=valid_y, weight=valid_weight)

    # XGBoost 파라미터 설정
    XGB_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 6,
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'gpu',
    }

    # 모델 훈련
    evals_result = {}
    model = xgb.train(
        params=XGB_PARAMS,
        dtrain=train_dmatrix,
        num_boost_round=1000,
        evals=[(train_dmatrix, 'train'), (valid_dmatrix, 'valid')],
        early_stopping_rounds=100,
        evals_result=evals_result,
        verbose_eval=50,
    )

    # 검증 세트 예측 및 R2 계산
    y_valid_pred = model.predict(valid_dmatrix)
    r2_score = calculate_r2(valid_y, y_valid_pred, valid_weight)
    print(f"Validation R2 score: {r2_score}")

    # 모델 저장
    if save_model:
        model_path = os.path.join(save_path, "xgb_model.json")
        model.save_model(model_path)
        print(f"Saved the model to {model_path}")

    return model, r2_score

def train_xgb_kfold_single(total_days=1699, n_splits=5, save_model=True, save_path='models/'):
    if save_model and not os.path.exists(save_path):
        os.makedirs(save_path)

    max_valid_days = 1200
    valid_days = min(total_days, max_valid_days)
    valid_start = 1699 - valid_days

    fold_size = valid_days // n_splits
    folds = [(valid_start + i * fold_size, valid_start + (i + 1) * fold_size - 1) for i in range(n_splits)]

    cv_scores = []
    model_group = []

    for fold_idx in range(n_splits):
        valid_range = folds[fold_idx]
        train_ranges = [folds[i] for i in range(n_splits) if i != fold_idx]
        print(f'Fold {fold_idx}: validation range {valid_range}, train parts: {train_ranges}')

        # 검증 데이터 로드
        valid_data = load_data(
            date_id_range=valid_range,
            columns=["date_id", "weight"] + FEAT_COLS + [TARGET],
            return_type='pl'
        )
        valid_weight = valid_data['weight'].to_pandas()
        valid_X = valid_data.select(FEAT_COLS).to_pandas()
        valid_y = valid_data[TARGET].to_pandas()

        # 학습 데이터 로드
        train_data = None
        for train_range in train_ranges:
            partial_train_data = load_data(
                date_id_range=train_range,
                columns=["date_id", "weight"] + FEAT_COLS + [TARGET],
                return_type='pl'
            )
            if train_data is None:
                train_data = partial_train_data
            else:
                train_data = train_data.vstack(partial_train_data)

        train_weight = train_data['weight'].to_pandas()
        train_X = train_data.select(FEAT_COLS).to_pandas()
        train_y = train_data[TARGET].to_pandas()

        # XGBoost DMatrix 생성
        train_dmatrix = xgb.DMatrix(data=train_X, label=train_y, weight=train_weight)
        valid_dmatrix = xgb.DMatrix(data=valid_X, label=valid_y, weight=valid_weight)

        # XGBoost 파라미터 설정
        XGB_PARAMS = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'gpu',
        }

        # 모델 훈련
        evals_result = {}
        model = xgb.train(
            params=XGB_PARAMS,
            dtrain=train_dmatrix,
            num_boost_round=1000,
            evals=[(train_dmatrix, 'train'), (valid_dmatrix, 'valid')],
            early_stopping_rounds=100,
            evals_result=evals_result,
            verbose_eval=50,
        )

        # 모델 저장
        model_group.append(model)

        # 검증 세트 예측 및 R2 계산
        y_valid_pred = model.predict(valid_dmatrix)
        r2_score = calculate_r2(valid_y, y_valid_pred, valid_weight)
        print(f"Fold {fold_idx} validation R2 score: {r2_score}")

        cv_scores.append(r2_score)

    # 모델 그룹 저장
    if save_model:
        for idx, model in enumerate(model_group):
            model_path = os.path.join(save_path, f"xgb_model_fold{idx}.json")
            model.save_model(model_path)
            print(f"Saved model for fold {idx} to {model_path}")

    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean R2 score: {np.mean(cv_scores)}, Std: {np.std(cv_scores)}")

    return model_group, np.mean(cv_scores), np.std(cv_scores)

total_days = 1699 # Total num of diff date_id = 1699
valid_days = 120
xgb_models_holdout_single, _ = train_xgb_holdout_single(total_days=total_days,valid_days=valid_days)

total_days = 500 # Total num of diff date_id = 1699
xgb_models_kfold_single, _, _ = train_xgb_kfold_single(total_days=total_days)