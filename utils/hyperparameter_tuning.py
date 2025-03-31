# xgboost functions

# structure of this file
# Top Level xgb_single_parameter_tuner
## kfold_rmse_with_early_stop
### xgb_model
### train_xgb_model
### rmse_predict

# visualisation functions specific to xgboost

### rmse_predict
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse_predict(model, X_val, log_y_val, dmatrix=False, y_log=False):
  # Evaluate on val
  if dmatrix:
    dval = xgb.DMatrix(X_val)
    y_pred = model.predict(dval)
  else:
    y_pred = model.predict(X_val)

  y_pred_flat = y_pred.flatten()

  if y_log:
    log_y_pred_flat = np.log(y_pred_flat)
  else:
    log_y_pred_flat = y_pred_flat

  mse_val = mean_squared_error(log_y_val, log_y_pred_flat)
  rmse_val = np.sqrt(mse_val)
  return rmse_val, log_y_pred_flat

### xgb_model
def xgb_model(learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric='rmse', objective='reg:squarederror', seed=42):
    return {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'eval_metric': eval_metric,
        'objective': objective,
        'seed': seed
    }

### train_xgb_model
def train_xgb_model(params, X_train, y_train, X_val, y_val, num_boost_round=500, early_stopping_rounds=10, verbose=False):

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose
    )
    
    return booster

## kfold_rmse_with_early_stop
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import KFold

def kfold_rmse_with_early_stop(model_init_func, X, y, n_splits=5, early_stopping_rounds=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        params = xgb_model()

        booster = train_xgb_model(params, X_tr, y_tr, X_val, y_val, verbose=False)

        dval = xgb.DMatrix(X_val, label=y_val)
        y_pred = booster.predict(dval)
        rmse, log_y_pred_flat = rmse_predict(booster, X_val, y_val,dmatrix=True, y_log=False)
        rmse_scores.append(rmse)

    return rmse_scores, np.mean(rmse_scores)

# Top Level xgb_single_parameter_tuner






# visualisation functions specific to xgboost
