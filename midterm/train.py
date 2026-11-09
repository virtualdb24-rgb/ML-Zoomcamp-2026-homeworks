import json
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

import xgboost as xgb

DATA = "insurance.csv"
TARGET = "charges"
MODEL_PATH = "model.pkl"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def cv_score(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    rmses, maes = [], []
    for train_idx, val_idx in kf.split(X):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[val_idx])
        rmses.append(rmse(y[val_idx], pred))
        maes.append(mae(y[val_idx], pred))
    return float(np.mean(rmses)), float(np.mean(maes))


def main():
    df = pd.read_csv(DATA)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    y_train = df_train[TARGET].values
    y_test = df_test[TARGET].values

    X_train_dicts = df_train.drop(columns=[TARGET]).to_dict(orient="records")
    X_test_dicts = df_test.drop(columns=[TARGET]).to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(X_train_dicts)
    X_test = dv.transform(X_test_dicts)

    candidates = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=1, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=1,
            n_jobs=-1,
        ),
    }

    results = {}
    print(f"{'model':22s} {'RMSE (CV)':>12s} {'MAE (CV)':>12s}")
    print("-" * 50)
    for name, model in candidates.items():
        r, m = cv_score(model, X_train, y_train)
        results[name] = {"rmse_cv": round(r, 2), "mae_cv": round(m, 2)}
        print(f"{name:22s} {r:12.2f} {m:12.2f}")

    best_name = min(results, key=lambda k: results[k]["rmse_cv"])
    print(f"\nBest model: {best_name}")

    best_model = clone(candidates[best_name])
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    test_mae = mae(y_test, y_pred)
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE : {test_mae:.2f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"dv": dv, "model": best_model}, f)
    print(f"Saved {MODEL_PATH}")

    with open("metrics.json", "w") as f:
        json.dump(
            {
                "cv": results,
                "best_model": best_name,
                "test_rmse": round(test_rmse, 2),
                "test_mae": round(test_mae, 2),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
