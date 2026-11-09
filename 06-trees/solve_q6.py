import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("car_fuel_efficiency_2026.csv").fillna(0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values

del df_train["fuel_efficiency_mpg"], df_val["fuel_efficiency_mpg"]

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.to_dict(orient="records"))
X_val = dv.transform(df_val.to_dict(orient="records"))

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(dv.get_feature_names_out()))
dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(dv.get_feature_names_out()))
watchlist = [(dtrain, "train"), (dval, "val")]

print("xgboost", xgb.__version__)

for eta in [0.3, 0.1]:
    params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
    }
    model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)
    pred = model.predict(dval)
    score = float(np.sqrt(((y_val - pred) ** 2).mean()))
    print(f"eta={eta}  val RMSE = {score:.6f}  -> {round(score, 4)}")
