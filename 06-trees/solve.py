import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("car_fuel_efficiency_2026.csv").fillna(0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

del df_train["fuel_efficiency_mpg"], df_val["fuel_efficiency_mpg"], df_test["fuel_efficiency_mpg"]

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.to_dict(orient="records"))
X_val = dv.transform(df_val.to_dict(orient="records"))
X_test = dv.transform(df_test.to_dict(orient="records"))
names = dv.get_feature_names_out()
print("X_train:", X_train.shape, type(X_train).__name__)
print("split sizes:", len(df_train), len(df_val), len(df_test))


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


print("\n===== Q1 : DecisionTreeRegressor(max_depth=1) =====")
dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)
feature = names[dt.tree_.feature[0]]
print("split feature :", feature)
print("original column:", feature.split("=")[0])

print("\n===== Q2 : RandomForestRegressor(n_estimators=10) =====")
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
q2 = rmse(y_val, rf.predict(X_val))
print("val RMSE =", q2, "->", round(q2, 4))

print("\n===== Q3 : n_estimators sweep =====")
q3 = {}
for n in [10, 50, 100, 150]:
    m = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    m.fit(X_train, y_train)
    r = rmse(y_val, m.predict(X_val))
    q3[n] = round(r, 3)
    print(f"  n_estimators={n:3d}  RMSE={r:.6f}  round3={round(r, 3)}")
print("lowest ->", min(q3, key=q3.get))

print("\n===== Q4 : max_depth x n_estimators =====")
q4 = {}
for d in [10, 15, 20, 25]:
    scores = []
    for n in [10, 50, 100, 150]:
        m = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=1, n_jobs=-1)
        m.fit(X_train, y_train)
        scores.append(rmse(y_val, m.predict(X_val)))
    q4[d] = float(np.mean(scores))
    print(f"  max_depth={d}: {[round(s, 6) for s in scores]}  mean={q4[d]:.6f}")
print("best max_depth ->", min(q4, key=q4.get))

print("\n===== Q5 : feature importance (n_estimators=10, max_depth=20) =====")
m5 = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
m5.fit(X_train, y_train)
imp = sorted(zip(names, m5.feature_importances_), key=lambda t: -t[1])
for name, score in imp[:8]:
    print(f"  {name:40s} {score:.6f}")
