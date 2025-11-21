# train.py
# Minimal training script: reads data/cars.csv, trains a simple pipeline,
# and writes model.joblib for the Flask app to use.

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

DATA_PATH = "data/cars.csv"
MODEL_PATH = "model.joblib"
CURRENT_YEAR = 2025

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def train_and_save(df):
    df = df.copy()
    df["age"] = CURRENT_YEAR - df["year"]
    X = df[["age", "mileage_km", "make", "model", "transmission", "fuel", "city"]]
    y = df["price"]

    num_cols = ["age", "mileage_km"]
    cat_cols = ["make", "model", "transmission", "fuel", "city"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    print(f"Test MAE: {mae:.2f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    df = load_data()
    train_and_save(df)
