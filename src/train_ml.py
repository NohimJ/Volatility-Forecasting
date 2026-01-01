import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

FEATURES = ["return_lag1", "return_lag2", "return_lag3",
            "rolling_mean_5", "rolling_std_5", "garch_lag1"]
TARGET = "realized_vol_21d"

def train_model(df):
    df = df.copy()
    
    # Ensure all required columns exist
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Drop rows with missing feature values
    df = df.dropna(subset=FEATURES + [TARGET])
    
    X = df[FEATURES]
    y = df[TARGET]

    # Train-test split (time series: no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Define hyperparameter search space
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 6, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    # Base model
    rf = RandomForestRegressor(random_state=42)

    # Randomized search
    rand_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,             # try 20 random combinations
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )

    # Fit hyperparameter search
    rand_search.fit(X_train, y_train)

    # Best model
    model = rand_search.best_estimator_

    # Predictions
    y_pred = model.predict(X_test)
    y_true = y_test
    y_garch = df["garch_vol"].iloc[-len(y_test):]

    # Compute MSE
    mse = mean_squared_error(y_true, y_pred)
    garch_mse = mean_squared_error(y_true, y_garch)
    
    print("Best hyperparameters:", rand_search.best_params_)
    print(f"ML Test MSE: {mse:.6f}")
    print(f"GARCH MSE: {garch_mse:.6f}")

    return model, X_test, y_test, y_pred
