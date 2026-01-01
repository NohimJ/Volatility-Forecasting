# run_pipeline_final.py
import pandas as pd
from src.Data_ingestion import load_price_data
from src.build_features import build_returns
from src.garch import fit_garch
from src.viz import plot_volatility
from src.forecast_tomorrow import forecast_tomorrow_volatility
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Feature and target columns
FEATURES = [
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "rolling_mean_5",
    "rolling_std_5",
    "garch_lag1",
]
TARGET = "realized_vol_21d"

# Best hyperparameters from tuning (update if neccesary)
BEST_PARAMS = {
    'n_estimators': 100,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': None,
    'random_state': 42
}

def train_final_model(df):
    """
    Train RandomForest with best hyperparameters on all data.
    """
    X = df[FEATURES]
    y = df[TARGET]

    model = RandomForestRegressor(**BEST_PARAMS)
    model.fit(X, y)
    return model

def evaluate_model(df, model):
    """
    Compare ML predictions vs GARCH on the last portion of data.
    """
    X_test = df[FEATURES].iloc[-int(0.2*len(df)):]
    y_test = df[TARGET].iloc[-int(0.2*len(df)):]
    y_pred = model.predict(X_test)
    y_garch = df["garch_vol"].iloc[-len(y_test):]

    ml_mse = mean_squared_error(y_test, y_pred)
    garch_mse = mean_squared_error(y_test, y_garch)
    print(f"ML Test MSE: {ml_mse:.6f}")
    print(f"GARCH MSE: {garch_mse:.6f}")

    return X_test, y_test, y_pred

def main():
    print("Downloading price data...")
    prices = load_price_data()

    print("Building features...")
    df = build_returns(prices)

    print("Fitting GARCH...")
    df = fit_garch(df)

    print("Training final RandomForest model...")
    model = train_final_model(df)

    # Save model for later use
    joblib.dump(model, "rf_volatility_model.pkl")
    print("Model saved to rf_volatility_model.pkl")

    # Evaluate
    print("Evaluating model...")
    X_test, y_test, y_pred = evaluate_model(df, model)

    # Forecast tomorrow's volatility
    print("Forecasting volatility for tomorrow...")
    tomorrow_vol = forecast_tomorrow_volatility(df, model)
    print(f"Tomorrow's forecasted volatility: {tomorrow_vol:.6f}")

    # Plot results
    print("Plotting results...")
    plot_volatility(df, X_test, y_test, y_pred)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
