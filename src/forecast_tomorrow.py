import pandas as pd

FEATURES = [
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "rolling_mean_5",
    "rolling_std_5",
    "garch_lag1",
]

def forecast_tomorrow_volatility(df, model):
    """
    Forecast next-day volatility using the most recent features.
    """

    # Take last available row (today)
    latest = df.iloc[-1]

    # Build feature vector
    X_latest = pd.DataFrame(
        [[latest[f] for f in FEATURES]],
        columns=FEATURES
    )

    # Predict tomorrow's volatility
    vol_forecast = model.predict(X_latest)[0] #translate into percentage change i.e multiply by 100 to get += deviation from price

    return vol_forecast
