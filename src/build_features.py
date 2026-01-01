import numpy as np

def build_returns(df):
    df = df.copy()
    df = df[df["price"] > 0]  # safety

    # Log returns
    df["returns"] = np.log(df["price"]).diff()  #log returns to see the percentage change

    # Lagged returns
    for i in range(1, 4):
        df[f"return_lag{i}"] = df["returns"].shift(i)   #lag the returns by 1 to 4 days, to be used as predictors

    # Rolling stats
    df["rolling_mean_5"] = df["returns"].rolling(5).mean()
    df["rolling_std_5"] = df["returns"].rolling(5).std()    #creates rolling means for features

    # 21-day realized volatility
    df["realized_vol_21d"] = df["returns"].rolling(21).std()    #uses 21-day vol as feature

    # Drop rows with any missing values
    df = df.dropna()

    return df
