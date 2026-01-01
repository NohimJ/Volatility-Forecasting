import matplotlib.pyplot as plt

def plot_volatility(df, X_test, y_test, y_pred, tomorrow_vol=None):
    dates = df.index[-len(y_test):]

    plt.figure(figsize=(12,6))

    # Actual realized volatility
    plt.plot(dates, y_test, label="Realized Vol")       #shows data from the test set only, (not training set)

    # ML prediction
    plt.plot(dates, y_pred, label="ML Prediction")

    # GARCH benchmark
    plt.plot(
        dates,
        df["garch_vol"].iloc[-len(y_test):],
        label="GARCH"
    )

    # Tomorrow's forecast 
    if tomorrow_vol is not None:
        plt.scatter(
            df.index[-1],
            tomorrow_vol,
            color="red",
            s=80,
            marker="X",
            label="Tomorrow Forecast"
        )

    plt.legend()
    plt.title("Volatility Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
