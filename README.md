# Volatility Forecasting Pipeline: GARCH + Machine Learning

This project implements a pipeline to forecast the next-day volatility of financial assets (currently crude oil, symbol `CL=F`). As well as plot the predictions of a engineered Random Forest Model against the actual volatility of crude oil, as well as the predictions from GARCH. It then prints the error for each model compared to the real volatility, and prints the forecasted volatility for tomorrow.

---

## Project Overview

This project forecasts the volatility of crude oil, which is the flucutation of price in that asset. This is important for

- Risk management  
- Options pricing  
- Portfolio hedging  
- Trading strategies

This pipeline demonstrates:

1. **Data ingestion** – Downloads crude oil historical price data using Yahoo Finance.  
2. **Feature engineering** – Computes log returns, lagged returns, and rolling statistics
3. **GARCH modeling** – Fits a GARCH(1,1) model to capture time-varying volatility.  
4. **Machine learning** – Trains a Random Forest Regressor which is then tuned using GridSearchCv to create engineered features and GARCH lag to predict future volatility.  
5. **Evaluation** – Compares ML predictions to GARCH forecasts and actual realized volatility.  
6. **Next-day forecast** – Outputs predicted volatility for tomorrow.  
7. **Visualization** – Plots actual volatility, ML predictions, and GARCH forecasts.

---

## 🛠 Features

| Feature | Description |
|---------|-------------|
| `return_lag1-3` | Lagged returns to capture momentum |
| `rolling_mean_5` | 5-day rolling mean of returns |
| `rolling_std_5` | 5-day rolling standard deviation of returns |
| `garch_lag1` | Previous-day GARCH volatility |
| `realized_vol_21d` | 21-day forward realized volatility (target) |


## ⚙️ Requirements

Python 3.12+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
