from arch import arch_model
import numpy as np

def fit_garch(df):
    df = df.copy()
    
    # Fit GARCH(1,1) on returns
    returns = df["returns"] * 100  # scale
    model = arch_model(returns, p=1, q=1)
    res = model.fit(disp="off")

    # Add fitted volatility to DataFrame
    df["garch_vol"] = res.conditional_volatility / 100  # scale back
    df["garch_lag1"] = df["garch_vol"].shift(1)
    
    return df.dropna()
