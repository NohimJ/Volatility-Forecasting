import yfinance as yf
import pandas as pd

def load_price_data(ticker="CL=F", start="2010-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Keep only adjusted close
    df = df[["Close"]].rename(columns={"Close": "price"})
    
    # Reset index if needed
    df = df.reset_index().set_index("Date")
    
    return df
