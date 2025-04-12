from arch import arch_model
import numpy as np
import pandas as pd


def RollingStDevFilter(file_name):
    df = pd.read_csv(f'{file_name}.csv')
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Volatility'] = df['Daily Return'].rolling(window=30).std()
    vol_thresh = df['Volatility'].quantile(0.75)
    df['high_volatility'] = df['Volatility'] > vol_thresh

    volatile_period = df[df['high_volatility']]
    volatile_period.to_csv(f'rollingstdev_volatile_period_{file_name}.csv', index=True)


def GarchVolatilityFilter(file_name):
    df = pd.read_csv(f'{file_name}.csv')
    df['Log Returns'] = np.log(df["Adj Close"] / df["Adj Close"].shift(1)) # calculates log returns
    log_returns = df['Log Returns'].dropna().values # gets rid of NaN values
    garch = arch_model(log_returns, vol='Garch', p=1, q=1, mean='Zero', dist='normal') 
    garch_fit = garch.fit(disp='off')
    returns_df = df.dropna(subset=['Log Returns']).copy()
    returns_df['Volatility'] = garch_fit.conditional_volatility # volatility measurement for each data point
    volatility_threshold = returns_df['Volatility'].quantile(0.75)
    returns_df['High Volatility'] = returns_df['Volatility'] > volatility_threshold # flags high volatility dataa

    volatile_period_garch = returns_df[returns_df['High Volatility']] #filters high volatility ddata
    volatile_period_garch.to_csv(f'garch_volatile_period_{file_name}.csv', index=True)

