import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
import pymc as pm
import arviz as az

def time_series_decomposition(df, column='Price', period=365):
    """Decompose time series into trend, seasonal, and residual components with plot."""
    df_indexed = df.set_index('Date')
    decomposition = seasonal_decompose(df_indexed[column], model='multiplicative', period=period)
    
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(10, 12))
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed, label='Observed', color='black')
    plt.title('Observed Data')
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label='Trend', color='blue')
    plt.title('Trend Component')
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label='Seasonal', color='red')
    plt.title('Seasonal Component')
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label='Residual', color='green')
    plt.title('Residual Component')
    plt.tight_layout(pad=3.0)
    plt.show()
    return decomposition

def arima_model(df, column='Price', order=(1, 1, 1), forecast_steps=30):
    """Fit ARIMA model, forecast future values, and plot."""
    model = ARIMA(df[column], order=order)
    results = model.fit()
    forecast = results.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_steps + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Price': forecast})
    
    print("ARIMA Model Summary:".rjust(30))
    print(results.summary().as_text().rjust(80))
    print("Forecast:".rjust(30))
    print(forecast_df.to_string(justify='right'))
    
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df[column], label='Brent Price', color='blue')
    plt.plot(forecast_df['Date'], forecast_df['Forecasted Price'], color='red', label='30-Day Forecast', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.title('Brent Oil Prices with ARIMA Forecast')
    plt.legend()
    plt.grid()
    plt.show()
    return results, forecast_df

def detect_change_points_l2(df, column='Price', pen=50):
    """Detect change points using Pelt L2 method and plot."""
    price_values = df[column].values
    algo = rpt.Pelt(model="l2").fit(price_values)
    result = algo.predict(pen=pen)
    if result[-1] == len(price_values):
        result = result[:-1]
    l2_dates = df['Date'].iloc[result].tolist()
    
    print("Pelt L2 Change Points:".rjust(30))
    for date in l2_dates:
        print(str(date).rjust(40))
    
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df[column], label='Brent Price', color='blue')
    plt.vlines(l2_dates, ymin=df[column].min(), ymax=df[column].max(), 
               color='green', linestyle='--', label='Pelt L2 Change Points')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.title('Brent Oil Prices with Pelt L2 Change Points')
    plt.legend()
    plt.grid()
    plt.show()
    return l2_dates

def detect_change_points_rbf(df, column='Price', pen=20):
    """Detect change points using Pelt RBF method and plot."""
    price_values = df[column].values
    algo = rpt.Pelt(model="rbf").fit(price_values)
    result = algo.predict(pen=pen)
    if result[-1] == len(price_values):
        result = result[:-1]
    rbf_dates = df['Date'].iloc[result].tolist()
    
    print("Pelt RBF Change Points:".rjust(30))
    for date in rbf_dates:
        print(str(date).rjust(40))
    
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df[column], label='Brent Price', color='blue')
    plt.vlines(rbf_dates, ymin=df[column].min(), ymax=df[column].max(), 
               color='purple', linestyle='-.', label='Pelt RBF Change Points')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.title('Brent Oil Prices with Pelt RBF Change Points')
    plt.legend()
    plt.grid()
    plt.show()
    return rbf_dates

def cusum_analysis(df, column='Price'):
    """Perform CUSUM analysis to detect deviations from the mean and plot."""
    mean_price = df[column].mean()
    cusum = np.cumsum(df[column] - mean_price)
    cusum_df = pd.DataFrame({'Date': df['Date'], 'CUSUM': cusum})
    
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 7))
    plt.plot(cusum_df['Date'], cusum_df['CUSUM'], label='CUSUM', color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('CUSUM Value')
    plt.title('CUSUM Analysis')
    plt.legend()
    plt.grid()
    plt.show()
    return cusum_df

def bayesian_change_point_detection(df, column='Price', max_samples=200, tune_samples=200, cores=1):
    """Efficient Bayesian Change Point Detection using PyMC with plot."""
    if len(df) > 1000:
        df_sampled = df.iloc[::20].reset_index(drop=True)
    else:
        df_sampled = df.copy()
    
    prices = df_sampled[column].values
    idx = np.arange(len(prices))
    
    with pm.Model() as model:
        mu_before = pm.Normal("mu_before", mu=np.mean(prices), sigma=10)
        mu_after = pm.Normal("mu_after", mu=np.mean(prices), sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        change_point = pm.DiscreteUniform("change_point", lower=0, upper=len(prices)-1)
        mu = pm.math.switch(change_point >= idx, mu_before, mu_after)
        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=prices)
        trace = pm.sample(max_samples, tune=tune_samples, cores=cores, target_accept=0.95, return_inferencedata=True)
    
    plt.style.use('seaborn-v0_8')
    az.plot_posterior(trace, var_names=["change_point"])
    plt.show()
    
    change_point_index = int(np.mean(trace.posterior["change_point"].values))
    scale_factor = 20 if len(df) > 1000 else 1
    bayesian_date = df['Date'].iloc[change_point_index * scale_factor]
    
    print(f"Bayesian Change Point: {str(bayesian_date)}".rjust(40))
    
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df[column], label='Brent Price', color='blue')
    plt.vlines(bayesian_date, ymin=df[column].min(), ymax=df[column].max(), 
               color='orange', linestyle=':', label='Bayesian Change Point')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.title('Brent Oil Prices with Bayesian Change Point')
    plt.legend()
    plt.grid()
    plt.show()
    return bayesian_date