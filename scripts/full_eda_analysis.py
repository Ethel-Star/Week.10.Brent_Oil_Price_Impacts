import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import pymc as pm
from statsmodels.tsa.seasonal import seasonal_decompose
import arviz as az  # Explicitly import arviz for plotting

# --- Functions for EDA ---
def check_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]

def summary_statistics(df):
    return df.describe()

def plot_histogram(df, ax):
    ax.hist(df['Price'], bins=50, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Brent Oil Prices')
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Frequency')

def plot_line_chart(df, ax):
    ax.plot(df['Date'], df['Price'], color='green', lw=2)
    ax.set_title('Brent Oil Price Trend Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True)

def plot_box_plot(df, ax):
    sns.boxplot(x=df['Price'], color='orange', ax=ax)
    ax.set_title('Box Plot of Brent Oil Prices')
    ax.set_xlabel('Price (USD)')

def eda_analysis(df):
    missing_values = check_missing_values(df)
    if not missing_values.empty:
        print("\nMissing Values Found:")
        print(missing_values)
    else:
        print("\nNo missing values found.")
    
    print("\nSummary Statistics:")
    print(summary_statistics(df))
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 24))
    plot_histogram(df, axs[0])
    plot_line_chart(df, axs[1])
    plot_box_plot(df, axs[2])
    plt.tight_layout(pad=3.0)
    plt.show()

# --- Time Series Decomposition ---
def time_series_decomposition(df):
    df_indexed = df.set_index('Date')
    decomposition = seasonal_decompose(df_indexed['Price'], model='multiplicative', period=365)
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    axs[0].plot(decomposition.observed, label='Observed', color='black')
    axs[0].set_title('Observed (Original) Data')
    axs[1].plot(decomposition.trend, label='Trend', color='blue')
    axs[1].set_title('Trend Component')
    axs[2].plot(decomposition.seasonal, label='Seasonal', color='red')
    axs[2].set_title('Seasonal Component')
    axs[3].plot(decomposition.resid, label='Residual', color='green')
    axs[3].set_title('Residual Component')
    plt.tight_layout(pad=3.0)
    plt.show()
    return decomposition

# --- Change Point Detection (Pelt L2 Method) ---
def detect_change_points_l2(df):
    price_values = df['Price'].values
    model = "l2"
    algo = rpt.Pelt(model=model).fit(price_values)
    result = algo.predict(pen=50)
    if result[-1] == len(price_values):
        result = result[:-1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], price_values, label="Price", color="blue")
    for cp in result:
        ax.axvline(x=df['Date'].iloc[cp], color='red', linestyle='--', label="L2 Change Point" if cp == result[0] else "")
    ax.set_title("Pelt L2 Change Point Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout(pad=3.0)
    plt.show()
    return result

# --- Change Point Detection (Pelt RBF Method) ---
def detect_change_points_rbf(df):
    price_values = df['Price'].values
    model = "rbf"
    algo = rpt.Pelt(model=model).fit(price_values)
    result = algo.predict(pen=20)  # Using pen=20 as in your snippet
    if result[-1] == len(price_values):
        result = result[:-1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], price_values, label="Price", color="blue")
    for cp in result:
        ax.axvline(x=df['Date'].iloc[cp], color='purple', linestyle='--', label="RBF Change Point" if cp == result[0] else "")
    ax.set_title("Pelt RBF Change Point Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout(pad=3.0)
    plt.show()
    return result

# --- CUSUM Visualization ---
def cusum_analysis(df):
    mean_price = df['Price'].mean()
    cusum = np.cumsum(df['Price'] - mean_price)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], cusum, label='CUSUM', color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('CUSUM Value')
    plt.title('CUSUM Analysis')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout(pad=3.0)
    plt.show()
    return cusum

# --- Bayesian Change Point Detection (Commented Existing Version) ---
'''
def bayesian_change_point_detection(df):
    with pm.Model() as model:
        # Priors
        mean_price = df['Price'].mean()
        mu_before = pm.Normal("mu_before", mu=mean_price, sigma=10)
        mu_after = pm.Normal("mu_after", mu=mean_price, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        change_point = pm.DiscreteUniform("change_point", lower=0, upper=min(100, len(df) - 1))
        
        # Dynamic mean based on change point
        prices = df['Price'].values
        idx = np.arange(len(prices))
        mu = pm.math.switch(idx < change_point, mu_before, mu_after)
        
        # Likelihood
        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=prices)
        
        # Inference with InferenceData output
        trace = pm.sample(500, tune=500, cores=1, return_inferencedata=True)
    
    # Plot posterior distributions using arviz
    az.plot_posterior(trace, var_names=["change_point", "mu_before", "mu_after"])
    plt.show()
    
    change_point_index = int(np.mean(trace.posterior["change_point"].values))
    return change_point_index
'''
def bayesian_change_point_detection(df, max_samples=1000, tune_samples=False, chains=4, cores=1):
    """
    Performs Bayesian Change Point Detection using PyMC with optimized parameters for speed.
    
    :param df: DataFrame with 'Price' column and 'Date' column.
    :param max_samples: Number of posterior samples per chain (default: 200).
    :param tune_samples: Number of tuning samples per chain (default: 200).
    :param chains: Number of chains (default: 2).
    :param cores: Number of cores for sampling (default: 1).
    :return: Change point index.
    """
    print("Using optimized parameters (fewer samples, chains, and aggressive subsampling) for faster runtime.")
    
    # Subsample data more aggressively if very large (e.g., every 20th row for testing)
    if len(df) > 1000:
        df_sampled = df.iloc[::20].reset_index(drop=True)  # Use every 20th row
    else:
        df_sampled = df.copy()
    
    with pm.Model() as model:
        # Priors
        mean_price = df_sampled['Price'].mean()
        mu_before = pm.Normal("mu_before", mu=mean_price, sigma=10)
        mu_after = pm.Normal("mu_after", mu=mean_price, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # Limit change point range to reduce computation (e.g., max 50 for even faster speed)
        max_change_point = min(100, len(df_sampled) - 1)  # Limit to 50 or less for speed
        change_point = pm.DiscreteUniform("change_point", lower=0, upper=max_change_point)
        
        # Dynamic mean based on change point
        prices = df_sampled['Price'].values
        idx = np.arange(len(prices))
        mu = pm.math.switch(idx < change_point, mu_before, mu_after)
        
        # Likelihood
        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=prices)
        
        # Inference with minimal tuning and fewer samples
        trace = pm.sample(max_samples, tune=tune_samples, chains=chains, cores=cores, return_inferencedata=True)
    
    # Diagnostics
    print("\nBayesian Diagnostics (Note: Results may be less reliable due to optimizations):")
    az.summary(trace, var_names=["change_point", "mu_before", "mu_after"])
    az.plot_trace(trace, var_names=["change_point", "mu_before", "mu_after"])
    plt.show()
    
    # Plot posterior distributions using arviz
    az.plot_posterior(trace, var_names=["change_point", "mu_before", "mu_after"])
    plt.show()
    
    change_point_index = int(np.mean(trace.posterior["change_point"].values))
    # Scale back to original index if subsampled
    change_point_index = change_point_index * 20 if len(df) > 1000 else change_point_index
    return change_point_index

def full_eda_analysis(df):
    print("Checking for missing values...")
    missing_values = check_missing_values(df)
    if not missing_values.empty:
        print(f"\nMissing values found in the following columns: {missing_values}")
    else:
        print("\nNo missing values found.")
    
    print("\nSummary Statistics:")
    print(summary_statistics(df))
    
    print("\nGenerating visualizations...")
    eda_analysis(df)
    
    print("\nPerforming time series decomposition...")
    time_series_decomposition(df)
    
    print("\nDetecting change points with Pelt L2...")
    pelt_l2_points = detect_change_points_l2(df)
    
    print("\nDetecting change points with Pelt RBF...")
    pelt_rbf_points = detect_change_points_rbf(df)
    
    print("\nPerforming CUSUM analysis...")
    cusum_analysis(df)
    
    print("\nDetecting Bayesian change point...")
    bayesian_point = bayesian_change_point_detection(df)
    
    # Plot all change points together
    print("\nPlotting all detected change points...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Brent Oil Price', color='blue')
    for cp in pelt_l2_points:
        plt.axvline(x=df['Date'].iloc[cp], color='red', linestyle='--', label='Pelt L2 Change Point' if cp == pelt_l2_points[0] else "")
    for cp in pelt_rbf_points:
        plt.axvline(x=df['Date'].iloc[cp], color='purple', linestyle='-.', label='Pelt RBF Change Point' if cp == pelt_rbf_points[0] else "")
    plt.axvline(x=df['Date'].iloc[bayesian_point], color='orange', linestyle='-.', label='Bayesian Change Point')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Brent Oil Prices with Detected Change Points')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout(pad=3.0)
    plt.show()
