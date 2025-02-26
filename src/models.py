import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
import itertools
import warnings
warnings.filterwarnings('ignore')

class OilPriceModels:
    def __init__(self, data):
        self.data = data

    def fit_arima_with_events(self, order=None, include_events=True):
        """Fit an ARIMA model with optional event dummies or sentiment as exogenous variables."""
        if order is None:
            order = self.optimize_arima()[0]
        exog = None
        if include_events:
            exog = self.data[['Gulf_War', 'Financial_Crisis', 'Covid_Impact', 'Oil_Price_War', 'Event_Sentiment']].dropna()
        model = ARIMA(self.data['Price'], order=order, exog=exog)
        results = model.fit()
        return results

    def fit_garch(self):
        """Fit a GARCH model to the returns series."""
        model = arch_model(self.data['Returns'].dropna(), vol='Garch', p=1, q=1)
        results = model.fit()
        return results

    def check_stationarity(self, series_name='Price', max_diffs=2):
        """
        Check if a series is stationary using ADF and KPSS tests, applying up to max_diffs differencing.
        
        Args:
            series_name (str): Name of the series to check ('Price' or 'Returns').
            max_diffs (int): Maximum number of differencing levels to try.
        
        Returns:
            tuple: (is_stationary, diff_level, series) - whether stationary, optimal differencing level, and differenced series.
        """
        series = self.data[series_name].dropna()
        diff_level = 0
        current_series = series.copy()

        for d in range(max_diffs + 1):
            # ADF test
            adf_result = adfuller(current_series)
            adf_p_value = adf_result[1]
            print(f'ADF Statistic for {series_name} (diff={d}):', adf_result[0])
            print(f'p-value for {series_name} (diff={d}):', adf_p_value)
            print(f'Critical Values for {series_name} (diff={d}):', adf_result[4])

            # KPSS test (complementary to ADF, checks for stationarity around a deterministic trend)
            kpss_result = kpss(current_series, regression='c')
            kpss_p_value = kpss_result[1]
            print(f'KPSS Statistic for {series_name} (diff={d}):', kpss_result[0])
            print(f'p-value for {series_name} (diff={d}):', kpss_p_value)
            print(f'Critical Values for {series_name} (diff={d}):', kpss_result[3])

            # Check if stationary (ADF p-value < 0.05 and KPSS p-value > 0.05)
            if adf_p_value < 0.05 and kpss_p_value > 0.05:
                print(f"{series_name} is stationary after {d} differencing(s).")
                return True, d, current_series
            elif d < max_diffs:
                current_series = current_series.diff().dropna()
                diff_level += 1
            else:
                print(f"{series_name} remains non-stationary after {max_diffs} differencing(s).")
                return False, diff_level, current_series

    def preprocess_exog_vars(self, data):
        """Preprocess exogenous variables for numerical stability, removing highly correlated variables."""
        scaler = StandardScaler()
        exog_cols = ['Event_Sentiment']  # Continuous exogenous vars
        event_dummies = ['Gulf_War', 'Financial_Crisis', 'Covid_Impact', 'Oil_Price_War']
        
        # Check correlations and remove highly correlated variables (e.g., correlation > 0.9)
        correlation_matrix = data[exog_cols + event_dummies].corr()
        print("Correlation Matrix for Exogenous Variables:\n", correlation_matrix)
        
        # Remove variables with correlation > 0.9 (example: keep only one if highly correlated)
        for col1 in correlation_matrix.columns:
            for col2 in correlation_matrix.columns:
                if col1 < col2 and abs(correlation_matrix.loc[col1, col2]) > 0.9:
                    if col2 in data.columns:
                        print(f"Removing {col2} due to high correlation with {col1}")
                        data = data.drop(columns=[col2])
        
        # Scale continuous exogenous variables
        if exog_cols and any(col in data.columns for col in exog_cols):
            data[exog_cols] = scaler.fit_transform(data[exog_cols])
        
        return data

    def fit_var(self, exog_vars=None, include_events=True, maxlags=None, use_differenced=False, max_diffs=2):
        """Fit a VAR model with endogenous variables (Price and Returns) and optional exogenous variables or events."""
        # Check stationarity for Price and Returns
        price_stationary, price_diffs, price_series = self.check_stationarity('Price', max_diffs)
        returns_stationary, returns_diffs, returns_series = self.check_stationarity('Returns', max_diffs)
        
        # Use Price and Returns as default endogenous variables (or their differences based on stationarity)
        if use_differenced or not (price_stationary and returns_stationary):
            print("Using differenced data due to non-stationarity.")
            endogenous_vars = pd.DataFrame({
                'Price': price_series.diff(max(price_diffs, returns_diffs)).dropna(),
                'Returns': returns_series.diff(max(price_diffs, returns_diffs)).dropna()
            })
        else:
            endogenous_vars = self.data[['Price', 'Returns']].dropna()
        
        if exog_vars is not None:
            if not isinstance(exog_vars, pd.DataFrame):
                raise ValueError("exog_vars must be a pandas DataFrame")
            data = pd.concat([endogenous_vars, exog_vars], axis=1).dropna()
        else:
            data = endogenous_vars
        
        # Optionally include event dummies and sentiment, excluding highly correlated variables
        if include_events:
            event_vars = self.data[['Gulf_War', 'Financial_Crisis', 'Covid_Impact', 'Oil_Price_War', 'Event_Sentiment']].dropna()
            data = pd.concat([data, event_vars], axis=1).dropna()
            data = self.preprocess_exog_vars(data)
        
        if data.empty or data.shape[1] < 2:
            raise ValueError("VAR requires at least two numerical variables. Ensure 'Price', 'Returns', and any exogenous variables are available and numerical.")
        
        model = VAR(data)
        if maxlags is None:
            # Select optimal lags using AIC (up to 4 for daily data, 2 for monthly to reduce complexity)
            lag_order = model.select_order(maxlags=4 if len(data) > 1000 else 2)  # Further reduced maxlags
            maxlags = lag_order.aic
        results = model.fit(maxlags=maxlags)
        return results

    def plot_prices_and_returns(self):
        """Plot Brent oil prices and daily returns."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Price'], label='Brent Price')
        plt.title('Brent Crude Oil Prices (1987–2022)')
        plt.xlabel('Date')
        plt.ylabel('Price ($/barrel)')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Returns'], label='Daily Returns', color='red')
        plt.title('Brent Crude Oil Daily Returns (1987–2022)')
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend()
        plt.show()

    def plot_arima(self, arima_results):
        """Plot ARIMA fitted values against actual prices."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Price'], label='Actual Price')
        plt.plot(arima_results.fittedvalues, label='ARIMA Fitted Values', color='red')  # Red as requested
        plt.title('ARIMA Model Fit for Brent Oil Prices (1987–2022)')
        plt.xlabel('Date')
        plt.ylabel('Price ($/barrel)')
        plt.legend()
        plt.show()

    def plot_garch(self, garch_results):
        """Plot GARCH conditional volatility."""
        plt.figure(figsize=(12, 6))
        plt.plot(garch_results.conditional_volatility, label='GARCH Conditional Volatility', color='purple')
        plt.title('GARCH Conditional Volatility for Brent Oil Returns (1987–2022)')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.show()

    def plot_var(self, var_results):
        """Plot VAR forecast for Price and Returns with longer forecast horizon."""
        # Determine data frequency to set appropriate forecast steps
        if len(self.data) > 1000:  # Assume daily data if >1000 observations
            steps = 252  # One year of daily data (252 trading days)
        else:  # Assume monthly data if <=1000 observations
            steps = 12  # One year of monthly data
    
        # Forecast for the number of steps
        forecast = var_results.forecast(var_results.endog[-var_results.k_ar:], steps=steps)
        
        # Plot Price forecast
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Price'], label='Actual Price')
        plt.plot(self.data.index[-len(forecast):], forecast[:, 0], label='VAR Price Forecast', color='orange')
        plt.title('VAR Model Forecast for Brent Oil Prices (1987–2022)')
        plt.xlabel('Date')
        plt.ylabel('Price ($/barrel)')
        plt.legend()
        plt.show()
        
        # Plot Returns forecast
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Returns'], label='Actual Returns')
        plt.plot(self.data.index[-len(forecast):], forecast[:, 1], label='VAR Returns Forecast', color='orange')
        plt.title('VAR Model Forecast for Brent Oil Returns (1987–2022)')
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend()
        plt.show()

    def plot_prices_with_events(self):
        """Plot Brent oil prices with shaded event periods."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Price'], label='Brent Price')
        plt.title('Brent Crude Oil Prices (1987–2022) with Events')
        plt.xlabel('Date')
        plt.ylabel('Price ($/barrel)')
        plt.legend()
        
        # Shade major events
        for event, years in [('Gulf_War', (1990, 1991)), ('Financial_Crisis', (2008, 2008)), 
                            ('Covid_Impact', (2020, 2021)), ('Oil_Price_War', (2020, 2020))]:
            plt.axvspan(pd.Timestamp(f'{years[0]}-01-01'), pd.Timestamp(f'{years[1]}-12-31'), 
                       color='gray', alpha=0.2, label=f'{event}' if event == 'Gulf_War' else "")
        plt.legend()
        plt.show()

    def plot_returns_with_events(self):
        """Plot Brent oil returns with shaded event periods."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Returns'], label='Daily Returns', color='red')
        plt.title('Brent Crude Oil Returns (1987–2022) with Events')
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend()
        
        # Shade major events
        for event, years in [('Gulf_War', (1990, 1991)), ('Financial_Crisis', (2008, 2008)), 
                            ('Covid_Impact', (2020, 2021)), ('Oil_Price_War', (2020, 2020))]:
            plt.axvspan(pd.Timestamp(f'{years[0]}-01-01'), pd.Timestamp(f'{years[1]}-12-31'), 
                       color='gray', alpha=0.2, label=f'{event}' if event == 'Gulf_War' else "")
        plt.legend()
        plt.show()

    def analyze_and_plot(self, include_events=True, use_differenced=False):
        """Perform full analysis and generate all plots, optionally including event impacts and differenced data."""
        # Fit models
        arima_results = self.fit_arima_with_events(include_events=include_events)
        garch_results = self.fit_garch()
        var_results = self.fit_var(include_events=include_events, use_differenced=use_differenced)
        
        # Generate plots
        self.plot_prices_and_returns()
        self.plot_arima(arima_results)
        self.plot_garch(garch_results)
        self.plot_var(var_results)
        self.plot_prices_with_events()
        self.plot_returns_with_events()

    def optimize_arima(self, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        """Find optimal ARIMA parameters using AIC."""
        best_aic = float('inf')
        best_order = None
        for p, d, q in itertools.product(p_range, d_range, q_range):
            try:
                model = ARIMA(self.data['Price'], order=(p, d, q))
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                continue
        return best_order, best_aic

    def evaluate_predictions(self, actual, predicted):
        """Evaluate model predictions with common metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        return {'MAE': mae, 'RMSE': rmse, 'R²': r2}
