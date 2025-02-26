import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class LSTMOilPredictor:
    def __init__(self, data, look_back=60):
        self.data = data['Price'].values.reshape(-1, 1)
        self.look_back = look_back
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self):
        """Prepare data for LSTM (scale and create sequences)."""
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i - self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def prepare_data_with_events(self, include_events=True):
        """Prepare data for LSTM, optionally including event features."""
        scaled_price = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(self.look_back, len(scaled_price)):
            price_seq = scaled_price[i - self.look_back:i, 0]
            if include_events:
                event_features = self.data.iloc[i - self.look_back:i][['Gulf_War', 'Financial_Crisis', 'Covid_Impact', 'Oil_Price_War', 'Event_Sentiment']].values
                combined = np.column_stack((price_seq, event_features.mean(axis=0)))  # Aggregate events over look_back
                X.append(combined)
            else:
                X.append(price_seq)
            y.append(scaled_price[i, 0])
        X, y = np.array(X), np.array(y)
        if include_events:
            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        else:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def prepare_data_from_subset(self, data):
        """Helper method to prepare data from a subset for cross-validation."""
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i - self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def build_model(self, units=50):
        """Build and compile the LSTM model with customizable units."""
        if self.model is None:
            self.model = Sequential([
                LSTM(units, return_sequences=True, input_shape=(self.look_back, 1 if not self.data.shape[1] > 1 else self.data.shape[1])),
                LSTM(units),
                Dense(1)
            ])
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return self.model

    def train_model(self, X, y, epochs=50, batch_size=32):
        """Train the LSTM model."""
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        """Make predictions and inverse scale."""
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)

    def tune_lstm(self, param_grid=None, n_splits=5, include_events=False):
        """Tune LSTM hyperparameters using time series cross-validation, optionally including events."""
        if param_grid is None:
            param_grid = {
                'look_back': [30, 60, 252],  # Daily: 30 days, 60 days, 1 year
                'units': [32, 50, 100],
                'epochs': [20, 50, 100],
                'batch_size': [16, 32, 64]
            }
        
        best_mse = float('inf')
        best_params = None
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for look_back in param_grid['look_back']:
            for units in param_grid['units']:
                for epochs in param_grid['epochs']:
                    for batch_size in param_grid['batch_size']:
                        mse_scores = []
                        for train_idx, test_idx in tscv.split(self.data):
                            train_data = self.data.iloc[train_idx]['Price'].values.reshape(-1, 1)
                            test_data = self.data.iloc[test_idx]['Price'].values.reshape(-1, 1)
                            
                            self.look_back = look_back
                            if include_events:
                                X_train, y_train = self.prepare_data_with_events(include_events=True)
                                X_test, y_test = self.prepare_data_with_events(include_events=True)
                            else:
                                X_train, y_train = self.prepare_data_from_subset(train_data)
                                X_test, y_test = self.prepare_data_from_subset(test_data)
                            
                            model = self.build_model(units=units)
                            self.train_model(X_train, y_train, epochs=epochs, batch_size=batch_size)
                            predictions = self.predict(X_test)
                            mse = mean_squared_error(self.scaler.inverse_transform(y_test), predictions)
                            mse_scores.append(mse)
                        
                        avg_mse = np.mean(mse_scores)
                        if avg_mse < best_mse:
                            best_mse = avg_mse
                            best_params = {'look_back': look_back, 'units': units, 'epochs': epochs, 'batch_size': batch_size}
        
        self.look_back = best_params['look_back']
        self.build_model(units=best_params['units'])
        return best_params, best_mse
