import React from 'react';

function MetricsPanel({ metrics }) {
    return (
        <div className="metrics-panel">
            <h2>Model Performance Metrics</h2>
            <p>ARIMA RMSE: {metrics.arima?.RMSE || 'N/A'}</p>
            <p>ARIMA MAE: {metrics.arima?.MAE || 'N/A'}</p>
            <p>ARIMA R2: {metrics.arima?.R2 || 'N/A'}</p>
            <p>VAR Returns RMSE: {metrics.var_returns?.RMSE || 'N/A'}</p>
            <p>VAR Returns MAE: {metrics.var_returns?.MAE || 'N/A'}</p>
            <p>VAR Returns R2: {metrics.var_returns?.R2 || 'N/A'}</p>
        </div>
    );
}

export default MetricsPanel;