import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import './App.css';

function App() {
  const [data, setData] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [volatility, setVolatility] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [dateRange, setDateRange] = useState({ start: '1987-01-01', end: '2022-12-31' });

  useEffect(() => {
    // Fetch historical data
    axios.get(`http://localhost:5000/api/data?start_date=${dateRange.start}&end_date=${dateRange.end}`)
      .then(res => setData(res.data));
    // Fetch LSTM forecast
    axios.get('http://localhost:5000/api/lstm_forecast')
      .then(res => setForecast(res.data));
    // Fetch volatility
    axios.get('http://localhost:5000/api/volatility')
      .then(res => setVolatility(res.data));
    // Fetch metrics
    axios.get('http://localhost:5000/api/metrics')
      .then(res => setMetrics(res.data));
  }, [dateRange]);

  const handleDateChange = (e) => {
    setDateRange({ ...dateRange, [e.target.name]: e.target.value });
  };

  return (
    <div className="App">
      <h1>Brent Oil Price Dashboard</h1>

      {/* Date Range Filter */}
      <div>
        <label>Start Date: </label>
        <input type="date" name="start" value={dateRange.start} onChange={handleDateChange} />
        <label>End Date: </label>
        <input type="date" name="end" value={dateRange.end} onChange={handleDateChange} />
      </div>

      {/* Price Chart with Events */}
      <h2>Historical Prices & Forecast</h2>
      <LineChart width={800} height={400} data={data.map((d, i) => ({
        date: d.Date,
        price: d.Price,
        forecast: forecast.forecast && i >= data.length - forecast.forecast.length ? forecast.forecast[i - (data.length - forecast.forecast.length)] : null,
        gulf_war: d.Gulf_War,
        financial_crisis: d.Financial_Crisis,
        covid_impact: d.Covid_Impact,
        oil_price_war: d.Oil_Price_War
      }))}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="price" stroke="#8884d8" name="Price" />
        <Line type="monotone" dataKey="forecast" stroke="#ff7300" name="LSTM Forecast" />
      </LineChart>

      {/* Volatility Chart */}
      <h2>Volatility (GARCH)</h2>
      <LineChart width={800} height={400} data={volatility.dates.map((d, i) => ({ date: d, volatility: volatility.volatility[i] }))}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="volatility" stroke="#82ca9d" name="Volatility" />
      </LineChart>

      {/* Metrics Display */}
      <h2>Model Performance</h2>
      <div>
        <p>MAE: {metrics.MAE?.toFixed(2)}</p>
        <p>RMSE: {metrics.RMSE?.toFixed(2)}</p>
        <p>R²: {metrics['R²']?.toFixed(2)}</p>
      </div>
    </div>
  );
}

export default App;