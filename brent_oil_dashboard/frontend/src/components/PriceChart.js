import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceArea } from 'recharts';

function PriceChart({ data, events, forecasts }) {
    // Prepare data for charting
    const chartData = data.map(item => ({
        date: item.Date,
        price: item.Price,
        returns: item.Returns,
    }));

    // Prepare event annotations
    const eventAnnotations = events.map(item => ({
        x1: item.Date,
        x2: item.Date,
        y1: 0,
        y2: 150, // Adjust max price range based on your data
        fill: 'gray',
        opacity: 0.2,
        label: item.Event_Description,
    }));

    // Prepare forecasts (simplified for Recharts)
    const forecastData = chartData.slice(-252).map((d, i) => ({
        ...d,
        arima: forecasts.arima[i] || d.price,
        var_price: forecasts.var_price[i] || d.price,
    }));

    return (
        <LineChart width={800} height={400} data={chartData.concat(forecastData)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" type="category" allowDuplication={false} />
            <YAxis domain={[0, 150]} /> // Adjust domain based on your data
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="price" stroke="#8884d8" name="Brent Price" />
            <Line type="monotone" dataKey="returns" stroke="#82ca9d" name="Returns" />
            <Line type="monotone" dataKey="arima" stroke="#ff7300" name="ARIMA Forecast" />
            <Line type="monotone" dataKey="var_price" stroke="#ff7300" name="VAR Price Forecast" />
            {eventAnnotations.map((event, index) => (
                <ReferenceArea key={index} {...event} />
            ))}
        </LineChart>
    );
}

export default PriceChart;