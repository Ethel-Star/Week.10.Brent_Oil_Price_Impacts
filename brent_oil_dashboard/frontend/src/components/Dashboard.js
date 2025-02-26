import React, { useState, useEffect } from 'react';
import axios from 'axios';
import PriceChart from './PriceChart';
import EventFilter from './EventFilter';
import MetricsPanel from './MetricsPanel';
import './styles/Dashboard.css';

function Dashboard() {
    const [data, setData] = useState({ prices: [], events: [], forecasts: {}, metrics: {} });
    const [filters, setFilters] = useState({ startDate: '1987-01-01', endDate: '2022-12-31', event: '' });

    useEffect(() => {
        fetchData();
    }, [filters]);

    const fetchData = async () => {
        try {
            const prices = await axios.get(`http://localhost:5000/api/historical_prices?start_date=${filters.startDate}&end_date=${filters.endDate}`);
            const events = await axios.get(`http://localhost:5000/api/events?event=${filters.event}`);
            const forecasts = await axios.get('http://localhost:5000/api/forecasts');
            const metrics = await axios.get('http://localhost:5000/api/metrics');
            setData({ prices: prices.data, events: events.data, forecasts: forecasts.data, metrics: metrics.data });
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    };

    return (
        <div className="dashboard">
            <h1>Brent Oil Price Dashboard (1987–2022)</h1>
            <EventFilter onFilterChange={setFilters} />
            <PriceChart data={data.prices} events={data.events} forecasts={data.forecasts} />
            <MetricsPanel metrics={data.metrics} />
        </div>
    );
}

export default Dashboard;