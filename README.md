
# Brent Oil Price Analysis

This project analyzes Brent oil prices (1987–2022) to detect significant changes and associate them with key events such as political decisions, conflicts, sanctions, and OPEC policies. Developed for Birhan Energies, a consultancy firm providing data-driven insights to energy sector stakeholders, it aims to support investors, policymakers, and energy companies in decision-making.

## Business Objective

- **Goal**: Investigate the impact of major events on Brent oil prices using change point analysis and statistical modeling.
- **Stakeholders**: Investors (risk management), policymakers (energy security), energy companies (operational planning).
- **Key Tasks**: Identify change points, measure event impacts, and forecast short-term price trends.

## Prerequisites

- **Python Version**: 3.9 or higher
- **Dependencies**: Listed in `requirements.txt`. Install with:
  ```bash
  pip install -r requirements.txt
  Week.10.Brent_Oil_Price_Impacts/
├── Data/
│   ├── BrentOilPrices.csv         # Input price data
│   ├── merged_brent_events.csv    # Merged price and event data (generated)
│   └── key_events_1987_2022.csv   # Event data (generated)
├── scripts/
│   ├── data_load.py              # Loads and cleans price data
│   ├── data_events.py            # Generates event data
│   ├── eda_utils.py              # EDA functions with plotting
│   └── timeseries_analysis.py    # Time series modeling and change point detection
├── data_merge.ipynb              # Merges price and event data
├── data_analysis.ipynb           # Main analysis and visualization
├── requirements.txt              # Python dependencies
└── README.md                     # This file