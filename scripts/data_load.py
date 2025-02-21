import pandas as pd

def load_brent_data(file_path):
    """
    Load Brent oil price data from a CSV with 'Date' and 'Price' columns.
    
    :param file_path: Path to CSV file (e.g., 'Date,Price\n20-May-87,18.63\n...').
    :return: DataFrame with 'Date' (datetime) and 'Price' (numeric) columns.
    """
    # Load CSV with explicit column names if needed, parsing 'Date'
    df = pd.read_csv(file_path)
    
    # Ensure 'Date' is parsed as datetime with DD-MMM-YY format
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%y", errors='coerce')
    
    # Ensure 'Price' is numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Drop rows with NaT in 'Date' or NaN in 'Price'
    df_cleaned = df.dropna(subset=['Date', 'Price'])
    
    # Save the cleaned data to a new CSV file
    df_cleaned.to_csv("cleaned_brent_oil_data.csv", index=False)
    
    return df_cleaned

