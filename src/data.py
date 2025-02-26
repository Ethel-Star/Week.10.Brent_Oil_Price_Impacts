import pandas as pd
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data (run once)
nltk.download('vader_lexicon')

class BrentDataLoader:
    def __init__(self, filepath):
        """
        Initialize the BrentDataLoader with a filepath to the data CSV.
        
        Args:
            filepath (str): Path to the merged_brent_events.csv file.
        """
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """
        Load Brent oil price data from CSV.
        
        Returns:
            pd.DataFrame: Loaded and indexed data.
        
        Raises:
            FileNotFoundError: If the data file is not found at the specified path.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Data file not found at {self.filepath}")
        self.data = pd.read_csv(self.filepath)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        return self.data

    def preprocess_data(self):
        """
        Clean and preprocess the data, including events, returns, and sentiment analysis.
        
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        if self.data is None:
            self.load_data()
        
        # Handle missing values for numerical columns
        self.data['Price'].fillna(method='ffill', inplace=True)
        self.data['Returns'] = self.data['Price'].pct_change() * 100
        
        # Process events
        self.process_events()
        self.add_event_dummies()
        self.analyze_event_sentiment()
        
        # Ensure 'Event_Type' and 'Event_Description' are strings
        self.data['Event_Type'] = self.data['Event_Type'].astype(str).fillna('None')
        self.data['Event_Description'] = self.data['Event_Description'].astype(str).fillna('None')
        
        return self.data

    def process_events(self):
        """
        Process Event_Type and Event_Description for analysis, cleaning and standardizing them.
        
        Returns:
            pd.DataFrame: Data with processed events.
        """
        self.data['Event_Type'] = self.data['Event_Type'].str.strip().str.title().fillna('None')
        self.data['Event_Description'] = self.data['Event_Description'].str.strip().fillna('None')
        return self.data

    def add_event_dummies(self):
        """
        Add dummy variables for major historical events affecting Brent oil prices.
        
        Returns:
            pd.DataFrame: Data with event dummy variables.
        """
        self.data['Gulf_War'] = ((self.data.index.year >= 1990) & (self.data.index.year <= 1991)).astype('int32')  # Gulf War (1990–1991)
        self.data['Financial_Crisis'] = (self.data.index.year == 2008).astype('int32')  # 2008 Financial Crisis
        self.data['Covid_Impact'] = ((self.data.index.year >= 2020) & (self.data.index.year <= 2021)).astype('int32')  # COVID-19 Impact (2020–2021)
        self.data['Oil_Price_War'] = (self.data.index.year == 2020).astype('int32')  # 2020 Oil Price War (specific to early 2020)
        return self.data

    def analyze_event_sentiment(self):
        """
        Analyze sentiment in Event_Description using VADER sentiment analyzer.
        
        Returns:
            pd.DataFrame: Data with added Event_Sentiment column (-1 to 1, compound score).
        """
        sia = SentimentIntensityAnalyzer()
        self.data['Event_Sentiment'] = self.data['Event_Description'].apply(lambda x: sia.polarity_scores(x)['compound'])
        return self.data

    def get_data(self):
        """
        Return the processed data, loading and preprocessing if not already done.
        
        Returns:
            pd.DataFrame: Fully processed data.
        """
        if self.data is None:
            self.preprocess_data()
        return self.data

    def load_external_data(self, external_filepath, merge_on='Date'):
        """
        Load external economic indicators or events and merge with Brent data.
        
        Args:
            external_filepath (str): Path to the external CSV file.
            merge_on (str): Column name to merge on (default 'Date').
        
        Returns:
            pd.DataFrame: Merged data.
        """
        external_data = pd.read_csv(external_filepath)
        external_data[merge_on] = pd.to_datetime(external_data[merge_on])
        external_data.set_index(merge_on, inplace=True)
        self.data = self.data.join(external_data, how='left').dropna()
        return self.data
