import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_missing_values(df):
    """Check for missing values in the DataFrame."""
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]

def summary_statistics(df):
    """Compute summary statistics for the DataFrame."""
    return df.describe()

def plot_histogram(df, column='Price', ax=None):
    """Plot histogram of a specified column."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[column], bins=50, color='skyblue', edgecolor='black')
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(f'{column} (USD)')
    ax.set_ylabel('Frequency')
    if ax is None:
        plt.show()

def plot_line_chart(df, x='Date', y='Price', ax=None):
    """Plot line chart of a specified column over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[x], df[y], color='green', lw=2)
    ax.set_title(f'{y} Trend Over Time')
    ax.set_xlabel(x)
    ax.set_ylabel(f'{y} (USD)')
    ax.grid(True)
    if ax is None:
        plt.show()

def plot_box_plot(df, column='Price', ax=None):
    """Plot box plot of a specified column."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[column], color='orange', ax=ax)
    ax.set_title(f'Box Plot of {column}')
    ax.set_xlabel(f'{column} (USD)')
    if ax is None:
        plt.show()

def eda_analysis(df, columns=['Price']):
    """Perform EDA with missing values, stats, and plots."""
    missing_values = check_missing_values(df)
    print("Missing Values:".rjust(30))
    if not missing_values.empty:
        print(missing_values.to_string().rjust(40))
    else:
        print("No missing values found.".rjust(40))
    
    print("Summary Statistics:".rjust(30))
    print(summary_statistics(df[columns]).to_string(justify='right'))
    
    plt.style.use('seaborn-v0_8')  # Updated style
    fig, axs = plt.subplots(3, 1, figsize=(10, 24))
    plot_histogram(df, columns[0], axs[0])
    plot_line_chart(df, 'Date', columns[0], axs[1])
    plot_box_plot(df, columns[0], axs[2])
    plt.tight_layout(pad=3.0)
    plt.show()