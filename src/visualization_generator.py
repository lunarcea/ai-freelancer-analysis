"""
Visualization Generator for Digital Labor Market Analysis

This script creates visualizations for skill metrics (depth and diversity),
showing trends over time and the relationship with various labor market outcomes.

This is a generic visualization tool that can be adapted for different research contexts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def remove_outliers(data, column, n_std=3):
    """
    Remove outliers using z-score method
    
    Parameters:
    - data: DataFrame
    - column: string, column name to check for outliers
    - n_std: number of standard deviations to use as threshold
    
    Returns:
    - DataFrame with outliers removed
    """
    z_scores = stats.zscore(data[column])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = abs_z_scores < n_std
    return data[filtered_entries]


def plot_time_series(df, metric_column, title, ylim=None, reference_date=None, 
                     start_date=None, end_date=None):
    """
    Create a time series plot for a metric, showing before/after trends around a reference date.
    
    Parameters:
    - df: DataFrame with the data
    - metric_column: column name containing the metric to plot
    - title: plot title
    - ylim: tuple with y-axis limits
    - reference_date: date to split before/after (e.g., policy change, technology adoption)
    - start_date: earliest date to include in the plot
    - end_date: latest date to include in the plot
    
    Returns:
    - matplotlib figure
    """
    # Create the figure and axis
    plt.figure(figsize=(15, 8))
    
    # Handle default dates
    if reference_date is None:
        # Use the middle of the date range
        reference_date = df['date'].min() + (df['date'].max() - df['date'].min()) / 2
    else:
        reference_date = pd.to_datetime(reference_date)
    
    if start_date is None:
        start_date = df['date'].min()
    else:
        start_date = pd.to_datetime(start_date)
        
    if end_date is None:
        end_date = df['date'].max()
    else:
        end_date = pd.to_datetime(end_date)
    
    # Filter data for the specified date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    plot_data = df[mask].copy()
    
    # Remove outliers from the entire dataset
    plot_data = remove_outliers(plot_data, metric_column, n_std=3)
    
    # Split data into before and after
    before_data = plot_data[plot_data['date'] < reference_date].copy()
    after_data = plot_data[plot_data['date'] >= reference_date].copy()
    
    # Print summary statistics before and after outlier removal
    print("Original data points:", len(df))
    print("Data points after outlier removal:", len(plot_data))
    print("\nSummary statistics after outlier removal:")
    print(plot_data[metric_column].describe())
    
    # Calculate daily means
    before_means = before_data.groupby('date')[metric_column].mean().reset_index()
    after_means = after_data.groupby('date')[metric_column].mean().reset_index()
    
    # Apply smoothing
    if len(before_means) > 5:
        before_smooth = savgol_filter(before_means[metric_column], 
                                    min(5, len(before_means) // 2 * 2 + 1), 3)
    else:
        before_smooth = before_means[metric_column]
        
    if len(after_means) > 5:
        after_smooth = savgol_filter(after_means[metric_column], 
                                   min(5, len(after_means) // 2 * 2 + 1), 3)
    else:
        after_smooth = after_means[metric_column]
    
    # Create the smoothed line plots
    plt.plot(before_means['date'], before_smooth, 
            color='blue', label='Before Reference Date', linewidth=2)
    plt.plot(after_means['date'], after_smooth, 
            color='orange', label='After Reference Date', linewidth=2)
    
    # Add vertical line for the reference date
    plt.axvline(x=reference_date, color='red', linestyle='-', label=f'Reference Date: {reference_date.strftime("%Y-%m-%d")}')
    
    # Calculate days relative to reference date for major tick marks
    days_range = (end_date - start_date).days
    day_interval = max(10, days_range // 10)  # Ensure we don't have too many ticks
    
    date_mapping = {}
    current_date = reference_date - timedelta(days=(days_range // 2))
    while current_date <= end_date:
        days_diff = (current_date - reference_date).days
        date_str = current_date.strftime('%Y-%m-%d')
        label = f'{days_diff:+d} days' if days_diff != 0 else '0'
        date_mapping[date_str] = label
        current_date += timedelta(days=day_interval)
    
    tick_dates = pd.to_datetime(list(date_mapping.keys()))
    plt.xticks(tick_dates, date_mapping.values(), rotation=45)
    
    # Customize the plot
    metric_name = ' '.join(metric_column.split('_')).title()
    plt.ylabel(metric_name)
    plt.xlabel('Days Relative to Reference Date')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis limits if provided
    if ylim:
        plt.ylim(ylim)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    return plt


def plot_relationship_skill_metric_outcome(df, x_metric, y_outcome, 
                                          x_label, y_label, title,
                                          xlim=None, ylim=None):
    """
    Create a scatter plot with trend lines showing the relationship between 
    a skill metric and a labor market outcome, separated by before/after periods.
    
    Parameters:
    - df: DataFrame with the data
    - x_metric: column name for the x-axis (skill metric)
    - y_outcome: column name for the y-axis (outcome) 
    - x_label, y_label: axis labels
    - title: plot title
    - xlim, ylim: axis limits tuples
    
    Returns:
    - matplotlib figure
    """
    # Create a figure with appropriate size
    plt.figure(figsize=(12, 8))
    
    # Ensure we have the 'period' column (before=0/after=1)
    if 'period' not in df.columns:
        # Check if we should use an alternative
        period_column = next((col for col in df.columns if 'period' in col.lower()), None)
        if period_column:
            df['period'] = df[period_column]
        else:
            # Try to infer from date if available
            if 'date' in df.columns:
                reference_date = df['date'].min() + (df['date'].max() - df['date'].min()) / 2
                df['period'] = (df['date'] >= reference_date).astype(int)
            else:
                raise ValueError("No period column found or could be inferred")
    
    # Split data by period
    before_period = df[df['period'] == 0]
    after_period = df[df['period'] == 1]
    
    # Calculate trend lines
    before_slope, before_intercept, before_r, _, _ = stats.linregress(
        before_period[x_metric], before_period[y_outcome]
    )
    
    after_slope, after_intercept, after_r, _, _ = stats.linregress(
        after_period[x_metric], after_period[y_outcome]
    )
    
    # Create x range for trend lines
    if xlim:
        x_range = np.linspace(xlim[0], xlim[1], 100)
    else:
        min_x = min(df[x_metric].min(), 0)
        max_x = df[x_metric].max() * 1.1
        x_range = np.linspace(min_x, max_x, 100)
    
    # Plot scatter points
    plt.scatter(before_period[x_metric], before_period[y_outcome], 
                color='blue', alpha=0.3, s=20, label='Before (Data)')
    plt.scatter(after_period[x_metric], after_period[y_outcome], 
                color='red', alpha=0.3, s=20, label='After (Data)')
    
    # Plot trend lines
    plt.plot(x_range, before_slope * x_range + before_intercept, 
             color='blue', linewidth=3, 
             label=f'Before (Trend, R²={before_r**2:.3f})')
    plt.plot(x_range, after_slope * x_range + after_intercept, 
             color='red', linewidth=3, 
             label=f'After (Trend, R²={after_r**2:.3f})')
    
    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Set axis limits if provided
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    return plt


def main():
    """Main function to demonstrate visualization capabilities."""
    try:
        # Load sample data - replace with your actual file path
        print("Loading data...")
        data_path = "data/input/skill_metrics.csv"
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Data loaded. Shape: {df.shape}")
        
        # Create directory for output if it doesn't exist
        output_dir = "output/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Example 1: Plot skill depth over time
        print("Generating skill depth time series plot...")
        reference_date = pd.to_datetime('2023-01-01')  # Example reference date
        depth_plot = plot_time_series(
            df=df,
            metric_column='individual_skill_depth',
            title='Skill Depth Over Time',
            ylim=(0, 0.3),
            reference_date=reference_date
        )
        depth_plot.savefig(f"{output_dir}/skill_depth_timeseries.png", dpi=300)
        
        # Example 2: Plot skill diversity over time
        print("Generating skill diversity time series plot...")
        diversity_plot = plot_time_series(
            df=df,
            metric_column='skill_diversity',
            title='Skill Diversity Over Time',
            ylim=(0, 1.0),
            reference_date=reference_date
        )
        diversity_plot.savefig(f"{output_dir}/skill_diversity_timeseries.png", dpi=300)
        
        # Example 3: Plot relationship between skill depth and outcome
        print("Generating skill depth vs outcome relationship plot...")
        skill_outcome_plot = plot_relationship_skill_metric_outcome(
            df=df,
            x_metric='skill_depth',
            y_outcome='success_rate',
            x_label='Skill Depth',
            y_label='Success Rate',
            title='Relationship Between Skill Depth and Success Rate',
            xlim=(0, 1),
            ylim=(0, 1)
        )
        skill_outcome_plot.savefig(f"{output_dir}/skill_depth_vs_success.png", dpi=300)
        
        print(f"All visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    main()