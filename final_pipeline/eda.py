import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import isodate
import os

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file (CSV or Parquet) and clean it by handling duplicates, missing values, and invalid entries.

    Args:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Detect file extension and load accordingly
    file_extension = os.path.splitext(file_path)[-1].lower()

    print("Loading data...")
    if file_extension == '.csv':
        data = pd.read_csv(file_path)
    elif file_extension == '.parquet':
        data = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}. Supported formats are .csv and .parquet.")

    print(f"Data loaded with shape: {data.shape}")

    # Drop duplicates
    data = data.drop_duplicates()

    # Convert datetime columns
    for col in ['video_published_at', 'video_trending__date', 'channel_published_at']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce', format='mixed')

    # Remove invalid rows
    data = data[~((data['channel_view_count'] == 0) & (data['channel_video_count'] == 0))]

    # Fill missing values
    data['video_description'].fillna("No description available", inplace=True)
    data['channel_description'].fillna("No description available", inplace=True)
    data['video_tags'].fillna("No tags", inplace=True)
    data['channel_country'].fillna("Unknown", inplace=True)
    data['channel_custom_url'].fillna("Unknown", inplace=True)
    data['channel_localized_description'].fillna("No localized description available", inplace=True)
    data['video_view_count'].fillna(0, inplace=True)
    data['video_like_count'].fillna(0, inplace=True)
    data['video_comment_count'].fillna(0, inplace=True)

    # Drop rows with missing critical information
    data = data.dropna(subset=['video_category_id', 'video_view_count'])
    data = data[data['video_view_count'] != 0]

    print(f"Data cleaned. Remaining rows: {len(data)}")
    return data

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features like engagement rate, time difference, and categorical time-of-day flags.

    Args:
        data (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Calculate engagement rate
    data['engagement_rate'] = (
        (data['video_like_count'] + data['video_comment_count']) /
        data['video_view_count']
    )

    # Calculate time difference to trending
    data['time_difference'] = (
        data['video_trending__date'] - data['video_published_at'].dt.tz_localize(None)
    )
    data['time_difference_days'] = data['time_difference'].dt.total_seconds() / (3600 * 24)

    # Extract video duration in seconds
    data['video_duration_seconds'] = data['video_duration'].apply(lambda x: isodate.parse_duration(x).total_seconds() if pd.notnull(x) else 0)
    data['is_weekend'] = pd.to_datetime(data['video_published_at']).dt.weekday >= 5
    # Time-of-day flags
    data['published_hour'] = data['video_published_at'].dt.hour
    data['published_morning'] = data['published_hour'].between(6, 12)
    data['published_afternoon'] = data['published_hour'].between(12, 18)
    data['published_evening'] = data['published_hour'].between(18, 24)
    data['published_night'] = data['published_hour'].between(0, 6)

    # Add channel-video publish difference in days
    data['channel_video_channel_publish_difference'] = (
        data['video_published_at'] - data['channel_published_at']
    ).dt.days

    # Binary is_trending feature
    data['is_trending'] = data['time_difference'].apply(lambda x: 1 if x <= pd.Timedelta(days=1) else 0)

    # Deduplicate videos by keeping the earliest trending record
    data = data.sort_values(by=['video_trending__date']).drop_duplicates(subset=['video_id'], keep='first')

    print(f"Data after deduplication has shape: {data.shape}")
    return data

def perform_eda(data: pd.DataFrame):
    """
    Perform basic exploratory data analysis on the dataset and generate visualizations.

    Args:
        data (pd.DataFrame): DataFrame to analyze.
    """
    print("Performing EDA...")

    # Distribution of engagement rate
    plt.figure(figsize=(8, 5))
    plt.hist(data['engagement_rate'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Engagement Rate')
    plt.xlabel('Engagement Rate')
    plt.ylabel('Frequency')
    plt.show()

    # Video duration vs. engagement rate
    plt.figure(figsize=(8, 5))
    plt.scatter(data['video_duration_seconds'], data['engagement_rate'], alpha=0.5)
    plt.title('Video Duration vs. Engagement Rate')
    plt.xlabel('Video Duration (seconds)')
    plt.ylabel('Engagement Rate')
    plt.show()

    # Time to trend vs. engagement rate
    plt.figure(figsize=(8, 5))
    plt.scatter(data['time_difference_days'], data['engagement_rate'], alpha=0.5)
    plt.title('Time to Trend vs. Engagement Rate')
    plt.xlabel('Time to Trend (days)')
    plt.ylabel('Engagement Rate')
    plt.show()

    print("EDA completed.")
