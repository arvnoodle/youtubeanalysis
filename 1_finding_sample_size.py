import pandas as pd 
import numpy as np

from scipy.stats import chisquare
import matplotlib.pyplot as plt

df = pd.read_csv('youtube_trending_videos_global.csv')
df = df.dropna(subset=['video_view_count', 'video_like_count', 'video_comment_count'])

# checking ano sample size, laki smyado
sample_sizes = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000]

results = []

for size in sample_sizes:

    sample = df.sample(n=size, random_state=42)
# Used to test if different populations have the same proportion of individuals with some characteristic.
    category_full = df['video_category_id'].value_counts(normalize=True)
    category_sample = sample['video_category_id'].value_counts(normalize=True)  # 

    category_sample = category_sample.reindex(category_full.index, fill_value=0)

    chi_stat, chi_p = chisquare(f_obs=category_sample, f_exp=category_full)
    results.append({'sample_size': size, 'Chi_stat': chi_stat, 'Chi_p': chi_p})

results_df = pd.DataFrame(results)

plt.plot(results_df['sample_size'], results_df['Chi_stat'], marker='o', label='Chi-square Stat')
plt.xlabel('Sample Size')
plt.ylabel('Chi-square Statistic')
plt.title('Chi-square Test for Category Proportions')
plt.legend()
plt.grid()
plt.show()
print(results_df) # pinili ko nalang 150000 haha # considering ano nalang kaya ng laptop ko


sample_size = 150000

#data_csv = df.sample(n=sample_size, random_state=42)
data_csv = df.copy()
print(data_csv.video_trending_country.value_counts())
# i think the dataset is large enough naman is be ok even at random_sample, authenticity reasons and what not
# but ito straitified 
# stratified_sample = df.groupby('video_trending_country', group_keys=False).apply(
#     lambda x: x.sample(min(len(x), sample_size // df['video_trending_country'].nunique()), random_state=42)
# )
print(data_csv.isnull().sum())

data_csv['video_published_at'] = pd.to_datetime(data_csv['video_published_at'], errors='coerce', format='mixed')
data_csv['video_trending__date'] = pd.to_datetime(data_csv['video_trending__date'], errors='coerce',format='mixed')
data_csv['channel_published_at'] = pd.to_datetime(data_csv['channel_published_at'], errors='coerce',format='mixed')

# sanity check alng
numeric_columns = ['video_view_count', 'video_like_count', 'video_comment_count',
                   'channel_view_count', 'channel_subscriber_count', 'channel_video_count']

invalid_values = {}
for col in numeric_columns:
    invalid_values[col] = {
        "negative_values": (data_csv[col] < 0).sum(),
        "zero_values": (data_csv[col] == 0).sum()
    }
    
    
print(invalid_values)

# checking baka may zero view/video count
zero_channel_views = data_csv[data_csv['channel_view_count'] == 0]
zero_channel_videos = data_csv[data_csv['channel_video_count'] == 0]

zero_channel_views_info = zero_channel_views[['channel_title', 'channel_view_count', 'channel_video_count', 'channel_subscriber_count']]
zero_channel_videos_info = zero_channel_videos[['channel_title', 'channel_view_count', 'channel_video_count', 'channel_subscriber_count']]

print(zero_channel_views_info)

data_cleaned = data_csv[~((data_csv['channel_view_count'] == 0) & (data_csv['channel_video_count'] == 0))]
rows_removed = len(data_csv) - len(data_cleaned)
remaining_rows = len(data_cleaned)
data_cleaned['video_description'].fillna("No description available", inplace=True)
data_cleaned['channel_description'].fillna("No description available", inplace=True)
data_cleaned['video_tags'].fillna("None", inplace=True)
data_cleaned['channel_country'].fillna("Unknown", inplace=True)
data_cleaned = data_cleaned.dropna(subset=['video_category_id'])
remaining_rows_after_cleaning = len(data_cleaned)

print(rows_removed)
print(remaining_rows)
print(remaining_rows_after_cleaning)
print(data_cleaned.isna().sum())

data_cleaned['channel_custom_url'].fillna("Unknown", inplace=True)
data_cleaned['channel_localized_description'].fillna("No localized description available", inplace=True)

data_cleaned.to_csv('post_processed_data_cleaned.csv', index=False) # done