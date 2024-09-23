import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'twitter_training.csv'
data = pd.read_csv(file_path)

# Check for missing values and unique values in the "feeling" column
print("Missing values in each column:")
print(data.isnull().sum())

print("\nUnique values in the 'feeling' column:")
print(data['feeling'].unique())

# Drop rows with missing comments
data_cleaned = data.dropna(subset=['comment'])

# Visualize the distribution of sentiments
sentiment_counts = data_cleaned['feeling'].value_counts()

# Plotting the sentiment distribution
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'gray','black'])
plt.title('Distribution of Sentiments in the Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.xticks(rotation=45)
plt.show()