import os
import pandas as pd

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'data', 'processed_reddit_posts.csv')

data = pd.read_csv(csv_path)

all_words = ' '.join(data['processed_text']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)
words, counts = zip(*common_words)

plt.figure(figsize=(12, 8))
sns.barplot(y=list(words), x=list(counts))
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()

data['text_length'] = data['processed_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], bins=50)
plt.title('Distribution of Post Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Number of Posts')
plt.show()
