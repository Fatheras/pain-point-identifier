import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

script_dir = os.path.dirname(__file__)
data_file = os.path.join(script_dir, '..', 'data', 'topic_modelled_posts.csv')

data = pd.read_csv(data_file)

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

data['sentiment'] = data['text'].apply(get_sentiment)

output_csv_path = os.path.join(script_dir, '..', 'data', 'sentiment_analysed_posts.csv')
data.to_csv(output_csv_path, index=False)

print(f"Sentiment analysis complete. Saved to {output_csv_path}.")
