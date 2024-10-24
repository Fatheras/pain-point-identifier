import os
import pandas as pd
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch

warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")

device = 0 if torch.cuda.is_available() else -1

# initialize sentiment analysis pipeline with truncation
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline(
    'sentiment-analysis',
    model=model,
    tokenizer=tokenizer,
    device=device
)

script_dir = os.path.dirname(__file__)
data_file = os.path.join(script_dir, '..', 'data', 'processed', 'topic_modelled_posts.csv')
data = pd.read_csv(data_file)

if 'text' not in data.columns:
    raise ValueError("The column 'text' was not found in the data.")

# remove rows with empty or NaN 'text'
data.dropna(subset=['text'], inplace=True)
data = data[data['text'].str.strip() != '']

# reset index after dropping rows
data.reset_index(drop=True, inplace=True)

# apply sentiment analysis in batches
def get_sentiments_and_scores(results):
    sentiments = []
    sentiment_scores = []
    for result in results:
        label = result['label']
        score = result['score']
        # map labels to 'Positive', 'Neutral', 'Negative' and assign scores
        if label in ['1 star', '2 stars']:
            sentiments.append('Negative')
            # assign negative scores
            if label == '1 star':
                sentiment_scores.append(-1 * score)
            else:
                sentiment_scores.append(-0.5 * score)
        elif label == '3 stars':
            sentiments.append('Neutral')
            sentiment_scores.append(0)
        else:
            sentiments.append('Positive')
            # assign positive scores
            if label == '5 stars':
                sentiment_scores.append(1 * score)
            else:
                sentiment_scores.append(0.5 * score)
    return sentiments, sentiment_scores

# process texts in batches
batch_size = 16
sentiments = []
sentiment_scores = []

print("Starting sentiment analysis...")
for i in tqdm(range(0, len(data), batch_size)):
    batch_texts = data['text'].iloc[i:i+batch_size].tolist()
    # limit each text to 10,000 characters to avoid extremely long inputs
    batch_texts = [text[:10000] for text in batch_texts]
    results = sentiment_pipeline(batch_texts, truncation=True, max_length=512)
    batch_sentiments, batch_scores = get_sentiments_and_scores(results)
    sentiments.extend(batch_sentiments)
    sentiment_scores.extend(batch_scores)

data['sentiment'] = sentiments
data['sentiment_score'] = sentiment_scores

output_csv_path = os.path.join(script_dir, '..', 'data', 'processed', 'sentiment_analysed_posts.csv')
data.to_csv(output_csv_path, index=False)
print(f"Sentiment analysis complete. Saved to {output_csv_path}.")
