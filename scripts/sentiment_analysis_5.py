import os
import pandas as pd
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch

# suppress specific warnings
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")

# check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# initialize sentiment analysis pipeline with truncation
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

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
def get_sentiments(texts):
    # tokenization and truncation will be handled automatically by the pipeline
    results = sentiment_pipeline(texts, truncation=True, max_length=512)
    sentiments = []
    for result in results:
        label = result['label']
        # map labels to 'Positive', 'Neutral', 'Negative'
        if label in ['1 star', '2 stars']:
            sentiments.append('Negative')
        elif label == '3 stars':
            sentiments.append('Neutral')
        else:
            sentiments.append('Positive')
    return sentiments

# process texts in batches
batch_size = 16
sentiments = []

print("Starting sentiment analysis...")
for i in tqdm(range(0, len(data), batch_size)):
    batch_texts = data['text'].iloc[i:i+batch_size].tolist()
    # limit each text to 10,000 characters to avoid extremely long inputs
    batch_texts = [text[:10000] for text in batch_texts]
    batch_sentiments = get_sentiments(batch_texts)
    sentiments.extend(batch_sentiments)

# add sentiments to the DataFrame
data['sentiment'] = sentiments

output_csv_path = os.path.join(script_dir, '..', 'data', 'processed', 'sentiment_analysed_posts.csv')
data.to_csv(output_csv_path, index=False)
print(f"Sentiment analysis complete. Saved to {output_csv_path}.")
