import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json

script_dir = os.path.dirname(__file__)
input_csv_path = os.path.join(script_dir, '..', 'data', 'processed', 'processed_reddit_posts.csv')
data = pd.read_csv(input_csv_path)

# remove rows where 'processed_text' is NaN or empty
data.dropna(subset=['processed_text'], inplace=True)
data = data[data['processed_text'].str.strip() != '']

if 'processed_text' not in data.columns:
    raise ValueError("The column 'processed_text' was not found in the data.")

# convert the 'processed_text' column to a list of documents
documents = data['processed_text'].astype(str).tolist()

topic_model = BERTopic(
    verbose=True,
    min_topic_size=10,
    nr_topics=None      # ensure topics are not merged
)

# fit the model
topics, probs = topic_model.fit_transform(documents)

data['topic'] = topics

topic_info = topic_model.get_topic_info()

topics_keywords = topic_model.get_topics()

custom_labels = {}

for topic_num in topics_keywords:
    if topic_num == -1:
        continue  # skip outliers
    top_words = [word for word, _ in topics_keywords[topic_num][:5]]
    if 'internet' in top_words or 'speed' in top_words:
        custom_labels[topic_num] = "Internet Speed Issues"
    elif 'service' in top_words or 'customer' in top_words:
        custom_labels[topic_num] = "Customer Service Complaints"
    elif 'billing' in top_words or 'charge' in top_words:
        custom_labels[topic_num] = "Billing Issues"
    elif 'outage' in top_words or 'disconnect' in top_words:
        custom_labels[topic_num] = "Service Outages"
    elif 'installation' in top_words or 'equipment' in top_words:
        custom_labels[topic_num] = "Installation and Equipment Problems"
    elif 'modem' in top_words or 'router' in top_words:
        custom_labels[topic_num] = "Modem and Router Issues"
    elif 'contract' in top_words or 'cancel' in top_words:
        custom_labels[topic_num] = "Contract and Cancellation Problems"
    else:
        custom_labels[topic_num] = "Other Issues"

data['topic_label'] = data['topic'].map(custom_labels)

def get_top_phrases(texts, n=5):
    if not texts:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(2,3), stop_words='english')
    try:
        X = vectorizer.fit_transform(texts)
        if X.shape[1] == 0:
            return []
        indices = X.sum(axis=0).A1.argsort()[-n:]
        features = vectorizer.get_feature_names_out()
        top_phrases = [features[i] for i in reversed(indices)]
        return top_phrases
    except ValueError as e:
        print(f"Error in get_top_phrases: {e}")
        return []

topic_phrases = {}

for topic_label in data['topic_label'].unique():
    topic_texts = data[data['topic_label'] == topic_label]['processed_text'].dropna()
    topic_texts = [text for text in topic_texts.tolist() if text.strip() != '']
    if not topic_texts:
        print(f"No valid texts found for topic '{topic_label}', skipping top phrases extraction.")
        topic_phrases[topic_label] = []
        continue
    top_phrases = get_top_phrases(topic_texts)
    topic_phrases[topic_label] = top_phrases

output_csv_path = os.path.join(script_dir, '..', 'data', 'processed', 'topic_modelled_posts.csv')
data.to_csv(output_csv_path, index=False)
print(f"Topic modeling complete. Saved to {output_csv_path}.")

phrases_file = os.path.join(script_dir, '..', 'data', 'processed', 'topic_phrases.json')
with open(phrases_file, 'w') as f:
    json.dump(topic_phrases, f)
print(f"Top phrases per topic saved to {phrases_file}.")

model_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'bertopic_model')
topic_model.save(model_path)
print(f"BERTopic model saved to {model_path}.")
