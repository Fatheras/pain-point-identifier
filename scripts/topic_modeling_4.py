# topic_modeling.py

import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
import json

script_dir = os.path.dirname(__file__)
input_csv_path = os.path.join(script_dir, '..', 'data', 'processed', 'processed_reddit_posts.csv')
data = pd.read_csv(input_csv_path)

data.dropna(subset=['processed_text'], inplace=True)
data = data[data['processed_text'].str.strip() != '']

documents = data['processed_text'].astype(str).tolist()

topic_model = BERTopic(
    verbose=True,
    min_topic_size=3,
    nr_topics=None
)

topics, probs = topic_model.fit_transform(documents)

data['topic'] = topics

topic_info = topic_model.get_topic_info()

custom_labels = {}

# automatically assign labels based on top words
for topic_num in topic_info['Topic'].unique():
    # get top words for the topic
    if topic_num == -1:
        custom_labels[topic_num] = "Miscellaneous Issues"
        continue
    top_words = [word for word, _ in topic_model.get_topic(topic_num)]
    top_words = top_words[:5]  # get the top 5 words

    # assign labels based on top words
    if any(word in top_words for word in ['internet', 'speed', 'slow', 'connection', 'lag', 'disconnect']):
        custom_labels[topic_num] = "Internet Speed and Connectivity Issues"
    elif any(word in top_words for word in ['service', 'customer', 'support', 'help', 'call', 'agent']):
        custom_labels[topic_num] = "Customer Service Complaints"
    elif any(word in top_words for word in ['billing', 'charge', 'bill', 'payment', 'fee', 'refund']):
        custom_labels[topic_num] = "Billing and Payment Issues"
    elif any(word in top_words for word in ['outage', 'disconnect', 'down', 'maintenance', 'interrupt', 'failure']):
        custom_labels[topic_num] = "Service Outages"
    elif any(word in top_words for word in ['installation', 'equipment', 'technician', 'setup', 'appointment', 'schedule']):
        custom_labels[topic_num] = "Installation and Equipment Problems"
    elif any(word in top_words for word in ['modem', 'router', 'wifi', 'gateway', 'signal', 'network']):
        custom_labels[topic_num] = "Modem and Router Issues"
    elif any(word in top_words for word in ['contract', 'cancel', 'agreement', 'term', 'policy', 'renew']):
        custom_labels[topic_num] = "Contract and Cancellation Problems"
    elif any(word in top_words for word in ['app', 'website', 'login', 'account', 'password', 'access']):
        custom_labels[topic_num] = "App and Website Issues"
    else:
        custom_labels[topic_num] = "Other Issues"

data['topic_label'] = data['topic'].map(custom_labels)

# extract common phrases for root cause analysis
def get_top_phrases(texts, n=5):
    if not texts:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')
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
