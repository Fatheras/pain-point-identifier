import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import json

script_dir = os.path.dirname(__file__)
input_csv_path = os.path.join(script_dir, '..', 'data', 'processed', 'processed_reddit_posts.csv')
data = pd.read_csv(input_csv_path)

# remove rows where 'processed_text' is NaN or empty
data.dropna(subset=['processed_text'], inplace=True)
data = data[data['processed_text'].str.strip() != '']

documents = data['processed_text'].astype(str).tolist()

# initialize BERTopic model with adjusted parameters
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=5,  # Reduced from 15 to 5
    verbose=True
)

topics, probs = topic_model.fit_transform(documents)

data['topic'] = topics
data['topic_probability'] = probs

topic_info = topic_model.get_topic_info()

# create a mapping from topic number to descriptive label
custom_labels = {}

# automatically assign labels based on top words
for topic_num in topic_info['Topic'].unique():
    # get top words for the topic
    if topic_num == -1:
        custom_labels[topic_num] = "Other Issues"
        continue
    top_words = [word for word, _ in topic_model.get_topic(topic_num)]
    top_words = top_words[:10]  # get the top 10 words

    # assign labels based on top words
    if any(word in top_words for word in ['internet', 'speed', 'slow', 'connection', 'lag', 'disconnect', 'drop', 'connectivity']):
        custom_labels[topic_num] = "Internet Speed and Connectivity Issues"
    elif any(word in top_words for word in ['service', 'customer', 'support', 'help', 'call', 'agent', 'representative', 'wait']):
        custom_labels[topic_num] = "Customer Service Complaints"
    elif any(word in top_words for word in ['billing', 'charge', 'bill', 'payment', 'fee', 'refund', 'pay', 'overcharged']):
        custom_labels[topic_num] = "Billing and Payment Issues"
    elif any(word in top_words for word in ['outage', 'disconnect', 'down', 'maintenance', 'interrupt', 'failure', 'unavailable']):
        custom_labels[topic_num] = "Service Outages"
    elif any(word in top_words for word in ['installation', 'equipment', 'technician', 'setup', 'appointment', 'schedule', 'install']):
        custom_labels[topic_num] = "Installation and Equipment Problems"
    elif any(word in top_words for word in ['modem', 'router', 'wifi', 'gateway', 'signal', 'network', 'wireless', 'connect']):
        custom_labels[topic_num] = "Modem and Router Issues"
    elif any(word in top_words for word in ['contract', 'cancel', 'agreement', 'term', 'policy', 'renew', 'termination']):
        custom_labels[topic_num] = "Contract and Cancellation Problems"
    elif any(word in top_words for word in ['app', 'website', 'login', 'account', 'password', 'access', 'site', 'portal']):
        custom_labels[topic_num] = "App and Website Issues"
    elif any(word in top_words for word in ['email', 'security', 'phishing', 'spam', 'hack', 'breach', 'privacy']):
        custom_labels[topic_num] = "Account and Security Issues"
    elif any(word in top_words for word in ['mobile', 'cell', 'phone', 'device', 'data', 'plan', 'text', 'call']):
        custom_labels[topic_num] = "Mobile Service Issues"
    elif any(word in top_words for word in ['cable', 'tv', 'channel', 'streaming', 'service', 'video', 'television']):
        custom_labels[topic_num] = "Cable and Streaming Issues"
    else:
        custom_labels[topic_num] = "Other Issues"

data['topic_label'] = data['topic'].map(custom_labels)

output_csv_path = os.path.join(script_dir, '..', 'data', 'processed', 'topic_modelled_posts.csv')
data.to_csv(output_csv_path, index=False)
print(f"Topic modeling complete. Saved to {output_csv_path}.")

# save the topic phrases for use in insights
from sklearn.feature_extraction.text import TfidfVectorizer

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

# collect top phrases for each topic
topic_phrases = {}

for topic_label in data['topic_label'].unique():
    topic_texts = data[data['topic_label'] == topic_label]['processed_text'].dropna()
    # convert to list and remove empty strings
    topic_texts = [text for text in topic_texts.tolist() if text.strip() != '']
    if not topic_texts:
        print(f"No valid texts found for topic '{topic_label}', skipping top phrases extraction.")
        topic_phrases[topic_label] = []
        continue
    top_phrases = get_top_phrases(topic_texts)
    topic_phrases[topic_label] = top_phrases

phrases_file = os.path.join(script_dir, '..', 'data', 'processed', 'topic_phrases.json')
with open(phrases_file, 'w') as f:
    json.dump(topic_phrases, f)
print(f"Top phrases per topic saved to {phrases_file}.")

model_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'bertopic_model')
topic_model.save(model_path)
print(f"BERTopic model saved to {model_path}.")
