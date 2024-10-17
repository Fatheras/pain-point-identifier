import os
import pandas as pd
import numpy as np
import re
import nltk
nltk.data.path.append('C:/Users/xset9/AppData/Roaming/nltk_data')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'data', 'reddit_posts.csv')

data = pd.read_csv(csv_path)
data.drop_duplicates(subset='id', inplace=True)
data.reset_index(drop=True, inplace=True)

# combine title and body for further processing
data['text'] = data['title'].fillna('') + ' ' + data['body'].fillna('')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    return text.strip()

data['clean_text'] = data['text'].apply(clean_text)

# download NLTK data files if not already present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text, preserve_line=True)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['processed_text'] = data['clean_text'].apply(preprocess_text)

data.to_csv('data/processed_reddit_posts.csv', index=False)
print("Data preprocessing complete. Saved to processed_reddit_posts.csv.")
