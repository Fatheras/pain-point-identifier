import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # remove newlines
    text = text.replace('\n', ' ')
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(tokens)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    input_csv_path = os.path.join(script_dir, '..', 'data', 'raw', 'reddit_posts.csv')
    data = pd.read_csv(input_csv_path)

    # remove any rows with missing 'text'
    data.dropna(subset=['text'], inplace=True)
    # remove non-English posts (simple heuristic)
    data = data[data['text'].apply(lambda x: re.match(r'^[a-zA-Z0-9\s\.,!?\'\"]+$', x) is not None)]
    data['processed_text'] = data['text'].apply(preprocess_text)

    output_csv_path = os.path.join(script_dir, '..', 'data', 'processed', 'processed_reddit_posts.csv')
    data.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")
