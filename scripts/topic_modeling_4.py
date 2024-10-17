import os
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'data', 'processed_reddit_posts.csv')

data = pd.read_csv(csv_path)

if 'processed_text' not in data.columns:
    raise ValueError("The column 'processed_text' was not found in the data.")

# convert the 'processed_text' column to a list of documents
documents = data['processed_text'].astype(str).tolist()

# convert documents to a matrix of token counts
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(documents)

# fit the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

# display the topics
for index, topic in enumerate(lda.components_):
    print(f"Top 10 words for Topic #{index}:")
    top_word_indices = topic.argsort()[-10:]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_word_indices]
    print(top_words)
    print("\n")
