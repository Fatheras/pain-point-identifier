import os
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'data', 'processed_reddit_posts.csv')

data = pd.read_csv(csv_path)

data['processed_text'] = data['processed_text'].astype(str)

documents = data['processed_text'].tolist()

# convert documents to a matrix of token counts
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(documents)

# fit the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

for index, topic in enumerate(lda.components_):
    print(f"Top 10 words for Topic #{index}:")
    top_word_indices = topic.argsort()[-10:]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_word_indices]
    print(top_words)
    print("\n")

topic_values = lda.transform(dtm)
data['dominant_topic'] = topic_values.argmax(axis=1)
data['topic_probabilities'] = topic_values.tolist()

output_csv_path = os.path.join(script_dir, '..', 'data', 'topic_modelled_posts.csv')
data.to_csv(output_csv_path, index=False)

print(f"Topic modeling complete. Saved to {output_csv_path}.")
