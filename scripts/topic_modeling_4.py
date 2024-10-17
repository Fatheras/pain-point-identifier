import os
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Determine the path to the processed data
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'data', 'processed_reddit_posts.csv')

# Load the data
data = pd.read_csv(csv_path)

# Ensure 'processed_text' column exists and convert to string
data['processed_text'] = data['processed_text'].astype(str)

# Convert the 'processed_text' column to a list of documents
documents = data['processed_text'].tolist()

# Convert documents to a matrix of token counts
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(documents)

# Fit the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

# Display the topics
for index, topic in enumerate(lda.components_):
    print(f"Top 10 words for Topic #{index}:")
    top_word_indices = topic.argsort()[-10:]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_word_indices]
    print(top_words)
    print("\n")

# Assign the dominant topic to each document
topic_values = lda.transform(dtm)
data['dominant_topic'] = topic_values.argmax(axis=1)

# Optionally, you can add the topic probabilities if needed
data['topic_probabilities'] = topic_values.tolist()

# Save the updated DataFrame to a CSV file
output_csv_path = os.path.join(script_dir, '..', 'data', 'topic_modelled_posts.csv')
data.to_csv(output_csv_path, index=False)

print(f"Topic modeling complete. Saved to {output_csv_path}.")
