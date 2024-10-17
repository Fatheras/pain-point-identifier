import os
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'data', 'processed_reddit_posts.csv')

data = pd.read_csv(csv_path)

data['processed_text'] = data['processed_text'].astype(str)

documents = data['processed_text'].tolist()

vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(documents)

n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(dtm)

def get_topics_keywords(lda_model, feature_names, n_top_words=5):
    topics_keywords = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics_keywords.append(top_features)
    return topics_keywords

feature_names = vectorizer.get_feature_names_out()

topics_keywords = get_topics_keywords(lda, feature_names, n_top_words=5)

topic_labels = {
    0: "Internet Connectivity Issues",
    1: "Customer Service Complaints",
    2: "Billing and Charges",
    3: "Service Outages",
    4: "Equipment and Installation"
}

data['dominant_topic'] = lda.transform(dtm).argmax(axis=1)
data['topic_keywords'] = data['dominant_topic'].map(lambda x: ', '.join(topics_keywords[x]))
data['topic_label'] = data['dominant_topic'].map(topic_labels)

output_csv_path = os.path.join(script_dir, '..', 'data', 'topic_modelled_posts.csv')
data.to_csv(output_csv_path, index=False)

print(f"Topic modeling complete. Saved to {output_csv_path}.")
