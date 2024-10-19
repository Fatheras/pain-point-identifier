import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from sklearn.feature_extraction.text import CountVectorizer


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sentiment_analysed_posts.csv')
phrases_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'topic_phrases.json')

data = pd.read_csv(data_file)

required_columns = ['topic_label', 'sentiment', 'text', 'created_utc', 'processed_text']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Required columns are missing from the data: {required_columns}")

with open(phrases_file, 'r') as f:
    topic_phrases = json.load(f)

# prepare data for visualization
def prepare_data():
    topic_counts = data['topic_label'].value_counts()
    topic_labels = topic_counts.index.tolist()
    topic_values = topic_counts.values.tolist()

    sentiment_counts = data['sentiment'].value_counts()
    sentiment_labels = sentiment_counts.index.tolist()
    sentiment_values = sentiment_counts.values.tolist()

    sentiment_by_topic = data.groupby('topic_label')['sentiment'].value_counts().unstack().fillna(0)
    sentiment_by_topic = sentiment_by_topic[['Negative', 'Neutral', 'Positive']]
    sentiment_by_topic_dict = sentiment_by_topic.to_dict('index')

    sample_posts = {}
    for topic in data['topic_label'].unique():
        posts = data[data['topic_label'] == topic]['text'].dropna().tolist()
        # remove empty strings
        posts = [post for post in posts if post.strip() != '']
        # get the top 5 posts
        posts = posts[:5]
        sample_posts[topic] = posts

    return {
        'topic_labels': topic_labels,
        'topic_values': topic_values,
        'sentiment_labels': sentiment_labels,
        'sentiment_values': sentiment_values,
        'sentiment_by_topic': sentiment_by_topic_dict,
        'sample_posts': sample_posts
    }

insights_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'insights.json')
with open(insights_file, 'r') as f:
    insights = json.load(f)

def calculate_topic_metrics():
    topic_metrics = {}
    for topic in data['topic_label'].unique():
        topic_data = data[data['topic_label'] == topic]
        complaint_count = len(topic_data)
        negative_count = len(topic_data[topic_data['sentiment'] == 'Negative'])
        negative_percentage = (negative_count / complaint_count) * 100 if complaint_count > 0 else 0
        topic_metrics[topic] = {
            'complaint_count': complaint_count,
            'negative_percentage': round(negative_percentage, 2)
        }
    return topic_metrics

def prepare_trend_data():
    # convert 'created_utc' to datetime
    data['created_date'] = pd.to_datetime(data['created_utc'], unit='s').dt.date

    # calculate daily complaint counts per topic
    daily_trends = data.groupby(['created_date', 'topic_label']).size().unstack(fill_value=0)

    # convert dates to strings for JSON serialization
    daily_trends.index = daily_trends.index.astype(str)

    # prepare data for the template
    return daily_trends.reset_index().to_dict('list')

def prepare_sentiment_trends():
    data['created_date'] = pd.to_datetime(data['created_utc'], unit='s').dt.date

    sentiment_trends = data.groupby(['created_date', 'topic_label', 'sentiment']).size().reset_index(name='count')

    # pivot the data for visualization
    sentiment_trends_pivot = sentiment_trends.pivot_table(
        index='created_date',
        columns=['topic_label', 'sentiment'],
        values='count',
        fill_value=0
    )

    # convert the index and columns to strings for JSON serialization
    sentiment_trends_pivot.index = sentiment_trends_pivot.index.astype(str)
    sentiment_trends_pivot.columns = [f"{topic}-{sentiment}" for topic, sentiment in sentiment_trends_pivot.columns]

    # convert to dictionary with dates as keys
    sentiment_trends_dict = sentiment_trends_pivot.to_dict(orient='index')

    return sentiment_trends_dict

def prepare_keyword_data():
    # for each topic, collect the most frequent words in negative posts
    keyword_data = {}
    for topic in data['topic_label'].unique():
        negative_texts = data[(data['topic_label'] == topic) & (data['sentiment'] == 'Negative')]['processed_text'].dropna().tolist()
        if not negative_texts:
            keyword_data[topic] = []
            continue
        vectorizer = CountVectorizer(stop_words='english', max_features=10)
        X = vectorizer.fit_transform(negative_texts)
        keywords = vectorizer.get_feature_names_out()
        keyword_data[topic] = keywords.tolist()
    return keyword_data

data_dict = prepare_data()
data_dict['insights'] = insights
data_dict['topic_metrics'] = calculate_topic_metrics()
data_dict['daily_trends'] = prepare_trend_data()
data_dict['sentiment_trends'] = prepare_sentiment_trends()
data_dict['keyword_data'] = prepare_keyword_data()

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        **data_dict
    })
