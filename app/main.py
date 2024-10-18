import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sentiment_analysed_posts.csv')
phrases_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'topic_phrases.json')

data = pd.read_csv(data_file)

required_columns = ['topic_label', 'sentiment', 'text', 'created_utc']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Required columns are missing from the data: {required_columns}")

with open(phrases_file, 'r') as f:
    topic_phrases = json.load(f)

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
        posts = data[data['topic_label'] == topic]['text'].head(3).tolist()
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
    data['created_date'] = pd.to_datetime(data['created_utc'], unit='s').dt.date

    daily_trends = data.groupby(['created_date', 'topic_label']).size().unstack(fill_value=0)

    daily_trends.index = daily_trends.index.astype(str)

    return daily_trends.reset_index().to_dict('list')

data_dict = prepare_data()
data_dict['insights'] = insights
data_dict['topic_metrics'] = calculate_topic_metrics()
data_dict['daily_trends'] = prepare_trend_data()

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        **data_dict
    })
