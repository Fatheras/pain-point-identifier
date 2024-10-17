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

data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_analysed_posts.csv')

data = pd.read_csv(data_file)

# convert 'created_utc' to datetime if it exists
if 'created_utc' in data.columns:
    data['created_date'] = pd.to_datetime(data['created_utc'], unit='s').dt.date

    # topic trends over time
    topic_trends = data.groupby(['created_date', 'topic_label']).size().unstack(fill_value=0)
    topic_trends = topic_trends.reset_index()
    topic_trends['created_date'] = topic_trends['created_date'].astype(str)
    topic_trends_dict = topic_trends.to_dict('list')
else:
    topic_trends_dict = {}

# ensure that the necessary columns exist
required_columns = ['dominant_topic', 'sentiment', 'topic_label', 'text']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Required columns are missing from the data: {required_columns}")

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

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    data_dict = prepare_data()
    data_dict['insights'] = insights
    data_dict['topic_trends'] = topic_trends_dict

    return templates.TemplateResponse("index.html", {
        "request": request,
        **data_dict
    })
