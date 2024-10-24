import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from sklearn.feature_extraction.text import CountVectorizer
from prophet import Prophet
from datetime import datetime

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sentiment_analysed_posts.csv')
phrases_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'topic_phrases.json')

data = pd.read_csv(data_file)

# remove entries with NaN in topic_label
data = data.dropna(subset=['topic_label'])

required_columns = ['topic_label', 'sentiment', 'text', 'created_utc', 'processed_text', 'score', 'num_comments']
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    raise ValueError(f"Required columns are missing from the data: {missing_cols}")

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
    for sentiment in ['Negative', 'Neutral', 'Positive']:
        if sentiment not in sentiment_by_topic.columns:
            sentiment_by_topic[sentiment] = 0
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

    # identify impactful posts by topic using sentiment_score
    impactful_posts = {}
    for topic in data['topic_label'].unique():
        topic_data = data[(data['topic_label'] == topic) & (data['sentiment'] == 'Negative')]
        topic_data = topic_data.sort_values(by='sentiment_score')

        # get the top 3 impactful posts
        posts = topic_data['text'].dropna().tolist()
        posts = [post for post in posts if post.strip() != '']
        posts = posts[:3]
        impactful_posts[topic] = posts

    return {
        'topic_labels': topic_labels,
        'topic_values': topic_values,
        'sentiment_labels': sentiment_labels,
        'sentiment_values': sentiment_values,
        'sentiment_by_topic': sentiment_by_topic_dict,
        'sample_posts': sample_posts,
        'impactful_posts': impactful_posts
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

# prioritize topics by severity (negative complaint count)
def prioritize_topics():
    priority_scores = {}
    for topic in data['topic_label'].unique():
        topic_data = data[data['topic_label'] == topic]
        negative_count = len(topic_data[topic_data['sentiment'] == 'Negative'])
        priority_scores[topic] = negative_count
    # sort topics by negative_count in descending order
    sorted_topics = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, _ in sorted_topics]

def prepare_trend_data():
    data['created_date'] = pd.to_datetime(data['created_utc'], unit='s')
    
    data['date'] = data['created_date'].dt.strftime('%Y-%m-%d')
    
    # calculate daily complaint counts per topic
    daily_trends = data.groupby(['date', 'topic_label']).size().unstack(fill_value=0)
    
    # prepare data for the template
    daily_trends = daily_trends.reset_index()
    return daily_trends.to_dict('list')

def prepare_sentiment_trends():
    data['created_date'] = pd.to_datetime(data['created_utc'], unit='s').dt.strftime('%Y-%m-%d')

    sentiment_trends = data.groupby(['created_date', 'topic_label', 'sentiment']).size().reset_index(name='count')

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

# Prepare data for forecasting
def prepare_forecasting_data():
    data['created_date'] = pd.to_datetime(data['created_utc'], unit='s')
    
    # create 'date' column for grouping
    data['date'] = data['created_date'].dt.date
    
    daily_counts = data.groupby('date').size().reset_index(name='complaint_count')
    
    prophet_data = daily_counts.rename(columns={'date': 'ds', 'complaint_count': 'y'})
    
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
    
    model = Prophet()
    model.fit(prophet_data)
    
    future = model.make_future_dataframe(periods=30)  # forecasting the next 30 days
    
    forecast = model.predict(future)
    
    # convert dates to strings for JSON serialization
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
    prophet_data['ds'] = prophet_data['ds'].dt.strftime('%Y-%m-%d')
    
    # prepare data for the template
    forecast_data = {
        'dates': forecast['ds'].tolist(),
        'forecast': forecast['yhat'].tolist(),
        'upper': forecast['yhat_upper'].tolist(),
        'lower': forecast['yhat_lower'].tolist(),
        'actual_dates': prophet_data['ds'].tolist(),
        'actual': prophet_data['y'].tolist()
    }
    
    return forecast_data

# prepare data and metrics
data_dict = prepare_data()
data_dict['insights'] = insights
data_dict['topic_metrics'] = calculate_topic_metrics()
data_dict['daily_trends'] = prepare_trend_data()
data_dict['sentiment_trends'] = prepare_sentiment_trends()
data_dict['keyword_data'] = prepare_keyword_data()
data_dict['prioritized_topics'] = prioritize_topics()
data_dict['forecast_data'] = prepare_forecasting_data()

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    data_dict['current_year'] = datetime.now().year
    return templates.TemplateResponse("index.html", {
        "request": request,
        **data_dict
    })
