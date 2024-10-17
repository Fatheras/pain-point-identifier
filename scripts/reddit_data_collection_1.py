import praw
import pandas as pd

reddit = praw.Reddit(
    client_id='client_id',
    client_secret='client_secret',
    user_agent='user_agent'
)

def fetch_posts(query, limit=1000):
    subreddit = reddit.subreddit('all')
    posts = []
    for submission in subreddit.search(query=query, sort='new', limit=limit):
        posts.append({
            'id': submission.id,
            'title': submission.title,
            'body': submission.selftext,
            'created_utc': submission.created_utc,
            'url': submission.url,
            'num_comments': submission.num_comments,
            'score': submission.score
        })
    return pd.DataFrame(posts)

if __name__ == '__main__':
    data = fetch_posts('Comcast OR Xfinity', limit=1000)
    data.to_csv('data/reddit_posts.csv', index=False)
    print("Data collection complete. Saved to reddit_posts.csv.")
