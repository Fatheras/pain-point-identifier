import praw
import pandas as pd
import os
import datetime
import time

reddit = praw.Reddit(
    client_id='client_id',
    client_secret='client_secret',
    user_agent='user_agent'
)

keywords = [
    'comcast', 'xfinity', 'internet outage', 'billing issue',
    'customer service', 'modem problem', 'slow internet', 'service interruption',
    'installation issue', 'network problem', 'connection drop', 'wifi issue'
]

subreddits = [
    'Comcast_Xfinity', 'cordcutters', 'techsupport', 'ISP',
    'internet', 'technology', 'HomeNetworking', 'sysadmin'
]

def collect_posts():
    posts = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Collecting posts from r/{subreddit_name}...")
        for keyword in keywords:
            print(f"Searching for keyword: '{keyword}'")
            try:
                # use PRAW's search function with pagination
                for submission in subreddit.search(keyword, limit=None, time_filter='month'):
                    posts.append({
                        'title': submission.title,
                        'text': submission.selftext,
                        'created_utc': submission.created_utc,
                        'subreddit': subreddit_name,
                        'url': submission.url,
                        'num_comments': submission.num_comments,
                        'score': submission.score
                    })
                    # respect Reddit's API rate limits
                    time.sleep(0.1)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
    return posts

if __name__ == "__main__":
    all_posts = collect_posts()
    data = pd.DataFrame(all_posts)
    print(f"Total posts collected: {len(data)}")

    data.drop_duplicates(subset=['title', 'text'], inplace=True)
    print(f"Total posts after removing duplicates: {len(data)}")

    script_dir = os.path.dirname(__file__)
    output_csv_path = os.path.join(script_dir, '..', 'data', 'raw', 'reddit_posts.csv')
    data.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")
