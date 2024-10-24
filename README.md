# Comcast Customer Pain Points Dashboard
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Data Processing Pipeline](#data-processing-pipeline)
  - [Data Preprocessing](#data-preprocessing)
  - [Topic Modeling](#topic-modeling)
  - [Sentiment Analysis](#sentiment-analysis)
- [Running the Application](#running-the-application)
- [Usage Instructions](#usage-instructions)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Notes](#notes)

## Introduction
The Comcast Customer Pain Points Dashboard is an interactive web application designed to analyze and visualize customer complaints extracted from Reddit posts. By leveraging advanced NLP techniques like topic modeling and sentiment analysis, this dashboard helps Comcast identify key areas of concern and prioritize issues that matter most to their customers.

## Features
- **Interactive Topic Filtering**: Select and filter topics to update the charts and focus on specific areas.
- **Topic Prioritization**: View a ranked list of topics based on the percentage of negative sentiments.
- **Visualizations**:
  - **Topic Distribution**: Understand the frequency of different topics.
  - **Sentiment Distribution**: Analyze the overall customer sentiment.
  - **Sentiment by Topic**: Dive deeper into sentiments associated with each topic.
  - **Trend Analysis**: Observe daily complaint trends and sentiment trends over time.
  - **Forecasting**: Predict future complaint volumes using time series forecasting.
- **Insights and Recommendations**: Access detailed insights for each topic, including frequent keywords and actionable recommendations.
- **Impactful Posts**: Review the most impactful negative posts to understand customer pain points in their own words.

## Project Structure
```
├── app
│   ├── main.py
│   ├── templates
│   │   └── index.html
│   ├── static
│   │   ├── style.css
│   │
├── data
│   ├── raw
│   │   ├── reddit_posts.csv
│   ├── processed
│   │   ├── processed_reddit_posts.csv
│   │   ├── topic_modelled_posts.csv
│   │   ├── sentiment_analysed_posts.csv
│   │   └── topic_phrases.json
├── models
│   └── bertopic_model.bin
├── scripts
│   ├── helpers
│   │   ├── download-pack-helper.py
│   │   └── exploratory_data_analysis.py
│   ├── data_preprocessing_2.py
│   ├── reddit_data_collection_1.py
│   ├── topic_modeling_3.py
│   └── sentiment_analysis_4.py
├── requirements.txt
└── README.md
```

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Virtual Environment (Optional but recommended)

## Installation and Setup
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/comcast-dashboard.git
cd comcast-dashboard
```

### 2. Create and Activate a Virtual Environment
On Windows:
```bash
python -m venv comcast_env
comcast_env\Scripts\activate
```

On macOS/Linux:
```bash
python3 -m venv comcast_env
source comcast_env/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (VADER Lexicon)
In a Python shell or script:
```python
import nltk
nltk.download('vader_lexicon')
```

### 5. Install Additional Dependencies for Prophet (if necessary)
On Windows:
```bash
pip install pystan~=2.14
pip install fbprophet
```

On macOS/Linux:
```bash
pip install prophet
```

## Data Collection
### 1. Data Retrieval From Reddit
**Script**: `scripts/reddit_data_collection_1.py`

This script uses PRAW (Python Reddit API Wrapper) to retrieve messages from one year ago. The output is saved as `data/raw/reddit_posts.csv`.

Run the script:
```bash
python scripts/reddit_data_collection_1.py
```

## Data Processing Pipeline
Before running the `scripts`, navigate to the `scripts` directory:
```bash
cd scripts
```

### 1. Data Preprocessing
**Script**: `scripts/data_preprocessing_2.py`

This script reads raw Reddit posts from `data/raw/reddit_posts.csv`, cleans the text data, removes stop words, and performs any necessary preprocessing steps. The output is saved as `data/processed/processed_reddit_posts.csv`.

Run the script:
```bash
python data_preprocessing_2.py
```

### 2. Topic Modeling
**Script**: `scripts/topic_modeling_3.py`

This script performs topic modeling using BERTopic on the preprocessed Reddit posts. It assigns a topic to each post and saves the results to `data/processed/topic_modelled_posts.csv`.

Run the script:
```bash
python topic_modeling_3.py
```

### 3. Sentiment Analysis
**Script**: `scripts/sentiment_analysis_4.py`

This script applies sentiment analysis to the topic-modeled posts, assigning a sentiment label and score to each post. The output is saved as `data/processed/sentiment_analysed_posts.csv`.

Run the script:
```bash
python sentiment_analysis_4.py
```

## Running the Application
### 1. Start the FastAPI Application
Navigate to the `app` directory:
```bash
cd app
```

Run the application using Uvicorn:
```bash
uvicorn main:app --reload
```

**Note**: The `--reload` flag enables hot-reloading, allowing the server to restart automatically when code changes are detected.

### 2. Access the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

## Usage Instructions
- **Filter Topics**:
  - Use the multi-select dropdown to choose topics of interest.
  - Click "Update Charts" to refresh the visualizations based on your selection.
- **Interact with Charts**:
  - Hover over charts to view detailed information.
  - Toggle legend items to show or hide specific data series.
- **Insights and Recommendations**:
  - Click on a topic header to expand and view detailed insights.
  - Use the "Expand All" and "Collapse All" buttons to control all sections at once.
- **Impactful Posts**:
  - Click "View Post X" to read specific customer complaints.
  - These posts are selected based on the most negative sentiment scores.

## Troubleshooting
- **Module Not Found Errors**:
  - Ensure all dependencies are installed correctly using `pip install -r requirements.txt`.
  - Check that you are using the correct Python environment (e.g., the virtual environment you created).
- **Data File Not Found Errors**:
  - Verify that all data processing scripts have been run in order.
  - Check the `data/processed` directory for the necessary CSV files.
- **Styles Not Updating**:
  - Clear your browser cache or perform a hard refresh (`Ctrl + F5` or `Cmd + Shift + R`).
  - Ensure the `style.css` file is correctly linked in `index.html`.
- **Charts Not Displaying**:
  - Confirm that your data files contain the required data.
  - Check the browser console for JavaScript errors.

## Contributing
Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -am 'Add a new feature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

## Acknowledgments
- **BERTopic**: For providing an excellent topic modeling framework.
- **NLTK and Hugging Face Transformers**: For powerful NLP tools and models.
- **Prophet**: For time series forecasting capabilities.
- **Plotly and Bootstrap**: For interactive charts and responsive UI components.
- **Reddit API**: For access to valuable customer feedback data.

## Notes
- Ensure that all scripts are run in the specified order to generate the necessary data files.
- Adjust parameters like `min_topic_size` in `topic_modeling.py` as needed for your dataset.
- The dashboard is designed to be extensible; feel free to customize it to suit your needs.
