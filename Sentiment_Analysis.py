!pip install yfinance vaderSentiment newsapi-python

# Import necessary libraries
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

# Step 1 - Get stock data
ticker = "TSLA"
end_date = datetime.today()
start_date = end_date - timedelta(days=30)
stock_df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                       end=end_date.strftime('%Y-%m-%d'))

stock_df.head(5)

# Step 2 - News data
## https://newsapi.org
newsapi = NewsApiClient(api_key='My API Secret Key')

all_articles = newsapi.get_everything(q='Tesla',
                                      from_param=start_date.strftime('%Y-%m-%d'),
                                      to=end_date.strftime('%Y-%m-%d'),
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=100)

texts, dates = [], []

for article in all_articles["articles"]:
    content = (article["title"] or "") + " " + (article["description"] or "")
    texts.append(content)
    dates.append(article["publishedAt"][:10])

# Step 3 - Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [analyzer.polarity_scores(t)["compound"] for t in texts]

sentiment_df = pd.DataFrame({
    "date": pd.to_datetime(dates),
    "sentiment": sentiment_scores
})
daily_sentiment = sentiment_df.groupby("date").mean()

daily_sentiment.head(5)

# Reset columns to remove MultiIndex from yfinance
stock_df.columns = stock_df.columns.droplevel(0)

stock_df.head(5)

# Step 4 - Merge with stock data
stock_df.index = pd.to_datetime(stock_df.index)
merged_df = stock_df.merge(daily_sentiment, left_index=True, 
                           right_index=True, how="left")

merged_df.head(5)
