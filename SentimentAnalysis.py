import pandas as pd
from textblob import TextBlob

# Load Dataset
df = pd.read_csv("reviews.csv")

# Function to get sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply Sentiment Analysis
df["Sentiment"] = df["Review"].apply(get_sentiment)

# Display Results
print(df)

# Save Output
df.to_csv("sentiment_output.csv", index=False)

print("Sentiment Analysis Completed Successfully!")
