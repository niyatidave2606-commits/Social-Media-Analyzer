# reddit_sports_wordcloud.py

"""
Reddit Sports Word Cloud Generator
----------------------------------
This script fetches Reddit posts about "sports" via RSS,
calculates TF-IDF scores for words in the titles,
excludes unwanted words, and generates a word cloud.
"""

# Install required libraries if not installed
# !pip install feedparser pandas scikit-learn wordcloud matplotlib

import feedparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1️⃣ Fetch Reddit posts about "sports"
url = "https://www.reddit.com/search.rss?q=sports&sort=hot"
feed = feedparser.parse(url)

# Extract post titles
titles = [entry.title for entry in feed.entries]

# Convert titles to a DataFrame
df = pd.DataFrame(titles, columns=["title"])

# 2️⃣ TF-IDF Vectorization
# Add custom stop words to exclude unwanted words
custom_stop_words = ["bra"]  # Add more words here if needed

# Combine with default English stop words
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
tfidf_matrix = vectorizer.fit_transform(df["title"])

# Create DataFrame of words and their TF-IDF scores
scores = tfidf_matrix.sum(axis=0).A1
words = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame({"word": words, "tfidf": scores})

# Remove unwanted words
tfidf_df = tfidf_df[~tfidf_df["word"].isin(custom_stop_words)]

# Sort words by TF-IDF
tfidf_df = tfidf_df.sort_values(by="tfidf", ascending=False)

# Optional: Display top 20 words
print("Top 20 words by TF-IDF:")
print(tfidf_df.head(20))

# 3️⃣ Generate Word Cloud
tfidf_dict = dict(zip(tfidf_df["word"], tfidf_df["tfidf"]))
wc = WordCloud(width=800, height=400, background_color="white", max_words=2000)
wc = wc.generate_from_frequencies(tfidf_dict)

# Display Word Cloud
plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Reddit Sports Word Cloud", fontsize=16)
plt.show()
