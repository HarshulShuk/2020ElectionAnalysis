# python3 -m IPython -i test.py
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import aspect_based_sentiment_analysis as absa
import time
import swifter


trump_df = pd.read_csv("trump_tweets_lang.csv")
biden_df = pd.read_csv("biden_tweets_lang.csv")
trump_df = trump_df[trump_df["lang"] == "en"]
biden_df = biden_df[biden_df["lang"] == "en"]
trump_df.tweet = trump_df.tweet.apply(lambda txt: re.sub("#", " ", txt.lower()))
biden_df.tweet = biden_df.tweet.apply(lambda txt: re.sub("#", " ", txt.lower()))

analyser = SentimentIntensityAnalyzer()
nlp = absa.load()
def find_sentiment_trump(txt):
	vader_scores = analyser.polarity_scores(txt)
	absa_scores = nlp(txt, aspects="trump").examples[0].scores
	return pd.Series({"vader_neutral": vader_scores["neu"],
					  "vader_negative": vader_scores["neg"],
					  "vader_pos": vader_scores["pos"],
					  "vader_compound": vader_scores["compound"],
					  "absa_neutral":absa_scores[0],
					  "absa_negative":absa_scores[1],
					  "absa_positive":absa_scores[2]})

def find_sentiment_biden(txt):
	vader_scores = analyser.polarity_scores(txt)
	absa_scores = nlp(txt, aspects="biden").examples[0].scores
	return pd.Series({"vader_neutral": vader_scores["neu"],
					  "vader_negative": vader_scores["neg"],
					  "vader_pos": vader_scores["pos"],
					  "vader_compound": vader_scores["compound"],
					  "absa_neutral":absa_scores[0],
					  "absa_negative":absa_scores[1],
					  "absa_positive":absa_scores[2]})

start = time.time()
new_columns = ["vader_neutral", "vader_negative", "vader_pos", "vader_compound", "absa_neutral", "absa_negative", "absa_positive"]
trump_df[new_columns] = trump_df.tweet.swifter.apply(find_sentiment_trump)
print("Elapsed: {}".format(time.time() - start))
biden_df[new_columns] = biden_df.tweet.swifter.apply(find_sentiment_biden)
print("Elapsed: {}".format(time.time() - start))

trump_df.to_pickle("trump_sentiment.pkl", protocol=0)
biden_df.to_pickle("biden_sentiment.pkl", protocol=0)
