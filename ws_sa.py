import tweepy
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import time

collection_btime = time.time()
api_key = 'rc1QwO5vwLEkxaKOKEL5fZcPa'
api_key_secret = '6vACF1sjbCtdR6MN2AtqVXQX8P26w8HZGIynnILmyA8yPeROAg'
access_token = '1232572193256591360-9XO7VBIwHqJuRgtsAs1KfB4SIUgWFv'
access_token_secret = 'g3lQgefWALnrdvnI20GK0UF1V7l5ZHMo3Sb5mGrXSCbZu'

auth_handler = tweepy.OAuthHandler(consumer_key = api_key, consumer_secret = api_key_secret)
auth_handler.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth_handler, wait_on_rate_limit=True)


search_term = 'Airlines'
no_of_tweets = 2000

tweets = tweepy.Cursor(api.search_tweets, q = search_term, lang = 'en').items(no_of_tweets)

# i =1
# for tweet in tweets:
# 	print(str(i)+') '+tweet.text+'\n')
# 	i=i+1

df  = pd.DataFrame([tweet.text for tweet in tweets], columns = ['Tweets'])
collection_etime = time.time()

collection_time = collection_etime - collection_btime

#print(df.head())
cleaning_btime = time.time()
def cleanTxt(text):
	text = re.sub(r'@[A-Za-z0-9]+', '', text)
	text = re.sub(r'#', '', text)
	text = re.sub(r'RT[\s]+','', text)
	text = re.sub(r'https?:\/\/\S+','', text)

	return text

df['Tweets'] = df['Tweets'].apply(cleanTxt)

#print(df.head())
cleaning_etime = time.time()
cleaning_time = cleaning_etime - cleaning_btime

classifying_btime = time.time()
def getSubjectivity(text):
	return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
	return TextBlob(text).sentiment.polarity

df['subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['polarity'] = df['Tweets'].apply(getPolarity)

classifying_etime = time.time()
classifying_time = classifying_etime - classifying_btime

print(df.head())

allWords = ''.join([twts for twts in df['Tweets']])
wordCloud = WordCloud(width =500, height =300, random_state = 21, max_font_size = 119).generate(allWords)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis('off')
plt.show()

def getAnalysis(score):
	if score < 0:
		return 'Negative'
	elif score == 0:
		return 'Neutral'
	else:
		return 'Positive'

df['Analysis'] = df['polarity'].apply(getAnalysis)

print(df.head())

print("number of tweets classified as positive reviews:",df[df.Analysis == 'Positive'].shape[0])
print("number of tweets classified as negative reviews:",df[df.Analysis == 'Negative'].shape[0])
print("number of tweets classified as neutral reviews:",df[df.Analysis == 'Neutral'].shape[0])

df.to_csv('tweets.csv', index = True)

print("collection time:", collection_time)
print("cleaning time:", cleaning_time)
print("classifying time:", classifying_time)