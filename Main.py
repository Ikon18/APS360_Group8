import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "elon musk min_replies:200 min_faves:200 "
tweets = []
raw_tweets = []
limit = 10

for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    if len(tweets) == limit:
        break
    else:
        tweets.append(tweet)
        print(tweet.user.favouritesCount,tweet.user.followersCount,tweet.username,tweet.user.description)
