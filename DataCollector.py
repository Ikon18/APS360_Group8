import snscrape.modules.twitter as sntwitter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

query = "tennis min_replies:750 min_faves:750 "
tweets = []
limit = 100
countVector = CountVectorizer(stop_words = 'english')

for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    if len(tweets) == limit:
        break
    else:
        if tweet.likeCount > 0:
            tweets.append([tweet.username, tweet.content, '%.4f' % (tweet.replyCount/tweet.likeCount)])

tennistweets = pd.DataFrame(tweets, columns=['User', 'Tweet', 'ratio'])
countVector_matrix = countVector.fit_transform(tennistweets['Tweet'])

ratio_workVector = pd.DataFrame(countVector_matrix.toarray(), index = tennistweets['ratio'], columns = countVector.get_feature_names())

#tennistweets.to_csv('Data.csv')
ratio_workVector.to_csv('Ratio_workVector.csv')