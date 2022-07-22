import snscrape.modules.twitter as sntwitter
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#'(#travel) min_faves:50 lang:en until:2022-07-22 since:2022-01-01'
#read from hashtags

topHashs = []

f = open("hashtags.txt", "r")
for tag in f.readlines():
    topHashs.append(tag.replace('\n','' ))
f.close()

tweets = []
raw_tweets = []
limit = 50000
countVector = CountVectorizer(stop_words = 'english')

for tag in topHashs:

    query ='(' +  tag + ") min_faves:50 lang:en until:2022-07-22 since:2022-01-01"

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():

        tweets.append([tweet.content, '%.4f' % (tweet.replyCount/(tweet.likeCount + tweet.retweetCount)), tweet.replyCount, tweet.likeCount, tweet.retweetCount])
        raw_tweets.append(tweet.content)
        if len(tweets) > limit:
            break

    if len(tweets) > limit:
        break

tweetsMatrix = pd.DataFrame(tweets, columns=['Tweet', 'ratio', 'replies', 'likes', 'retweets'])
#countVector_matrix = countVector.fit_transform(tweetsMatrix['Tweet'])

#ratio_workVector = pd.DataFrame(countVector_matrix.toarray(), index = tweetsMatrix['ratio'], columns = countVector.get_feature_names_out())

#vectorizer = TfidfVectorizer()
#df_matrix = vectorizer.fit_transform(raw_tweets)
#docScore = []

#for doc in df_matrix:
#    docScore.append(doc.sum())

#tfidf_matrix = tweetsMatrix
#tfidf_matrix['Tweet'] = docScore
#tfidf_matrix.rename(columns={'Tweet':'tfidf_score'}, inplace=True)

tweetsMatrix.to_csv('Data.csv')
#ratio_workVector.to_csv('Ratio_workVector.csv')
#tfidf_matrix.to_csv('tfidf_ratio.csv')
