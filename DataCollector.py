import snscrape.modules.twitter as sntwitter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


query = "tennis min_replies:750 min_faves:750 "
tweets = []
raw_tweets = []
limit = 100
countVector = CountVectorizer(stop_words = 'english')

for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    if len(tweets) == limit:
        break
    else:
        if tweet.likeCount > 0:
            tweets.append([tweet.content, '%.4f' % (tweet.replyCount/tweet.likeCount)])
            raw_tweets.append(tweet.content)


tennistweets = pd.DataFrame(tweets, columns=['Tweet', 'ratio'])
countVector_matrix = countVector.fit_transform(tennistweets['Tweet'])

ratio_workVector = pd.DataFrame(countVector_matrix.toarray(), index = tennistweets['ratio'], columns = countVector.get_feature_names_out())

vectorizer = TfidfVectorizer()
df_matrix = vectorizer.fit_transform(raw_tweets)
docScore = []

for doc in df_matrix:
    docScore.append(doc.sum())

tfidf_matrix = tennistweets
tfidf_matrix['Tweet'] = docScore
tfidf_matrix.rename(columns={'Tweet':'tfidf_score'}, inplace=True)

print(tfidf_matrix)
tennistweets.to_csv('Data.csv')
ratio_workVector.to_csv('Ratio_workVector.csv')
tfidf_matrix.to_csv('tfidf_ratio.csv')
