import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from transformers import BertTokenizer

path =r"train.csv"
train_data = pd.read_csv(path)
train_data = train_data.iloc[:,3:]

#print(train_data.head())

texts = train_data.text.values
labels = train_data.target.values
# using the low level BERT for our task.

# initialize
cv = CountVectorizer(stop_words='english')

cv_matrix = cv.fit_transform(texts)
# create document term matrix
df_dtm = pd.DataFrame(cv_matrix.toarray(), labels, columns=cv.get_feature_names_out())
texts = df_dtm.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.33, random_state=66)
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# predictions


rfc_predict = rfc.predict(X_test)

#rfc_cv_score = cross_val_score(rfc, texts, labels, cv=10, scoring= 'roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')

#test_data = pd.read_csv('test.csv')
#test_data = test_data.iloc[:,3:]

#x_test = test_data.text.values
#cv_matrix = cv.fit_transform(x_test)
# create document term matrix
#df_dtm = pd.DataFrame(cv_matrix.toarray(), labels, columns=cv.get_feature_names_out())
#x_test = df_dtm.values.tolist()