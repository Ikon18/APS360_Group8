import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

path =r"tfidf_ratio.csv"
tfidf_data = pd.read_csv(path)
tfidf_data = tfidf_data.iloc[:,1:]

#print(tfidf_data.head())

x = tfidf_data['tfidf_score']
y = tfidf_data['ratio']

# create the scatter plot.
plt.plot(x, y, 'o', color ='b', label = 'score to ratio')

# make sure it's formatted.
plt.title("Tweet Tfidf Score Vs Ratio")
plt.xlabel("tfidf_score")
plt.ylabel("ratio")
plt.legend()

plt.show()
print(tfidf_data.corr())

Y = tfidf_data.drop('ratio', axis = 1)
X = tfidf_data[['ratio']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

intercept = regression_model.intercept_[0]
coefficient = regression_model.coef_[0][0]

print("The Coefficient for our model is {:.2}".format(coefficient))
print("The intercept for our model is {:.4}".format(intercept))

y_predict = regression_model.predict(X_test)

model_mse = mean_squared_error(y_test, y_predict)

# calculate the mean absolute error.
model_mae = mean_absolute_error(y_test, y_predict)

# calulcate the root mean squared error
model_rmse =  math.sqrt(model_mse)

# display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))

model_r2 = r2_score(y_test, y_predict)
print("R2: {:.2}".format(model_r2))

plt.scatter(X_test, y_test,  color='gainsboro', label = 'score to ratio')
plt.plot(X_test, y_predict, color='royalblue', linewidth = 3, linestyle= '-',label ='Regression Line')

plt.title("Linear Regression tweet tfidf score to ratio")
plt.xlabel("ratio")
plt.ylabel("tfidf score")
plt.legend()
plt.show()