#!C:\Users\KAMRAN\AppData\Local\Programs\Python\Python38-32\python.exe
import pickle

import pandas as pd

train_filepath = r'C:\Users\KAMRAN\PycharmProjects\fakeOrNot\dataSet\train.csv'
test_filepath = r'C:\Users\KAMRAN\PycharmProjects\fakeOrNot\dataSet\test.csv'
insta_train = pd.read_csv(train_filepath)
insta_test = pd.read_csv(test_filepath)

insta_train.head()
insta_train.describe()
insta_train.info()

train_Y = insta_train.fake
train_Y = pd.DataFrame(train_Y)
train_Y.tail(12)

train_X = insta_train.drop(columns="fake")
train_X.head()
test_Y = insta_test.fake
test_Y = pd.DataFrame(test_Y)
test_Y.tail(12)

test_X = insta_test.drop(columns="fake")
test_X.head()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
model1 = logreg.fit(train_X, train_Y)
logreg_predict = model1.predict(test_X)

with open("pickle_model", "wb") as f:
    pickle.dump(model1, f)

print(accuracy_score(logreg_predict, test_Y))
