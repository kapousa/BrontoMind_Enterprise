#  Copyright (c) 2021. Slonos Labs. All rights Reserved.

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('LaptopPricePrediction.csv')
def extract_nan_columns():
    return 0

def vectorizer_columns(df: DataFrame):
    vectorizers_list = []
    for i in df.columns:
        vectorizer = CountVectorizer()
        docs = np.array(df[i])
        bag = vectorizer.fit_transform(docs)
        vectorizers_list.append(bag)
    return vectorizers_list

# vectorizer_columns(data)

def devectorizer_columns(result):
    return 0


vectorizer = CountVectorizer()
#
# Create sample set of documents
#
docs = np.array(['Mirabai has won a silver medal in weight lifting in Tokyo olympics 2021',
                 'Sindhu has won a bronze medal in badminton in Tokyo olympics',
                 'Indian hockey team is in top four team in Tokyo olympics 2021 after 40 years'])
#
# Fit the bag-of-words model
#
bag = vectorizer.fit_transform(docs)
print(vectorizer.decode(docs))
print(bag.data)
print(bag.indptr)
print(bag.indices)
#
# Get unique words / tokens found in all the documents. The unique words / tokens represents
# the features
#
#print(vectorizer.get_feature_names())
#
# Associate the indices with each unique word
#
#print(vectorizer.vocabulary_)
#
# Print the numerical feature vector
#
#print(bag.toarray())

#
# Creating training data set from bag-of-words  and dummy label
#
X = bag.toarray()
y = bag.toarray()

#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# Create an instance of LogisticRegression classifier
#
# lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2), n_jobs=-1)
#
# Fit the model
#
lr.fit(X_train, y_train)
#
# Create the predictions
#
y_predict = lr.predict(X_test)

# Use metrics.accuracy_score to measure the score
print(vectorizer.decode(y_predict))
