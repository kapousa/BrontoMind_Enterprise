#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
import numpy as np
from pandas import DataFrame
from sklearn.cluster import MeanShift, KMeans
from sklearn.linear_model import SGDRegressor, SGDClassifier, LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC, SVC


class ClassificationModelProcessor:
    test_value = ''

    def __init__(self):
        self.test_value = '_'


