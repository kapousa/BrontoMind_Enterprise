#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor, SGDClassifier, LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC, SVC

from app import config_parser


class ModelProcessor:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def regressionmodelselector(self, df: DataFrame, modellabels):
        number_of_predictions = len(modellabels.axes[1])
        numberofrecords = len(df.axes[0])
        numberofrecordsedge = 100000
        labeldatatype = modellabels.dtypes

        if (numberofrecords <= 50):  # no enough data
            return 0;

        if (number_of_predictions == 0):  # Clustering
            cls = KMeans(n_clusters=2, random_state=0)
            return cls

        # if (number_of_predictions == 1 and labeldatatype[0] == np.object):  # Classification
        #     cls = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        #     return cls

        if (number_of_predictions > 0):  # Multi-Output Classification
            cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), n_jobs=-1)
            return cls

        cls = LinearRegression()  # Prediction
        return cls

        # if (number_of_predictions == 1):
        #     if (labeldatatype[0] == np.object):  # Classification
        #         if (numberofrecords < numberofrecordsedge):
        #             try:
        #                 cls = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        #             except:
        #                 cls = SGDClassifier(max_iter=5)
        #         else:
        #             try:
        #                 cls = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
        #             except:
        #                 cls = GaussianNB()
        #     else:  # Prediction (Regression)
        #         if (labeldatatype[0] == np.int64 and numberofrecords < numberofrecordsedge):
        #             cls = SGDRegressor()
        #         elif (labeldatatype[0] == np.int64 and numberofrecords >= numberofrecordsedge):
        #             if (number_of_features < 10):
        #                 cls = Lasso(alpha=1.0)
        #             else:
        #                 try:
        #                     cls = SVR(kernal='linear')
        #                 except:
        #                     cls = SVR(kernal='rbf')
        #         else:
        #             cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), n_jobs=-1)
        # else:
        #     cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        #                                 n_jobs=-1)

    def classificationmodelselector(self, numberofrecords):
        numberofrecordsedge = 100000

        # Classification
        if (numberofrecords < numberofrecordsedge):
            try:
                cls = LinearSVC(verbose=0)
                LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                          verbose=0)
                return cls
            except:
                try:
                    cls = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                    return cls
                except:
                    try:
                        cls = SVC(probability=True, decision_function_shape='ovo', kernel='rbf', gamma=0.0078125, C=8)
                        return cls
                    except Exception as e:
                        print(e)
                        return 0

        if (numberofrecords >= numberofrecordsedge):
            try:
                cls = SGDClassifier(max_iter=1000, tol=0.01)
                return cls
            except Exception as e:
                print(e)
                return 0
        return 0

    def predictionmodelselector(self, df: DataFrame, modelfeatures, modellabels):
        number_of_predictions = len(modellabels.axes[1])
        number_of_features = len(modelfeatures.axes[1])
        numberoflabels = modellabels
        numberofrecords = len(df.axes[0])
        numberofrecordsedge = 100000
        if (number_of_predictions == 1):
            labeldatatype = modellabels.dtypes
            # Prediction (Regression)
            if ((labeldatatype[0] == np.int64 or labeldatatype[
                0] == np.float) and numberofrecords < numberofrecordsedge):
                cls = SGDRegressor()
            elif ((labeldatatype[0] == np.int64 or labeldatatype[
                0] == np.float) and numberofrecords >= numberofrecordsedge):
                if (number_of_features < 10):
                    cls = Lasso(alpha=1.0)
                else:
                    try:
                        cls = SVR(kernal='linear')
                    except:
                        cls = SVR(kernal='rbf')
            else:
                cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
                                            n_jobs=-1)
        else:
            cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
                                        n_jobs=-1)
        return cls

    def clustering_model_selector(self):
        try:
            cls = KMeans(
                n_clusters=5)
            return cls
        except Exception as e:
            return config_parser.get('ErrorMessages', 'ErrorMessages.fail_create_model')
