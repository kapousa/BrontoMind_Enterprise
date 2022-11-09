#  Copyright (c) 2022. Slonos Labs. All rights Reserved.
import math
import pickle
from _socket import herror
from datetime import datetime, timedelta
from random import randint

import dateutil
import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlabel
from pandas import DataFrame, Series
#from scipy.linalg import pinv2
#import scipy.linalg.pinv2
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from datetime import datetime
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller

from app.base.db_models.ModelForecastingResults import ModelForecastingResults

from app import db
from app.base.db_models.ModelProfile import ModelProfile
from bm.datamanipulation.AdjustDataFrame import convert_data_to_sample
from bm.db_helper.AttributesHelper import add_features, add_labels, add_api_details, update_api_details_id, add_forecasting_results
from bm.utiles.CVSReader import get_only_file_name
from bm.utiles.Helper import Helper
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
import itertools



class TimeForecastingController:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def analyize_dataset(self, file_location):

        # 1- is there any date column
        ds = pd.read_csv(file_location)
        forecasting_columns_arr = []
        depended_columns_arr = []
        datetime_columns_arr = []
        for col in ds.columns:
            if ds[col].dtype == 'object':
                try:
                    #ds[col] = pd.to_datetime(ds[col], dayfirst=True, format="%d/%m/%Y %H:%M")
                    ds[col] = pd.to_datetime(ds[col], yearfirst=True, format="%Y/%m/%d")
                    datetime_columns_arr.append(col)
                except ValueError:
                    forecasting_columns_arr.append(col)
                    pass
            elif (ds[col].dtype == 'float64' or ds[col].dtype == 'int64'):
                depended_columns_arr.append(col)
            else:
                forecasting_columns_arr.append(col)

        # 2- Suggested forcasting values
        return forecasting_columns_arr, depended_columns_arr, datetime_columns_arr;

    def create_ts_forecating_model(self, csv_file_location, forecasting_factor, depended_factor, time_factor, ds_source, ds_goal):
        try:
            # Prepare training and testing data
            dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
            # data = pd.read_csv(csv_file_location, usecols=[forecasting_factor, depended_factor, time_factor], header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=dateparse)
            data = pd.read_csv(csv_file_location, usecols=[forecasting_factor, depended_factor, time_factor], header=0,
                               squeeze=True)
            model_id = randint(0, 10)
            file_name = get_only_file_name(csv_file_location)
            data = data.set_index(time_factor)
            data = data.sort_index()

            # no_of_modeling_records = data.iloc[df_start_date: training_end_date,:]
            forecasting_categories = numpy.array(pd.Categorical(data[forecasting_factor]).categories)
            forecasting_category = forecasting_categories[0]
            # forecasting_category = forecasting_categories[np.where(forecasting_categories == forecasting_factor)]
            forecasting_dataframe = data[[forecasting_factor, depended_factor]].copy()
            forecasting_dataframe = forecasting_dataframe[
                forecasting_dataframe[forecasting_factor] == forecasting_category].copy()
            forecasting_dataframe = forecasting_dataframe.drop(forecasting_factor, 1)
            if not forecasting_dataframe.index.is_unique:  # Remove duplicated index
                forecasting_dataframe = forecasting_dataframe.loc[~forecasting_dataframe.index.duplicated(), :]
            # forecasting_dataframe[time_factor] = forecasting_dataframe.index
            data = forecasting_dataframe

            data = data.astype(float)
            #data.plot(style='k.', ylabel='Sales')
            #plt.show()

            data = np.array(data)
            data = np.reshape(data, (-1, 1))

            # For predicting the daily sales data at any time step, we need to provide values of previous time steps data as input (which is known as the lag). Here, we have chosen the lag size as 14, i.e. sales data of the previous 14 days is used to predict the next day’s sales data
            # Data normalization is an essential preprocessing task to be performed in any timeforecasting job. In this code, the time series data has been normalized by subtracting the mean and then dividing by the standard deviation of data. The predictions will need to be transformed back to the original scale using the corresponding denormalization equation.
            total_records = len(data)
            m = 14  #
            number_of_data_sales, forecasting_dates, forecasting_dates_freq = self.get_useful_model_info(
                forecasting_dataframe)  # number of slaes record before timeforecasting period (last 3 months)
            per = (number_of_data_sales - m) / total_records
            size = int(len(data) * per)
            d_train, d_test = data[0:size], data[size:len(data)]
            mean_train = np.mean(d_train)
            sd_train = np.std(d_train)
            d_train = (d_train - mean_train) / sd_train
            d_test = (d_test - mean_train) / sd_train

            X_train = np.array([d_train[i][0] for i in range(m)])
            y_train = np.array(d_train[m][0])
            for i in range(1, (d_train.shape[0] - m)):
                l = np.array([d_train[j][0] for j in range(i, i + m)])
                X_train = np.vstack([X_train, l])
                y_train = np.vstack([y_train, d_train[i + m]])
            X_test = np.array([d_test[i][0] for i in range(m)])
            y_test = np.array(d_test[m - 1][0])
            for i in range(1, (d_test.shape[0] - m)):
                l = np.array([d_test[j][0] for j in range(i, i + m)])
                X_test = np.vstack([X_test, l])
                y_test = np.vstack([y_test, d_test[i + m]])
            print(X_train.shape)
            print(y_train.shape)
            print(X_test.shape)
            print(y_test.shape)

            input_size = X_train.shape[1]
            hidden_size = 100  # no. of hidden neurons
            mu, sigma = 0, 1
            w_lo = -1
            w_hi = 1
            b_lo = -1
            b_hi = 1
            # initialising input weights and biases randomly drawn from a truncated normal distribution
            input_weights = stats.truncnorm.rvs((w_lo - mu) / sigma, (w_hi - mu) / sigma, loc=mu, scale=sigma,
                                                size=[input_size, hidden_size])
            biases = stats.truncnorm.rvs((b_lo - mu) / sigma, (b_hi - mu) / sigma, loc=mu, scale=sigma,
                                         size=[hidden_size])

            output_weights = np.dot(pinv2(self.hidden_nodes(X_train, input_weights, biases)), y_train)
            prediction = self.predict(X_test, input_weights, biases, output_weights)
            correct = 0
            total = X_test.shape[0]
            y_test = (y_test * sd_train) + mean_train
            prediction = (prediction * sd_train) + mean_train
            # evaluate forecasts
            rmse = math.sqrt(mean_squared_error(y_test, prediction))
            print('Test RMSE: %.3f' % rmse)
            mape_sum = 0
            for i, j in zip(y_test, prediction):
                mape_sum = mape_sum + (abs((i - j) / i))
            mape = (mape_sum / total) * 100
            mpe_sum = 0
            for i, j in zip(y_test, prediction):
                mpe_sum = mpe_sum + ((i - j) / i)
            mpe = (mpe_sum / total) * 100
            print('Test MAPE: %.3f' % mape)
            print('Test MPE: %.3f' % mpe)

            # plot forecasts against actual outcomes
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.plot(y_test, label='Actual')
            # ax.plot(prediction, color='red', label='Predictions')
            # ax.legend(loc='upper right', frameon=False)
            # plt.xlabel('Days', fontname="Arial", fontsize=24, style='italic', fontweight='bold')
            # plt.ylabel('Sales Data', fontname="Arial", fontsize=24, style='italic', fontweight='bold')
            # plt.title('Forecasting for last 3 months with ELM (100 hidden nodes)', fontname="Arial", fontsize=24, style='italic', fontweight='bold')
            # plt.xticks([0, 35, 70, 100], forecasting_dates_freq, fontname="Arial", fontsize=20, style='italic')
            # plt.yticks(fontname="Arial", fontsize=22, style='italic')
            # plt.show()

            # X_testt = ['2017-10-02', '2017-10-22', '2017-11-11', '2017-12-01', '2017-12-21']

            # ------------------Predict values from the model-------------------------#
            # mean_Squared_Error / root mean squared error (RMSE)
            # mean_absolute_error / mean percentage error (MPE)
            # root_mean_squared_error / mean absolute percentage error (MAPE)
            now = datetime.now()
            all_return_values = {'accuracy': '', 'confusion_matrix': '', 'plot_image_path': '',
                                 'file_name': file_name,
                                 'Root_Mean_Squared_Error': str(rmse),
                                 'Mean_Percentage_Error': str(mpe),
                                 'Mean_Absolute_Percentage_Error': str(mape),
                                 'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                                 'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                                 'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S")}

            # Add model profile to the database
            modelmodel = {'model_id': model_id,
                          'model_name': file_name,
                          'user_id': 1,
                          'model_headers': '',
                          'prediction_results_accuracy': '',
                          'mean_percentage_error': str(mpe),
                          'root_mean_squared_error': str(rmse),
                          'mean_absolute_percentage_error': str(mape),
                          'plot_image_path': '',
                          'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'ds_source': ds_source,
                          'ds_goal': ds_goal,
                          'depended_factor': depended_factor,
                          'forecasting_category': forecasting_category}
            model_model = ModelProfile(**modelmodel)
            # Delete current profile
            model_model.query.filter().delete()
            db.session.commit()
            # Add new profile
            db.session.add(model_model)
            db.session.commit()

            # Add features, labels, and APIs details
            add_features_list = add_features(model_id, [time_factor])
            add_labels_list = add_labels(model_id, [depended_factor])
            add_forecasting_results_list = add_forecasting_results(model_id, y_test.flatten(), prediction.flatten(), numpy.array(forecasting_dates))
            api_details_id = randint(0, 22)
            api_details_list = add_api_details(model_id, api_details_id, 'v1')
            api_details_list = update_api_details_id(api_details_id)
            #db.session.commit()
            #db.session.expunge_all()
            # db.close_all_sessions

            # APIs details and create APIs document

            #convert_data_to_sample(csv_file_location, 5)

            return forecasting_category, forecasting_dates, y_test.flatten(), prediction.flatten(), all_return_values;

        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e.__str__())

    def create_forecating_model_(self, csv_file_location, forecasting_factor, depended_factor, time_factor):
        try:
            # Prepare training and testing data
            data = pd.read_csv(csv_file_location, usecols=[forecasting_factor, depended_factor, time_factor])

            # Prepare training and testing data
            data[time_factor] = pd.to_datetime(data[time_factor])
            data = data.set_index(time_factor)
            data = data.sort_index()
            forecasting_categories = numpy.array(pd.Categorical(data[forecasting_factor]).categories)
            forecasting_category = forecasting_categories[0]
            forecasting_dataframe = data[[forecasting_factor, depended_factor]].copy()
            forecasting_dataframe = forecasting_dataframe[
                forecasting_dataframe[forecasting_factor] == forecasting_category].copy()
            forecasting_dataframe = forecasting_dataframe.drop(forecasting_factor, 1)

            if not forecasting_dataframe.index.is_unique:  # Remove duplicated index
                forecasting_dataframe = forecasting_dataframe.loc[~forecasting_dataframe.index.duplicated(), :]
            # forecasting_dataframe = forecasting_dataframe.asfreq('MS') #--- To be uncomment later
            # forecasting_dataframe = forecasting_dataframe.astype(float) #--- To be uncomment later
            forecasting_dataframe[time_factor] = forecasting_dataframe.index

            X = forecasting_dataframe[time_factor].values
            y = forecasting_dataframe[depended_factor].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

            # Fitting Random Forest Regression to the dataset
            regressor = RandomForestRegressor(n_estimators=10, random_state=0)
            regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

            y_pred = regressor.predict(X_test.reshape(-1, 1))

            # # Visualising the Random Forest Regression Results
            # Prepare thick of X axis
            last_3_months = ['0 days', '18 days', '36 days', '58 days', '72 days',
                             '90 days']  # pd.DatetimeIndex(forecasting_dataframe[time_factor]).month
            # last_3_months = last_3_months.drop_duplicates()
            # no_of_intervals = math.floor(100 / len(last_3_months))
            interval_thick = [0, 20, 40, 60, 80, 100]
            # for i in range(len(last_3_months)):
            # interval_thick.append(i * no_of_intervals)

            X = pd.to_datetime(X)
            X_grid = np.arange(X.min(), X.max(), dtype='datetime64[h]')
            X_grid = X_grid.reshape((len(X_grid), 1))
            plt.plot(y_test, color='blue', label="Data")
            plt.plot(y_pred, color='red', label="Prediction")
            plt.title('Random Forest Regression')
            plt.xticks(interval_thick, last_3_months, rotation='vertical')
            plt.xlabel(time_factor.replace('_', ' '))
            plt.legend(loc='best')
            plt.ylabel(depended_factor.replace('_', ' '))
            plt.show()

            return forecasting_category, X_test, y_test, y_pred;
        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e.__str__())

    def relu(self, x):  # hidden layer activation function
        return np.maximum(x, 0, x)

    def hidden_nodes(self, X, input_weights, biases):
        G = np.dot(X, input_weights)
        G = G + biases
        H = self.relu(G)
        return H

    def predict(self, X, input_weights, biases, output_weights):
        out = self.hidden_nodes(X, input_weights, biases)
        out = np.dot(out, output_weights)
        return out

    def parser(x):
        return datetime.strptime(x, "%Y-%m-%d")

    def get_useful_model_info(self, data: DataFrame):
        df_start_date = data.first_valid_index()
        df_start_date = pd.to_datetime(df_start_date)
        df_end_date = data.last_valid_index()
        df_end_date = pd.to_datetime(df_end_date)
        forecasting_start_date = df_end_date - timedelta(days=90)  # last three month (90 days)
        training_end_date = forecasting_start_date - timedelta(days=1)
        training_data_rows = data.loc[pd.to_datetime(data.index.values) <= training_end_date]
        forecasting_dates = pd.to_datetime(data.index[pd.to_datetime(data.index.values) > training_end_date]).tolist()
        # forecasting_dates_freq = pd.date_range(forecasting_dates[0], forecasting_dates[-1], freq='M')
        optimized_forecasting_dates_freq = Helper.previous_n_months(4)
        # for i in forecasting_dates_freq:
        #    str_1 = str(i)
        #    optimized_forecasting_dates_freq.append(str_1[0:10])
        # forecasting_dates = forecasting_dates.year
        # months_list = Helper.previous_n_months(4)
        # print(months_list)

        return len(training_data_rows.index), forecasting_dates, optimized_forecasting_dates_freq

    def get_forecasting_results(self):
        try:
            forecasting_results = ModelForecastingResults.query.all()
            period_dates=[]
            actual=[]
            predicted=[]
            for profile in forecasting_results:
                predicted.append(profile.predicted)
                actual.append(profile.actual)
                period_dates.append(profile.period_dates)
            return actual, predicted, period_dates
        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e)
            return 0


# csv_file = pd.read_csv('Sales_Data.csv')
fc = TimeForecastingController()
#o = fc.create_forecating_model('Sales_Data_1.csv', 'Product', 'Price', 'Transaction_date')
