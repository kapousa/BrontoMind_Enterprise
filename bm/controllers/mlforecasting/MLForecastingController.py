# Data manipulation
# ==============================================================================
from datetime import datetime
import os
import pickle
import random
from random import randint

import matplotlib
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import json

from app import db
from app.base.constants.BM_CONSTANTS import pkls_location, html_plots_location, html_short_path
from app.base.db_models.ModelProfile import ModelProfile
from bm.datamanipulation.AdjustDataFrame import convert_data_to_sample
from bm.db_helper.AttributesHelper import add_api_details, update_api_details_id, add_features, add_labels, \
    add_forecasting_results
from bm.utiles.CVSReader import get_only_file_name

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5

# Modeling and Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

from joblib import dump, load

# Warnings configuration
# ==============================================================================
import warnings


# warnings.filterwarnings('ignore')


class MLForecastingController:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def run_mlforecasting_model(self, csv_file_location, forecasting_factor, depended_factor, time_factor, ds_source,
                                ds_goal):
        # Data download
        # ==============================================================================
        data = pd.read_csv(csv_file_location, usecols=[depended_factor, time_factor], sep=',', header=0)
        model_id = Helper.generate_model_id()
        file_name = get_only_file_name(csv_file_location)

        # Data preparation
        # ==============================================================================
        data = data.rename(columns={time_factor: 'date'})
        data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
        # dropping ALL duplicate values
        data.drop_duplicates(subset="date", keep='first', inplace=True)
        data = data.set_index('date')
        data = data.rename(columns={depended_factor: depended_factor})
        data = data.asfreq('MS')
        data = data.sort_index()
        data.head()

        print(f'Number of rows with missing values: {data.isnull().any(axis=1).mean()}')

        # Verify that a temporary index is complete
        # ==============================================================================
        (data.index == pd.date_range(start=data.index.min(),
                                     end=data.index.max(),
                                     freq=data.index.freq)).all()

        # Fill gaps in a temporary index
        # ==============================================================================
        # data.asfreq(freq='30min', fill_value=np.nan)

        # Split data into train-test
        # ==============================================================================
        steps = int(round(len(data.index) / 4, 0))
        data_train = data[:-steps]
        data_test = data[-steps:]

        print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
        print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

        fig, ax = plt.subplots(figsize=(9, 4))
        data_train[depended_factor].plot(ax=ax, label='train')
        data_test[depended_factor].plot(ax=ax, label='test')
        ax.legend();
        # plt.show()

        # Create and train forecaster
        # ==============================================================================
        forecaster = ForecasterAutoreg(
            regressor=RandomForestRegressor(random_state=123),
            lags=6
        )

        forecaster.fit(y=data_train[depended_factor])
        forecaster

        # Predictions
        # ==============================================================================
        # steps = 36
        predictions = forecaster.predict(steps=steps)
        predictions.head(5)

        # Plot
        # ==============================================================================
        fig, ax = plt.subplots(figsize=(9, 4))
        data_train[depended_factor].plot(ax=ax, label='train')
        data_test[depended_factor].plot(ax=ax, label='test')
        predictions.plot(ax=ax, label='predictions')
        ax.legend()
        # plt.show()

        # Test error
        # ==============================================================================
        error_mse = mean_squared_error(
            y_true=data_test[depended_factor],
            y_pred=predictions
        )

        print(f"Test error (mse): {error_mse}")

        # Hyperparameter Grid search
        # ==============================================================================
        # steps = 36
        forecaster = ForecasterAutoreg(
            regressor=RandomForestRegressor(random_state=123),
            lags=12  # This value will be replaced in the grid search
        )

        # Lags used as predictors
        lags_grid = [10, 20]

        # Regressor's hyperparameters
        # removed go to the source

        # ==============================================================================
        #                               FINAL MODEL
        # ==============================================================================
        # Create and train forecaster with the best hyperparameters
        # ==============================================================================
        regressor = RandomForestRegressor(max_depth=3, n_estimators=500, random_state=123)
        forecaster = ForecasterAutoreg(
            regressor=regressor,
            lags=20
        )

        forecaster.fit(y=data_train[depended_factor])

        # Save model details
        # ==============================================================================
        # model_file_name = pkls_location + file_name + '_model.pkl'
        # pickle.dump(forecaster, open(model_file_name, 'wb'))

        # Predictions
        # ==============================================================================
        predictions = forecaster.predict(steps=steps)

        # Test error
        # ==============================================================================
        error_mse = mean_squared_error(
            y_true=data_test[depended_factor],
            y_pred=predictions
        )

        print(f"Test error (mse): {error_mse}")

        # Plot
        # ==============================================================================
        # Delete old visualization html
        dir = html_plots_location
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        fig, ax = plt.subplots(figsize=(9, 4))
        data_train[depended_factor].plot(ax=ax, label='train')
        data_test[depended_factor].plot(ax=ax, label='test')
        predictions.plot(ax=ax, label='predictions')
        ax.legend();
        # plt.show()
        df = self.prepare_ploting_data(data_train, data_test, predictions, depended_factor)
        fig = px.line(df, x=df.index, y=depended_factor, color="label", labels={
            depended_factor: depended_factor,
            "index": "Date",
            "label": "Data type"
        }, title="")
        fig.update_traces(textposition="bottom right")
        # fig.show()
        html_file_location = html_plots_location + file_name + ".html"
        html_path = html_short_path + file_name + ".html"
        plotly.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)


        # ------------------Predict values from the model-------------------------#
        now = datetime.now()
        all_return_values = {'model_id': model_id,
                            'plot_image_path': html_path,
                             'file_name': file_name,
                             'forecasting_factor': forecasting_factor,
                             'depended_factor': depended_factor,
                             'chart_title': "The forecasting of the " + depended_factor + " of the " + forecasting_factor + " - ( Mean squared error =" + str(round(error_mse, 3)) + ")",
                             'Mean_Squared_Error': error_mse,
                             'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                             'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                             'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S")}

        # Add model profile to the database
        modelmodel = {'model_id': model_id,
                      'model_name': file_name,
                      'user_id': 1,
                      'mean_squared_error': error_mse,
                      'forecasting_category': forecasting_factor,
                      'depended_factor': depended_factor,
                      'plot_image_path': html_path,
                      'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'ds_source': ds_source,
                      'ds_goal': ds_goal}
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
        # add_forecasting_results_list = add_forecasting_results(model_id, y_test.flatten(), prediction.flatten(), numpy.array(forecasting_dates))
        api_details_id = randint(0, 22)
        api_details_list = add_api_details(model_id, api_details_id, 'v1')
        api_details_list = update_api_details_id(api_details_id)
        # db.session.commit()
        # db.session.expunge_all()
        # db.close_all_sessions

        # APIs details and create APIs document

        convert_data_to_sample(csv_file_location, 5)

        return all_return_values;

    def prepare_ploting_data(self, train_y, test_y, predict_y, depended_factor):
        train_y.insert(0, 'label', 'Train')
        test_y.insert(0, 'label', 'Test')
        prediction_df = predict_y.to_frame()
        prediction_df.rename(columns={'pred': depended_factor}, inplace=True)
        prediction_df.insert(0, 'label', 'predictions')
        df = pd.concat([train_y, test_y, prediction_df], axis=0)
        return df

# cc = MLForecastingController()
# url = 'h2o_exog.csv'
# bb = cc.run_mlforecasting_model(url, '', 'y', 'fecha', 1, 0)
