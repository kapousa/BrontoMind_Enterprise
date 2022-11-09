import os

import numpy
from flask import render_template, request, current_app, session
from markupsafe import Markup

from app.base.constants.BM_CONSTANTS import progress_icon_path, loading_icon_path
from bm.controllers.BaseController import BaseController
from bm.controllers.timeforecasting.TimeForecastingController import TimeForecastingController
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor
from bm.db_helper.AttributesHelper import get_labels, get_features, get_model_name
from bm.controllers.classification.ClassificationController import ClassificationController
from bm.utiles.Helper import Helper


class ForecastingDirector:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def specify_forecating_properties(self, file_location, headersArray, message):
        time_forecasting_controller = TimeForecastingController()
        forecasting_columns, depended_columns, datetime_columns = time_forecasting_controller.analyize_dataset(
            file_location)
        message = (message if ((len(forecasting_columns) != 0) and (
                len(datetime_columns) != 0) and (
                                       len(depended_columns) != 0)) else 'Your data file doesn not have one or more required fields to build the timeforecasting model. The file should have:<ul><li>One or more ctaegoires columns</li><li>One or more time series columns</li><li>One or more columns with numerical values.</li></ul><br/>Please check your file and upload it again.')
        return render_template('applications/pages/forecasting/dsfileanalysis.html',
                               headersArray=headersArray,
                               segment='createmodel', message=Markup(message),
                               forecasting_columns=forecasting_columns,
                               depended_columns=depended_columns,
                               datetime_columns=datetime_columns)