import os

import numpy
import pandas as pd
from flask import render_template, request, current_app, session
from markupsafe import Markup

from app.base.constants.BM_CONSTANTS import progress_icon_path, loading_icon_path, df_location, docs_templates_folder, \
    output_docs
from app.base.db_models.ModelAPIDetails import ModelAPIDetails

from app.base.db_models.ModelProfile import ModelProfile
from bm.apis.v1.APIHelper import APIHelper
from bm.controllers.BaseController import BaseController
from bm.controllers.mlforecasting.MLForecastingController import MLForecastingController
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

    def create_forecasting_model(self, request):
        fname = request.form.get('fname')
        ds_source = request.form.get('ds_source')
        ds_goal = request.form.get('ds_goal')
        data_file_path = "%s%s" % (df_location, fname)
        df = pd.read_csv(data_file_path, sep=",")
        data_sample = (df.sample(n=5))

        ml_forecasting_controller = MLForecastingController()
        forecastingfactor = request.form.get('forecastingfactor')
        dependedfactor = request.form.get('dependedfactor')
        timefactor = request.form.get('timefactor')
        all_return_values = ml_forecasting_controller.run_mlforecasting_model(
            data_file_path, forecastingfactor,
            dependedfactor, timefactor, ds_source, ds_goal)
        # Forecasting webpage details
        page_url = request.host_url + "embedforecasting?m=" + str(all_return_values['model_id'])
        page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

        # APIs details and create APIs document
        model_api_details = ModelAPIDetails.query.filter_by( model_id = str(all_return_values['model_id'])).first()
        apihelper = APIHelper()
        model_head = ModelProfile.query.with_entities(ModelProfile.model_id, ModelProfile.model_name).filter_by(
            model_id=all_return_values['model_id']).first()
        generate_apis_docs = apihelper.generateapisdocs(model_head.model_id,
                                                        str(request.host_url + 'api/' + model_api_details.api_version),
                                                        docs_templates_folder, output_docs)

        return render_template('applications/pages/forecasting/modelstatus.html',
                               depended_factor=all_return_values['depended_factor'],
                               forecasting_category=all_return_values['forecasting_factor'],
                               plot_image_path=all_return_values['plot_image_path'],
                               sample_data=[
                                   data_sample.to_html(border=0, classes='table table-hover', header="false",
                                                       justify="center").replace("<th>",
                                                                                 "<th class='text-warning'>")],
                               fname=model_head.model_name, model_id = model_head.model_id,
                               segment='createmodel', page_url=page_url, page_embed=page_embed,
                               created_on=all_return_values['created_on'],
                               updated_on=all_return_values['updated_on'],
                               last_run_time=all_return_values['last_run_time']
                               )