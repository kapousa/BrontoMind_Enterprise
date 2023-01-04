import os

import flask
from flask import request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
from app.base.constants.BM_CONSTANTS import df_location
from base.constants.BM_CONSTANTS import api_data_filename
from bm.controllers.BaseController import BaseController
from bm.datamanipulation.AdjustDataFrame import export_mysql_query_to_csv, export_api_respose_to_csv
from bm.utiles.CVSReader import getcvsheader
from bm.utiles.Helper import Helper



class BaseDirector:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    @staticmethod
    def get_data_details(request):
        #f = request.files['filename']
        f = flask.request.files.getlist('filename[]')
        number_of_files = len(f)
        ds_source = session['ds_source']
        ds_goal = session['ds_goal']
        if number_of_files == 1:
            file_name = f[0].filename
            filePath = os.path.join(df_location, secure_filename(file_name))
            f[0].save(filePath)

            # Remove empty columns
            data = Helper.remove_empty_columns(filePath)

            # Check if the dataset if engough
            count_row = data.shape[0]
            message = 'No'

            if (count_row < 5):
                message = 'Uploaded data document does not have enough data, the document must have minimum 50 records of data for accurate processing.'
                return render_template('applications/pages/dashboard.html',
                                       message=message,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel')

            # Get the DS file header
            headersArray = getcvsheader(filePath)
            fname = secure_filename(f[0].filename)
            session['fname'] = fname

            return fname, filePath, headersArray, data, message
        else:
            print("Hi...")
            return 0



    @staticmethod
    def prepare_query_results(request):
        host_name = request.form.get('host_name')
        username = request.form.get('username')
        password = request.form.get('password')
        database_name = request.form.get('database_name')
        sql_query = request.form.get('sql_query')
        file_location, count_row = export_mysql_query_to_csv(host_name, username, password, database_name, sql_query)

        if (count_row < 50):
            return render_template('applications/pages/dashboard.html',
                                   message='Uploaded data document does not have enough data, the document must have minimum 50 records of data for accurate processing.',
                                   segment='createmodel')
        # Get the DS file header
        session['fname'] = database_name + ".csv"
        message = 'No'
        filelocation = '%s' % (file_location)
        headersArray = getcvsheader(filelocation)

        return database_name, file_location,headersArray, count_row, message

    @staticmethod
    def prepare_api_results(request):
        api_url = request.form.get('api_url')
        request_type = request.form.get('request_type')
        root_node = request.form.get('root_node')
        request_parameters = request.form.get('request_parameters')
        file_location, count_row = export_api_respose_to_csv(api_url, request_type, root_node, request_parameters)

        if (count_row < 50):
            return render_template('applications/pages/dashboard.html',
                                   message='Uploaded data document does not have enough data, the document must have minimum 50 records of data for accurate processing.',
                                   segment='createmodel')
        # Get the DS file header
        session['fname'] = api_data_filename
        message = 'No'
        filelocation = '%s' % (file_location)
        headersArray = getcvsheader(filelocation)

        return api_data_filename, file_location, headersArray, count_row, message
    @staticmethod
    def analyse_request(request):
        session['ds_goal'] = None
        model_desc = request.form.get('text_value')
        basecontroller = BaseController()
        results = basecontroller.detectefittedmodels(model_desc.strip())
        return render_template('applications/pages/suggestions.html',
                               message= 'analysis_result', results = results,
                               segment='idea')

    @staticmethod
    def update_model_info(request):
        model_id = request.args.get('param')
        n = request.args.get('n')
        if (n == '1'):
            profile = BaseController.get_model_status(model_id)
            return render_template('applications/pages/updateinfo.html',
                                   message='You do not have any running model yet.', profile=profile, modid=model_id,
                                   segment='showdashboard')

        model_id = request.form.get('modid')
        model_name = request.form.get('mname')
        model_description= "{}".format(request.form.get('mdesc'))

        basecontroller =BaseController()
        updatemodelinfo = basecontroller.updatemodelinfo(model_id, model_name, model_description)

        return redirect(url_for('base_blueprint.showmodels'))

    @staticmethod
    def change_model_status(model_id):
        basecontroller = BaseController()
        suspendmodel = basecontroller.changemodelstatus(model_id)
        return redirect(url_for('base_blueprint.showmodels'))

    def deploy_model(model_id):
        basecontroller = BaseController()
        deploy_statu = basecontroller.deploymodel(model_id)
        return redirect(url_for('base_blueprint.showmodels', message=deploy_statu))
