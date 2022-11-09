import os

import pandas as pd
from flask import request, render_template, session
from werkzeug.utils import secure_filename

from app.base.constants.BM_CONSTANTS import df_location
from base.constants.BM_CONSTANTS import api_data_filename
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
        f = request.files['filename']
        ds_source = session['ds_source']
        ds_goal = session['ds_goal']
        filePath = os.path.join(df_location, secure_filename(f.filename))
        f.save(filePath)

        # Remove empty columns
        data = Helper.remove_empty_columns(filePath)

        # Check if the dataset if engough
        count_row = data.shape[0]
        message = 'No'

        if (count_row < 50):
            message = 'Uploaded data document does not have enough data, the document must have minimum 50 records of data for accurate processing.'
            return render_template('applications/pages/dashboard.html',
                                   message=message,
                                   ds_source=ds_source, ds_goal=ds_goal,
                                   segment='createmodel')

        # Get the DS file header
        headersArray = getcvsheader(filePath)
        fname = secure_filename(f.filename)
        session['fname'] = fname

        return fname, filePath, headersArray, data, message

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
