# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import io
import os
import shutil
import sys

import numpy
import pandas as pd
from flask import Markup, session
from flask import redirect, send_file, Response, Flask, \
    current_app
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
from matplotlib.backends.backend_template import FigureCanvas

from app import login_manager
from app.base import blueprint
from app.base.app_routes.directors.BaseDirector import BaseDirector
from app.base.app_routes.directors.ClassificationDirector import ClassificationDirector
from app.base.app_routes.directors.ForecastingDirector import ForecastingDirector
from app.base.app_routes.directors.PredictionDirector import PredictionDirector
from app.base.constants.BM_CONSTANTS import plot_zip_download_location, progress_icon_path, \
    loading_icon_path, df_location
from app.base.db_models.ModelAPIDetails import ModelAPIDetails
from app.base.db_models.ModelLabels import ModelLabels
from app.base.db_models.ModelProfile import ModelProfile
from bm.apis.v1.APIHelper import APIHelper
from bm.apis.v1.APIsPredictionServices import predictvalues
from bm.controllers.BaseController import BaseController
from bm.controllers.classification.ClassificationController import ClassificationController
from bm.controllers.mlforecasting.MLForecastingController import MLForecastingController
from bm.controllers.prediction.ModelController import run_prediction_model, predict_values_from_model
from bm.controllers.timeforecasting.TimeForecastingController import TimeForecastingController
from bm.core.DocumentProcessor import DocumentProcessor
from bm.core.engine.factories.ClassificationFactory import ClassificationFactory
from bm.core.engine.factories.PredictionFactory import PredictionFactory
from bm.datamanipulation.AdjustDataFrame import create_figure, import_mysql_table_csv, \
    export_mysql_query_to_csv
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor
from bm.db_helper.AttributesHelper import get_features, get_labels, get_model_name
from bm.utiles.CVSReader import getcvsheader, adjust_csv_file
from bm.utiles.CVSReader import improve_data_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = 'app/data/'
app.config['DOCS_TEMPLATES_FOLDER'] = 'docs_templates/'
app.config['APPS_DATA_FOLDER'] = 'app/docs_templates/apps_data_sources'
app.config['DOWNLOAD_APPS_DATA_FOLDER'] = 'docs_templates/apps_data_sources'
app.config['OUTPUT_DOCS'] = 'app/base/output_docs/'
app.config['OUTPUT_PDF_DOCS'] = '/output_docs/'
app.config['DEMO_KEY'] = 'DEMO'
app.config['APP_ROOT'] = '/app'
root_path = app.root_path


@blueprint.route('/index')
@login_required
def index():
    return render_template('base/index.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/base/FILE.html
        return render_template("base/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('base/page-404.html'), 404

    except:
        return render_template('base/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


@blueprint.route('/selectmodelgoal')
@login_required
def selectmodelgoal():
    return render_template('applications/pages/selectmodelgoal.html', segment='selectmodelgoal')


@blueprint.route('/analysedata')
@login_required
def analysedata():
    return render_template('applications/pages/selectdssource.html', segment='selectmodelgoal')


@blueprint.route('/createmodel')
@login_required
def createmodel():
    ds_goal = request.args.get("t")
    session['ds_goal'] = ds_goal
    return render_template('applications/pages/selectdssource.html', ds_goal=ds_goal, segment='createmodel')


@blueprint.route('/updatemodel')
@login_required
def updatemodel():
    model_profile_row = ModelProfile.query.all()
    session['ds_goal'] = model_profile_row['ds_goal']
    return render_template('applications/pages/selectdssource.html', segment='createmodel')


@blueprint.route('/selectcsvds', methods=['GET', 'POST'])
@login_required
def selectcsvds():
    session['ds_source'] = request.form.get('ds_source')
    return selectds()


@blueprint.route('/selectmysqlds', methods=['GET', 'POST'])
@login_required
def selectmysqlds():
    session['ds_source'] = request.form.get('ds_source')
    return selectds()


@blueprint.route('/selectsfds', methods=['GET', 'POST'])
@login_required
def selectsfds():
    session['ds_source'] = request.form.get('ds_source')
    return selectds()


@blueprint.route('/selectgsds', methods=['GET', 'POST'])
@login_required
def selectgsds():
    session['ds_source'] = request.form.get('ds_source')
    return selectds()


@blueprint.route('/selectphysicalfiles', methods=['GET', 'POST'])
@login_required
def selectphysicalfiles():
    session['ds_source'] = request.form.get('ds_source')
    return selectds()


def selectds():
    return render_template('applications/pages/connecttods.html', ds_id=session['ds_source'], segment='createmodel')


@blueprint.route('/uploadcsvds', methods=['GET', 'POST'])
@login_required
def uploadcsvds():
    if request.method == 'POST':
        fname, filePath, headersArray, data, message = BaseDirector.get_data_details(request)
        cc = session['ds_goal']
        if (session['ds_goal'] == current_app.config['PREDICTION_MODULE']):  # Prediction
            prediction_director = PredictionDirector()
            return prediction_director.fetch_data(fname, headersArray, message)

        if (session['ds_goal'] == current_app.config['CLASSIFICATION_MODULE']):  # Classification
            session['fname'] = fname
            return render_template('applications/pages/classification/selectfields.html', headersArray=headersArray,
                                   segment='createmodel', message=message)

        if (session['ds_goal'] == current_app.config['FORECASTING_MODULE']):  # Forecasting
            forecasting_director = ForecastingDirector()
            return forecasting_director.specify_forecating_properties(filePath, headersArray, message)

        if (session['ds_goal'] == current_app.config['ROBOTIC_MODULE']):  # Robotics
            return render_template('applications/pages/robotics/selectfields.html', headersArray=headersArray,
                                   fname=fname,
                                   ds_source=session['ds_source'], ds_goal=session['ds_goal'],
                                   segment='createmodel', message=message)

        # ds_goal = '' means user can't decide
        document_processor = DocumentProcessor()
        columns_list, nan_cols, final_columns_list, final_total_rows, numric_columns, datetime_columns = document_processor.document_analyzer(
            filePath)
        return render_template('applications/pages/analysisreport.html', headersArray=headersArray,
                               columns_list=columns_list, nan_cols=nan_cols,
                               final_columns_list=final_columns_list, final_total_rows=final_total_rows,
                               numric_columns=numric_columns,
                               datetime_columns=datetime_columns,
                               fname=fname,
                               ds_source=session['ds_source'], ds_goal=session['ds_goal'],
                               segment='createmodel', message=message)


@blueprint.route('/dffromdb', methods=['GET', 'POST'])
@login_required
def dffromdb():
    try:
        if request.method == 'POST':

            database_name, file_location, headersArray, count_row, message = BaseDirector.prepare_query_results(request)
            cc = session['ds_goal']

            if (session['ds_goal'] == current_app.config['PREDICTION_MODULE']):
                prediction_director = PredictionDirector()
                return prediction_director.fetch_data(session['fname'], headersArray, message)

            if (session['ds_goal'] == current_app.config['FORECASTING_MODULE']):
                forecasting_director = ForecastingDirector()
                return forecasting_director.specify_forecating_properties(file_location, headersArray, message)

            if (session['ds_goal'] == current_app.config['CLASSIFICATION_MODULE']):
                return render_template('applications/pages/classification/selectfields.html', headersArray=headersArray,
                                       segment='createmodel', message=message)

            if (session['ds_goal'] == current_app.config['ROBOTIC_MODULE']):
                return render_template('applications/dashboard.html')

            return render_template('applications/dashboard.html')

    except Exception as e:
        print(e)
    return render_template('page-501.html', error=e.with_traceback(), segment='error')


@blueprint.route('/dffrompf', methods=['GET', 'POST'])
@login_required
def dffrompf():
    try:
        if request.method == 'POST':
            ds_source = session['ds_source']
            ds_goal = session['ds_goal']
            ftb_server = request.form.get('ftb_server')
            ftb_username = request.form.get('ftb_username')
            ftb_password = request.form.get('ftb_password')
            is_local_data = request.form.get("is_local_data")
            session['is_local_data'] = request.form.get("is_local_data")

            return render_template('applications/pages/classification/creatingclassificationmodel.html',
                                   location=ftb_server,
                                   name=ftb_username,
                                   session_token=ftb_password,
                                   progress_icon_path=progress_icon_path, fname='',
                                   is_local_data=is_local_data,
                                   loading_icon_path=loading_icon_path,
                                   ds_source=ds_source, ds_goal=ds_goal,
                                   segment='createmodel')

    except Exception as e:
        tb = sys.exc_info()[2]
        print(e)
        return render_template('page-501.html', error=e.with_traceback(tb))


@blueprint.route('/creatingthemodel', methods=['GET', 'POST'])
@login_required
def creatingthemodel():  # The function of showing the gif page
    try:
        if request.method == 'POST':
            ds_source = session['ds_source']
            ds_goal = session['ds_goal']

            if (ds_goal == current_app.config[
                'CLASSIFICATION_MODULE'] and ds_source == '11'):  # data source: Physical files
                ftb_server = request.form.get('ftb_server')
                ftb_username = request.form.get('ftb_username')
                ftb_password = request.form.get('ftb_password')
                is_local_data = request.form.get("is_local_data")
                session['is_local_data'] = request.form.get("is_local_data")
                classification_label = numpy.array(request.form.getlist('classification_label'))
                classification_features = numpy.array(request.form.getlist('classification_features'))
                return render_template('applications/pages/classification/creatingclassificationmodel.html',
                                       location=ftb_server,
                                       name=ftb_username,
                                       session_token=ftb_password,
                                       progress_icon_path=progress_icon_path, fname='',
                                       is_local_data=is_local_data,
                                       loading_icon_path=loading_icon_path,
                                       classification_label=classification_label,
                                       classification_features=classification_features,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel')

            fname = session['fname']
            # loading_icon_path = os.path.join('images/', 'loading_icon.gif')
            # progress_icon_path = os.path.join('images/', 'progress_icon_2.gif')
            if (ds_goal == current_app.config['PREDICTION_MODULE']):
                predication_director = PredictionDirector()
                return predication_director.creatingthemodel(request, fname, ds_goal, ds_source)

            if (ds_goal == current_app.config['CLASSIFICATION_MODULE']):
                classification_label = numpy.array(request.form.getlist('classification_label'))
                classification_features = numpy.array(request.form.getlist('classification_features'))
                acf = adjust_csv_file(session['fname'], classification_features, classification_label)
                session['is_local_data'] = request.form.get("is_local_data")
                return render_template('applications/pages/classification/creatingclassificationmodel.html',
                                       classification_label=classification_label,
                                       classification_features=classification_features,
                                       progress_icon_path=progress_icon_path, fname=fname,
                                       loading_icon_path=loading_icon_path,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel')

            if (ds_goal == current_app.config['FORECASTING_MODULE']):
                timefactor = request.form.get('timefactor')
                forecastingfactor = request.form.get('forecastingfactor')
                dependedfactor = request.form.get('dependedfactor')
                return render_template('applications/pages/forecasting/creatingforecastingmodel.html',
                                       timefactor=timefactor, dependedfactor=dependedfactor,
                                       forecastingfactor=forecastingfactor,
                                       progress_icon_path=progress_icon_path, fname=fname,
                                       loading_icon_path=loading_icon_path,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel')

            return 0
        else:
            return 0
    except Exception as e:
        tb = sys.exc_info()[2]
        print(e)
        return render_template('page-501.html', error=e.with_traceback(tb))


@blueprint.route('/sendvalues', methods=['GET', 'POST'])
@login_required
def sendvalues():  # The main function of creating the model
    try:
        if request.method == 'POST':

            ds_source = session['ds_source']
            ds_goal = session['ds_goal']

            if (ds_goal == current_app.config['CLASSIFICATION_MODULE'] and ds_source == '11'):
                classificationfactory = ClassificationFactory()
                return classificationfactory.create_classification_text_model(request)

            fname = session['fname']

            if (ds_goal == current_app.config['PREDICTION_MODULE']):
                predictionfactory = PredictionFactory()
                return predictionfactory.create_prediction_csv_model(request)

            if (ds_goal == current_app.config['CLASSIFICATION_MODULE']):
                classification_label = numpy.array(request.form.getlist('classification_label'))
                idf = improve_data_file(fname, app.config['UPLOAD_FOLDER'],
                                        classification_label)  # Reorder the columns of the data file
                classificationfactory = ClassificationFactory()
                return classificationfactory.create_classification_text_model(request)

            if (ds_goal == current_app.config['FORECASTING_MODULE']):
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
                page_url = request.host_url + "embedforecasting?m=" + all_return_values['model_id']
                page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

                # APIs details and create APIs document
                model_api_details = ModelAPIDetails.query.first()
                apihelper = APIHelper()
                model_head = ModelProfile.query.with_entities(ModelProfile.model_id, ModelProfile.model_name).filter_by(
                    model_id=all_return_values['model_id']).first()
                generate_apis_docs = apihelper.generateapisdocs(0, model_head.model_id, model_head.model_name,
                                                                str(request.host_url + 'api/' + model_api_details.api_version),
                                                                app.config['DOCS_TEMPLATES_FOLDER'],
                                                                app.config['OUTPUT_DOCS'])

                return render_template('applications/pages/forecasting/modelstatus.html',
                                       depended_factor=all_return_values['depended_factor'],
                                       forecasting_category=all_return_values['forecasting_factor'],
                                       plot_image_path=all_return_values['plot_image_path'],
                                       sample_data=[
                                           data_sample.to_html(border=0, classes='table table-hover', header="false",
                                                               justify="center").replace("<th>",
                                                                                         "<th class='text-warning'>")],
                                       fname=model_name,
                                       segment='createmodel', page_url=page_url, page_embed=page_embed,
                                       created_on=all_return_values['created_on'],
                                       updated_on=all_return_values['updated_on'],
                                       last_run_time=all_return_values['last_run_time']
                                       )
            return 0
    except Exception as e:
        # tb = sys.exc_info()[2]
        # profile = get_model_status()
        # if ((len(profile) > 0) and ((profile['ds_goal'] == current_app.config['PREDICTION_MODULE']) or (
        #         profile['ds_goal'] == current_app.config['FORECASTING_MODULE']))):
        #     return redirect(url_for('base_blueprint.showdashboard'))
        # else:
        return render_template('page-501.html',
                               # error="There is no enugh data to build the model after removing empty rows. The data set should have mimimum 50 records to buld the model.",
                               error=e,
                               segment='error')


@blueprint.route('/runthemodel', methods=['GET', 'POST'])
@login_required
def runthemodel():
    try:
        ds_source = request.form.get('ds_source')
        return redirect('pages/connecttods.html', ds_id=ds_source, segment='createmodel')

    except Exception as e:
        print(e)
        return render_template('page-501.html', error=e.with_traceback(), segment='error')


@app.route('/plot.png')
@login_required
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png', segment='createmodel')


@blueprint.route('/predictevalues', methods=['GET', 'POST'])
def predictevalues():
    try:
        model_status = BaseController.get_model_status()

        if (len(model_status) == 0):
            return render_template('applications/pages/nomodeltopredictevalues.html',
                                   message='There is no active model')

        ds_goal = request.args.get("t")

        if (ds_goal == current_app.config['PREDICTION_MODULE']):
            prediction_director = PredictionDirector()
            return prediction_director.predict_labels(request)

        if (ds_goal == current_app.config['CLASSIFICATION_MODULE']):
            classification_directory = ClassificationDirector()
            return classification_directory.classify_inputs_from_model(request)

    except Exception as e:
        return render_template('applications/pages/nomodeltopredictevalues.html',
                               error=str(e),
                               message="Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset",
                               segment='message')


@blueprint.route('/embedforecasting', methods=['GET', 'POST'])
def embedforecasting():
    try:
        profile = BaseController.get_model_status()
        if len(profile) == 0:
            # response = make_response()
            return render_template('applications/pages/forecasting/embedforecasting.html',
                                   message='There is no active model')
        else:
            # Forecasting webpage details
            page_url = request.host_url + "embedforecasting?m=" + profile['model_id']
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            return render_template('applications/pages/forecasting/embedforecasting.html',
                                   depended_factor=profile['depended_factor'],
                                   forecasting_factor=profile['forecasting_category'],
                                   plot_image_path=profile['plot_image_path'], message='No', )
    except Exception as e:
        print(e)
        return render_template('page-501.html',
                               error='Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset',
                               segment='message')


@blueprint.route('/showdashboard')
@login_required
def showdashboard():
    profile = BaseController.get_model_status()

    if len(profile) > 0:

        if (profile['ds_goal'] == current_app.config['CLASSIFICATION_MODULE'] and str(profile['ds_source']) == '11'):
            classification_director = ClassificationDirector()
            return classification_director.show_text_model_dashboard()

        if profile['ds_goal'] == current_app.config['CLASSIFICATION_MODULE']:
            # Webpage details
            classification_director = ClassificationDirector()
            return classification_director.show_text_model_dashboard()

        fname = profile['model_name'] + '.csv'
        data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        df = pd.read_csv(data_file_path, sep=",")
        data_sample = (df.sample(n=5, replace=True))

        if profile['ds_goal'] == current_app.config['PREDICTION_MODULE']:
            # Webpage details
            page_url = request.host_url + "predictevalues?t=" + str(profile['ds_goal']) + "&s=" + str(
                profile["ds_source"]) + "&m=" + str(profile['model_id'])
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            return render_template('applications/pages/prediction/dashboard.html',
                                   accuracy=profile['prediction_results_accuracy'],
                                   confusion_matrix='',
                                   plot_image_path=profile['plot_image_path'], sample_data=[
                    data_sample.to_html(border=0, classes='table table-hover', header="false",
                                        justify="center").replace(
                        "<th>", "<th class='text-warning'>")], Mean_Absolute_Error=profile['mean_absolute_error'],
                                   Mean_Squared_Error=profile['mean_squared_error'],
                                   Root_Mean_Squared_Error=profile['root_mean_squared_error'], message='No',
                                   fname=profile['model_name'], page_url=page_url, page_embed=page_embed,
                                   segment='showdashboard', created_on=profile['created_on'],
                                   ds_goal=profile['ds_goal'],
                                   updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])

        if profile['ds_goal'] == current_app.config['CLASSIFICATION_MODULE']:
            # Webpage details
            page_url = request.host_url + "predictevalues?m=" + profile['model_id']
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            return render_template('applications/pages/classification/dashboard.html',
                                   accuracy=profile['prediction_results_accuracy'],
                                   confusion_matrix='',
                                   plot_image_path=profile['plot_image_path'], sample_data=[
                    data_sample.to_html(border=0, classes='table table-hover', header="false",
                                        justify="center").replace(
                        "<th>", "<th class='text-warning'>")], Mean_Absolute_Error=profile['mean_absolute_error'],
                                   Mean_Squared_Error=profile['mean_squared_error'],
                                   Root_Mean_Squared_Error=profile['root_mean_squared_error'], message='No',
                                   fname=profile['model_name'], page_url=page_url, page_embed=page_embed,
                                   segment='showdashboard', created_on=profile['created_on'],
                                   ds_goal=profile['ds_goal'],
                                   updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])

        if profile['ds_goal'] == current_app.config['FORECASTING_MODULE']:
            # Forecasting webpage details
            page_url = request.host_url + "embedforecasting?m=" + profile['model_id']
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            return render_template('applications/pages/forecasting/dashboard.html',
                                   accuracy=profile['prediction_results_accuracy'],
                                   confusion_matrix='', depended_factor=profile['depended_factor'],
                                   forecasting_factor=profile['forecasting_category'],
                                   error_mse=profile['mean_squared_error'],
                                   plot_image_path=profile['plot_image_path'], sample_data=[
                    data_sample.to_html(border=0, classes='table table-hover', header="false",
                                        justify="center").replace(
                        "<th>", "<th class='text-warning'>")], message='No',
                                   fname=profile['model_name'], page_url=page_url, page_embed=page_embed,
                                   segment='showdashboard', created_on=profile['created_on'],
                                   ds_goal=profile['ds_goal'],
                                   updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])
        else:
            return 0
    else:
        return render_template('applications/pages/dashboard.html', message='You do not have any running model yet.',
                               segment='showdashboard')


@blueprint.route('/deletemodel', methods=['GET', 'POST'])
@login_required
def deletemodel():
    bc = BaseController()
    delete_model = bc.deletemodels()
    return render_template('applications/pages/dashboard.html', message='You do not have any running model yet.',
                           segment='deletemodel')


@blueprint.route('/applications', methods=['GET', 'POST'])
@login_required
def applications():
    return render_template('applications/applications.html', segment='applications')


@blueprint.route('/installapp', methods=['GET', 'POST'])
@login_required
def installapp():
    if request.method == 'POST':
        bc = BaseController()
        delete_model = bc.deletemodels()
        f = request.form.get('filename')
        original = os.path.join(app.config['APPS_DATA_FOLDER'], f)
        target = os.path.join(app.config['UPLOAD_FOLDER'], f)

        shutil.copyfile(original, target)

        # Remove empty columns
        data = pd.read_csv(target)
        data = data.dropna(axis=1, how='all')
        data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        data.to_csv(target, index=False)
        data = pd.read_csv(target)

        # Check if the dataset if engough
        count_row = data.shape[0]
        if (count_row > 50):
            # Get the DS file header
            headersArray = getcvsheader(target)
            fname = f
            return render_template('applications/pages/prediction/selectfields.html', headersArray=headersArray,
                                   fname=fname,
                                   segment='createmodel', message='No')
        else:
            return render_template('applications/pages/prediction/selectfields.html',
                                   message='Uploaded data document does not have enough data, the document must have minimum 50 records of data for accurate processing.',
                                   segment='createmodel')


@blueprint.route('/downloaddsfile', methods=['GET', 'POST'])
@login_required
def downloaddsfile():
    if request.method == 'POST':
        f = request.form.get('filename')
        path = os.path.join(app.config['DOWNLOAD_APPS_DATA_FOLDER'], f)
        return send_file(path, as_attachment=True)


@blueprint.route('/downloadapisdocument', methods=['GET', 'POST'])
@login_required
def downloadapisdocument():
    # For windows you need to use drive name [ex: F:/Example.pdf]
    fname = ModelProfile.query.with_entities(ModelProfile.model_name).first()[0]
    path = root_path + app.config['OUTPUT_PDF_DOCS'] + '/' + fname + '_BrontoMind_APIs_document.docx'
    return send_file(path, as_attachment=True)


@blueprint.route('/downloadplots', methods=['GET', 'POST'])
@login_required
def downloadplots():
    # For windows you need to use drive name [ex: F:/Example.pdf]
    model_profile = BaseController.get_model_status()
    path = plot_zip_download_location + model_profile['model_name'] + '.zip'
    return send_file(path, as_attachment=True)


@blueprint.route('/uploaddatafiles', methods=['GET', 'POST'])
@login_required
def uploaddatafiles():
    if request.method == 'POST':
        # folderfiles = request.files.getlist('folderfiles[]')
        # mapfile = request.files['mapfile']
        ds_source = request.form.get('ds_source')
        ds_goal = request.form.get('ds_goal')
        classification_director = ClassificationDirector()
        return classification_director.create_text_classification_model(ds_goal)


## Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('page-403.html', segment='error'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('page-403.html', segment='error'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('page-404.html', segment='error'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template(('page-500.html'), error=error, segment='error'), 500


@blueprint.errorhandler(501)
def internal_error(error):
    return render_template(('page-501.html'), error=error, segment='error'), 501


@blueprint.errorhandler(413)
def request_entity_too_large_error(error):
    return render_template(('page-413.html'), segment='error'), 413
