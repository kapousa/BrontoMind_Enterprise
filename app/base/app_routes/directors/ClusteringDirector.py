import os

import numpy
from flask import render_template, request, current_app, session, send_file

from app.base.constants.BM_CONSTANTS import progress_icon_path, loading_icon_path, docs_templates_folder, output_docs, \
    labeled_data_filename, labeled_data_filename_download_path
from app.base.db_models.ModelAPIDetails import ModelAPIDetails
from app.base.db_models.ModelProfile import ModelProfile

from base.constants.BM_CONSTANTS import html_short_path, output_docs_location
from bm.apis.v1.APIHelper import APIHelper
from bm.controllers.BaseController import BaseController
from bm.controllers.clustering.ClusteringController import ClusteringController
from bm.controllers.clustering.ClusteringControllerHelper import ClusteringControllerHelper
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor
from bm.db_helper.AttributesHelper import get_labels, get_features, get_model_name
from bm.controllers.classification.ClassificationController import ClassificationController
from bm.utiles.Helper import Helper


class ClusteringDirector:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def prepare_date_files(self):
        # 1. Collect uploaded data files
        # 1. Collect uploaded data files
        return 0

    def create_text_clustering_model(self, request):
        try:
            #upload_files = Helper.upload_data_files(folderfiles, mapfile)
            #create_data_bunch = ''
            ds_source = session['ds_source']
            ds_goal = session['ds_goal']
            location_details = {
                'host': request.form.get('location'),
                'username': request.form.get('name'),
                'password': request.form.get('session_token')
            }
            is_local_data = request.form.get('is_local_data') if session['is_local_data'] != 'csv' else session['is_local_data']
            clustering_features = numpy.array((request.form.getlist('clustering_features')))
            clusteringcontroller = ClusteringController()
            return_values =  clusteringcontroller.run_clustering_model(location_details, ds_goal, ds_source, is_local_data, clustering_features)
            page_url = request.host_url + "getdatacluster?t=" + ds_goal + "&s=" + ds_source + "&m=" + str(return_values['model_id'])
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

            # APIs details and create APIs document
            model_api_details = ModelAPIDetails.query.first()
            apihelper = APIHelper()
            model_head = ModelProfile.query.with_entities(ModelProfile.model_id, ModelProfile.model_name).filter_by(model_id = return_values['model_id']).first()
            generate_apis_docs = apihelper.generateapisdocs(0, model_head.model_id, model_head.model_name,
                                                            str(request.host_url + 'api/' + model_api_details.api_version),
                                                            docs_templates_folder, output_docs)

            return render_template('applications/pages/clustering/modelstatus.html',
                                   fname=return_values['file_name'],
                                   clusters_keywords=return_values['clusters_keywords'],
                                   segment='createmodel',
                                   created_on=return_values['created_on'],
                                   updated_on=return_values['updated_on'],
                                   last_run_time=return_values['last_run_time'],
                                   page_url=page_url, page_embed=page_embed,
                                   ds_goal=ds_goal, ds_sourc=ds_source,
                                   plot_image_path = html_short_path + "file_name.html"
                                   )

        except Exception as e:
            return render_template('page-501.html', error=e, segment='message')

    def show_model_status(self):
        try:
            model_profile = BaseController.get_model_status()
            page_url = request.host_url + "getdatacluster?" + + "&m=" + str(model_profile['model_id'])

            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

            return render_template('applications/pages/clustering/modelstatus.html',
                                       train_precision=model_profile['train_precision'],
                                       train_recall=model_profile['train_recall'],
                                       train_f1=model_profile['train_f1'],
                                       test_precision=model_profile['test_precision'],
                                       test_recall=model_profile['test_recall'],
                                       test_f1=model_profile['test_f1'],
                                       segment='createmodel', page_url=page_url, page_embed=page_embed,
                                       created_on=model_profile['created_on'],
                                       updated_on=model_profile['updated_on'],
                                       last_run_time=model_profile['last_run_time'],
                                       fname=model_profile['file_name'])

        except Exception as e:
            return render_template('page-501.html', error=e, segment='message')

    def show_text_model_dashboard(self):
        profile = BaseController.get_model_status()
        page_url = request.host_url + "getdatacluster?t=" + str(profile['ds_goal']) + "&s=" + str(profile['ds_source'])  + "&m=" + str(profile['model_id'])
        page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

        return render_template('applications/pages/classification/textdashboard.html',
                               train_precision=profile['train_precision'],
                               train_recall=profile['train_recall'],
                               train_f1=profile['train_f1'],
                               test_precision=profile['test_precision'],
                               test_recall=profile['test_recall'],
                               test_f1=profile['test_f1'],
                               message='No',
                               fname=profile['model_name'], page_url=page_url, page_embed=page_embed,
                               segment='showdashboard', created_on=profile['created_on'],
                               ds_goal=profile['ds_goal'],
                               updated_on=profile['updated_on'], last_run_time=profile['last_run_time'],
                               plot_image_path = html_short_path + "file_name.html")

    def show_clustermodel_dashboard(self, request, profile):
        # Webpage details
        page_url = request.host_url + "getdatacluster"  + "&m=" + str(profile['model_id'])
        page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
        clusters_keywords = ClusteringControllerHelper.get_clustering_keywords()
        return render_template('applications/pages/clustering/dashboard.html',
                               plot_image_path=profile['plot_image_path'], message='No',
                               fname=profile['model_name'], page_url=page_url, page_embed=page_embed,
                               segment='showdashboard', created_on=profile['created_on'],
                               ds_goal=profile['ds_goal'],
                               clusters_keywords=clusters_keywords,
                               updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])

    @staticmethod
    def download_labeled_datafile():
        """
        Download the labeled data file
        @param request:
        @return:
        """
        try:
            path = "%s%s" %  (labeled_data_filename_download_path, labeled_data_filename)
            return send_file(path, as_attachment=True)

        except Exception as e:
            return render_template('page-501.html', error=e, segment='message')

    def get_clusters(self, request):
        """
        Get cluster of provided data
        @param request:
        @return: Cluster name
        """
        try:
            opt_param = len(request.form)
            ds_goal = request.args.get("t")
            ds_source = request.args.get("s")

            if opt_param == 0:
                # response = make_response()
                return render_template('applications/pages/clustering/clusterdata.html', ds_goal=ds_goal, ds_source=ds_source, testing_values='nothing',
                                       clusters_data=[], predicted='Nothing', message='No')
            else:
                data = (request.form.get('text_value')).lstrip()
                clusteringcontroller = ClusteringController()
                clusters_dic = clusteringcontroller.get_data_cluster([data])
                return render_template('applications/pages/clustering/clusterdata.html', ds_goal=ds_goal, ds_source=ds_source, testing_values=data,
                                       clusters_dic=clusters_dic, predicted='Yes', message='No')

            return clusters
        except Exception as e:
            return render_template('page-501.html', error=e, segment='message')