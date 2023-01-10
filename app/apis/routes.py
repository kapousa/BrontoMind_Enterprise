# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import render_template, request

from app import login_manager
from app.apis import blueprint
from bm.apis.v1.APIsClusteringServices import APIsClusteringServices
from bm.apis.v1.APIsPredictionServices import predictvalues
from bm.apis.v1.APIsClassificationServices import APIsClassificationServices
from bm.apis.v1.cvision.APIsObjectDetectionServices import APIsObjectDetectionServices


## APIs

@blueprint.route('/api/v1/<model_id>/predictevalues', methods=['POST'])
def predictevalues_api(model_id):
    content = request.get_json(silent=True)
    apireturn_json = predictvalues(model_id, content)
    return apireturn_json


@blueprint.route('/api/v1/<model_id>/classifydata', methods=['POST'])
def classifydata_api(model_id):
    content = request.json
    apis_classification_services = APIsClassificationServices()
    apireturn_json = apis_classification_services.classify_data(content, model_id)
    return apireturn_json

@blueprint.route('/api/v1/<model_id>/labeldata', methods=['POST'])
def labeldata_api(model_id):
    content = request.json
    apis_clustering_services = APIsClusteringServices()
    apireturn_json = apis_clustering_services.cluster_data(content, model_id)
    return apireturn_json

@blueprint.route('/api/v1/<model_id>//cvision/objdect/labelfiles', methods=['POST'])
def labelfiles_api(model_id):
    content = request.json
    apisObjectdetectionservices = APIsObjectDetectionServices()
    apireturn_json = apisObjectdetectionservices.label_files(content, model_id)
    return apireturn_json

# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('page-500.html'), 500
