# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os

from flask import request, send_file
from flask_login import login_required

from app.base.app_routes.directors.cvision.ObjectDetectionDirector import ObjectDetectionDirector
from app.base.constants.BM_CONSTANTS import results_path
from app.cvision import blueprint
from bm.controllers.BaseController import BaseController
from bm.controllers.cvision.ObjectDetectionCotroller import ObjectDetectionCotroller
from bm.core.engine.factories.cvision.ObjectDetectionFactory import ObjectDetectionFactory


## CVision

@blueprint.route('/selectvision', methods=['GET', 'POST'])
@login_required
def selectvision():
    objectdetectionfactory = ObjectDetectionFactory()
    return objectdetectionfactory.selectcvisiontype(request)

@blueprint.route('/createmodel/objectdetection', methods=['GET', 'POST'])
@login_required
def createobjectdetection():
    objectdetectionfactory = ObjectDetectionFactory()
    return objectdetectionfactory.createobjectdetectionmodel(request)

@blueprint.route('<model_id>/showobjdetectemodel', methods=['GET', 'POST'])
@login_required
def showobjdetectemodel(model_id):
    profile = BaseController.get_model_status(model_id)
    objectdetectionfactory = ObjectDetectionFactory()

    return objectdetectionfactory.showobjdetectrmodeldashboard(profile)

@blueprint.route('<model_id>/objtdetect/detect', methods=['GET', 'POST'])
@login_required
def detect(model_id):
    objectdetectionfactory = ObjectDetectionFactory()
    return objectdetectionfactory.detectobjects(model_id, request)

@blueprint.route('<model_id>/<run_id>/downloadresults', methods=['GET', 'POST'])
def downloadresults(model_id, run_id):
    f = '{0}_{1}{2}'.format(model_id, run_id, '.zip')
    path = os.path.join(results_path, f)
    return send_file(path, as_attachment=True)


