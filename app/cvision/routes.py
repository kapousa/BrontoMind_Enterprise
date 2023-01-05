# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import request
from flask_login import login_required

from app.base.app_routes.directors.cvision.ObjectDetectionDirector import ObjectDetectionDirector
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

@blueprint.route('objtdetect/detect', methods=['GET', 'POST'])
@login_required
def detect():
    aa = ObjectDetectionCotroller()
    cc = aa.lable_files()
    objectdetectionfactory = ObjectDetectionFactory()
    return objectdetectionfactory.showobjdetectrmodeldashboard(profile)



